import config

args, unparsed = config.get_args()

import os
import random
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from data import trainSet

# from data_prostate import trainSet
from util_evaluation import calc_psnr, calc_ssim
from select_model import select_model
import optim
from select_loss import Select_Loss
from model_zoo.flowseek.core.utils.flow_viz import flow_to_image


def to_wandb_image(x):
    """Convert [B,H,W,C] / [B,C,H,W] / [H,W] tensor to HxW or HxWx3 uint8 for wandb.

    wandb/PIL 吃不了原始 [B,H,W,1]：会把 batch/通道维搞乱，最后变成类似
    (1,1,256) 这种非法图像 shape。这里统一抽第 0 个样本并压成 HxW / HxWx3。
    """
    x = x.detach().float().cpu()
    if x.ndim == 4:
        # prefer HWC (model output); fallback to CHW
        if x.shape[-1] in (1, 3):
            x = x[0, ..., 0] if x.shape[-1] == 1 else x[0]
        else:
            x = x[0, 0] if x.shape[1] == 1 else x[0].permute(1, 2, 0)
    elif x.ndim == 3:
        if x.shape[-1] == 1:
            x = x[..., 0]
        elif x.shape[0] == 1:
            x = x[0]
    x = x.clamp(0, 1).numpy()
    if x.ndim == 2:
        return (x * 255.0).astype(np.uint8)
    return (x * 255.0).astype(np.uint8)


def flow_to_wandb_image(flow):
    """flow: [B,2,H,W] or [2,H,W] -> HxWx3 uint8 color wheel."""
    flow = flow.detach().float().cpu()
    if flow.ndim == 4:
        flow = flow[0]
    flow_np = flow.permute(1, 2, 0).numpy()  # H,W,2
    return flow_to_image(flow_np)


def sample_caption(t, meta, sample_idx=0):
    """Build caption: n_mid / mid_idx / t / endpoint gap."""
    t0 = t[sample_idx].detach().float().cpu()
    n_mid = int(meta[sample_idx, 0].item())
    mid_idx = int(meta[sample_idx, 1].item())
    t_val = float(t0[0].item())
    gap = float(t0[1].item())
    return (
        f"n_mid={n_mid}, mid_idx={mid_idx}/{n_mid}, "
        f"t={t_val:.4f}, gap={gap:g}"
    )


def main():
    device = torch.device("cuda" if args.cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    GLOBAL_SEED = 777
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(GLOBAL_SEED)
        torch.cuda.manual_seed_all(GLOBAL_SEED)

    wandb_name = args.ckpt_dir

    args.ckpt_dir = "experiments/" + args.ckpt_dir
    os.makedirs(args.ckpt_dir, exist_ok=True)

    trainset = trainSet(data_root=args.traindata_path, args=args)

    dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = select_model(args)

    if args.resume:
        print("load weight")
        load_ckpt = torch.load(args.ckpt, map_location="cpu")
        state = load_ckpt["state_dict"] if isinstance(load_ckpt, dict) and "state_dict" in load_ckpt else load_ckpt
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        print("load weight success")
        if isinstance(load_ckpt, dict) and "epoch" in load_ckpt:
            args.start_epoch = load_ckpt["epoch"]

    model = model.to(device)

    optimizer = optim.select_optim(args, model)
    scheduler = optim.select_scheduler(args, optimizer)
    loss_function = Select_Loss(args)

    wandb.init(
        project="multi_random",
        name=wandb_name,
        config=args.__dict__,
    )

    # amp
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
    else:
        scaler = None
        autocast = None

    best_psnr = 0
    last_80_start = int(0.7 * args.max_epoch)

    model.train()
    for epoch in range(args.start_epoch, args.max_epoch):
        loss_epoch = 0
        psnr_epoch = 0

        loader_iter = tqdm(enumerate(dataloader), total=len(dataloader))

        for iter, (lr, gt, t, meta) in loader_iter:
            lr = lr.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            if lr.ndim == 5:
                B, N, H, W, C = lr.shape
                lr = lr.view(B * N, H, W, C)
                gt = gt.view(B * N, H, W, gt.shape[-1])
                t = t.view(B * N, t.shape[-1])
                meta = meta.view(B * N, meta.shape[-1])

            optimizer.zero_grad()

            flow_fn = model.flow_estimator._get_base_flow
            if use_amp:
                with autocast():
                    model_out = model(lr, t)
                    loss_iter = loss_function(
                        model_out, gt, cond=t, lr=lr, flow_fn=flow_fn
                    )
                    out = model_out["out"] if isinstance(model_out, dict) else model_out
                    loss = loss_iter
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                model_out = model(lr, t)
                loss_iter = loss_function(
                    model_out, gt, cond=t, lr=lr, flow_fn=flow_fn
                )
                out = model_out["out"] if isinstance(model_out, dict) else model_out
                loss = loss_iter
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                psnr_iter = calc_psnr(gt, out).item()

            loss_epoch += loss_iter.detach().item()
            psnr_epoch += psnr_iter

            lr_tmp = optimizer.state_dict()["param_groups"][0]["lr"]
            n_mid = int(meta[0, 0].item())
            mid_idx = int(meta[0, 1].item())
            log = (
                f"epoch[{epoch + 1}/{args.max_epoch}] "
                f"iter[{iter + 1}/{len(dataloader)}] "
                f"n_mid:{n_mid} mid:{mid_idx} "
                f"psnr:{psnr_iter:.6f} loss:{loss_epoch:.12f} lr:{lr_tmp:.12f}"
            )
            now = str(datetime.datetime.now())
            print(now + " " + log)

        n_iter = max(len(dataloader), 1)
        loss_epoch = loss_epoch / n_iter
        psnr_epoch = psnr_epoch / n_iter

        if args.schedule == "step":
            scheduler.step()
        elif args.schedule == "cos_lr":
            scheduler.step_update(epoch)
        elif args.schedule == "Tmin":
            scheduler.step(loss_epoch)
        elif args.schedule == "Tmax":
            scheduler.step(loss_epoch)

        lr_tmp = optimizer.state_dict()["param_groups"][0]["lr"]

        with torch.no_grad():
            cap = sample_caption(t, meta, sample_idx=0)
            log_dict = {
                "train/psnr_epoch": psnr_epoch,
                "train/loss_epoch": loss_epoch,
                "train/lr": lr_tmp,
                "epoch": epoch,
                "vis/out": wandb.Image(to_wandb_image(out), caption=f"Pred | {cap}"),
                "vis/gt": wandb.Image(to_wandb_image(gt), caption=f"GT | {cap}"),
            }
            if isinstance(model_out, dict):
                if "img0" in model_out:
                    log_dict["vis/img0"] = wandb.Image(
                        to_wandb_image(model_out["img0"]), caption=f"LR0 | {cap}"
                    )
                if "img1" in model_out:
                    log_dict["vis/img1"] = wandb.Image(
                        to_wandb_image(model_out["img1"]), caption=f"LR1 | {cap}"
                    )
                if "warped0" in model_out:
                    log_dict["vis/warped0"] = wandb.Image(
                        to_wandb_image(model_out["warped0"]), caption=f"Warp0 | {cap}"
                    )
                if "warped1" in model_out:
                    log_dict["vis/warped1"] = wandb.Image(
                        to_wandb_image(model_out["warped1"]), caption=f"Warp1 | {cap}"
                    )
                if "flow0t" in model_out:
                    log_dict["vis/flow0t"] = wandb.Image(
                        flow_to_wandb_image(model_out["flow0t"]),
                        caption=f"Flow0t | {cap}",
                    )
                if "flow1t" in model_out:
                    log_dict["vis/flow1t"] = wandb.Image(
                        flow_to_wandb_image(model_out["flow1t"]),
                        caption=f"Flow1t | {cap}",
                    )
                if "mask" in model_out:
                    log_dict["vis/mask"] = wandb.Image(
                        to_wandb_image(model_out["mask"]), caption=f"Mask | {cap}"
                    )
            wandb.log(log_dict)

        log = (
            f"epoch[{epoch + 1}/{args.max_epoch}] "
            f"psnr:{psnr_epoch:.6f} loss:{loss_epoch:.12f} lr:{lr_tmp:.12f}"
        )
        now = str(datetime.datetime.now())
        print(now + " " + log)

        os.makedirs(args.ckpt_dir + "/pth", exist_ok=True)

        if epoch + 1 > last_80_start and psnr_epoch > best_psnr:
            best_psnr = psnr_epoch
            state_dict = model.state_dict()
            torch.save(
                state_dict,
                args.ckpt_dir + "/pth/best_{:04d}.pth".format(epoch + 1),
            )
            print(f"Saved best checkpoint at epoch {epoch + 1}, psnr={best_psnr:.6f}")

    wandb.finish()


if __name__ == "__main__":
    main()
