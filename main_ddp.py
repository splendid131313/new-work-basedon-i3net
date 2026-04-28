import config

args, unparsed = config.get_args()

import os
import random
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import wandb

from torch.utils.data.distributed import DistributedSampler

from data import trainSet

# from data_prostate import trainSet
from util_evaluation import calc_psnr, calc_ssim
from select_model import select_model
import optim
from select_loss import compute_reprojection_loss, compute_consistency_loss


def main():
    dist.init_process_group(backend="nccl")

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    is_main = rank == 0

    GLOBAL_SEED = 777
    random.seed(GLOBAL_SEED + rank)
    np.random.seed(GLOBAL_SEED + rank)
    torch.manual_seed(GLOBAL_SEED + rank)
    torch.cuda.manual_seed(GLOBAL_SEED + rank)
    torch.cuda.manual_seed_all(GLOBAL_SEED + rank)

    args.ckpt_dir = "experiments/" + args.ckpt_dir
    if is_main:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    trainset = trainSet(data_root=args.traindata_path, args=args)
    train_sampler = DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    model = select_model(args)

    if args.resume:
        if is_main:
            print("load weight")
        load_ckpt = torch.load(args.ckpt, map_location="cpu")
        state = load_ckpt["state_dict"]
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        if is_main:
            print("load weight success")
        args.start_epoch = load_ckpt["epoch"]

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.select_optim(args, model)
    scheduler = optim.select_scheduler(args, optimizer)
    loss_reproj = compute_reprojection_loss
    loss_cons = compute_consistency_loss

    # if is_main:
    #     wandb.init(
    #         project="i3net",
    #         name="flow_module",
    #         config=args.__dict__,
    #     )

    # amp
    use_amp = args.amp
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
    else:
        scaler = None
        autocast = None

    model.train()
    for epoch in range(args.start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch)

        loss_epoch = 0
        psnr_local_epoch = 0
        psnr_global_epoch = 0
        psnr_pred_local_epoch = 0
        psnr_pred_global_epoch = 0

        if is_main:
            loader_iter = tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            loader_iter = enumerate(dataloader)

        for iter, hr in loader_iter:
            # hr: [B, N, H, W, S] 或 [B, H, W, S]
            if len(hr.shape) == 5:
                gt = torch.cat([i for i in hr], 0)  # [B, H, W, S]
            else:
                gt = hr
            lr = gt[..., :: args.upscale]  # [B, H, W, lr_slice_patch]

            gt = gt.to(device, non_blocking=True)
            lr = lr.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    out_local, out_global = model(lr)
                    loss_iter = loss_reproj(out_local, gt) + loss_reproj(out_global, gt) + loss_cons(out_local, out_global)
                    loss = loss_iter
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_local, out_global = model(lr)
                loss_iter = loss_reproj(out_local, gt) + loss_reproj(out_global, gt) + loss_cons(out_local, out_global)
                loss = loss_iter
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                psnr_local_iter = 0.0
                psnr_global_iter = 0.0
                psnr_pred_local_iter = 0.0
                psnr_pred_global_iter = 0.0
                for bz in range(gt.shape[0]):
                    psnr_local_iter += calc_psnr(gt[bz, :, :, :], out_local[bz, :, :, :]).item()
                    psnr_global_iter += calc_psnr(gt[bz, :, :, :], out_global[bz, :, :, :]).item()
                    psnr_pred_local_iter += calc_psnr(
                        gt[bz, :, :, 1 :: args.upscale],
                        out_local[bz, :, :, 1 :: args.upscale],
                    ).item()
                    psnr_pred_global_iter += calc_psnr(
                        gt[bz, :, :, 1 :: args.upscale],
                        out_global[bz, :, :, 1 :: args.upscale],
                    ).item()
                psnr_local_iter /= gt.shape[0]
                psnr_global_iter /= gt.shape[0]
                psnr_pred_local_iter /= gt.shape[0]
                psnr_pred_global_iter /= gt.shape[0]

            loss_epoch += loss_iter.detach().item()
            psnr_local_epoch += psnr_local_iter
            psnr_global_epoch += psnr_global_iter
            psnr_pred_local_epoch += psnr_pred_local_iter
            psnr_pred_global_epoch += psnr_pred_global_iter

            if is_main:
                lr_tmp = optimizer.state_dict()["param_groups"][0]["lr"]
                log = (
                    f"epoch[{epoch + 1}/{args.max_epoch}] "
                    f"iter[{iter + 1}/{len(dataloader)}] "
                    f"psnrTrLocal:{psnr_local_iter:.6f} psnrTrGlobal:{psnr_global_iter:.6f} psnrPredLocal:{psnr_pred_local_iter:.6f} psnrPredGlobal:{psnr_pred_global_iter:.6f} lossTr:{loss_epoch:.12f} lr:{lr_tmp:.12f}"
                )
                now = str(datetime.datetime.now())
                print(now + " " + log)

        loss_tensor = torch.tensor(loss_epoch, device=device)
        psnr_local_tensor = torch.tensor(psnr_local_epoch, device=device)
        psnr_global_tensor = torch.tensor(psnr_global_epoch, device=device)
        psnr_pred_local_tensor = torch.tensor(psnr_pred_local_epoch, device=device)
        psnr_pred_global_tensor = torch.tensor(psnr_pred_global_epoch, device=device)
        count_tensor = torch.tensor(len(dataloader), device=device, dtype=torch.float32)

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_local_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_global_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_pred_local_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_pred_global_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        loss_epoch = (loss_tensor / count_tensor).item()
        psnr_local_epoch = (psnr_local_tensor / count_tensor).item()
        psnr_global_epoch = (psnr_global_tensor / count_tensor).item()
        psnr_pred_local_epoch = (psnr_pred_local_tensor / count_tensor).item()
        psnr_pred_global_epoch = (psnr_pred_global_tensor / count_tensor).item()

        if args.schedule == "step":
            scheduler.step()
        elif args.schedule == "cos_lr":
            scheduler.step_update(epoch)
        elif args.schedule == "Tmin":
            scheduler.step(loss_epoch)
        elif args.schedule == "Tmax":
            scheduler.step(psnr_local_epoch)

        if is_main:
            lr_tmp = optimizer.state_dict()["param_groups"][0]["lr"]

            with torch.no_grad():
                b, h, w, s = gt.shape
                mid_s = s // 2
                out_local = torch.clamp(out_local, 0, 1)
                out_global = torch.clamp(out_global, 0, 1)
                gt = torch.clamp(gt, 0, 1)
                out_local_mid = out_local[0, :, :, mid_s].detach().cpu().float().numpy()
                out_global_mid = (
                    out_global[0, :, :, mid_s].detach().cpu().float().numpy()
                )
                gt_mid = gt[0, :, :, mid_s].detach().cpu().float().numpy()

                wandb.log(
                    {
                        "train/psnr_local_epoch": psnr_local_epoch,
                        "train/psnr_global_epoch": psnr_global_epoch,
                        "train/psnr_pred_local_epoch": psnr_pred_local_epoch,
                        "train/psnr_pred_global_epoch": psnr_pred_global_epoch,
                        "train/loss_epoch": loss_epoch,
                        "train/lr": lr_tmp,
                        "epoch": epoch,
                        "vis/out_local": wandb.Image(out_local_mid, caption="Local pred"),
                        "vis/out_global": wandb.Image(out_global_mid, caption="Global pred"),
                        "vis/gt": wandb.Image(gt_mid, caption="GT"),
                    }
                )

            log = (
                f"epoch[{epoch + 1}/{args.max_epoch}] "
                f"psnrTrLocal:{psnr_local_epoch:.6f} psnrTrGlobal:{psnr_global_epoch:.6f} psnrPredLocal:{psnr_pred_local_epoch:.6f} psnrPredGlobal:{psnr_pred_global_epoch:.6f} lossTr:{loss_epoch:.12f} lr:{lr_tmp:.12f}"
            )
            now = str(datetime.datetime.now())
            print(now + " " + log)

            if epoch + 1 > int(0.99 * args.max_epoch):
                os.makedirs(args.ckpt_dir + "/pth", exist_ok=True)
                state_dict = model.module.state_dict()  # DDP
                torch.save(
                    {"epoch": epoch + 1, "state_dict": state_dict},
                    args.ckpt_dir + "/pth/" + str(epoch + 1).zfill(4) + ".pth",
                )

    # if is_main:
    #     wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
