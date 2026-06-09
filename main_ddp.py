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
from select_loss import TotalLoss


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

    wandb_name = args.ckpt_dir

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
    loss_function = TotalLoss(args, device=device).to(device)

    if is_main:
        wandb.init(
            project="i3net_flowseek",
            name=wandb_name,
            config=args.__dict__,
        )

    # amp
    use_amp = args.amp
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
    else:
        scaler = None
        autocast = None

    best_psnr = 0.0
    last_80_start = int(0.8 * args.max_epoch)
    last_99_start = int(0.99 * args.max_epoch)

    model.train()
    for epoch in range(args.start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch)

        loss_iter_epoch = 0
        psnr_epoch = 0
        psnr_pred_epoch = 0

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
                    sr, flow = model(lr)
                with autocast(enabled=False):
                    loss_iter = loss_function(sr, gt)
                    loss = loss_iter
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sr, flow = model(lr)
                loss_iter = loss_function(sr, gt)
                loss = loss_iter
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                psnr_iter = 0.0
                psnr_pred_iter = 0.0
                for bz in range(gt.shape[0]):
                    psnr_iter += calc_psnr(gt[bz, :, :, :], sr[bz, :, :, :]).item()
                    # TODO:
                    # upscale > 2的时候，这里的索引是不合理的
                    psnr_pred_iter += calc_psnr(gt[bz, :, :, 1::args.upscale], sr[bz, :, :, 1::args.upscale]).item()
                psnr_iter /= gt.shape[0]
                psnr_pred_iter /= gt.shape[0]

            loss_iter_epoch += loss_iter.detach().item()
            psnr_epoch += psnr_iter
            psnr_pred_epoch += psnr_pred_iter

            if is_main:
                lr_tmp = optimizer.state_dict()["param_groups"][0]["lr"]
                log = (
                    f"epoch[{epoch + 1}/{args.max_epoch}] "
                    f"iter[{iter + 1}/{len(dataloader)}] "
                    f"psnrTr:{psnr_iter:.6f} psnrPred:{psnr_pred_iter:.6f} lossTr:{loss_iter_epoch:.12f} lr:{lr_tmp:.12f}"
                )
                now = str(datetime.datetime.now())
                print(now + " " + log)

        loss_iter_tensor = torch.tensor(loss_iter_epoch, device=device)
        psnr_tensor = torch.tensor(psnr_epoch, device=device)
        psnr_pred_tensor = torch.tensor(psnr_pred_epoch, device=device)
        count_tensor = torch.tensor(len(dataloader), device=device, dtype=torch.float32)

        dist.all_reduce(loss_iter_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_pred_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        loss_iter_epoch = (loss_iter_tensor / count_tensor).item()
        loss_epoch = loss_iter_epoch
        psnr_epoch = (psnr_tensor / count_tensor).item()
        psnr_pred_epoch = (psnr_pred_tensor / count_tensor).item()

        if args.schedule == "step":
            scheduler.step()
        elif args.schedule == "cos_lr":
            scheduler.step_update(epoch)
        elif args.schedule == "Tmin":
            scheduler.step(loss_epoch)
        elif args.schedule == "Tmax":
            scheduler.step(psnr_epoch)

        if is_main:
            lr_tmp = optimizer.state_dict()["param_groups"][0]["lr"]

            with torch.no_grad():
                b, h, w, s = gt.shape
                mid_s = s // 2
                sr = torch.clamp(sr, 0, 1)
                gt = torch.clamp(gt, 0, 1)
                sr_mid = sr[0, :, :, mid_s].detach().cpu().float().numpy()
                gt_mid = gt[0, :, :, mid_s].detach().cpu().float().numpy()

                wandb.log(
                    {
                        "train/psnr_epoch": psnr_epoch,
                        "train/psnr_pred_epoch": psnr_pred_epoch,
                        "train/loss_epoch": loss_iter_epoch,
                        "train/lr": lr_tmp,
                        "epoch": epoch,
                        "vis/sr_slice": wandb.Image(sr_mid, caption="SR pred"),
                        "vis/gt_slice": wandb.Image(gt_mid, caption="GT"),
                    }
                )

            log = (
                f"epoch[{epoch + 1}/{args.max_epoch}] "
                f"psnrTr:{psnr_epoch:.6f} lossTr:{loss_epoch:.12f} lr:{lr_tmp:.12f}"
            )
            now = str(datetime.datetime.now())
            print(now + " " + log)

            os.makedirs(args.ckpt_dir + "/pth", exist_ok=True)

            if epoch + 1 > last_80_start and psnr_epoch > best_psnr:
                best_psnr = psnr_epoch
                state_dict = model.module.state_dict()
                torch.save(
                    state_dict,
                    args.ckpt_dir + "/pth/best_{:04d}.pth".format(epoch + 1),
                )
                print(f"Saved best checkpoint at epoch {epoch + 1}, psnr={best_psnr:.6f}")

            if epoch + 1 > last_99_start:
                state_dict = model.module.state_dict()
                torch.save(state_dict,
                           args.ckpt_dir + "/pth/" + str(epoch + 1).zfill(4) + ".pth",
                           )

    if is_main:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
