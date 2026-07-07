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
from select_loss import Select_Loss


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
    loss_function = Select_Loss(args)

    if is_main:
        wandb.init(
            project="multi_random",
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

    best_psnr = 0
    last_80_start = int(0.7 * args.max_epoch)

    model.train()
    for epoch in range(args.start_epoch, args.max_epoch):
        train_sampler.set_epoch(epoch)

        loss_epoch = 0
        psnr_epoch = 0

        if is_main:
            loader_iter = tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            loader_iter = enumerate(dataloader)

        for iter, (lr, gt, t) in loader_iter:
            lr = lr.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    out = model(lr, t)
                    loss_iter = loss_function(out, gt)
                    loss = loss_iter
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(lr, t)
                loss_iter = loss_function(out, gt)
                loss = loss_iter
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                psnr_iter = calc_psnr(gt, out).item()

            loss_epoch += loss_iter.detach().item()
            psnr_epoch += psnr_iter

            if is_main:
                lr_tmp = optimizer.state_dict()["param_groups"][0]["lr"]
                log = (
                    f"epoch[{epoch + 1}/{args.max_epoch}] "
                    f"iter[{iter + 1}/{len(dataloader)}] "
                    f"psnr:{psnr_iter:.6f} loss:{loss_epoch:.12f} lr:{lr_tmp:.12f}"
                )
                now = str(datetime.datetime.now())
                print(now + " " + log)

        loss_tensor = torch.tensor(loss_epoch, device=device)
        psnr_tensor = torch.tensor(psnr_epoch, device=device)
        count_tensor = torch.tensor(len(dataloader), device=device, dtype=torch.float32)

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(psnr_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        loss_epoch = (loss_tensor / count_tensor).item()
        psnr_epoch = (psnr_tensor / count_tensor).item()

        if args.schedule == "step":
            scheduler.step()
        elif args.schedule == "cos_lr":
            scheduler.step_update(epoch)
        elif args.schedule == "Tmin":
            scheduler.step(loss_epoch)
        elif args.schedule == "Tmax":
            scheduler.step(loss_epoch)

        if is_main:
            lr_tmp = optimizer.state_dict()["param_groups"][0]["lr"]

            with torch.no_grad():
                out = torch.clamp(out, 0, 1)
                gt = torch.clamp(gt, 0, 1)

                wandb.log(
                    {
                        "train/psnr_epoch": psnr_epoch,
                        "train/loss_epoch": loss_epoch,
                        "train/lr": lr_tmp,
                        "epoch": epoch,
                        "vis/out": wandb.Image(out, caption="Pred"),
                        "vis/gt": wandb.Image(gt, caption="GT"),
                    }
                )

            log = (
                f"epoch[{epoch + 1}/{args.max_epoch}] "
                f"psnr:{psnr_epoch:.6f} loss:{loss_epoch:.12f} lr:{lr_tmp:.12f}"
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

    if is_main:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
