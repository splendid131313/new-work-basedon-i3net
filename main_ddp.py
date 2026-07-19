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
from data import trainSet
import util

from util_evaluation import calc_psnr, calc_ssim
from select_model import select_model
import optim
import select_loss


def main():
    GLOBAL_SEED = 777
    random.seed(GLOBAL_SEED)

    wandb_name = args.ckpt_dir

    args.ckpt_dir = "experiments/" + args.ckpt_dir
    os.makedirs(args.ckpt_dir, exist_ok=True)

    trainset = trainSet(data_root=args.traindata_path, args=args)

    dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=False,
    )

    model = select_model(args)
    model = model.cuda()
    model.weight_cycle = float(args.weight_cycle)

    optimizer = optim.select_optim(args, model)
    scheduler = optim.select_scheduler(args, optimizer)
    criterion_ncc = select_loss.NCC()
    criterion_cha = select_loss.CharbonnierLoss
    criterion_reg = select_loss.Grad3d(penalty="l2")
    criterion_l1n = select_loss.L1_norm()
    epsilon = 1e-3

    wandb.init(
        project="multi_random",
        name=wandb_name,
        config=args.__dict__,
    )

    best_psnr = 0
    last_80_start = int(0.7 * args.max_epoch)

    model.train()
    for epoch in range(args.start_epoch, args.max_epoch):
        loss_all = util.AverageMeter()
        loss_all_full = util.AverageMeter()
        loss_ncc_all_full = util.AverageMeter()
        loss_cha_all_full = util.AverageMeter()
        loss_reg_all_full = util.AverageMeter()
        loss_all_cycle = util.AverageMeter()
        loss_diff_all = util.AverageMeter()

        loss_epoch = 0
        psnr_epoch = 0

        loader_iter = tqdm(enumerate(dataloader), total=len(dataloader))

        for iter, (lr, gt, t) in loader_iter:
            lr = lr.cuda()
            gt = gt.cuda()
            t = t.cuda()

            if lr.ndim == 5:
                B, N, H, W, C = lr.shape
                lr = lr.view(B * N, H, W, C)
                gt = gt.view(B * N, H, W, gt.shape[-1])
                t = t.view(B * N, t.shape[-1])

            optimizer.zero_grad()

            i0 = lr.permute(0, 3, 1, 2)[:, 0:1].contiguous()
            i1 = lr.permute(0, 3, 1, 2)[:, 1:2].contiguous()
            gt_nchw = gt.permute(0, 3, 1, 2).contiguous()

            if args.weight_cycle == 0:
                # supervised: interpolate at t, supervise with middle-slice GT
                model_out = model(lr, t)
                pred = model_out["out"]
                i_0_1 = model_out["i_0_1"]
                i_1_0 = model_out["i_1_0"]
                flow_0_1 = model_out["flow_0_1"]
                flow_1_0 = model_out["flow_1_0"]

                loss_sup_ncc = criterion_ncc(pred, gt_nchw) * args.weight_ncc
                loss_sup_cha = criterion_cha(pred, gt_nchw, eps=epsilon) * args.weight_cha
                loss_ncc_1 = criterion_ncc(i_0_1, i1) * args.weight_ncc
                loss_cha_1 = criterion_cha(i_0_1, i1, eps=epsilon) * args.weight_cha
                loss_reg_1 = criterion_reg(flow_0_1, None)
                loss_ncc_0 = criterion_ncc(i_1_0, i0) * args.weight_ncc
                loss_cha_0 = criterion_cha(i_1_0, i0, eps=epsilon) * args.weight_cha
                loss_reg_0 = criterion_reg(flow_1_0, None)
                loss_full = (
                    loss_sup_ncc
                    + loss_sup_cha
                    + loss_ncc_1
                    + loss_cha_1
                    + loss_reg_1
                    + loss_ncc_0
                    + loss_cha_0
                    + loss_reg_0
                )

                loss_all_full.update(loss_full.item(), gt_nchw.numel())
                loss_ncc_all_full.update(loss_sup_ncc.item(), gt_nchw.numel())
                loss_cha_all_full.update(loss_sup_cha.item(), gt_nchw.numel())
                loss_reg_all_full.update(
                    (loss_reg_0 + loss_reg_1).item(), gt_nchw.numel()
                )

                with torch.no_grad():
                    psnr_iter = calc_psnr(pred, gt_nchw).item()
                psnr_epoch += psnr_iter

                loss_full.backward()
                optimizer.step()
                loss_all.update(loss_full.item(), gt_nchw.numel())
                continue

            # unsupervised cycle path
            model_out = model(lr)
            i_0_1, i_1_0, flow_0_1, flow_1_0, i0_out, i1_out, i0_out_diff, i1_out_diff = (
                model_out
            )

            loss_ncc_1 = criterion_ncc(i_0_1, i1) * args.weight_ncc
            loss_cha_1 = criterion_cha(i_0_1, i1, eps=epsilon) * args.weight_cha
            loss_reg_1 = criterion_reg(flow_0_1, None)
            loss_ncc_0 = criterion_ncc(i_1_0, i0) * args.weight_ncc
            loss_cha_0 = criterion_cha(i_1_0, i0, eps=epsilon) * args.weight_cha
            loss_reg_0 = criterion_reg(flow_1_0, None)
            loss_full = (
                loss_ncc_1
                + loss_cha_1
                + loss_reg_1
                + loss_ncc_0
                + loss_cha_0
                + loss_reg_0
            )

            loss_all_full.update(loss_full.item(), i1.numel())
            loss_ncc_all_full.update(loss_ncc_1.item(), i1.numel())
            loss_cha_all_full.update(loss_cha_1.item(), i1.numel())
            loss_reg_all_full.update(loss_reg_1.item(), i1.numel())
            loss_ncc_all_full.update(loss_ncc_0.item(), i1.numel())
            loss_cha_all_full.update(loss_cha_0.item(), i1.numel())
            loss_reg_all_full.update(loss_reg_0.item(), i1.numel())

            with torch.no_grad():
                psnr_iter = 0.5 * (
                    calc_psnr(i_0_1, i1).item() + calc_psnr(i_1_0, i0).item()
                )
            psnr_epoch += psnr_iter

            loss_diff_0 = criterion_l1n(i0_out_diff)
            loss_diff_1 = criterion_l1n(i1_out_diff)
            loss_diff = (loss_diff_0 + loss_diff_1) * args.weight_diff

            loss_cyc_ncc_0 = criterion_ncc(i0_out, i0) * args.weight_ncc
            loss_cyc_cha_0 = criterion_cha(i0_out, i0, eps=epsilon) * args.weight_cha
            loss_cyc_ncc_1 = criterion_ncc(i1_out, i1) * args.weight_ncc
            loss_cyc_cha_1 = criterion_cha(i1_out, i1, eps=epsilon) * args.weight_cha

            loss_cycle_0 = loss_cyc_ncc_0 + loss_cyc_cha_0
            loss_cycle_1 = loss_cyc_ncc_1 + loss_cyc_cha_1
            loss_cycle = (loss_cycle_0 + loss_cycle_1) * args.weight_cycle

            loss_diff_all.update(loss_diff_0.item(), i1.numel())
            loss_diff_all.update(loss_diff_1.item(), i1.numel())
            loss_all_cycle.update(loss_cycle_0.item(), i1.numel())
            loss_all_cycle.update(loss_cycle_1.item(), i1.numel())

            loss = loss_full + loss_cycle + loss_diff

            loss.backward()
            optimizer.step()

            loss_all.update(loss.item(), i0.numel())

        n_iter = max(len(dataloader), 1)
        psnr_epoch = psnr_epoch / n_iter
        loss_epoch = loss_all.avg

        if args.schedule == "step":
            scheduler.step()
        elif args.schedule == "cos_lr":
            scheduler.step_update(epoch)
        elif args.schedule == "Tmin":
            scheduler.step(loss_epoch)
        elif args.schedule == "Tmax":
            scheduler.step(loss_epoch)

        lr_tmp = optimizer.state_dict()["param_groups"][0]["lr"]

        wandb.log({"Loss_all/train": loss_all.avg})
        wandb.log({"Loss_full/train_all": loss_all_full.avg})
        wandb.log({"Loss_full/train_img_ncc": loss_ncc_all_full.avg})
        wandb.log({"Loss_full/train_img_cha": loss_cha_all_full.avg})
        wandb.log({"Loss_full/train_reg": loss_reg_all_full.avg})
        wandb.log({"Loss_cycle/train_all": loss_all_cycle.avg})
        wandb.log({"Loss_cycle/train_diff": loss_diff_all.avg})
        wandb.log({"Metrics/PSNR": psnr_epoch})

        log = (
            f"epoch[{epoch + 1}/{args.max_epoch}] "
            f"psnr:{psnr_epoch:.6f} loss:{loss_epoch:.12f} lr:{lr_tmp:.12f}"
        )
        now = str(datetime.datetime.now())
        print(now + " " + log)

        os.makedirs(args.ckpt_dir + "/pth", exist_ok=True)

        if epoch + 1 > last_80_start and psnr_epoch > best_psnr:
            best_psnr = psnr_epoch
            state_dict = (
                model.module.state_dict()
                if isinstance(model, DDP)
                else model.state_dict()
            )
            torch.save(
                state_dict,
                args.ckpt_dir + "/pth/best_{:04d}.pth".format(epoch + 1),
            )
            print(f"Saved best checkpoint at epoch {epoch + 1}, psnr={best_psnr:.6f}")

    wandb.finish()



if __name__ == "__main__":
    main()
