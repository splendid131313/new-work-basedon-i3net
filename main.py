import config

args, unparsed = config.get_args()

import os
import random
import datetime
import numpy as np
from tqdm import tqdm

import torch

from data import trainSet, collate_train_spans

# from data_prostate import trainSet
from util_evaluation import calc_psnr, calc_ssim
from select_model import select_model
import optim
from select_loss import TotalLoss


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
        collate_fn=collate_train_spans,
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
    loss_function = TotalLoss(args, device=device).to(device)

    # log
    with open(args.ckpt_dir + '/logs.txt', mode='a+') as f:
        s = "START EXPERIMENT\n"
        f.write(s)
        for i, j in args.__dict__.items():
            f.write(str(i) + ' : ' + str(j) + '\n')

    # amp
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast
    else:
        scaler = None
        autocast = None

    best_psnr = 0
    last_80_start = int(0.8 * args.max_epoch)
    last_99_start = int(0.99 * args.max_epoch)

    model.train()
    for epoch in range(args.start_epoch, args.max_epoch):
        loss_epoch = 0
        psnr_epoch = 0

        loader_iter = tqdm(enumerate(dataloader), total=len(dataloader))

        for iter, (lr, gt, t, span_ids) in loader_iter:
            lr = lr.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    model_out = model(lr, t)
                    out = model_out["out"] if isinstance(model_out, dict) else model_out
                    loss_iter = loss_function(out, gt)
                    loss = loss_iter
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                model_out = model(lr, t)
                out = model_out["out"] if isinstance(model_out, dict) else model_out
                loss_iter = loss_function(out, gt)
                loss = loss_iter
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                psnr_iter = calc_psnr(gt, out).item()

            loss_epoch += loss_iter.detach().item()
            psnr_epoch += psnr_iter

            lr_tmp = optimizer.state_dict()["param_groups"][0]["lr"]
            gap = int(round(t[0, 1].item()))
            mid_idx = int(round(t[0, 0].item() * gap))
            log = (
                f"epoch[{epoch + 1}/{args.max_epoch}] "
                f"iter[{iter + 1}/{len(dataloader)}] "
                f"gap:{gap} mid:{mid_idx} queries:{t.shape[0]} "
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

        log = (
            f"epoch[{epoch + 1}/{args.max_epoch}] "
            f"psnr:{psnr_epoch:.6f} loss:{loss_epoch:.12f} lr:{lr_tmp:.12f}"
        )
        now = str(datetime.datetime.now())
        print(now + " " + log)
        with open(args.ckpt_dir + '/logs.txt', mode='a+') as f:
            f.write('\n' + now + log)

        os.makedirs(args.ckpt_dir + "/pth", exist_ok=True)

        if epoch + 1 > last_80_start and psnr_epoch > best_psnr:
            best_psnr = psnr_epoch
            state_dict = model.state_dict()
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


if __name__ == "__main__":
    main()
