import config
args, unparsed = config.get_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
from tqdm import tqdm
import wandb

from data import trainSet
# from data_prostate import trainSet
from util_evaluation import calc_psnr, calc_ssim
import datetime
from select_model import select_model
import optim
from select_loss import Select_Loss

####################################################################
# seed
GLOBAL_SEED = 777
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)

wandb_name = args.ckpt_dir

args.ckpt_dir = 'experiments/' + args.ckpt_dir
os.makedirs(args.ckpt_dir, exist_ok=True)

args.parallel = len(args.gpu_id.split(',')) > 1

# data
trainset = trainSet(data_root=args.traindata_path,args=args)
# batch_size = args.batch_size*len(device_ids)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,\
                shuffle=False, num_workers=args.num_workers, pin_memory=False)

# model
model = select_model(args)

if args.resume:
    print('load weight')
    load_ckpt = torch.load(args.ckpt, map_location=torch.device('cpu'))
    state = load_ckpt['state_dict']
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    print('load weight success')
    args.start_epoch = load_ckpt['epoch']

if args.parallel:
    model = nn.DataParallel(model)
model = model.cuda()

#### optim ####
optimizer = optim.select_optim(args,model)
# optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
scheduler = optim.select_scheduler(args,optimizer)

#### loss ####
loss_function = Select_Loss(args).cuda()

########################### train ###################################
# log
wandb.init(
    project="new backbone",
    name=wandb_name,
    config=args.__dict__,
)

# amp
if args.amp:
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

best_psnr = 0.0
last_80_start = int(0.8 * args.max_epoch)
last_99_start = int(0.99 * args.max_epoch)

model.train()
for epoch in tqdm(range(args.start_epoch,args.max_epoch)):
    loss_iter_epoch = 0
    psnr_epoch = 0
    psnr_pred_epoch = 0

    for iter, hr in tqdm(enumerate(dataloader)):
                 
        if len(hr.shape) == 5:
            gt = torch.cat([i for i in hr],0) #[bz,h,w,7]
        else: gt = hr
        lr = gt[...,::args.upscale] # [bz,h,w,4]
        if torch.cuda.is_available():
            gt = Variable(gt.cuda())
            lr = Variable(lr.cuda())
        
        optimizer.zero_grad()

        if args.amp:
            with autocast():
                sr = model(lr)
                loss_iter = loss_function(sr,gt)
                scaler.scale(loss_iter).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            sr = model(lr)
            loss_iter = loss_function(sr,gt)
            loss_iter.backward()
            optimizer.step()

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

        #### log ####    
        lr_tmp = optimizer.state_dict()['param_groups'][0]['lr']
        log = (
            f"epoch[{epoch + 1}/{args.max_epoch}] "
            f"iter[{iter + 1}/{len(dataloader)}] "
            f"psnrTr:{psnr_iter:.6f} psnrPred:{psnr_pred_iter:.6f} lossTr:{loss_iter.item():.12f} lr:{lr_tmp:.12f}"
        )
        now = str(datetime.datetime.now())
        print(now + " " + log)

    num_batches = len(dataloader)
    loss_epoch = loss_iter_epoch / num_batches
    psnr_epoch = psnr_epoch / num_batches
    psnr_pred_epoch = psnr_pred_epoch / num_batches

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
                "train/loss_epoch": loss_epoch,
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

    #### lr schedule ####
    if args.schedule == 'step':
        scheduler.step()
    elif args.schedule == 'cos_lr':
        #### torch.cos_lr ####
        # scheduler.step()
        #### timm.cos_lr ####
        scheduler.step_update(epoch)
    elif args.schedule == 'Tmin':
        scheduler.step(loss_epoch)
    elif args.schedule == 'Tmax':
        scheduler.step(psnr_epoch)

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
        state_dict = model.state_dict()
        torch.save(state_dict,
                    args.ckpt_dir + "/pth/" + str(epoch + 1).zfill(4) + ".pth",
                )


wandb.finish()
