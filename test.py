import os 
import config
args, unparsed = config.get_args()

import numpy as np
import torch
from torchvision.utils import save_image
from data import testSet
from util_evaluation import calc_psnr,calc_ssim
from select_model import select_model


def main():
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f'device: {device}')

    args.ckpt_dir = 'experiments/'+args.model+'/'+args.ckpt_dir
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        s = "\n\n\n\n\nSTART EXPERIMENT\n"
        f.write(s)
        f.write('testdata:'+args.testdata_path+'\n')
        f.write('checkpoint:'+args.ckpt+'\n')

    model = select_model(args)
    checkpoint = torch.load(args.ckpt, map_location=device)
    if 'module' in checkpoint['state_dict']:
        checkpoint['state_dict'] = checkpoint['state_dict']['module']
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    print(f'load:{args.ckpt}')
    model.eval()

    testset = testSet(data_root=args.testdata_path)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1,
    drop_last=False, shuffle=False, num_workers=4, pin_memory=False)

    average_psnr=0
    average_psnr_x_y=0
    average_psnr_x_z=0
    average_psnr_y_z=0
    total_x_y_ssim=0
    total_x_z_ssim=0
    total_y_z_ssim=0

    for id, (name,volume) in enumerate(dataloader):
        # volume [bz=1,h,w,s]
        gt = volume.squeeze(0) #[h,w,s]
        psnr=0
        x_y_psnr=0
        x_z_psnr=0
        y_z_psnr=0
        x_y_ssim=0 
        x_z_ssim=0
        y_z_ssim=0

        m = (gt.shape[2]-1) % args.upscale 
        if m != 0:
            gt = gt[...,:-m]
        lr = gt[...,::args.upscale]
        time_list = args.time_list[1:-1]

        # prepare save dirs for sr and I_t
        save_root = os.path.join(args.ckpt_dir, 'results')
        os.makedirs(save_root, exist_ok=True)
        if isinstance(name, (list, tuple)):
            name_str = str(name[0])
        else:
            name_str = str(name)
        name_str = os.path.splitext(os.path.basename(name_str))[0]
        case_dir = os.path.join(save_root, name_str)
        os.makedirs(case_dir, exist_ok=True)
        xy_dir = os.path.join(case_dir, 'xy')
        xz_dir = os.path.join(case_dir, 'xz')
        yz_dir = os.path.join(case_dir, 'yz')
        os.makedirs(xy_dir, exist_ok=True)
        os.makedirs(xz_dir, exist_ok=True)
        os.makedirs(yz_dir, exist_ok=True)

        sr = torch.zeros_like(gt)
        sr_cnt = torch.zeros_like(gt)

        for tmp_s in range(lr.shape[2]-args.lr_slice_patch+1):
            tmp_lr = lr[...,tmp_s:tmp_s+args.lr_slice_patch]
            tmp_lr = tmp_lr.unsqueeze(0).to(device) #[1,s,h,w]
            with torch.no_grad():
                tmp_sr = model(tmp_lr)

            tmp_sr = torch.clamp(tmp_sr.squeeze(0),0,1).cpu()
            sr[...,tmp_s*args.upscale:tmp_s*args.upscale+((args.lr_slice_patch-1)*args.upscale+1)] += tmp_sr
            sr_cnt[...,tmp_s*args.upscale:tmp_s*args.upscale+((args.lr_slice_patch-1)*args.upscale+1)] += 1

        sr = sr[...,args.upscale : -1*args.upscale]
        sr_cnt = sr_cnt[...,args.upscale : -1*args.upscale]
        gt = gt[...,args.upscale : -1*args.upscale]

        sr /= sr_cnt #[h,w,s]

        # save output slices as images (sr): xy, xz, yz 三个方向
        def _save_slice(slice_2d, save_path):
            slice_i = slice_2d.unsqueeze(0)  # [1, H, W]
            s_min, s_max = slice_i.min(), slice_i.max()
            if s_max > s_min:
                slice_i = (slice_i - s_min) / (s_max - s_min)
            save_image(slice_i, save_path)

        sr_cpu = sr.detach().cpu()
        # 保存 SR 体积 [H, W, S]，与 testSet 中 np.load 的轴顺序一致
        np.save(os.path.join(case_dir, 'sr.npy'), sr_cpu.numpy().astype(np.float32))

        # xy 方向 (axial)
        for i in range(sr_cpu.shape[2]):
            _save_slice(sr_cpu[:, :, i], os.path.join(xy_dir, f'{i:03d}.png'))
        # xz 方向 (sagittal)
        for i in range(sr_cpu.shape[0]):
            _save_slice(sr_cpu[i, :, :], os.path.join(xz_dir, f'{i:03d}.png'))
        # yz 方向 (coronal)
        for i in range(sr_cpu.shape[1]):
            _save_slice(sr_cpu[:, i, :], os.path.join(yz_dir, f'{i:03d}.png'))

        # volume-level PSNR（保持原有计算）
        sr = sr.to(device)
        gt = gt.to(device)
        print("sr.shape:", sr.shape)
        print("gt.shape:", gt.shape)
        psnr = calc_psnr(sr, gt).item()

        # slice-level PSNR（新增）
        for i in range(gt.shape[2]):
            x_y_psnr += calc_psnr(sr[:,:,i], gt[:,:,i]).item()
        x_y_psnr /= gt.shape[2]
        for i in range(gt.shape[0]):
            x_z_psnr += calc_psnr(sr[i,:,:], gt[i,:,:]).item()
        x_z_psnr /= gt.shape[0]
        for i in range(gt.shape[1]):
            y_z_psnr += calc_psnr(sr[:,i,:], gt[:,i,:]).item()
        y_z_psnr /= gt.shape[1]

        for i in range(gt.shape[2]):
            x_y_ssim += calc_ssim(gt[:,:,i], sr[:,:,i])
        x_y_ssim /= gt.shape[2]
        for i in range(gt.shape[0]):
            x_z_ssim += calc_ssim(gt[i,:,:], sr[i,:,:])
        x_z_ssim /= gt.shape[0]
        for i in range(gt.shape[1]):
            y_z_ssim += calc_ssim(gt[:,i,:], sr[:,i,:])
        y_z_ssim /= gt.shape[1]

        log = r"[{} / {}] NAME:{} PSNR:{:.2f} x_y_psnr:{:.2f} x_z_psnr:{:.2f} y_z_psnr:{:.2f} x_y_ssim:{:.4f} x_z_ssim:{:.4f} y_z_ssim:{:.4f} "\
            .format(id+1,dataloader.__len__(),name,psnr,x_y_psnr,x_z_psnr,y_z_psnr,x_y_ssim,x_z_ssim,y_z_ssim)
        print(log)
        with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
            f.write(log+'\n') 

        average_psnr += psnr
        average_psnr_x_y += x_y_psnr
        average_psnr_x_z += x_z_psnr
        average_psnr_y_z += y_z_psnr
        total_x_y_ssim += x_y_ssim
        total_x_z_ssim += x_z_ssim
        total_y_z_ssim += y_z_ssim

    n = id + 1
    average_psnr /= n
    average_psnr_x_y /= n
    average_psnr_x_z /= n
    average_psnr_y_z /= n
    total_x_y_ssim /= n
    total_x_z_ssim /= n
    total_y_z_ssim /= n
    print("average_psnr (volume):", average_psnr)
    print("average_x_y_psnr (slice):", average_psnr_x_y)
    print("average_x_z_psnr (slice):", average_psnr_x_z)
    print("average_y_z_psnr (slice):", average_psnr_y_z)
    print("average_x_y_ssim:", total_x_y_ssim)
    print("average_x_z_ssim:", total_x_z_ssim)
    print("average_y_z_ssim:", total_y_z_ssim)

    with open(args.ckpt_dir + '/logs_test.txt', mode='a+') as f:
        log = r"PSNR: {:.4f} x_y_psnr: {:.4f} x_z_psnr: {:.4f} y_z_psnr: {:.4f} x_y_ssim: {:.6f} x_z_ssim: {:.6f} y_z_ssim: {:.6f}".format(
            average_psnr, average_psnr_x_y, average_psnr_x_z, average_psnr_y_z, total_x_y_ssim, total_x_z_ssim, total_y_z_ssim)
        f.write(log)


if __name__ == "__main__":
    main()
