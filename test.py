import os 
import config
args, unparsed = config.get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import torch
from torchvision.utils import save_image
import numpy as np
from data import testSet
from util_evaluation import calc_psnr,calc_ssim
from select_model import select_model
import util


def main():
    args.ckpt_dir = 'experiments/'+args.ckpt_dir
    os.makedirs(args.ckpt_dir,exist_ok=True)
    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        s = "\n\n\n\n\nSTART EXPERIMENT\n"
        f.write(s)
        f.write('testdata:'+args.testdata_path+'\n')
        f.write('checkpoint:'+args.ckpt+'\n')

    model = select_model(args)
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and any(
        k.endswith(".weight") for k in checkpoint.keys()
    ):
        state = checkpoint
    else:
        print(f"Error: cannot find weight in ckpt: {checkpoint.keys()}")
        raise KeyError("cannot find weight in ckpt")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f'load:{args.ckpt}')
    model = model.cuda()
    model.eval()

    testset = testSet(data_root=args.testdata_path, image_size=args.image_size)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1,
    drop_last=False, shuffle=False, num_workers=4, pin_memory=False)

    average_psnr=0
    total_x_y_ssim = 0
    total_x_z_ssim = 0
    total_y_z_ssim = 0
    average_psnr_slice=[]
    average_ssim_slice=[]

    for id, (name, volume, vmin, vmax) in enumerate(dataloader):
        # volume [bz=1,h,w,s]
        gt = volume.squeeze(0) #[h,w,s]
        x_y_ssim = 0
        x_z_ssim = 0
        y_z_ssim = 0
        # stats are per-volume; squeeze batch dim
        vmin = float(vmin.squeeze(0).item()) if hasattr(vmin, "squeeze") else float(vmin)
        vmax = float(vmax.squeeze(0).item()) if hasattr(vmax, "squeeze") else float(vmax)

        m = (gt.shape[2]-1) % args.upscale 
        if m != 0:
            gt = gt[...,:-m]
        lr = gt[...,::args.upscale]

        # prepare save dirs for sr and I_t
        save_root = os.path.join(args.ckpt_dir, 'results')
        os.makedirs(save_root, exist_ok=True)
        if isinstance(name, (list, tuple)):
            name_str = str(name[0])
        else:
            name_str = str(name)
        name_str = os.path.splitext(os.path.basename(name_str))[0]

        sr = torch.zeros_like(gt)
        sr_cnt = torch.zeros_like(gt)

        for i in range(lr.shape[2]-args.lr_slice_patch+1):
            tmp_lr = lr[...,i:i+args.lr_slice_patch]
            tmp_lr = tmp_lr.unsqueeze(0).cuda() #[1,s,h,w]
            gt_i = i * args.upscale
            tmp_gt = gt[...,gt_i:gt_i+args.hr_slice_patch] #[h,w,s]
            with torch.no_grad():
                tmp_sr = model(tmp_lr)

            psnr_slice = []
            ssim_slice = []

            tmp_sr_cpu = torch.clamp(tmp_sr.squeeze(0), 0, 1).detach().cpu()
            tmp_gt_cpu = tmp_gt.detach().cpu()
            
            psnr_volume = calc_psnr(tmp_gt_cpu, tmp_sr_cpu).item()

            pred_slices = [
                slice_idx
                for slice_idx in range(tmp_gt_cpu.shape[-1])
                if slice_idx % args.upscale != 0
            ]
            for slice_idx in pred_slices:
                psnr = calc_psnr(tmp_gt_cpu[..., slice_idx], tmp_sr_cpu[..., slice_idx]).item()
                ssim = calc_ssim(tmp_gt_cpu[..., slice_idx], tmp_sr_cpu[..., slice_idx])
                psnr_slice.append(psnr)
                ssim_slice.append(ssim)
            psnr_slice = sum(psnr_slice) / max(len(psnr_slice), 1)
            ssim_slice = sum(ssim_slice) / max(len(ssim_slice), 1)
            average_psnr_slice.append(psnr_slice)
            average_ssim_slice.append(ssim_slice)

            tmp_sr = tmp_sr_cpu
            
            log = r"[{} : {}] NAME:{} psnr_volume:{} psnr_slice:{} ssim_slice:{}"\
                .format(id+1,i+1,name,psnr_volume,psnr_slice,ssim_slice)
            print(log)
            with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
                f.write(log+'\n')


            sr[..., gt_i : gt_i + args.hr_slice_patch] += tmp_sr
            sr_cnt[..., gt_i : gt_i + args.hr_slice_patch] += 1

        mask = sr_cnt > 0
        sr = torch.where(mask, sr / torch.clamp(sr_cnt, min=1), sr)

        # save output slices (sr) scaled back to original intensity range
        sr_cpu = sr.detach().cpu()
        sr_raw = util.denormalize(sr_cpu, vmin, vmax).numpy().astype(np.float32)
        save_path = os.path.join(save_root, f"{name_str}.npy")
        np.save(save_path, sr_raw)

        # print(sr.shape) # h w s
        sr = sr.cuda()
        gt = gt.cuda()
        psnr = calc_psnr(sr,gt).item()
        average_psnr += psnr
        
        gt = gt.cuda()
        sr = sr.cuda()
        for i in range(gt.shape[2]):
            ssim = calc_ssim(gt[:, :, i], sr[:, :, i])
            x_y_ssim += ssim
        x_y_ssim /= i + 1
        for i in range(gt.shape[0]):
            ssim = calc_ssim(gt[i, :, :], sr[i, :, :])
            x_z_ssim += ssim
        x_z_ssim /= i + 1
        for i in range(gt.shape[1]):
            ssim = calc_ssim(gt[:, i, :], sr[:, i, :])
            y_z_ssim += ssim
        y_z_ssim /= i + 1
        
        log = r"[{} / {}] NAME:{} PSNR:{} x_y_ssim:{:.4f} x_z_ssim:{:.4f} y_z_ssim:{:.4f}".format(
            id + 1, dataloader.__len__(), name, psnr, x_y_ssim, x_z_ssim, y_z_ssim
        )
        print(log)
        with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
            f.write(log+'\n') 
        
        total_x_y_ssim += x_y_ssim
        total_x_z_ssim += x_z_ssim
        total_y_z_ssim += y_z_ssim

    average_psnr /= (id+1)
    total_x_y_ssim /= (id+1)
    total_x_z_ssim /= (id+1)
    total_y_z_ssim /= (id+1)
    average_psnr_slice = sum(average_psnr_slice) / max(len(average_psnr_slice), 1)
    average_ssim_slice = sum(average_ssim_slice) / max(len(average_ssim_slice), 1)
    print("average_psnr:",average_psnr) 
    print("total_x_y_ssim:",total_x_y_ssim)
    print("total_x_z_ssim:",total_x_z_ssim)
    print("total_y_z_ssim:",total_y_z_ssim)
    print("average_psnr_slice:",average_psnr_slice)
    print("average_ssim_slice:",average_ssim_slice)

    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        log = r"PSNR: {} x_y_ssim: {:.4f} x_z_ssim: {:.4f} y_z_ssim: {:.4f} PSNR_SLICE: {} SSIM_SLICE: {}".format(average_psnr,total_x_y_ssim,total_x_z_ssim,total_y_z_ssim,average_psnr_slice,average_ssim_slice)
        f.write(log+'\n')


if __name__ == "__main__":
    main()
