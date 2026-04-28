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

    average_psnr_local = 0
    average_psnr_global = 0
    total_x_y_ssim_local = 0
    total_x_y_ssim_global = 0
    total_x_z_ssim_local = 0
    total_x_z_ssim_global = 0
    total_y_z_ssim_local = 0
    total_y_z_ssim_global = 0
    average_psnr_slice_local = []
    average_psnr_slice_global = []
    average_ssim_slice_local = []
    average_ssim_slice_global = []

    for id, (name, volume, vmin, vmax) in enumerate(dataloader):
        # volume [bz=1,h,w,s]
        gt = volume.squeeze(0) #[h,w,s]
        x_y_ssim_local = 0
        x_y_ssim_global = 0
        x_z_ssim_local = 0
        x_z_ssim_global = 0
        y_z_ssim_local = 0
        y_z_ssim_global = 0
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
        save_root_local = os.path.join(save_root, "local")
        save_root_global = os.path.join(save_root, "global")
        os.makedirs(save_root_local, exist_ok=True)
        os.makedirs(save_root_global, exist_ok=True)
        if isinstance(name, (list, tuple)):
            name_str = str(name[0])
        else:
            name_str = str(name)
        name_str = os.path.splitext(os.path.basename(name_str))[0]

        sr_local = torch.zeros_like(gt)
        sr_global = torch.zeros_like(gt)
        sr_cnt = torch.zeros_like(gt)

        psnr_local_slice = []
        psnr_global_slice = []
        ssim_local_slice = []
        ssim_global_slice = []

        for i in range(lr.shape[2]-args.lr_slice_patch+1):
            tmp_lr = lr[...,i:i+args.lr_slice_patch]
            tmp_lr = tmp_lr.unsqueeze(0).cuda() #[1,s,h,w]
            gt_i = i * args.upscale
            tmp_gt = gt[...,gt_i:gt_i+args.hr_slice_patch] #[h,w,s]
            with torch.no_grad():
                tmp_local, tmp_global = model(tmp_lr)

            tmp_local_cpu = torch.clamp(tmp_local.squeeze(0), 0, 1).detach().cpu()
            tmp_global_cpu = torch.clamp(tmp_global.squeeze(0), 0, 1).detach().cpu()
            tmp_gt_cpu = tmp_gt.detach().cpu()

            pred_local_slices = [
                slice_idx
                for slice_idx in range(tmp_gt_cpu.shape[-1])
                if slice_idx % args.upscale != 0
            ]
            for slice_idx in pred_local_slices:
            # for slice_idx in range(tmp_gt_cpu.shape[-1]):
                psnr = calc_psnr(
                    tmp_gt_cpu[..., slice_idx], tmp_local_cpu[..., slice_idx]
                ).item()
                ssim = calc_ssim(
                    tmp_gt_cpu[..., slice_idx], tmp_local_cpu[..., slice_idx]
                )
                # log = f"LOCAL--slice {slice_idx} psnr: {psnr:.4f} ssim: {ssim:.4f}"
                # print(log)
                # with open(args.ckpt_dir + '/logs_test.txt', mode='a+') as f:
                #     f.write(log + '\n')
                psnr_local_slice.append(psnr)
                ssim_local_slice.append(ssim)
            
            pred_global_slices = [
                slice_idx
                for slice_idx in range(1, tmp_gt_cpu.shape[-1]-1)
            ]
            for slice_idx in pred_global_slices:
            # for slice_idx in range(tmp_gt_cpu.shape[-1]):
                psnr = calc_psnr(
                    tmp_gt_cpu[..., slice_idx], tmp_global_cpu[..., slice_idx]
                ).item()
                ssim = calc_ssim(
                    tmp_gt_cpu[..., slice_idx], tmp_global_cpu[..., slice_idx]
                )
                # log = f"GLOBAL--slice {slice_idx} psnr: {psnr:.4f} ssim: {ssim:.4f}"
                # print(log)
                # with open(args.ckpt_dir + '/logs_test.txt', mode='a+') as f:
                #     f.write(log + '\n')
                psnr_global_slice.append(psnr)
                ssim_global_slice.append(ssim)

            tmp_sr_local = tmp_local_cpu
            tmp_sr_global = tmp_global_cpu
            sr_local[..., gt_i : gt_i + args.hr_slice_patch] += tmp_sr_local
            sr_global[..., gt_i : gt_i + args.hr_slice_patch] += tmp_sr_global
            sr_cnt[..., gt_i : gt_i + args.hr_slice_patch] += 1

        mask = sr_cnt > 0
        sr_local = torch.where(mask, sr_local / torch.clamp(sr_cnt, min=1), sr_local)
        sr_global = torch.where(mask, sr_global / torch.clamp(sr_cnt, min=1), sr_global)

        # save output slices (sr) scaled back to original intensity range
        sr_local_cpu = sr_local.detach().cpu()
        sr_global_cpu = sr_global.detach().cpu()
        sr_local_raw = util.denormalize(sr_local_cpu, vmin, vmax).numpy().astype(np.float32)
        sr_global_raw = util.denormalize(sr_global_cpu, vmin, vmax).numpy().astype(np.float32)
        save_path_local = os.path.join(save_root_local, f"{name_str}.npy")
        save_path_global = os.path.join(save_root_global, f"{name_str}.npy")
        np.save(save_path_local, sr_local_raw)
        np.save(save_path_global, sr_global_raw)

        # print(sr.shape) # h w s
        sr_local = sr_local.cuda()
        sr_global = sr_global.cuda()
        gt = gt.cuda()
        psnr_local = calc_psnr(sr_local,gt).item()
        psnr_global = calc_psnr(sr_global,gt).item()
        average_psnr_local += psnr_local
        average_psnr_global += psnr_global
        
        gt = gt.cuda()
        sr_local = sr_local.cuda()
        sr_global = sr_global.cuda()
        for i in range(gt.shape[2]):
            ssim_local = calc_ssim(gt[:, :, i], sr_local[:, :, i])
            ssim_global = calc_ssim(gt[:, :, i], sr_global[:, :, i])
            x_y_ssim_local += ssim_local
            x_y_ssim_global += ssim_global
        x_y_ssim_local /= i + 1
        x_y_ssim_global /= i + 1
        for i in range(gt.shape[0]):
            ssim_local = calc_ssim(gt[i, :, :], sr_local[i, :, :])
            ssim_global = calc_ssim(gt[i, :, :], sr_global[i, :, :])
            x_z_ssim_local += ssim_local
            x_z_ssim_global += ssim_global
        x_z_ssim_local /= i + 1
        x_z_ssim_global /= i + 1
        for i in range(gt.shape[1]):
            ssim_local = calc_ssim(gt[:, i, :], sr_local[:, i, :])
            ssim_global = calc_ssim(gt[:, i, :], sr_global[:, i, :])
            y_z_ssim_local += ssim_local
            y_z_ssim_global += ssim_global
        y_z_ssim_local /= i + 1
        y_z_ssim_global /= i + 1

        psnr_slice_local = sum(psnr_local_slice) / max(len(psnr_local_slice), 1)
        psnr_slice_global = sum(psnr_global_slice) / max(len(psnr_global_slice), 1)
        ssim_slice_local = sum(ssim_local_slice) / max(len(ssim_local_slice), 1)
        ssim_slice_global = sum(ssim_global_slice) / max(len(ssim_global_slice), 1)
        average_psnr_slice_local.append(psnr_slice_local)
        average_psnr_slice_global.append(psnr_slice_global)
        average_ssim_slice_local.append(ssim_slice_local)
        average_ssim_slice_global.append(ssim_slice_global)
        log = r"[{} / {}] NAME:{} psnr_slice_local:{} psnr_slice_global:{} ssim_slice_local:{} ssim_slice_global:{}" \
            .format(id + 1, dataloader.__len__(), name, psnr_slice_local, psnr_slice_global, ssim_slice_local, ssim_slice_global)
        print(log)
        with open(args.ckpt_dir + '/logs_test.txt', mode='a+') as f:
            f.write(log + '\n')

        log = r"[{} / {}] NAME:{} PSNR_local:{} PSNR_global:{} x_y_ssim_local:{:.4f} x_y_ssim_global:{:.4f} x_z_ssim_local:{:.4f} x_z_ssim_global:{:.4f} y_z_ssim_local:{:.4f} y_z_ssim_global:{:.4f}".format(
            id + 1, dataloader.__len__(), name, psnr_local, psnr_global, x_y_ssim_local, x_y_ssim_global, x_z_ssim_local, x_z_ssim_global, y_z_ssim_local, y_z_ssim_global
        )
        print(log)
        with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
            f.write(log+'\n') 
        
        total_x_y_ssim_local += x_y_ssim_local
        total_x_y_ssim_global += x_y_ssim_global
        total_x_z_ssim_local += x_z_ssim_local
        total_x_z_ssim_global += x_z_ssim_global
        total_y_z_ssim_local += y_z_ssim_local
        total_y_z_ssim_global += y_z_ssim_global

    average_psnr_local /= (id+1)
    average_psnr_global /= (id+1)
    total_x_y_ssim_local /= (id+1)
    total_x_y_ssim_global /= (id+1)
    total_x_z_ssim_local /= (id+1)
    total_x_z_ssim_global /= (id+1)
    total_y_z_ssim_local /= (id+1)
    total_y_z_ssim_global /= (id+1)
    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        log = r"PSNR_local: {} x_y_ssim_local: {:.4f} x_z_ssim_local: {:.4f} y_z_ssim_local: {:.4f} PSNR_SLICE_local: {} SSIM_SLICE_local: {}".format(average_psnr_local,total_x_y_ssim_local,total_x_z_ssim_local,total_y_z_ssim_local,average_psnr_slice_local,average_ssim_slice_local)
        f.write(log+'\n')
        log = r"PSNR_global: {} x_y_ssim_global: {:.4f} x_z_ssim_global: {:.4f} y_z_ssim_global: {:.4f} PSNR_SLICE_global: {} SSIM_SLICE_global: {}".format(average_psnr_global,total_x_y_ssim_global,total_x_z_ssim_global,total_y_z_ssim_global,average_psnr_slice_global,average_ssim_slice_global)
        f.write(log+'\n')


if __name__ == "__main__":
    main()
