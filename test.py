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

    average_psnr = 0
    total_x_y_ssim = 0
    total_x_z_ssim = 0
    total_y_z_ssim = 0
    average_psnr_slice = []
    average_ssim_slice = []

    for id, (name, volume, vmin, vmax) in enumerate(dataloader):
        # volume [bz=1,h,w,s]
        gt = volume.squeeze(0) #[h,w,s]
        x_y_ssim = 0
        x_z_ssim = 0
        y_z_ssim = 0
        # stats are per-volume; squeeze batch dim
        vmin = float(vmin.squeeze(0).item()) if hasattr(vmin, "squeeze") else float(vmin)
        vmax = float(vmax.squeeze(0).item()) if hasattr(vmax, "squeeze") else float(vmax)
        
        upscale = args.max_mid_slices + 1
        surplus = (gt.shape[2]-1) % upscale -1
        surplus_with_interpolation = False if surplus == 1 else True
         
        lr_mid = gt[...,:-surplus:upscale]
        lr_surplus = gt[...,-surplus::surplus]

        # prepare save dirs for sr and I_t
        save_root = os.path.join(args.ckpt_dir, 'results')
        os.makedirs(save_root, exist_ok=True)
        if isinstance(name, (list, tuple)):
            name_str = str(name[0])
        else:
            name_str = str(name)
        name_str = os.path.splitext(os.path.basename(name_str))[0]

        sr = torch.zeros_like(gt)

        psnr_slice = []
        ssim_slice = []

        t_list_mid = [k/upscale for k in range(1, upscale)]
        t_list_surplus = [k/surplus for k in range(1, surplus)]

        lr_mid = lr_mid.unsqueeze(0).cuda() #[1,s,h,w]
        lr_surplus = lr_surplus.unsqueeze(0).cuda() #[1,s,h,w]
        with torch.no_grad():
            sr_mid = model.inference(lr_mid, t_list_mid) #[1,s,h,w]
            if surplus_with_interpolation:
                sr_surplus = model.inference(lr_surplus, t_list_surplus) #[1,s,h,w]
            else:
                sr_surplus = lr_surplus
        
        sr = torch.cat([sr_mid, sr_surplus[...,1:]], dim=1)
        sr_cpu = torch.clamp(sr.squeeze(0), 0, 1).detach().cpu()
        gt_cpu = gt.detach().cpu()

        sr_raw = util.denormalize(sr_cpu, vmin, vmax).numpy().astype(np.float32)
        save_path = os.path.join(save_root, f"{name_str}.npy")
        np.save(save_path, sr_raw)

        # print(sr.shape) # h w s
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
            id + 1, dataloader.__len__(), name, average_psnr, x_y_ssim, x_z_ssim, y_z_ssim
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
    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        log = r"PSNR: {} x_y_ssim: {:.4f} x_z_ssim: {:.4f} y_z_ssim: {:.4f} PSNR_SLICE: {} SSIM_SLICE: {}".format(average_psnr,total_x_y_ssim,total_x_z_ssim,total_y_z_ssim,average_psnr_slice,average_ssim_slice)
        f.write(log+'\n')


if __name__ == "__main__":
    main()
