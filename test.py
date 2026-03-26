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
    args.ckpt_dir = 'experiments/'+args.model+'/'+args.ckpt_dir
    os.makedirs(args.ckpt_dir,exist_ok=True)
    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        s = "\n\n\n\n\nSTART EXPERIMENT\n"
        f.write(s)
        f.write('testdata:'+args.testdata_path+'\n')
        f.write('checkpoint:'+args.ckpt+'\n')

    model = select_model(args)
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    if 'module' in checkpoint['state_dict']:
        checkpoint['state_dict'] = checkpoint['state_dict']['module']
    model.load_state_dict(checkpoint['state_dict'])
    print(f'load:{args.ckpt}')
    model = model.cuda()
    model.eval()

    testset = testSet(data_root=args.testdata_path)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1,
    drop_last=False, shuffle=False, num_workers=4, pin_memory=False)

    average_psnr=0

    for id, (name, volume, vmin, vmax) in enumerate(dataloader):
        # volume [bz=1,h,w,s]
        gt = volume.squeeze(0) #[h,w,s]
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
                psnr_slice.append(psnr)
            psnr_slice = sum(psnr_slice) / max(len(psnr_slice), 1)
            
            tmp_sr = tmp_sr_cpu
            
            log = r"[{} : {}] NAME:{} psnr_volume:{} psnr_slice:{}"\
                .format(id+1,i+1,name,psnr_volume,psnr_slice)
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

        log = r"[{} / {}] NAME:{} PSNR:{}"\
            .format(id+1,dataloader.__len__(),name,psnr)
        print(log)
        with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
            f.write(log+'\n') 

    average_psnr /= (id+1)
    print("average_psnr:",average_psnr) 

    with open(args.ckpt_dir + '/logs_test.txt',mode='a+') as f:
        log = r"PSNR: {}".format(average_psnr)
        f.write(log)


if __name__ == "__main__":
    main()
