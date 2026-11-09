import os
import config

args, unparsed = config.get_args()

import torch
import numpy as np

from data import testSet
from util_evaluation import calc_psnr, calc_ssim
from select_model import select_model
import util


@torch.no_grad()
def infer_volume(model, volume, gap):
    """沿 z 以 gap 取端点，对相邻端点查询全部中间 t。"""
    device = next(model.parameters()).device
    volume = volume.to(device)

    surplus = (volume.shape[2] - 1) % gap
    if surplus != 0:
        volume = volume[..., :-surplus]

    lr = volume[..., ::gap].unsqueeze(0)  # [1, H, W, N_lr]
    if lr.shape[3] < 2:
        raise ValueError(f"need at least 2 LR slices after gap={gap}, got {lr.shape[3]}")

    t_list = [k / gap for k in range(1, gap)]
    sr = model.inference(lr, t_list, gap=gap)  # [1, S, H, W]
    sr = torch.clamp(sr.squeeze(0).permute(1, 2, 0), 0, 1)  # [H, W, S]
    return volume, sr


def main():
    device = torch.device("cuda" if args.cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    args.ckpt_dir = "experiments/" + args.ckpt_dir
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(args.ckpt_dir + "/logs_test.txt", mode="a+") as f:
        f.write("\n\n\n\n\nSTART EXPERIMENT\n")
        f.write("testdata:" + args.testdata_path + "\n")
        f.write("checkpoint:" + args.ckpt + "\n")
        f.write(f"estimator:{args.estimator}\n")
        f.write(f"test_gap:{args.test_gap}\n")

    model = select_model(args)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and any(k.endswith(".weight") for k in checkpoint.keys()):
        state = checkpoint
    else:
        print(f"Error: cannot find weight in ckpt: {checkpoint.keys()}")
        raise KeyError("cannot find weight in ckpt")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"load:{args.ckpt} estimator={args.estimator}")
    model = model.to(device)
    model.eval()

    testset = testSet(data_root=args.testdata_path, image_size=args.image_size)
    dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        drop_last=False,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    average_psnr = 0.0
    total_x_y_ssim = 0.0
    total_x_z_ssim = 0.0
    total_y_z_ssim = 0.0
    n_case = 0

    gap = args.test_gap
    if gap < 2:
        raise ValueError(f"test_gap must be >= 2, got {gap}")

    for id, (name, volume) in enumerate(dataloader):
        gt = volume.squeeze(0)  # [H, W, S]
        x_y_ssim = 0.0
        x_z_ssim = 0.0
        y_z_ssim = 0.0

        save_root = os.path.join(args.ckpt_dir, "results")
        os.makedirs(save_root, exist_ok=True)
        if isinstance(name, (list, tuple)):
            name_str = str(name[0])
        else:
            name_str = str(name)
        name_str = os.path.splitext(os.path.basename(name_str))[0]

        gt, sr = infer_volume(model, gt, gap=gap)
        sr_cpu = sr.detach().cpu()

        # 与 util.normalize 的固定 0~4095 对齐
        sr_raw = (sr_cpu * 4095.0).numpy().astype(np.float32)
        save_path = os.path.join(save_root, f"{name_str}.npy")
        np.save(save_path, sr_raw)

        psnr = calc_psnr(gt, sr).item()
        for i in range(gt.shape[2]):
            x_y_ssim += calc_ssim(gt[:, :, i], sr[:, :, i])
        x_y_ssim /= gt.shape[2]
        for i in range(gt.shape[0]):
            x_z_ssim += calc_ssim(gt[i, :, :], sr[i, :, :])
        x_z_ssim /= gt.shape[0]
        for i in range(gt.shape[1]):
            y_z_ssim += calc_ssim(gt[:, i, :], sr[:, i, :])
        y_z_ssim /= gt.shape[1]

        log = (
            f"[{id + 1} / {len(dataloader)}] NAME:{name_str} "
            f"PSNR:{psnr:.4f} x_y_ssim:{x_y_ssim:.4f} "
            f"x_z_ssim:{x_z_ssim:.4f} y_z_ssim:{y_z_ssim:.4f}"
        )
        print(log)
        with open(args.ckpt_dir + "/logs_test.txt", mode="a+") as f:
            f.write(log + "\n")

        average_psnr += psnr
        total_x_y_ssim += x_y_ssim
        total_x_z_ssim += x_z_ssim
        total_y_z_ssim += y_z_ssim
        n_case += 1

    average_psnr /= n_case
    total_x_y_ssim /= n_case
    total_x_z_ssim /= n_case
    total_y_z_ssim /= n_case
    with open(args.ckpt_dir + "/logs_test.txt", mode="a+") as f:
        log = (
            f"PSNR: {average_psnr} x_y_ssim: {total_x_y_ssim:.4f} "
            f"x_z_ssim: {total_x_z_ssim:.4f} y_z_ssim: {total_y_z_ssim:.4f}"
        )
        f.write(log + "\n")


if __name__ == "__main__":
    main()
