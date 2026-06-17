"""
I3Net 中间层特征图可视化 — 逐通道展示各阶段特征。

用法示例:
    python visualize_features.py \\
        --ckpt experiments/xxx/pth/1500.pth \\
        --testdata_path /path/to/test \\
        --vis_out_dir experiments/feature_vis \\
        --gpu_id 0

输出:
    vis_out_dir/feature_maps/{volume_name}/
        pre_head.png          head 前 (LR 输入各 slice 通道)
        post_head.png         head 后 (64 通道)
        body_group_first.png  第一个 I2Group 后 (64 通道)
        body_group_last.png   最后一个 I2Group 后 (64 通道)
        pre_fuse.png          fuse 前 (head + align3 + align7, 各 64 通道)
        post_fuse.png         fuse 后 (64 通道)
        post_tail.png         tail 后 (HR slice 通道)
        final_out.png         最终输出 (LR 注入后)
        body_group_first_block{j}_inter.png  第一组 I2Block inter 分支 (64 通道)
        body_group_first_block{j}_intra.png       第一组 intra（deviation 归一化，推荐）
        body_group_first_block{j}_intra_minmax.png  第一组 intra（min-max，易呈纯色）
        body_group_last_block{j}_inter.png   最后一组 inter
        body_group_last_block{j}_intra.png        最后一组 intra（deviation）
        body_group_last_block{j}_intra_minmax.png 最后一组 intra（min-max）
"""

import math
import os

import config

config.parser.add_argument(
    "--vis_out_dir", type=str, default="experiments/feature_vis",
    help="特征可视化结果保存目录",
)
config.parser.add_argument(
    "--num_samples", type=int, default=1,
    help="可视化的 volume 数量",
)
config.parser.add_argument(
    "--sample_idx", type=int, default=-1,
    help="指定 volume 索引, -1 表示按 num_samples 依次取",
)
config.parser.add_argument(
    "--feat_ncols", type=int, default=8,
    help="特征通道网格每行列数 (64 通道默认 8x8)",
)

args, _ = config.get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import testSet
from model_zoo.i3net.basic_model import extract_i3net_features
from select_model import select_model


def load_model(args):
    model = select_model(args)
    checkpoint = torch.load(args.ckpt, map_location=torch.device("cpu"))
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    elif isinstance(checkpoint, dict) and any(k.endswith(".weight") for k in checkpoint.keys()):
        state = checkpoint
    else:
        raise KeyError(f"cannot find weight in ckpt: {list(checkpoint.keys())}")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.ckpt}")
    if args.cuda:
        model = model.cuda()
    model.eval()
    return model


def _normalize_channel(arr):
    arr = arr.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def _normalize_spatial_deviation(arr, clip_sigma=3.0):
    """去空间均值后按 robust 标准差缩放，突出 intra 的微弱空间结构。"""
    arr = arr.astype(np.float32)
    mu = arr.mean()
    dev = arr - mu
    sigma = dev.std()
    if sigma < 1e-8:
        return None, mu, sigma
    dev = np.clip(dev / sigma, -clip_sigma, clip_sigma)
    return (dev + clip_sigma) / (2 * clip_sigma), mu, sigma


def save_channel_grid(feat, save_path, title, ncols=None, cmap="viridis", channel_titles=None):
    """feat: [C, H, W] numpy"""
    c, h, w = feat.shape
    if ncols is None:
        ncols = max(1, int(math.ceil(math.sqrt(c))))
    nrows = int(math.ceil(c / ncols))

    fig_w = min(ncols * 1.6, 24)
    fig_h = min(nrows * 1.6, 24)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(nrows * ncols):
        r, col = divmod(idx, ncols)
        ax = axes[r, col]
        ax.axis("off")
        if idx < c:
            img = _normalize_channel(feat[idx])
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            if channel_titles is not None:
                ax.set_title(channel_titles[idx], fontsize=6)
            else:
                ax.set_title(f"ch{idx}", fontsize=7)
        else:
            ax.axis("off")

    fig.suptitle(f"{title}  ({c} ch, {h}x{w})", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_intra_channel_grid(feat, save_path, title, ncols=8):
    """intra 专用：去均值 + 标准化，避免空间近似常数时 min-max 退化为纯色。"""
    c, h, w = feat.shape
    nrows = int(math.ceil(c / ncols))
    fig_w = min(ncols * 1.6, 24)
    fig_h = min(nrows * 1.6, 24)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    flat_count = 0
    for idx in range(nrows * ncols):
        r, col = divmod(idx, ncols)
        ax = axes[r, col]
        ax.axis("off")
        if idx >= c:
            continue
        ch = feat[idx]
        ch_range = ch.max() - ch.min()
        ch_std = ch.std()
        norm, mu, sigma = _normalize_spatial_deviation(ch)
        if norm is None:
            flat_count += 1
            ax.imshow(np.zeros((h, w)), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"ch{idx} flat μ={mu:.2g}", fontsize=6, color="red")
        else:
            ax.imshow(norm, cmap="RdBu_r", vmin=0, vmax=1)
            ax.set_title(f"ch{idx} σ={sigma:.2e} r={ch_range:.2e}", fontsize=6)

    fig.suptitle(
        f"{title}  (deviation norm, flat={flat_count}/{c})  blue↓ red↑ vs spatial mean",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return flat_count, c


def save_pre_fuse_grid(feat, save_path, n_feats, ncols=8):
    """pre_fuse: [3*n_feats, H, W], 按 head / align3 / align7 分块展示。"""
    labels = ["head (align_0)", "align_3", "align_7"]
    fig, axes = plt.subplots(3, 1, figsize=(min(ncols * 1.6, 20), 3 * min(ncols * 1.5, 14)))

    for block_idx, label in enumerate(labels):
        start = block_idx * n_feats
        block = feat[start : start + n_feats]
        c = block.shape[0]
        nrows = int(math.ceil(c / ncols))
        ax = axes[block_idx]
        ax.axis("off")

        mosaic = np.zeros((nrows * block.shape[1], ncols * block.shape[2]), dtype=np.float32)
        for ch in range(c):
            r, col = divmod(ch, ncols)
            y0, x0 = r * block.shape[1], col * block.shape[2]
            mosaic[y0 : y0 + block.shape[1], x0 : x0 + block.shape[2]] = _normalize_channel(block[ch])

        ax.imshow(mosaic, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"pre_fuse — {label} ({n_feats} ch)", fontsize=10)

    fig.suptitle(f"pre_fuse concat  ({feat.shape[0]} ch total)", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def tensor_to_numpy(feat, batch_idx=0):
    """[B, C, H, W] -> [C, H, W] numpy"""
    return feat[batch_idx].detach().cpu().float().numpy()


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_inter_intra_branches(model, feats, out_dir, group_ids, ncols):
    """保存指定 I2Group 内各 block 的 inter / intra 分支特征。"""
    net = _unwrap_model(model)
    group_labels = {group_ids[0]: "body_group_first", group_ids[-1]: "body_group_last"}

    for gid in group_ids:
        group = net.body[gid]
        label = group_labels.get(gid, f"body_group_{gid}")
        num_blocks = len(group.body)

        for bid in range(num_blocks):
            block = group.body[bid]
            inter_key = f"body_group_{gid}_block_{bid}_inter"
            intra_key = f"body_group_{gid}_block_{bid}_intra"

            if inter_key in feats:
                inter = tensor_to_numpy(feats[inter_key]) * block.res_scale
                save_channel_grid(
                    inter,
                    os.path.join(out_dir, f"{label}_block{bid}_inter.png"),
                    f"{label} block{bid} inter (×res_scale={block.res_scale})",
                    ncols=ncols,
                    cmap="plasma",
                )

            if intra_key in feats:
                intra = tensor_to_numpy(feats[intra_key])
                # min-max 版（与 inter 一致，空间常数通道会呈纯色）
                save_channel_grid(
                    intra,
                    os.path.join(out_dir, f"{label}_block{bid}_intra_minmax.png"),
                    f"{label} block{bid} intra (min-max)",
                    ncols=ncols,
                    cmap="cividis",
                )
                # deviation 版（推荐：突出相对空间结构）
                flat, total = save_intra_channel_grid(
                    intra,
                    os.path.join(out_dir, f"{label}_block{bid}_intra.png"),
                    f"{label} block{bid} intra",
                    ncols=ncols,
                )
                print(
                    f"  {label} block{bid} intra: "
                    f"flat={flat}/{total}, "
                    f"global std={intra.std():.4e}, "
                    f"median spatial σ={np.median(intra.std(axis=(1, 2))):.4e}"
                )


@torch.no_grad()
def visualize_one_sample(model, lr_patch, out_dir, n_feats, ncols):
    """lr_patch: [1, H, W, C_lr]"""
    out, feats = extract_i3net_features(model, lr_patch, no_grad=True)
    os.makedirs(out_dir, exist_ok=True)

    stages = [
        ("pre_head", "pre_head", feats.get("pre_head")),
        ("post_head", "post_head", feats.get("head")),
        ("body_group_first", "body_group_first", feats.get("body_group_first")),
        ("body_group_last", "body_group_last", feats.get("body_group_last")),
        ("post_fuse", "post_fuse", feats.get("fuse")),
        ("post_tail", "post_tail", feats.get("tail")),
    ]

    for fname, title, tensor in stages:
        if tensor is None:
            print(f"  skip {fname}: feature not found")
            continue
        feat = tensor_to_numpy(tensor)
        save_channel_grid(
            feat,
            os.path.join(out_dir, f"{fname}.png"),
            title,
            ncols=ncols if feat.shape[0] > 4 else feat.shape[0],
        )

    if feats.get("pre_fuse") is not None:
        pre_fuse = tensor_to_numpy(feats["pre_fuse"])
        save_pre_fuse_grid(
            pre_fuse,
            os.path.join(out_dir, "pre_fuse.png"),
            n_feats=n_feats,
            ncols=ncols,
        )

    net = _unwrap_model(model)
    num_groups = len(net.body)
    if num_groups > 0:
        group_ids = [0]
        if num_groups > 1:
            group_ids.append(num_groups - 1)
        save_inter_intra_branches(model, feats, out_dir, group_ids, ncols)

    final = out[0].detach().cpu().float().numpy().transpose(2, 0, 1)  # [C_hr, H, W]
    save_channel_grid(
        final,
        os.path.join(out_dir, "final_out.png"),
        "final_out (after LR inject)",
        ncols=final.shape[0],
    )


def get_lr_patch(gt, args):
    m = (gt.shape[2] - 1) % args.upscale
    if m != 0:
        gt = gt[..., :-m]
    lr = gt[..., ::args.upscale]

    i = (lr.shape[2] - args.lr_slice_patch) // 2
    tmp_lr = lr[..., i : i + args.lr_slice_patch].unsqueeze(0)
    if args.cuda:
        tmp_lr = tmp_lr.cuda()
    return tmp_lr


def main():
    if not args.ckpt:
        raise ValueError("请通过 --ckpt 指定模型权重路径")

    out_root = os.path.join(args.vis_out_dir, "feature_maps")
    os.makedirs(out_root, exist_ok=True)

    model = load_model(args)
    testset = testSet(data_root=args.testdata_path, image_size=args.image_size)
    n_feats = args.n_feats

    if args.sample_idx >= 0:
        indices = [args.sample_idx]
    else:
        indices = list(range(min(args.num_samples, len(testset))))

    for idx in indices:
        name, volume, _, _ = testset[idx]
        tag = os.path.splitext(os.path.basename(name))[0]
        print(f"Processing [{idx + 1}/{len(indices)}] {tag} ...")

        gt = volume
        lr_patch = get_lr_patch(gt, args)
        sample_dir = os.path.join(out_root, tag)
        visualize_one_sample(model, lr_patch, sample_dir, n_feats, args.feat_ncols)
        print(f"  saved to {sample_dir}/")

    print(f"\nDone. Feature maps saved to: {out_root}")


if __name__ == "__main__":
    main()
