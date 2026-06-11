"""
I3NET 中间特征可视化脚本（新 Net 架构）。

展示内容:
    - 光流 (flow_list, 按 HR 网格每个 T 一张)
    - 重采样图 (warped0 / warped1, 每个 T 一张)
    - Head 输出
    - Alignment 每一层 (模块输出 + 残差后)
    - Body 每一层 (fuse 之前)
    - FuseAlign / Align(LocalMMF) / before_tail
    - Mask / Delta / 融合结果 (每个 T 一张)

用法示例:
    python visualize_features.py \\
        --ckpt experiments/xxx/best.pth \\
        --testdata_path /path/to/test \\
        --vis_out_dir experiments/feature_vis \\
        --gpu_id 0
"""

import os

import config

config.parser.add_argument(
    "--vis_out_dir",
    type=str,
    default="experiments/feature_vis",
    help="特征可视化结果保存目录",
)
config.parser.add_argument(
    "--num_samples",
    type=int,
    default=3,
    help="可视化的 volume 数量 (从 test set 前 N 个取)",
)
config.parser.add_argument(
    "--sample_idx",
    type=int,
    default=-1,
    help="指定 volume 索引, -1 表示按 num_samples 依次取",
)
config.parser.add_argument(
    "--slice_idx",
    type=int,
    default=-1,
    help="HR 网格上的 slice 索引, -1 表示取中间插值 slice (拼图参考帧)",
)

args, _ = config.get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import testSet
from select_model import select_model
from model_zoo.flowseek.core.utils.flow_viz import flow_to_image


# ---------------------------------------------------------------------------
# 模型 & 数据
# ---------------------------------------------------------------------------


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
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.ckpt}")
    if args.cuda:
        model = model.cuda()
    model.eval()
    return model


def pick_hr_slice_idx(args):
    if args.slice_idx >= 0:
        return args.slice_idx
    mid = args.upscale // 2
    return mid if mid > 0 else 1


def prepare_lr_patch(gt, args):
    m = (gt.shape[2] - 1) % args.upscale
    if m != 0:
        gt = gt[..., :-m]
    lr = gt[..., :: args.upscale]
    i = (lr.shape[2] - args.lr_slice_patch) // 2
    tmp_lr = lr[..., i : i + args.lr_slice_patch].unsqueeze(0)
    gt_i = i * args.upscale
    tmp_gt = gt[..., gt_i : gt_i + args.hr_slice_patch]
    return tmp_lr, tmp_gt, i, gt_i


# ---------------------------------------------------------------------------
# 特征可视化工具
# ---------------------------------------------------------------------------


def _to_numpy(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    return img.astype(np.float32)


def feat_to_heatmap(feat, reduce="mean"):
    """将 (C, H, W) 或 (B, C, H, W) 特征压成单张 2D heatmap。"""
    arr = _to_numpy(feat)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if reduce == "mean":
            arr = arr.mean(axis=0)
        elif reduce == "max":
            arr = arr.max(axis=0)
        else:
            arr = np.linalg.norm(arr, axis=0)
    arr = arr - arr.min()
    denom = arr.max() + 1e-8
    return (arr / denom).astype(np.float32)


def flow_tensor_to_rgb(flow):
    """(2, H, W) tensor -> RGB numpy [H, W, 3]."""
    arr = _to_numpy(flow)
    if arr.ndim == 4:
        arr = arr[0]
    uv = np.transpose(arr, (1, 2, 0))
    return flow_to_image(uv)


def get_flow_for_hr_t(flow_list, t, upscale):
    """
    为 HR 网格第 t 个 slice 取对应光流。
    关键帧显示 LR 区间基准光流；插值帧显示与 warp 一致的缩放光流 flow*time。
    """
    pair_idx = min(t // upscale, len(flow_list) - 1)
    flow = flow_list[pair_idx]
    offset = t % upscale
    if offset == 0:
        return flow, flow, pair_idx, 0.0, True
    time = offset / upscale
    flow_w0 = flow * time
    flow_w1 = flow * (1.0 - time)
    return flow_w0, flow_w1, pair_idx, time, False


def save_panel_grid(panels, save_path, ncols=4, figsize_per_col=3.5, suptitle=None):
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_col * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (title, img, cmap, vmin, vmax) in zip(axes, panels):
        if img.ndim == 3 and img.shape[-1] == 3:
            ax.imshow(img)
        else:
            kwargs = {"cmap": cmap}
            if vmin is not None:
                kwargs["vmin"] = vmin
            if vmax is not None:
                kwargs["vmax"] = vmax
            ax.imshow(img, **kwargs)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    if suptitle:
        plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_single_image(img, save_path, title=None, cmap="gray", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    if img.ndim == 3 and img.shape[-1] == 3:
        ax.imshow(img)
    else:
        kwargs = {"cmap": cmap}
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax
        ax.imshow(img, **kwargs)
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_spatial_feat(feat, save_dir, name, tag, title_prefix=None):
    """保存 (B, C, H, W) 空间特征的单张 heatmap。"""
    os.makedirs(save_dir, exist_ok=True)
    hm = feat_to_heatmap(feat)
    prefix = title_prefix or name
    save_single_image(
        hm,
        os.path.join(save_dir, f"{tag}_{name}.png"),
        title=prefix,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )


def save_per_t_bt_series(vis, gt_vol, pred_vol, save_dir, tag, upscale):
    """对 B,T,H,W 张量按每个 T 保存单张图。"""
    T = vis["mask"].shape[1]
    delta_global_vmax = max(float(np.abs(_to_numpy(vis["delta"][0])).max()), 1e-6)

    subdirs = {
        "flow_w0": os.path.join(save_dir, "flow_w0"),
        "flow_w1": os.path.join(save_dir, "flow_w1"),
        "flow_rgb": os.path.join(save_dir, "flow_rgb"),
        "warped0": os.path.join(save_dir, "warped0"),
        "warped1": os.path.join(save_dir, "warped1"),
        "mask": os.path.join(save_dir, "mask"),
        "delta": os.path.join(save_dir, "delta"),
        "fused": os.path.join(save_dir, "fused"),
        "gt": os.path.join(save_dir, "gt"),
        "pred": os.path.join(save_dir, "pred"),
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    flow_w0_strip, flow_w1_strip = [], []
    warped0_strip, warped1_strip = [], []
    mask_strip, delta_strip = [], []
    gt_strip, pred_strip = [], []

    for t in range(T):
        flow_w0, flow_w1, pair_idx, time_frac, is_key = get_flow_for_hr_t(
            vis["flow_list"], t, upscale
        )
        key_tag = "key" if is_key else f"interp_t{time_frac:.2f}"
        title_suffix = f"T{t:02d} pair{pair_idx} {key_tag}"

        f0_rgb = flow_tensor_to_rgb(flow_w0)
        f1_rgb = flow_tensor_to_rgb(flow_w1)
        save_single_image(
            f0_rgb,
            os.path.join(subdirs["flow_w0"], f"{tag}_t{t:02d}.png"),
            title=f"flow_w0 {title_suffix}",
        )
        save_single_image(
            f1_rgb,
            os.path.join(subdirs["flow_w1"], f"{tag}_t{t:02d}.png"),
            title=f"flow_w1 {title_suffix}",
        )
        save_single_image(
            np.concatenate([f0_rgb, f1_rgb], axis=1),
            os.path.join(subdirs["flow_rgb"], f"{tag}_t{t:02d}.png"),
            title=f"flow {title_suffix}",
        )
        flow_w0_strip.append(f0_rgb)
        flow_w1_strip.append(f1_rgb)

        w0 = _to_numpy(vis["warped0"][0, t])
        w1 = _to_numpy(vis["warped1"][0, t])
        save_single_image(
            w0,
            os.path.join(subdirs["warped0"], f"{tag}_t{t:02d}.png"),
            title=f"warped0 {title_suffix}",
            vmin=0,
            vmax=1,
        )
        save_single_image(
            w1,
            os.path.join(subdirs["warped1"], f"{tag}_t{t:02d}.png"),
            title=f"warped1 {title_suffix}",
            vmin=0,
            vmax=1,
        )
        warped0_strip.append(w0)
        warped1_strip.append(w1)

        mask = _to_numpy(vis["mask"][0, t])
        delta = _to_numpy(vis["delta"][0, t])
        fused = _to_numpy(vis["fused"][0, t])
        gt_slice = _to_numpy(gt_vol[..., t])
        pred_slice = _to_numpy(pred_vol[..., t])

        save_single_image(
            mask,
            os.path.join(subdirs["mask"], f"{tag}_t{t:02d}.png"),
            title=f"mask {title_suffix}",
            vmin=0,
            vmax=1,
        )
        save_single_image(
            delta,
            os.path.join(subdirs["delta"], f"{tag}_t{t:02d}.png"),
            title=f"delta {title_suffix}",
            cmap="seismic",
            vmin=-delta_global_vmax,
            vmax=delta_global_vmax,
        )
        save_single_image(
            fused,
            os.path.join(subdirs["fused"], f"{tag}_t{t:02d}.png"),
            title=f"fused {title_suffix}",
            vmin=0,
            vmax=1,
        )
        save_single_image(
            gt_slice,
            os.path.join(subdirs["gt"], f"{tag}_t{t:02d}.png"),
            title=f"GT {title_suffix}",
            vmin=0,
            vmax=1,
        )
        save_single_image(
            pred_slice,
            os.path.join(subdirs["pred"], f"{tag}_t{t:02d}.png"),
            title=f"Pred {title_suffix}",
            vmin=0,
            vmax=1,
        )

        mask_strip.append(mask)
        delta_strip.append(delta)
        gt_strip.append(gt_slice)
        pred_strip.append(pred_slice)

    def _save_t_strip(images, save_path, title, cmap="gray", vmin=None, vmax=None):
        panels = [(f"T{i:02d}", img, cmap, vmin, vmax) for i, img in enumerate(images)]
        save_panel_grid(panels, save_path, ncols=T, figsize_per_col=2.2, suptitle=title)

    strip_dir = os.path.join(save_dir, "strips")
    os.makedirs(strip_dir, exist_ok=True)
    _save_t_strip(flow_w0_strip, os.path.join(strip_dir, f"{tag}_flow_w0_all_T.png"), "Flow w0 — all T")
    _save_t_strip(flow_w1_strip, os.path.join(strip_dir, f"{tag}_flow_w1_all_T.png"), "Flow w1 — all T")
    _save_t_strip(warped0_strip, os.path.join(strip_dir, f"{tag}_warped0_all_T.png"), "Warped0 — all T", vmin=0, vmax=1)
    _save_t_strip(warped1_strip, os.path.join(strip_dir, f"{tag}_warped1_all_T.png"), "Warped1 — all T", vmin=0, vmax=1)
    _save_t_strip(mask_strip, os.path.join(strip_dir, f"{tag}_mask_all_T.png"), "Mask — all T", vmin=0, vmax=1)
    _save_t_strip(
        delta_strip,
        os.path.join(strip_dir, f"{tag}_delta_all_T.png"),
        "Delta — all T",
        cmap="seismic",
        vmin=-delta_global_vmax,
        vmax=delta_global_vmax,
    )
    _save_t_strip(gt_strip, os.path.join(strip_dir, f"{tag}_gt_all_T.png"), "GT — all T", vmin=0, vmax=1)
    _save_t_strip(pred_strip, os.path.join(strip_dir, f"{tag}_pred_all_T.png"), "Pred — all T", vmin=0, vmax=1)


def save_all_features(vis, hr_idx, gt_vol, pred_vol, save_dir, tag):
    os.makedirs(save_dir, exist_ok=True)
    T = vis["mask"].shape[1]

    # 1) 光流 / 重采样 / mask / tail / delta — 每个 T 一张
    save_per_t_bt_series(vis, gt_vol, pred_vol, save_dir, tag, args.upscale)

    # 2) Head / Alignment / Body / AlignLocal / FuseAlign — 空间特征 (B,C,H,W)
    feat_root = os.path.join(save_dir, "features")
    save_spatial_feat(vis["head"], os.path.join(feat_root, "head"), "head", tag, "Head output")

    for i, feat in enumerate(vis["alignment_out"]):
        save_spatial_feat(
            feat,
            os.path.join(feat_root, "alignment_out"),
            f"align{i}_out",
            tag,
            f"Alignment[{i}] module out",
        )
    for i, feat in enumerate(vis["alignment_res"]):
        save_spatial_feat(
            feat,
            os.path.join(feat_root, "alignment_res"),
            f"align{i}_res",
            tag,
            f"Alignment[{i}] + residual",
        )

    for i, feat in enumerate(vis["body_out"]):
        save_spatial_feat(
            feat,
            os.path.join(feat_root, "body"),
            f"body{i:02d}",
            tag,
            f"Body[{i}] output",
        )

    save_spatial_feat(
        vis["fuse_align"],
        os.path.join(feat_root, "fuse_align"),
        "fuse_align",
        tag,
        "FuseAlign (before align)",
    )
    save_spatial_feat(
        vis["align_out"],
        os.path.join(feat_root, "align"),
        "align",
        tag,
        "Align LocalMMF (before tail)",
    )

    # 3) 参考 slice 拼图
    gt_slice = _to_numpy(gt_vol[..., hr_idx])
    pred_slice = _to_numpy(pred_vol[..., hr_idx])
    flow_w0, flow_w1, pair_idx, time_frac, is_key = get_flow_for_hr_t(
        vis["flow_list"], hr_idx, args.upscale
    )
    warped0 = _to_numpy(vis["warped0"][0, hr_idx])
    warped1 = _to_numpy(vis["warped1"][0, hr_idx])
    mask = _to_numpy(vis["mask"][0, hr_idx])
    delta = _to_numpy(vis["delta"][0, hr_idx])
    fused = _to_numpy(vis["fused"][0, hr_idx])
    delta_vmax = max(np.abs(delta).max(), 1e-6)
    key_tag = "key" if is_key else f"interp_t{time_frac:.2f}"

    align_panels = [
        (f"Align[{i}] out", feat_to_heatmap(f), "viridis", 0, 1)
        for i, f in enumerate(vis["alignment_out"])
    ]
    save_panel_grid(
        align_panels,
        os.path.join(save_dir, f"{tag}_03_alignment_out.png"),
        ncols=min(3, len(align_panels)),
        suptitle="Alignment — module outputs",
    )

    body_panels = [
        (f"Body[{i}]", feat_to_heatmap(f), "viridis", 0, 1)
        for i, f in enumerate(vis["body_out"])
    ]
    save_panel_grid(
        body_panels,
        os.path.join(save_dir, f"{tag}_04_body.png"),
        ncols=4,
        suptitle="Body — each layer output (before tail)",
    )

    save_panel_grid(
        [
            ("FuseAlign", feat_to_heatmap(vis["fuse_align"]), "viridis", 0, 1),
            ("Align (LocalMMF)", feat_to_heatmap(vis["align_out"]), "viridis", 0, 1),
        ],
        os.path.join(save_dir, f"{tag}_05_fuse_align.png"),
        ncols=2,
        suptitle="FuseAlign → Align (LocalMMF, win=5)",
    )

    save_panel_grid(
        [
            ("Warped w0", warped0, "gray", 0, 1),
            ("Warped w1", warped1, "gray", 0, 1),
            ("|w0 - w1|", np.abs(warped0 - warped1), "hot", 0, None),
        ],
        os.path.join(save_dir, f"{tag}_02_resample.png"),
        ncols=3,
        suptitle=f"Resampled Slices (T={hr_idx})",
    )

    save_panel_grid(
        [
            ("Head", feat_to_heatmap(vis["head"]), "viridis", 0, 1),
            ("FuseAlign", feat_to_heatmap(vis["fuse_align"]), "viridis", 0, 1),
            ("Align", feat_to_heatmap(vis["align_out"]), "viridis", 0, 1),
            ("Mask", mask, "gray", 0, 1),
            ("Delta", delta, "seismic", -delta_vmax, delta_vmax),
            ("Fused", fused, "gray", 0, 1),
            ("GT", gt_slice, "gray", 0, 1),
            ("Pred", pred_slice, "gray", 0, 1),
            ("|Pred-GT|", np.abs(pred_slice - gt_slice), "hot", 0, None),
        ],
        os.path.join(save_dir, f"{tag}_08_tail_mask_delta.png"),
        ncols=4,
        suptitle=f"Head / FuseAlign / Align / Mask / Delta (T={hr_idx})",
    )

    overview = [
        (f"Flow w0 {key_tag}", flow_tensor_to_rgb(flow_w0), None, None, None),
        (f"Flow w1 {key_tag}", flow_tensor_to_rgb(flow_w1), None, None, None),
        ("Warped w0", warped0, "gray", 0, 1),
        ("Warped w1", warped1, "gray", 0, 1),
        ("Head", feat_to_heatmap(vis["head"]), "viridis", 0, 1),
        ("FuseAlign", feat_to_heatmap(vis["fuse_align"]), "viridis", 0, 1),
        ("Align", feat_to_heatmap(vis["align_out"]), "viridis", 0, 1),
        ("Mask", mask, "gray", 0, 1),
        ("Delta", delta, "seismic", -delta_vmax, delta_vmax),
        ("GT", gt_slice, "gray", 0, 1),
        ("Pred", pred_slice, "gray", 0, 1),
    ]
    if vis["alignment_out"]:
        overview.insert(4, ("Align[0]", feat_to_heatmap(vis["alignment_out"][0]), "viridis", 0, 1))
    if len(vis["body_out"]) > 0:
        overview.insert(5, ("Body[0]", feat_to_heatmap(vis["body_out"][0]), "viridis", 0, 1))
    if len(vis["body_out"]) > 3:
        overview.insert(6, ("Body[3]", feat_to_heatmap(vis["body_out"][3]), "viridis", 0, 1))
    save_panel_grid(
        overview,
        os.path.join(save_dir, f"{tag}_00_overview.png"),
        ncols=4,
        figsize_per_col=3.2,
        suptitle=f"I3NET Feature Overview — {tag} (HR slice {hr_idx})",
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


@torch.no_grad()
def process_one_volume(model, gt, name, out_root, args):
    tmp_lr, tmp_gt, _, _ = prepare_lr_patch(gt, args)
    hr_idx = pick_hr_slice_idx(args)

    if args.cuda:
        tmp_lr = tmp_lr.cuda()
    pred, vis = model.forward_with_vis(tmp_lr)
    pred = torch.clamp(pred.squeeze(0), 0, 1).cpu()
    tmp_gt_np = _to_numpy(tmp_gt)
    pred_np = _to_numpy(pred)
    tag = str(name).replace(os.sep, "_").replace(".", "_")

    vol_dir = os.path.join(out_root, tag)
    save_all_features(vis, hr_idx, tmp_gt_np, pred_np, vol_dir, tag)

    mse_all = float(np.mean((pred_np - tmp_gt_np) ** 2))
    psnr_all = 10 * np.log10(1.0 / mse_all) if mse_all > 1e-12 else 99.0
    T = vis["mask"].shape[1]
    print(f"  [{tag}] T={T} | PSNR(all)={psnr_all:.3f} dB | saved to {vol_dir}")
    return {"name": tag, "num_T": T, "psnr": f"{psnr_all:.3f}"}


def main():
    if not args.ckpt:
        raise ValueError("请通过 --ckpt 指定模型权重路径")

    out_root = args.vis_out_dir
    os.makedirs(out_root, exist_ok=True)

    model = load_model(args)
    if not hasattr(model, "forward_with_vis"):
        raise AttributeError(
            "当前模型不支持 forward_with_vis，请确认使用的是 model_zoo/net.py 中的 Net"
        )

    testset = testSet(data_root=args.testdata_path, image_size=args.image_size)
    if args.sample_idx >= 0:
        indices = [args.sample_idx]
    else:
        indices = list(range(min(args.num_samples, len(testset))))

    rows = []
    for idx in indices:
        name, volume, _, _ = testset[idx]
        print(f"Processing [{idx + 1}/{len(indices)}] {name} ...")
        row = process_one_volume(model, volume, name, out_root, args)
        rows.append(row)

    summary_path = os.path.join(out_root, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("I3NET Feature Visualization Summary (New Net)\n")
        f.write("=" * 40 + "\n")
        f.write(f"checkpoint: {args.ckpt}\n")
        f.write(f"testdata:   {args.testdata_path}\n")
        f.write(f"ref_slice:  {pick_hr_slice_idx(args)} (拼图参考帧)\n\n")
        for row in rows:
            f.write(f"{row['name']}: T={row['num_T']}, PSNR={row['psnr']} dB\n")

    print(f"\nDone. Results saved to: {out_root}")
    print("  - summary.txt")
    print("  - <volume_name>/flow_w0|flow_w1/t00.png ...  (每个 T 光流)")
    print("  - <volume_name>/mask|delta/t00.png ...")
    print("  - <volume_name>/features/head|body|alignment_*|fuse_align|align")
    print("  - <volume_name>/00_overview.png")


if __name__ == "__main__":
    main()
