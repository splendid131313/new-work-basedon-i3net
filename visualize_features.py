"""
I3NET 中间特征可视化脚本。

展示内容:
    - 光流 (flow01 / flow10)
    - 重采样图 (warped0 / warped1)
    - 融合消融指标 (warp0/warp1/warp_avg/fused_no_delta/final 的 PSNR/SSIM, delta 统计)
    - Encoder 浅层 / 深层特征 (通道均值 heatmap + 全通道网格)
    - 频率分支 (MidPFG) 浅层 / 深层特征 + 频率门控图 (全通道网格)
    - Decoder 浅层 / 深层特征 (全通道网格)
    - Tail 前输出 (before_tail)、mask、delta、融合结果

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
    help="HR 网格上的 slice 索引, -1 表示取中间插值 slice",
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
from util_evaluation import calc_psnr, calc_ssim


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
    model.load_state_dict(state)
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


def bt_index(t, batch_size=1):
    """B,T 展平为 B*T 时的行索引 (默认 B=1)。"""
    return batch_size * t + 0


def feat_to_heatmap(feat, reduce="mean"):
    """
    将 (C, H, W) 或 (H, W) 特征图转为可显示的 2D heatmap。
    """
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


def normalize_channels(feat):
    """(C, H, W) 或 (H, W) -> 逐通道 min-max 归一化。"""
    arr = _to_numpy(feat)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4:
        arr = arr[0]
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        ch = arr[c]
        ch = ch - ch.min()
        out[c] = ch / (ch.max() + 1e-8)
    return out


def save_channel_grid(
    feat,
    save_path,
    title=None,
    ncols=8,
    cmap="viridis",
    max_channels=None,
    cell_size=1.5,
):
    """
    将 (C, H, W) 特征的所有通道排列成网格图保存。
    (H, W) 单通道输入会显示为 1 格。
    """
    arr = normalize_channels(feat)
    if max_channels is not None:
        arr = arr[: max(1, min(max_channels, arr.shape[0]))]
    c = arr.shape[0]
    nrows = (c + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * cell_size, nrows * cell_size),
        squeeze=False,
    )
    for idx in range(nrows * ncols):
        r, col = divmod(idx, ncols)
        ax = axes[r, col]
        if idx < c:
            ax.imshow(arr[idx], cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"c{idx:02d}", fontsize=6)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def flow_tensor_to_rgb(flow):
    """(2, H, W) tensor -> RGB numpy [H, W, 3]."""
    arr = _to_numpy(flow)
    if arr.ndim == 4:
        arr = arr[0]
    uv = np.transpose(arr, (1, 2, 0))
    return flow_to_image(uv)


def save_panel_grid(panels, save_path, ncols=4, figsize_per_col=3.5, suptitle=None):
    n = len(panels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_col * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (title, img, cmap, vmin, vmax) in zip(axes, panels):
        kwargs = {"cmap": cmap}
        if vmin is not None:
            kwargs["vmin"] = vmin
        if vmax is not None:
            kwargs["vmax"] = vmax
        if img.ndim == 3 and img.shape[-1] == 3:
            ax.imshow(img)
        else:
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


def save_freq_maps(freq_maps, save_path, title_prefix="Freq"):
    """
    freq_maps: (3, H, W) — PFGA 的 grad / laplacian / variance 门控输入。
    """
    arr = _to_numpy(freq_maps)
    if arr.ndim == 4:
        arr = arr[0]
    names = ["Gradient", "Laplacian", "Variance"]
    panels = []
    for i, name in enumerate(names):
        ch = arr[i]
        ch = ch - ch.min()
        ch = ch / (ch.max() + 1e-8)
        panels.append((f"{title_prefix} | {name}", ch, "magma", 0, 1))
    combined = arr.mean(axis=0)
    combined = combined - combined.min()
    combined = combined / (combined.max() + 1e-8)
    panels.append((f"{title_prefix} | Combined", combined, "magma", 0, 1))
    save_panel_grid(panels, save_path, ncols=4, suptitle=f"{title_prefix} Gating Maps (PFGA)")


def save_channel_grids_for_vis(vis, hr_idx, save_dir, tag, ncols=8):
    """
    保存 encoder / decoder / freq 全通道网格，以及 tail 前单通道输出。
    参考 slice (hr_idx) 存汇总图；每个 T 在 channel_grids/<name>/ 下各存一张。
    """
    grid_root = os.path.join(save_dir, "channel_grids")
    T = vis["mask"].shape[1]
    bt = bt_index(hr_idx)

    grid_specs = (
        ("encoder_shallow", vis["encoder_shallow"], "Encoder shallow (skip)"),
        ("encoder_deep", vis["encoder_deep"], "Encoder deep (embed)"),
        ("freq_shallow", vis["freq_shallow_feat"], "Freq shallow (MSInit)"),
        ("freq_deep", vis["freq_deep_feat"], "Freq deep (PFG blocks)"),
        ("decoder_shallow", vis["decoder_shallow"], "Decoder shallow"),
        ("decoder_deep", vis["decoder_deep"], "Decoder deep"),
    )

    for name, _, _ in grid_specs:
        os.makedirs(os.path.join(grid_root, name), exist_ok=True)
    os.makedirs(os.path.join(grid_root, "before_tail"), exist_ok=True)
    os.makedirs(os.path.join(grid_root, "freq_maps_shallow"), exist_ok=True)
    os.makedirs(os.path.join(grid_root, "freq_maps_deep"), exist_ok=True)

    for name, tensor, label in grid_specs:
        save_channel_grid(
            tensor[bt],
            os.path.join(save_dir, f"{tag}_grid_{name}_T{hr_idx:02d}.png"),
            title=f"{label} — all channels (T{hr_idx:02d})",
            ncols=ncols,
        )

    before_tail_slice = vis["before_tail"][0, hr_idx]
    save_channel_grid(
        before_tail_slice,
        os.path.join(save_dir, f"{tag}_grid_before_tail_T{hr_idx:02d}.png"),
        title=f"Before tail — T{hr_idx:02d}",
        ncols=1,
        cmap="gray",
    )
    save_channel_grid(
        vis["freq_shallow_maps"][bt],
        os.path.join(save_dir, f"{tag}_grid_freq_maps_shallow_T{hr_idx:02d}.png"),
        title=f"Freq gating maps shallow — T{hr_idx:02d}",
        ncols=4,
        cmap="magma",
    )
    save_channel_grid(
        vis["freq_deep_maps"][bt],
        os.path.join(save_dir, f"{tag}_grid_freq_maps_deep_T{hr_idx:02d}.png"),
        title=f"Freq gating maps deep — T{hr_idx:02d}",
        ncols=4,
        cmap="magma",
    )

    for t in range(T):
        bi = bt_index(t)
        for name, tensor, label in grid_specs:
            save_channel_grid(
                tensor[bi],
                os.path.join(grid_root, name, f"{tag}_t{t:02d}.png"),
                title=f"{label} T{t:02d}",
                ncols=ncols,
            )
        save_channel_grid(
            vis["before_tail"][0, t],
            os.path.join(grid_root, "before_tail", f"{tag}_t{t:02d}.png"),
            title=f"Before tail T{t:02d}",
            ncols=1,
            cmap="gray",
        )
        save_channel_grid(
            vis["freq_shallow_maps"][bi],
            os.path.join(grid_root, "freq_maps_shallow", f"{tag}_t{t:02d}.png"),
            title=f"Freq maps shallow T{t:02d}",
            ncols=4,
            cmap="magma",
        )
        save_channel_grid(
            vis["freq_deep_maps"][bi],
            os.path.join(grid_root, "freq_maps_deep", f"{tag}_t{t:02d}.png"),
            title=f"Freq maps deep T{t:02d}",
            ncols=4,
            cmap="magma",
        )


def _interp_slice_indices(num_slices, upscale):
    return [i for i in range(num_slices) if i % upscale != 0]


def _slice_metric(pred, gt):
    """(H, W) numpy -> PSNR / SSIM。"""
    pred_t = torch.from_numpy(_to_numpy(pred)).double()
    gt_t = torch.from_numpy(_to_numpy(gt)).double()
    return {
        "psnr": float(calc_psnr(pred_t, gt_t).item()),
        "ssim": float(calc_ssim(pred_t, gt_t)),
    }


def _avg_metrics(per_slice_metrics):
    if not per_slice_metrics:
        return {"psnr": float("nan"), "ssim": float("nan")}
    return {
        "psnr": float(np.mean([m["psnr"] for m in per_slice_metrics])),
        "ssim": float(np.mean([m["ssim"] for m in per_slice_metrics])),
    }


def compute_fusion_ablation_metrics(vis, gt_vol, pred_vol, upscale):
    """
    对比各融合阶段相对 GT 的质量，用于判断 warp 提升是否被 delta 吸收。

    阶段:
        warp0 / warp1 / warp_avg / warp_best(oracle)
        fused_no_delta = mask*w0 + (1-mask)*w1
        pre_anchor     = fused_no_delta + delta (替换关键帧前)
        final          = 模型最终输出 (含关键帧锚定)
    """
    warped0 = _to_numpy(vis["warped0"][0])
    warped1 = _to_numpy(vis["warped1"][0])
    mask = _to_numpy(vis["mask"][0])
    delta = _to_numpy(vis["delta"][0])
    gt = _to_numpy(gt_vol)
    pred = _to_numpy(pred_vol)
    T = mask.shape[0]

    warp_avg_vol = 0.5 * (warped0 + warped1)
    fused_no_delta_vol = mask * warped0 + (1.0 - mask) * warped1
    pre_anchor_vol = fused_no_delta_vol + delta

    per_t = []
    for t in range(T):
        gt_slice = gt[..., t]
        m_w0 = _slice_metric(warped0[t], gt_slice)
        m_w1 = _slice_metric(warped1[t], gt_slice)
        row = {
            "t": t,
            "is_interp": t % upscale != 0,
            "warp0": m_w0,
            "warp1": m_w1,
            "warp_avg": _slice_metric(warp_avg_vol[t], gt_slice),
            "warp_best": m_w0 if m_w0["ssim"] >= m_w1["ssim"] else m_w1,
            "fused_no_delta": _slice_metric(fused_no_delta_vol[t], gt_slice),
            "pre_anchor": _slice_metric(pre_anchor_vol[t], gt_slice),
            "final": _slice_metric(pred[..., t], gt_slice),
            "delta_mean_abs": float(np.mean(np.abs(delta[t]))),
            "delta_max_abs": float(np.max(np.abs(delta[t]))),
            "delta_rmse": float(np.sqrt(np.mean(delta[t] ** 2))),
            "mask_mean": float(np.mean(mask[t])),
        }
        per_t.append(row)

    interp_rows = [r for r in per_t if r["is_interp"]]
    key_rows = [r for r in per_t if not r["is_interp"]]

    def _collect(stage):
        return _avg_metrics([r[stage] for r in per_t])

    def _collect_interp(stage):
        return _avg_metrics([r[stage] for r in interp_rows])

    delta_interp = interp_rows or per_t
    return {
        "num_T": T,
        "num_interp": len(interp_rows),
        "num_key": len(key_rows),
        "per_t": per_t,
        "avg_all": {stage: _collect(stage) for stage in (
            "warp0", "warp1", "warp_avg", "warp_best", "fused_no_delta", "pre_anchor", "final"
        )},
        "avg_interp": {stage: _collect_interp(stage) for stage in (
            "warp0", "warp1", "warp_avg", "warp_best", "fused_no_delta", "pre_anchor", "final"
        )},
        "delta_stats": {
            "mean_abs": float(np.mean([r["delta_mean_abs"] for r in delta_interp])),
            "max_abs": float(np.max([r["delta_max_abs"] for r in delta_interp])),
            "rmse": float(np.mean([r["delta_rmse"] for r in delta_interp])),
            "mask_mean": float(np.mean([r["mask_mean"] for r in delta_interp])),
        },
        "volumes": {
            "warp_avg": warp_avg_vol,
            "fused_no_delta": fused_no_delta_vol,
            "pre_anchor": pre_anchor_vol,
        },
    }


def save_fusion_metrics_report(metrics, save_path):
    """将融合消融指标写入文本报告。"""
    lines = [
        "Fusion Ablation Metrics (vs GT)",
        "=" * 72,
        f"T={metrics['num_T']} | interp={metrics['num_interp']} | key={metrics['num_key']}",
        "",
        "Average on ALL slices:",
        f"  {'stage':<18} {'PSNR':>10} {'SSIM':>10}",
    ]
    for stage, vals in metrics["avg_all"].items():
        lines.append(f"  {stage:<18} {vals['psnr']:10.4f} {vals['ssim']:10.6f}")

    lines.extend([
        "",
        "Average on INTERP slices only (插值帧, 与 test.py ssim_slice 一致):",
        f"  {'stage':<18} {'PSNR':>10} {'SSIM':>10}",
    ])
    for stage, vals in metrics["avg_interp"].items():
        lines.append(f"  {stage:<18} {vals['psnr']:10.4f} {vals['ssim']:10.6f}")

    ds = metrics["delta_stats"]
    lines.extend([
        "",
        "Delta stats (interp slices):",
        f"  mean(|delta|) = {ds['mean_abs']:.6f}",
        f"  max(|delta|)  = {ds['max_abs']:.6f}",
        f"  rmse(delta)   = {ds['rmse']:.6f}",
        f"  mean(mask)    = {ds['mask_mean']:.6f}",
        "",
        "Per-slice detail (interp only):",
        f"  {'T':>3} {'warp_avg':>10} {'fused':>10} {'pre_anchor':>10} {'final':>10} "
        f"{'|delta|':>10} {'mask':>8}",
    ])
    for row in metrics["per_t"]:
        if not row["is_interp"]:
            continue
        lines.append(
            f"  {row['t']:3d} "
            f"{row['warp_avg']['ssim']:10.6f} "
            f"{row['fused_no_delta']['ssim']:10.6f} "
            f"{row['pre_anchor']['ssim']:10.6f} "
            f"{row['final']['ssim']:10.6f} "
            f"{row['delta_mean_abs']:10.6f} "
            f"{row['mask_mean']:8.4f}"
        )

    lines.extend([
        "",
        "解读提示:",
        "  - 若 warp_avg/fused 提升但 final 不变 -> delta 吸收了 warp 收益",
        "  - 若仅 warp 提升、fused/final 都不变 -> 下游 mask 未利用更好 warp",
        "  - 对比两次实验的 delta_stats 可验证补偿是否减弱",
    ])

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_ablation_comparison_strip(metrics, gt_vol, pred_vol, save_dir, tag, upscale):
    """保存插值帧的 warp_avg / fused_no_delta / final / GT 横向对比条。"""
    interp_idx = _interp_slice_indices(metrics["num_T"], upscale)
    if not interp_idx:
        return

    vols = metrics["volumes"]
    gt = _to_numpy(gt_vol)
    pred = _to_numpy(pred_vol)

    panels = []
    for t in interp_idx:
        panels.append((f"T{t:02d} warp_avg", vols["warp_avg"][t], "gray", 0, 1))
        panels.append((f"T{t:02d} fused", vols["fused_no_delta"][t], "gray", 0, 1))
        panels.append((f"T{t:02d} final", pred[..., t], "gray", 0, 1))
        panels.append((f"T{t:02d} GT", gt[..., t], "gray", 0, 1))

    os.makedirs(save_dir, exist_ok=True)
    save_panel_grid(
        panels,
        os.path.join(save_dir, f"{tag}_ablation_interp_compare.png"),
        ncols=4,
        figsize_per_col=2.5,
        suptitle="Interp slices: warp_avg | fused(no delta) | final | GT",
    )


def get_flow_for_hr_t(flow01_list, flow10_list, t, upscale):
    """
    为 HR 网格上第 t 个 slice 取对应光流。
    关键帧 (t % upscale == 0) 显示该 LR 区间的双向基准光流；
    插值帧显示与 warp 一致的缩放光流: -flow01*time, -flow10*(1-time)。
    """
    pair_idx = min(t // upscale, len(flow01_list) - 1)
    flow01 = flow01_list[pair_idx]
    flow10 = flow10_list[pair_idx]
    offset = t % upscale
    if offset == 0:
        return flow01, flow10, pair_idx, 0.0, True
    time = offset / upscale
    return -flow01 * time, -flow10 * (1.0 - time), pair_idx, time, False


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


def save_per_t_bt_series(vis, gt_vol, pred_vol, save_dir, tag, upscale, ablation_vols=None):
    """
    对 B,T,H,W 张量按每个 T 保存单张图。
    包含: 光流、重采样、mask、tail、delta、融合消融中间结果。
    """
    T = vis["mask"].shape[1]
    delta_global_vmax = max(
        float(np.abs(_to_numpy(vis["delta"][0])).max()), 1e-6
    )
    if ablation_vols is None:
        warped0_np = _to_numpy(vis["warped0"][0])
        warped1_np = _to_numpy(vis["warped1"][0])
        mask_np = _to_numpy(vis["mask"][0])
        ablation_vols = {
            "warp_avg": 0.5 * (warped0_np + warped1_np),
            "fused_no_delta": mask_np * warped0_np + (1.0 - mask_np) * warped1_np,
        }

    subdirs = {
        "flow01": os.path.join(save_dir, "flow01"),
        "flow10": os.path.join(save_dir, "flow10"),
        "flow_rgb": os.path.join(save_dir, "flow_rgb"),
        "warped0": os.path.join(save_dir, "warped0"),
        "warped1": os.path.join(save_dir, "warped1"),
        "warp_avg": os.path.join(save_dir, "warp_avg"),
        "fused_no_delta": os.path.join(save_dir, "fused_no_delta"),
        "before_tail": os.path.join(save_dir, "before_tail"),
        "mask": os.path.join(save_dir, "mask"),
        "delta": os.path.join(save_dir, "delta"),
        "fused": os.path.join(save_dir, "fused"),
        "gt": os.path.join(save_dir, "gt"),
        "pred": os.path.join(save_dir, "pred"),
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    flow01_strip, flow10_strip = [], []
    warped0_strip, warped1_strip = [], []
    warp_avg_strip, fused_no_delta_strip = [], []
    mask_strip, before_tail_strip, delta_strip = [], [], []
    gt_strip, pred_strip = [], []

    for t in range(T):
        flow01, flow10, pair_idx, time_frac, is_key = get_flow_for_hr_t(
            vis["flow01_list"], vis["flow10_list"], t, upscale
        )
        key_tag = "key" if is_key else f"interp_t{time_frac:.2f}"
        title_suffix = f"T{t:02d} pair{pair_idx} {key_tag}"

        f01_rgb = flow_tensor_to_rgb(flow01)
        f10_rgb = flow_tensor_to_rgb(flow10)
        save_single_image(
            f01_rgb,
            os.path.join(subdirs["flow01"], f"{tag}_t{t:02d}_flow01.png"),
            title=f"flow01 {title_suffix}",
        )
        save_single_image(
            f10_rgb,
            os.path.join(subdirs["flow10"], f"{tag}_t{t:02d}_flow10.png"),
            title=f"flow10 {title_suffix}",
        )
        save_single_image(
            np.concatenate([f01_rgb, f10_rgb], axis=1),
            os.path.join(subdirs["flow_rgb"], f"{tag}_t{t:02d}_flow01_flow10.png"),
            title=f"flow {title_suffix}",
        )
        flow01_strip.append(f01_rgb)
        flow10_strip.append(f10_rgb)

        w0 = _to_numpy(vis["warped0"][0, t])
        w1 = _to_numpy(vis["warped1"][0, t])
        save_single_image(
            w0, os.path.join(subdirs["warped0"], f"{tag}_t{t:02d}.png"),
            title=f"warped0 {title_suffix}", vmin=0, vmax=1,
        )
        save_single_image(
            w1, os.path.join(subdirs["warped1"], f"{tag}_t{t:02d}.png"),
            title=f"warped1 {title_suffix}", vmin=0, vmax=1,
        )
        warped0_strip.append(w0)
        warped1_strip.append(w1)

        warp_avg = ablation_vols["warp_avg"][t]
        fused_no_delta = ablation_vols["fused_no_delta"][t]
        save_single_image(
            warp_avg,
            os.path.join(subdirs["warp_avg"], f"{tag}_t{t:02d}.png"),
            title=f"warp_avg {title_suffix}", vmin=0, vmax=1,
        )
        save_single_image(
            fused_no_delta,
            os.path.join(subdirs["fused_no_delta"], f"{tag}_t{t:02d}.png"),
            title=f"fused(no delta) {title_suffix}", vmin=0, vmax=1,
        )
        warp_avg_strip.append(warp_avg)
        fused_no_delta_strip.append(fused_no_delta)

        before_tail = _to_numpy(vis["before_tail"][0, t])
        mask = _to_numpy(vis["mask"][0, t])
        delta = _to_numpy(vis["delta"][0, t])
        fused = _to_numpy(vis["fused"][0, t])
        gt_slice = _to_numpy(gt_vol[..., t])
        pred_slice = _to_numpy(pred_vol[..., t])

        save_single_image(
            before_tail,
            os.path.join(subdirs["before_tail"], f"{tag}_t{t:02d}.png"),
            title=f"before tail {title_suffix}", vmin=0, vmax=1,
        )
        save_single_image(
            mask,
            os.path.join(subdirs["mask"], f"{tag}_t{t:02d}.png"),
            title=f"mask {title_suffix}", vmin=0, vmax=1,
        )
        save_single_image(
            delta,
            os.path.join(subdirs["delta"], f"{tag}_t{t:02d}.png"),
            title=f"delta {title_suffix}",
            cmap="seismic", vmin=-delta_global_vmax, vmax=delta_global_vmax,
        )
        save_single_image(
            fused,
            os.path.join(subdirs["fused"], f"{tag}_t{t:02d}.png"),
            title=f"fused {title_suffix}", vmin=0, vmax=1,
        )
        save_single_image(
            gt_slice,
            os.path.join(subdirs["gt"], f"{tag}_t{t:02d}.png"),
            title=f"GT {title_suffix}", vmin=0, vmax=1,
        )
        save_single_image(
            pred_slice,
            os.path.join(subdirs["pred"], f"{tag}_t{t:02d}.png"),
            title=f"Pred {title_suffix}", vmin=0, vmax=1,
        )

        before_tail_strip.append(before_tail)
        mask_strip.append(mask)
        delta_strip.append(delta)
        gt_strip.append(gt_slice)
        pred_strip.append(pred_slice)

    def _save_t_strip(images, save_path, title, cmap="gray", vmin=None, vmax=None):
        panels = [(f"T{i:02d}", img, cmap, vmin, vmax) for i, img in enumerate(images)]
        save_panel_grid(panels, save_path, ncols=T, figsize_per_col=2.2, suptitle=title)

    strip_dir = os.path.join(save_dir, "strips")
    os.makedirs(strip_dir, exist_ok=True)
    _save_t_strip(flow01_strip, os.path.join(strip_dir, f"{tag}_flow01_all_T.png"), "Flow01 — all T")
    _save_t_strip(flow10_strip, os.path.join(strip_dir, f"{tag}_flow10_all_T.png"), "Flow10 — all T")
    _save_t_strip(warped0_strip, os.path.join(strip_dir, f"{tag}_warped0_all_T.png"), "Warped0 — all T", vmin=0, vmax=1)
    _save_t_strip(warped1_strip, os.path.join(strip_dir, f"{tag}_warped1_all_T.png"), "Warped1 — all T", vmin=0, vmax=1)
    _save_t_strip(warp_avg_strip, os.path.join(strip_dir, f"{tag}_warp_avg_all_T.png"), "Warp avg — all T", vmin=0, vmax=1)
    _save_t_strip(
        fused_no_delta_strip,
        os.path.join(strip_dir, f"{tag}_fused_no_delta_all_T.png"),
        "Fused (no delta) — all T",
        vmin=0, vmax=1,
    )
    _save_t_strip(before_tail_strip, os.path.join(strip_dir, f"{tag}_before_tail_all_T.png"), "Before Tail — all T", vmin=0, vmax=1)
    _save_t_strip(mask_strip, os.path.join(strip_dir, f"{tag}_mask_all_T.png"), "Mask — all T", vmin=0, vmax=1)
    _save_t_strip(
        delta_strip, os.path.join(strip_dir, f"{tag}_delta_all_T.png"), "Delta — all T",
        cmap="seismic", vmin=-delta_global_vmax, vmax=delta_global_vmax,
    )
    _save_t_strip(gt_strip, os.path.join(strip_dir, f"{tag}_gt_all_T.png"), "GT — all T", vmin=0, vmax=1)
    _save_t_strip(pred_strip, os.path.join(strip_dir, f"{tag}_pred_all_T.png"), "Pred — all T", vmin=0, vmax=1)


def save_all_features(vis, hr_idx, gt_vol, pred_vol, save_dir, tag, ablation_metrics=None):
    """
    保存全部中间结果。B,T,H,W 量 (光流/重采样/mask/tail/delta) 按每个 T 各存一张；
    encoder/decoder/freq 特征同样按 T 索引一一对应。
    """
    os.makedirs(save_dir, exist_ok=True)
    T = vis["mask"].shape[1]
    ablation_vols = None if ablation_metrics is None else ablation_metrics["volumes"]

    # 1) 光流 / 重采样 / mask / tail / delta — 每个 T 一张
    save_per_t_bt_series(
        vis, gt_vol, pred_vol, save_dir, tag, args.upscale, ablation_vols=ablation_vols
    )

    # 1b) encoder / decoder / freq / before_tail 全通道网格
    save_channel_grids_for_vis(vis, hr_idx, save_dir, tag, ncols=8)

    # 2) 以下特征图仍额外保存 hr_idx 参考 slice 的对比拼图
    bt = bt_index(hr_idx)
    gt_slice = _to_numpy(gt_vol[..., hr_idx])
    pred_slice = _to_numpy(pred_vol[..., hr_idx])
    flow01, flow10, pair_idx, time_frac, is_key = get_flow_for_hr_t(
        vis["flow01_list"], vis["flow10_list"], hr_idx, args.upscale
    )
    warped0 = _to_numpy(vis["warped0"][0, hr_idx])
    warped1 = _to_numpy(vis["warped1"][0, hr_idx])

    enc_shallow = feat_to_heatmap(vis["encoder_shallow"][bt])
    enc_deep = feat_to_heatmap(vis["encoder_deep"][bt])
    freq_shallow = feat_to_heatmap(vis["freq_shallow_feat"][bt])
    freq_deep = feat_to_heatmap(vis["freq_deep_feat"][bt])
    dec_shallow = feat_to_heatmap(vis["decoder_shallow"][bt])
    dec_deep = feat_to_heatmap(vis["decoder_deep"][bt])

    before_tail = _to_numpy(vis["before_tail"][0, hr_idx])
    raw_tail = _to_numpy(vis["raw_tail"][0, hr_idx])
    mask = _to_numpy(vis["mask"][0, hr_idx])
    delta = _to_numpy(vis["delta"][0, hr_idx])
    fused = _to_numpy(vis["fused"][0, hr_idx])

    # 每个 T 的 encoder/decoder/freq 特征
    feat_root = os.path.join(save_dir, "features")
    for name in ("encoder_shallow", "encoder_deep", "freq_shallow", "freq_deep", "decoder_shallow", "decoder_deep"):
        os.makedirs(os.path.join(feat_root, name), exist_ok=True)
    for t in range(T):
        bi = bt_index(t)
        enc_s = feat_to_heatmap(vis["encoder_shallow"][bi])
        enc_d = feat_to_heatmap(vis["encoder_deep"][bi])
        fq_s = feat_to_heatmap(vis["freq_shallow_feat"][bi])
        fq_d = feat_to_heatmap(vis["freq_deep_feat"][bi])
        dc_s = feat_to_heatmap(vis["decoder_shallow"][bi])
        dc_d = feat_to_heatmap(vis["decoder_deep"][bi])
        for fname, img in (
            ("encoder_shallow", enc_s), ("encoder_deep", enc_d),
            ("freq_shallow", fq_s), ("freq_deep", fq_d),
            ("decoder_shallow", dc_s), ("decoder_deep", dc_d),
        ):
            save_single_image(
                img,
                os.path.join(feat_root, fname, f"{tag}_t{t:02d}.png"),
                title=f"{fname} T{t:02d}",
                cmap="viridis", vmin=0, vmax=1,
            )

    # 3) Encoder (参考 slice)
    save_panel_grid(
        [
            ("Encoder shallow (skip)", enc_shallow, "viridis", 0, 1),
            ("Encoder deep (embed)", enc_deep, "viridis", 0, 1),
        ],
        os.path.join(save_dir, f"{tag}_03_encoder.png"),
        ncols=2,
        suptitle="Encoder Features",
    )

    # 4) 频率分支 (参考 slice)
    save_panel_grid(
        [
            ("Freq shallow (MSInit)", freq_shallow, "viridis", 0, 1),
            ("Freq deep (PFG blocks)", freq_deep, "viridis", 0, 1),
        ],
        os.path.join(save_dir, f"{tag}_04_freq_feat.png"),
        ncols=2,
        suptitle="Frequency Branch Features (MidPFG)",
    )
    save_freq_maps(
        vis["freq_shallow_maps"][bt],
        os.path.join(save_dir, f"{tag}_05_freq_shallow_maps.png"),
        title_prefix=f"Shallow T{hr_idx:02d}",
    )
    save_freq_maps(
        vis["freq_deep_maps"][bt],
        os.path.join(save_dir, f"{tag}_06_freq_deep_maps.png"),
        title_prefix=f"Deep T{hr_idx:02d}",
    )
    freq_map_root = os.path.join(save_dir, "features", "freq_maps")
    os.makedirs(os.path.join(freq_map_root, "shallow"), exist_ok=True)
    os.makedirs(os.path.join(freq_map_root, "deep"), exist_ok=True)
    for t in range(T):
        bi = bt_index(t)
        save_freq_maps(
            vis["freq_shallow_maps"][bi],
            os.path.join(freq_map_root, "shallow", f"{tag}_t{t:02d}.png"),
            title_prefix=f"Shallow T{t:02d}",
        )
        save_freq_maps(
            vis["freq_deep_maps"][bi],
            os.path.join(freq_map_root, "deep", f"{tag}_t{t:02d}.png"),
            title_prefix=f"Deep T{t:02d}",
        )

    # 5) Decoder (参考 slice)
    save_panel_grid(
        [
            ("Decoder shallow", dec_shallow, "viridis", 0, 1),
            ("Decoder deep", dec_deep, "viridis", 0, 1),
        ],
        os.path.join(save_dir, f"{tag}_07_decoder.png"),
        ncols=2,
        suptitle="Decoder Features",
    )

    # 6) Tail / mask / delta (参考 slice 拼图)
    delta_vis = delta.copy()
    delta_vmax = max(np.abs(delta_vis).max(), 1e-6)
    save_panel_grid(
        [
            ("Decoder out (before tail)", before_tail, "gray", 0, 1),
            ("Tail raw (ch0)", raw_tail, "gray", None, None),
            ("Mask (sigmoid)", mask, "gray", 0, 1),
            ("Delta", delta_vis, "seismic", -delta_vmax, delta_vmax),
            ("Fused (before anchor)", fused, "gray", 0, 1),
            ("GT", gt_slice, "gray", 0, 1),
            ("Pred", pred_slice, "gray", 0, 1),
            ("|Pred - GT|", np.abs(pred_slice - gt_slice), "hot", 0, None),
        ],
        os.path.join(save_dir, f"{tag}_08_tail_mask_delta.png"),
        ncols=4,
        suptitle="Tail / Mask / Delta / Output",
    )

    # 7) 总览拼图 (参考 slice)
    key_tag = "key" if is_key else f"interp_t{time_frac:.2f}"
    overview = [
        (f"Flow01 {key_tag}", flow_tensor_to_rgb(flow01), None, None, None),
        ("Warped w0", warped0, "gray", 0, 1),
        ("Warped w1", warped1, "gray", 0, 1),
        ("Enc shallow", enc_shallow, "viridis", 0, 1),
        ("Enc deep", enc_deep, "viridis", 0, 1),
        ("Freq shallow", freq_shallow, "viridis", 0, 1),
        ("Freq deep", freq_deep, "viridis", 0, 1),
        ("Dec shallow", dec_shallow, "viridis", 0, 1),
        ("Dec deep", dec_deep, "viridis", 0, 1),
        ("Before tail", before_tail, "gray", 0, 1),
        ("Mask", mask, "gray", 0, 1),
        ("Delta", delta_vis, "seismic", -delta_vmax, delta_vmax),
        ("GT", gt_slice, "gray", 0, 1),
        ("Pred", pred_slice, "gray", 0, 1),
    ]
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
    tmp_gt = tmp_gt.cpu()

    tmp_gt_np = _to_numpy(tmp_gt)
    pred_np = _to_numpy(pred)
    tag = str(name).replace(os.sep, "_").replace(".", "_")

    ablation_metrics = compute_fusion_ablation_metrics(
        vis, tmp_gt_np, pred_np, args.upscale
    )

    vol_dir = os.path.join(out_root, tag)
    save_all_features(vis, hr_idx, tmp_gt_np, pred_np, vol_dir, tag, ablation_metrics)
    save_fusion_metrics_report(
        ablation_metrics, os.path.join(vol_dir, "fusion_ablation_metrics.txt")
    )
    save_ablation_comparison_strip(
        ablation_metrics, tmp_gt_np, pred_np, vol_dir, tag, args.upscale
    )

    mse_all = float(np.mean((pred_np - tmp_gt_np) ** 2))
    psnr_all = 10 * np.log10(1.0 / mse_all) if mse_all > 1e-12 else 99.0
    T = vis["mask"].shape[1]
    interp_avg = ablation_metrics["avg_interp"]
    print(
        f"  [{tag}] T={T} | PSNR(all)={psnr_all:.3f} dB | "
        f"SSIM(interp): warp_avg={interp_avg['warp_avg']['ssim']:.4f} "
        f"fused={interp_avg['fused_no_delta']['ssim']:.4f} "
        f"final={interp_avg['final']['ssim']:.4f} | "
        f"|delta|={ablation_metrics['delta_stats']['mean_abs']:.5f}"
    )
    return {
        "name": tag,
        "num_T": T,
        "psnr": f"{psnr_all:.3f}",
        "ablation": ablation_metrics,
    }


def main():
    if not args.ckpt:
        raise ValueError("请通过 --ckpt 指定模型权重路径")

    out_root = args.vis_out_dir
    os.makedirs(out_root, exist_ok=True)

    model = load_model(args)
    if not hasattr(model, "forward_with_vis"):
        raise AttributeError(
            "当前模型不支持 forward_with_vis，请确认使用的是 I3Net (model_zoo/net.py)"
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

    def _mean_interp_metric(rows, stage, key):
        vals = [r["ablation"]["avg_interp"][stage][key] for r in rows]
        return float(np.mean(vals)) if vals else float("nan")

    summary_path = os.path.join(out_root, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("I3NET Feature Visualization Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"checkpoint: {args.ckpt}\n")
        f.write(f"flow_ckpt:  {args.flow_ckpt}\n")
        f.write(f"testdata:   {args.testdata_path}\n")
        f.write(f"ref_slice:  {pick_hr_slice_idx(args)} (拼图参考帧)\n\n")

        f.write("Per-volume PSNR (all slices):\n")
        for row in rows:
            f.write(f"  {row['name']}: T={row['num_T']}, PSNR={row['psnr']} dB\n")

        f.write("\nFusion ablation — avg SSIM on INTERP slices:\n")
        f.write(f"  {'volume':<24} {'warp_avg':>10} {'fused':>10} {'final':>10} {'|delta|':>10}\n")
        for row in rows:
            ia = row["ablation"]["avg_interp"]
            ds = row["ablation"]["delta_stats"]
            f.write(
                f"  {row['name']:<24} "
                f"{ia['warp_avg']['ssim']:10.6f} "
                f"{ia['fused_no_delta']['ssim']:10.6f} "
                f"{ia['final']['ssim']:10.6f} "
                f"{ds['mean_abs']:10.6f}\n"
            )

        if len(rows) > 1:
            f.write("\nDataset mean (interp SSIM):\n")
            for stage in ("warp0", "warp1", "warp_avg", "warp_best", "fused_no_delta", "pre_anchor", "final"):
                f.write(f"  {stage:<18} SSIM={_mean_interp_metric(rows, stage, 'ssim'):.6f}\n")
            mean_delta = float(np.mean([r["ablation"]["delta_stats"]["mean_abs"] for r in rows]))
            f.write(f"  {'mean(|delta|)':<18} {mean_delta:.6f}\n")

    print(f"\nDone. Results saved to: {out_root}")
    print("  - summary.txt (含融合消融 SSIM 汇总)")
    print("  - <volume_name>/flow01/t00.png ...     (每个 T 一张光流)")
    print("  - <volume_name>/mask/t00.png ...       (每个 T 一张 mask)")
    print("  - <volume_name>/before_tail/t00.png ...")
    print("  - <volume_name>/delta/t00.png ...")
    print("  - <volume_name>/warped0|warped1|warp_avg|fused_no_delta/t00.png ...")
    print("  - <volume_name>/fusion_ablation_metrics.txt  (各阶段 PSNR/SSIM)")
    print("  - <volume_name>/*_ablation_interp_compare.png  (插值帧对比条)")
    print("  - <volume_name>/strips/*_all_T.png     (所有 T 横向拼接)")
    print("  - <volume_name>/00_overview.png        (参考 slice 总览拼图)")
    print("  - <volume_name>/*_grid_*.png           (参考 slice 全通道网格)")
    print("  - <volume_name>/channel_grids/         (每个 T 的全通道网格)")


if __name__ == "__main__":
    main()
