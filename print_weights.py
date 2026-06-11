"""加载 I3NET 模型/checkpoint 并打印权重参数信息。"""

import os

import config
from select_model import select_model

args, _ = config.get_args()


def extract_state_dict(checkpoint):
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
    return state


def print_params(model, title="模型参数"):
    total = 0
    trainable = 0
    print("=" * 90)
    print(title)
    print(f"{'参数名':<62} {'形状':<22} {'参数量':>12}")
    print("-" * 90)
    for name, param in model.named_parameters():
        num = param.numel()
        total += num
        if param.requires_grad:
            trainable += num
        print(f"{name:<62} {str(tuple(param.shape)):<22} {num:>12,}")
    print("-" * 90)
    print(
        f"总参数量: {total:,}  |  可训练: {trainable:,}  |  冻结: {total - trainable:,}"
    )
    print("=" * 90)


def print_module_summary(model):
    summary = {}
    for name, param in model.named_parameters():
        top = name.split(".", 1)[0]
        summary.setdefault(top, {"total": 0, "trainable": 0})
        summary[top]["total"] += param.numel()
        if param.requires_grad:
            summary[top]["trainable"] += param.numel()

    print("\n按顶层模块汇总:")
    print(f"{'模块':<20} {'总参数量':>14} {'可训练':>14} {'冻结':>14}")
    print("-" * 66)
    for name, stat in summary.items():
        frozen = stat["total"] - stat["trainable"]
        print(f"{name:<20} {stat['total']:>14,} {stat['trainable']:>14,} {frozen:>14,}")


def main():
    import torch

    model = select_model(args)

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        state = extract_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
    else:
        print("未指定 --ckpt，仅打印初始化后的模型参数（含 flowseek 预训练权重）")

    print_params(model)
    print_module_summary(model)


if __name__ == "__main__":
    main()
