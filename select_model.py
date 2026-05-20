from opt import *
import torch
import torch.nn as nn
import json
from importlib import import_module

# dav2 (Depth Anything V2) 始终冻结；以下为可独立解冻的 FlowSeek 子模块
FLOWSEEK_TRAINABLE_PREFIXES = (
    "merge_head",   # 深度特征 -> 光流特征
    "cnet",         # 双帧 context 编码
    "bnet",         # 几何 base 编码
    "init_conv",
    "flow_head",    # 光流/置信度头
    "upsample_weight",
    "fnet",         # 迭代相关特征 (参数量大)
    "update_block", # 迭代更新 (参数量中等)
)

# 预设模式: 从轻到重, 显存/参数量递增
FLOWSEEK_FINETUNE_MODES = {
    "frozen": [],
    "merge_head": ["merge_head"],
    "heads": ["merge_head", "init_conv", "flow_head", "upsample_weight"],
    "adapter": ["merge_head", "bnet", "init_conv"],
    "lite": [
        "merge_head", "cnet", "bnet", "init_conv", "flow_head", "upsample_weight"
    ],
    "refine": [
        "merge_head", "bnet", "init_conv", "flow_head", "upsample_weight", "update_block"
    ],
    "feature": ["merge_head", "fnet"],
    "full": list(FLOWSEEK_TRAINABLE_PREFIXES),
}


def load_flowseek_ckpt(args):
    ckpt = torch.load(args.flow_ckpt, map_location="cpu")
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and any(k.endswith(".weight") for k in ckpt.keys()):
        state = ckpt
    else:
        print(f"Error: cannot find weight in ckpt: {ckpt.keys()}")
        raise KeyError("cannot find weight in ckpt")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


def args_add_additional_attr(args, json_path):
    dic = json.load(open(json_path, "r"))
    for key, value in dic.items():
        if key == "//":
            continue
        setattr(args, key, value)


def resolve_flowseek_train_prefixes(args):
    mode = getattr(args, "flowseek_finetune_mode", "lite").strip().lower()
   
    if mode not in FLOWSEEK_FINETUNE_MODES:
        raise ValueError(f"wrong flowseek_finetune_mode!!")
    else:
        prefixes = FLOWSEEK_FINETUNE_MODES[mode]

    unknown = [p for p in prefixes if p not in FLOWSEEK_TRAINABLE_PREFIXES]
    if unknown:
        raise ValueError(f"wrong flowseek train prefixes!!")
    return mode, prefixes


def _flowseek_param_trainable(param_name, train_prefixes):
    if param_name.startswith("dav2."):
        return False
    if not train_prefixes:
        return False
    return any(
        param_name == prefix or param_name.startswith(prefix + ".")
        for prefix in train_prefixes
    )


def setup_flowseek_finetune(flowseek, args):
    mode, train_prefixes = resolve_flowseek_train_prefixes(args)
    flowseek.dav2.eval()

    trainable, frozen = 0, 0
    group_stats = {p: 0 for p in FLOWSEEK_TRAINABLE_PREFIXES}
    group_stats["dav2"] = 0
    group_stats["_other_frozen"] = 0

    for name, param in flowseek.named_parameters():
        train = _flowseek_param_trainable(name, train_prefixes)
        param.requires_grad = train
        n = param.numel()
        if train:
            trainable += n
        else:
            frozen += n

        matched = False
        for prefix in FLOWSEEK_TRAINABLE_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                group_stats[prefix] += n
                matched = True
                break
        if not matched:
            if name.startswith("dav2."):
                group_stats["dav2"] += n
            else:
                group_stats["_other_frozen"] += n

    active = train_prefixes if train_prefixes else ["(none)"]
    print(f"flowseek finetune mode={mode}, trainable prefixes={active}")
    print(
        f"  total: trainable={trainable / 1e6:.2f}M, frozen={frozen / 1e6:.2f}M"
    )
    for prefix in FLOWSEEK_TRAINABLE_PREFIXES:
        n = group_stats[prefix]
        if n == 0:
            continue
        flag = "ON " if prefix in train_prefixes else "off"
        print(f"  [{flag}] {prefix}: {n / 1e6:.2f}M")


def select_model(args):
    opt_path = f"opt/{args.model}.json"
    args_add_additional_attr(args, opt_path)
    flow_path = f"model_zoo/flowseek/config/eval/{args.flow_cfg}"
    args_add_additional_attr(args, flow_path)
    module = import_module("model_zoo.net")
    model = module.make_model(args)

    flow_state = load_flowseek_ckpt(args)
    missing_keys, unexpected_keys = model.flowseek.load_state_dict(
        flow_state, strict=True
    )
    if len(missing_keys) > 0:
        print(f"Warning: Missing keys: {missing_keys}")
    setup_flowseek_finetune(model.flowseek, args)
    print("load flowseek weight success")
    return model
