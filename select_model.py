from opt import *
import torch
import torch.nn as nn
import json
from importlib import import_module

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


def apply_flowseek_requires_grad(flowseek, args):
    finetune = getattr(args, "finetune_flowseek", False)
    if not finetune:
        for p in flowseek.parameters():
            p.requires_grad = False
        return

    scope = getattr(args, "flowseek_finetune_scope", "refine")

    if scope == "full":
        for p in flowseek.parameters():
            p.requires_grad = True
        for p in flowseek.dav2.parameters():
            p.requires_grad = False
        return

    for p in flowseek.parameters():
        p.requires_grad = False

    def _unfreeze(module):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = True

    if scope == "minimal":
        _unfreeze(flowseek.upsample_weight)
        _unfreeze(flowseek.flow_head)
        _unfreeze(getattr(flowseek, "update_block", None))
        return

    if scope == "refine":
        _unfreeze(flowseek.init_conv)
        _unfreeze(flowseek.upsample_weight)
        _unfreeze(flowseek.flow_head)
        _unfreeze(getattr(flowseek, "update_block", None))
        return

    raise ValueError(f"unknown flowseek_finetune_scope: {scope}")


def args_add_additional_attr(args,json_path):
    dic = json.load(open(json_path,'r',))
    for key,value in dic.items():
        if key == '//':
            continue
        setattr(args,key,value)

def select_model(args):
    opt_path = f'opt/{args.model}.json'
    args_add_additional_attr(args, opt_path)
    flow_path = f'model_zoo/flowseek/config/eval/{args.flow_cfg}'
    args_add_additional_attr(args, flow_path)
    module = import_module('model_zoo.net')
    model = module.make_model(args)

    flow_state = load_flowseek_ckpt(args)
    missing_keys, unexpected_keys = model.motion.flowseek.load_state_dict(
        flow_state, strict=True
    )
    if len(missing_keys) > 0:
        print(f"Warning: Missing keys: {missing_keys}")
    apply_flowseek_requires_grad(model.motion.flowseek, args)
    finetune = getattr(args, "finetune_flowseek", False)
    sc = getattr(args, "flowseek_finetune_scope", "refine")
    if not finetune:
        print("load flowseek weight success (frozen)")
    else:
        n_train = sum(p.numel() for p in model.motion.flowseek.parameters() if p.requires_grad)
        n_tot = sum(p.numel() for p in model.motion.flowseek.parameters())
        print(
            f"load flowseek weight success (finetune scope={sc}, "
            f"trainable params {n_train / 1e6:.2f}M / {n_tot / 1e6:.2f}M)"
        )
    return model


