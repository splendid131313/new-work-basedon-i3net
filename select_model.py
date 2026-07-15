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
    missing_keys, unexpected_keys = model.flow_estimator.flowseek.load_state_dict(
        flow_state, strict=True
    )
    if len(missing_keys) > 0:
        print(f"Warning: Missing keys: {missing_keys}")
    for param in model.flow_estimator.flowseek.parameters():
        param.requires_grad = False
    print("load flowseek weight success")
    return model


