import torch
import torch.nn as nn
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.optim.lr_scheduler as lrs


def unwrap_model(net):
    return net.module if isinstance(net, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else net


###### optim ######
def _flowseek_finetune_param_groups(opt, net):
    lr = opt.lr
    ratio = getattr(opt, "flow_lr_ratio", 0.1)
    raw = unwrap_model(net)
    flowseek_trainable = [
        p for p in raw.motion.flowseek.parameters() if p.requires_grad
    ]
    if not flowseek_trainable:
        return None
    fs_set = set(flowseek_trainable)
    other = [p for p in net.parameters() if p.requires_grad and p not in fs_set]
    if not other:
        return [{"params": flowseek_trainable, "lr": lr * ratio}]
    return [
        {"params": other, "lr": lr},
        {"params": flowseek_trainable, "lr": lr * ratio},
    ]


def select_optim(opt,net):
    lr = opt.lr
    finetune_fs = getattr(opt, "finetune_flowseek", False)
    groups = _flowseek_finetune_param_groups(opt, net) if finetune_fs else None

    if opt.optim == 'SGD':
        if groups is not None:
            optimizer = torch.optim.SGD(
                groups, weight_decay=opt.wd, momentum=0.9
            )
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                        lr=lr, weight_decay=opt.wd, momentum=0.9)

    elif opt.optim == 'Adam':
        if groups is not None:
            optimizer = torch.optim.Adam(
                groups, betas=(opt.beta1, opt.beta2), eps=opt.eps
            )
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                          lr=lr, betas=(opt.beta1,opt.beta2), eps=opt.eps)

    elif opt.optim == 'AdamW':
        if groups is not None:
            optimizer = torch.optim.AdamW(groups, weight_decay=opt.wd)
        else:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                            lr=lr, weight_decay=opt.wd)

    return optimizer

###### optim ######
def select_scheduler(opt,optimizer):
    if opt.schedule == 'step':
        scheduler = lrs.StepLR(
            optimizer,
            step_size=opt.lr_decay,
            gamma=opt.gamma
        )
    if opt.schedule == 'cos_lr':
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.Tmax, \
        #                                                eta_min=opt.lr / opt.lr_gap)
        scheduler = CosineLRScheduler(optimizer,
                                      t_initial=opt.max_epoch,
                                      lr_min=opt.lr/10,
                                      warmup_lr_init=opt.lr/100,
                                      warmup_t=int(opt.max_epoch * opt.warmup_epoch),
                                      cycle_limit=1,
                                      t_in_epochs=False,
        )

    elif opt.schedule == 'Tmin':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               patience=opt.patience, threshold=0.000001)
    elif opt.schedule == 'Tmax':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                               patience=opt.patience, threshold=0.000001)

    return scheduler