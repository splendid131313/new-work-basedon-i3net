import argparse
import os
import torch
arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Dataset
data_arg = add_argument_group('Dataset')
data_arg.add_argument('--data_type', type=str, default='direct')
data_arg.add_argument('--lr_slice_patch', type=int, default=4, help='每个lr样本的slice个数,插值为中间3个slice')
data_arg.add_argument('--traindata_path', type=str, default='/remote-home/share/Medical/i3net_dataset/Task06_Lung/train')
data_arg.add_argument('--testdata_path', type=str, default='/remote-home/share/Medical/i3net_dataset/Task10_Colon/test')
data_arg.add_argument('--image_size', type=int, default=256)


# Model
model_arg = add_argument_group('Model')
model_arg.add_argument('--model', type=str, default='i3net', help='select model')
model_arg.add_argument('--upscale', type=int, default=2, help='scale_factor')
model_arg.add_argument("--resume", type=bool, default=False, help='run resume or not')
model_arg.add_argument('--ckpt', type=str, default='', help='pretrained model path')
model_arg.add_argument("--flow_cfg", type=str, default="flowseek-S.json")
model_arg.add_argument("--flow_ckpt", type=str, default="./model_zoo/flowseek/weights/flowseek_T_CT.pth")


# Training / test parameters
learn_arg = add_argument_group('Learning')
#### optim ####
learn_arg.add_argument('--optim', type=str, default='Adam') 
learn_arg.add_argument('--lr', type=float, default=(3e-4)) # 0.0003
learn_arg.add_argument('--wd', type=float, default=(1e-4), help='weight decay')
learn_arg.add_argument('--beta1', type=float, default=0.9, help='Adam-beta1')
learn_arg.add_argument('--beta2', type=float, default=0.999, help='Adam-beta2')
learn_arg.add_argument('--eps', type=float, default=1e-08)
learn_arg.add_argument('--flood', type=bool, default=False)
#### schedule ####
learn_arg.add_argument('--schedule', type=str, default='cos_lr', help='step/cos_lr/Tmax/Tmin') 
learn_arg.add_argument('--lr_decay', type=int, default=400)
learn_arg.add_argument('--gamma', type=float, default='0.5', help='下降速度')
#### epoch/bs ####
learn_arg.add_argument('--batch_size', type=int, default=6)
learn_arg.add_argument('--one_batch_n_sample', type=int, default=1, help='smapling n times of each volume')
learn_arg.add_argument('--start_epoch', type=int, default=0)
learn_arg.add_argument('--max_epoch', type=int, default=800)
learn_arg.add_argument('--warmup_epoch', type=float, default=0.05, help='warm up epoch ratio')
#### loss ####
learn_arg.add_argument('--lambda_l1', type=float, default=1.0)
learn_arg.add_argument('--lambda_lap', type=float, default=0.0)
learn_arg.add_argument('--lambda_gra', type=float, default=0.0)
learn_arg.add_argument('--lambda_tissue', type=float, default=0.1)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--ckpt_dir', type=str, default='default',help='saved filename')
misc_arg.add_argument('--gpu_id', type=str, default='0')
misc_arg.add_argument('--num_workers', type=int, default=8)
misc_arg.add_argument('--parallel', type=bool, default=True, help="parallel training")
misc_arg.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
misc_arg.add_argument("--amp", default=True, type=bool, help='autocast')

def get_args():
    """Parses all of the arguments above
    """
    args, unparsed = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    gpu_ok = args.gpu_id.strip().lower() not in ('', '-1', 'none')
    setattr(args, 'cuda', gpu_ok and torch.cuda.is_available())
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    
    args.hr_slice_patch = args.upscale * (args.lr_slice_patch - 1) + 1
    args.lr_time_list = torch.linspace(0, 1, args.upscale - 1 + 2)
    return args, unparsed

