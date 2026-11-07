import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import util

def collate_train_spans(batch):

    lr_list, gt_list, cond_list, span_id_list = [], [], [], []
    span_offset = 0
    for lr, gt, cond, span_ids in batch:
        n_spans = int(span_ids.max().item()) + 1
        batch_span_ids = span_ids + span_offset
        span_offset += n_spans
        lr_list.append(lr)
        gt_list.append(gt)
        cond_list.append(cond)
        span_id_list.append(batch_span_ids)
    return tuple(torch.cat(items, dim=0) for items in (
        lr_list, gt_list, cond_list, span_id_list
    ))


class trainSet(Dataset):
    def __init__(
        self,
        data_root,
        args=None,
        random_crop=None,
        resize=None,
        augment_s=True,
        augment_t=True,
    ):
        self.args = args
        self.data_root = data_root
        self.image_size = args.image_size
        self.augment_s = augment_s
        self.augment_t = augment_t
        self.random_crop = random_crop

        self.volume_list = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".pt")]
        random.shuffle(self.volume_list)
        self.file_len = len(self.volume_list)

    def _augment_xy(self, volume):
        if self.augment_s and random.random() >= 0.5:
            volume = volume[:, ::-1, :].copy()
        if self.augment_s and random.random() >= 0.5:
            volume = volume[::-1, :, :].copy()
        return volume

    def _sample_dynamic_span(self, volume):
        z_size = volume.shape[2]
        train_gaps = [int(gap) for gap in getattr(self.args, "train_gaps", [4])]
        valid_gaps = [gap for gap in train_gaps if gap >= 2 and gap < z_size]

        gap = random.choice(valid_gaps)
        n_mid = gap - 1
        span_len = gap + 1
        z0 = random.randint(0, z_size - span_len)
        z1 = z0 + gap

        hr_span = volume[:, :, z0 : z1 + 1]
        hr_span = self._augment_xy(hr_span)
        hr_span = util.crop_center(hr_span, self.image_size, self.image_size)

        mid_indices = torch.arange(1, gap, dtype=torch.float32)
        endpoints = torch.from_numpy(hr_span[:, :, [0, -1]].copy())
        lr = endpoints.unsqueeze(0).repeat(n_mid, 1, 1, 1)
        gt = torch.from_numpy(hr_span[:, :, 1:-1].copy())
        gt = gt.permute(2, 0, 1).unsqueeze(-1).contiguous()
        cond = torch.stack((mid_indices / gap, torch.full_like(mid_indices, gap)), dim=1)
        return lr, gt, cond

    def __getitem__(self, index):
        with open(self.volume_list[index], "rb") as f:
            volume = pickle.load(f)["image"]
        volume = util.normalize(volume).astype(np.float32)

        lr_list, gt_list, t_list, span_id_list = [], [], [], []
        for span_id in range(self.args.one_batch_n_sample):
            lr, gt, t = self._sample_dynamic_span(volume)
            span_ids = torch.full((t.shape[0],), span_id, dtype=torch.long)
            lr_list.append(lr)
            gt_list.append(gt)
            t_list.append(t)
            span_id_list.append(span_ids)

        lr = torch.cat(lr_list, 0)
        gt = torch.cat(gt_list, 0)
        t = torch.cat(t_list, 0)
        span_ids = torch.cat(span_id_list, 0)
        return lr, gt, t, span_ids

    def __len__(self):
        return self.file_len


class testSet(Dataset):
    def __init__(self, data_root, image_size):
        self.data_root = data_root
        self.image_size = image_size
        self.trainlist = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".pt")]
        self.file_len = len(self.trainlist)

    def __getitem__(self, index):
        volumepath = self.trainlist[index]
        with open(volumepath, "rb") as f:
            volumeIn = pickle.load(f)["image"]
        volumeIn = util.crop_center(volumeIn, self.image_size, self.image_size)
        volumeIn = util.normalize(volumeIn)
        volumeIn = volumeIn.astype(np.float32)
        volumeIn = torch.from_numpy(volumeIn)

        name = os.path.splitext(os.path.basename(volumepath))[0]
        return name, volumeIn

    def __len__(self):
        return self.file_len
