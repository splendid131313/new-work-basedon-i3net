import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import util


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
        self.hr_slice_patch = args.hr_slice_patch

        self.volume_list = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".npy")]
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
        targets_per_span = max(1, int(getattr(self.args, "targets_per_span", 1)))
        train_gaps = [int(gap) for gap in getattr(self.args, "train_gaps", [2, 3, 4])]
        valid_gaps = [gap for gap in train_gaps if gap >= 2 and gap < z_size and (gap - 1) >= targets_per_span]
        if not valid_gaps:
            raise ValueError(
                f"volume z={z_size} has no valid train gap in {train_gaps} "
                f"for targets_per_span={targets_per_span}"
            )

        # Sample endpoint distance directly. This keeps the meaning and frequency
        # of each scale independent of max_mid_slices.
        gap = random.choice(valid_gaps)
        n_mid = gap - 1
        span_len = gap + 1
        z0 = random.randint(0, z_size - span_len)
        z1 = z0 + gap

        hr_span = volume[:, :, z0 : z1 + 1]
        hr_span = self._augment_xy(hr_span)
        hr_span = util.crop_center(hr_span, self.image_size, self.image_size)

        all_mid = list(range(1, n_mid + 1))
        mid_indices = random.sample(all_mid, targets_per_span)

        lr_base = hr_span[:, :, [0, -1]]
        lr_list, gt_list, cond_list, meta_list = [], [], [], []
        for mid_idx in mid_indices:
            lr_list.append(torch.from_numpy(lr_base.copy()))
            gt_list.append(
                torch.from_numpy(hr_span[:, :, mid_idx : mid_idx + 1].copy())
            )
            cond_list.append(
                torch.tensor([mid_idx / gap, float(gap)], dtype=torch.float32)
            )
            # [n_mid, mid_idx]：中间切片总数、监督目标是第几张（相对 span，1..n_mid）
            meta_list.append(
                torch.tensor([n_mid, mid_idx], dtype=torch.float32)
            )

        return (
            torch.stack(lr_list, 0),
            torch.stack(gt_list, 0),
            torch.stack(cond_list, 0),
            torch.stack(meta_list, 0),
        )

    def __getitem__(self, index):
        volume = np.load(self.volume_list[index])
        volume = util.normalize(volume).astype(np.float32)

        lr_list, gt_list, t_list, meta_list = [], [], [], []
        for _ in range(self.args.one_batch_n_sample):
            lr, gt, t, meta = self._sample_dynamic_span(volume)
            lr_list.append(lr)
            gt_list.append(gt)
            t_list.append(t)
            meta_list.append(meta)

        lr = torch.cat(lr_list, 0)
        gt = torch.cat(gt_list, 0)
        t = torch.cat(t_list, 0)
        meta = torch.cat(meta_list, 0)
        return lr, gt, t, meta

    def __len__(self):
        return self.file_len


class testSet(Dataset):
  def __init__(self, data_root, image_size):
    self.data_root = data_root
    self.image_size = image_size
    self.trainlist = [(data_root + "/" + f) for f in os.listdir(data_root)]

    self.file_len = len(self.trainlist)

  def __getitem__(self, index):
    volumepath = self.trainlist[index]
    volumeIn = np.load(volumepath)
    volumeIn = util.crop_center(volumeIn, self.image_size, self.image_size)
    volumeIn, vmin, vmax = util.normalize(volumeIn, return_stats=True)
    volumeIn = volumeIn.astype(np.float32)
    volumeIn = torch.from_numpy(volumeIn)

    name = volumepath.split("/")[-1].split(".")[0]
    return name, volumeIn, np.float32(vmin), np.float32(vmax)

  def __len__(self):
    return self.file_len
