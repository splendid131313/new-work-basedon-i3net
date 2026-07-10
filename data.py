import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import util


class trainSet(Dataset):
  def __init__(self, data_root, args=None, random_crop=None, resize=None, augment_s=True, augment_t=True):
    self.args = args
    self.data_root = data_root
    self.image_size = args.image_size
    self.augment_s = augment_s
    self.augment_t = augment_t
    self.random_crop = random_crop

    self.volume_list = [
      os.path.join(data_root, f)
      for f in os.listdir(data_root)
      if f.endswith(".npy")
    ]
    random.shuffle(self.volume_list)
    self.file_len = len(self.volume_list)

  def _load_volume(self, volumepath):
    volume = np.load(volumepath)
    if volume.ndim == 4:
      volume = volume[:, :, :, 0]
    if volume.ndim != 3:
      raise ValueError(
        f"expects 3D volume [H,W,Z], got shape {volume.shape} from {volumepath}"
      )
    return volume

  def _augment_xy(self, volume):
    if self.augment_s and random.random() >= 0.5:
      volume = volume[:, ::-1, :].copy()
    if self.augment_s and random.random() >= 0.5:
      volume = volume[::-1, :, :].copy()
    return volume

  def _sample_dynamic_span(self, volume):
    z_size = volume.shape[2]
    n_upper = min(z_size - 2, getattr(self.args, "max_mid_slices", z_size - 2))
    if n_upper < 1:
      raise ValueError(f"volume z={z_size} too short for dynamic span (need >= 3 slices)")

    large_gap_prob = float(getattr(self.args, "large_gap_prob", 0.0))
    if n_upper > 1 and random.random() < large_gap_prob:
      n_mid = random.randint(max(1, (n_upper + 1) // 2), n_upper)
    else:
      n_mid = random.randint(1, n_upper)
    span_len = n_mid + 2
    z0 = random.randint(0, z_size - span_len)
    z1 = z0 + n_mid + 1

    hr_span = volume[:, :, z0:z1+1].astype(np.float32)
    hr_span = util.normalize(hr_span)
    hr_span = self._augment_xy(hr_span)
    hr_span = util.crop_center(hr_span, self.image_size, self.image_size)

    targets_per_span = max(1, int(getattr(self.args, "targets_per_span", 1)))
    all_mid = list(range(1, n_mid + 1))
    if targets_per_span <= n_mid:
      mid_indices = random.sample(all_mid, targets_per_span)
    else:
      mid_indices = [random.choice(all_mid) for _ in range(targets_per_span)]

    gap = n_mid + 1
    max_gap = max(1, int(getattr(self.args, "max_mid_slices", n_upper)) + 1)
    gap_norm = min(float(gap) / float(max_gap), 1.0)

    lr_base = hr_span[:, :, [0, -1]]
    lr_list, gt_list, cond_list = [], [], []
    for mid_idx in mid_indices:
      lr_list.append(torch.from_numpy(lr_base.copy()))
      gt_list.append(torch.from_numpy(hr_span[:, :, mid_idx : mid_idx + 1].copy()))
      cond_list.append(torch.tensor([mid_idx / gap, gap_norm], dtype=torch.float32))

    return (
      torch.stack(lr_list, 0),
      torch.stack(gt_list, 0),
      torch.stack(cond_list, 0),
    )

  def __getitem__(self, index):
    volume = self._load_volume(self.volume_list[index])

    lr_list, gt_list, t_list = [], [], []
    for _ in range(self.args.one_batch_n_sample):
      lr, gt, t = self._sample_dynamic_span(volume)
      lr_list.append(lr)
      gt_list.append(gt)
      t_list.append(t)

    lr = torch.cat(lr_list, 0)
    gt = torch.cat(gt_list, 0)
    t = torch.cat(t_list, 0)
    return lr, gt, t

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
