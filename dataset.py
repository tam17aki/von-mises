# -*- coding: utf-8 -*-
"""Dataset definition for von Mises DNNs.

Copyright (C) 2024 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import functools
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VonMisesDataset(Dataset):
    """Dataset for von Mises DNN."""

    def __init__(self, feat_paths):
        """Initialize class."""
        self.logabs_paths = feat_paths["logabs"]
        self.phase_paths = feat_paths["phase"]

    def __getitem__(self, idx):
        """Get a pair of input and target.

        Args:
            idx (int): index of the pair

        Returns:
            tuple: input and target in numpy format
        """
        return (np.load(self.logabs_paths[idx]), np.load(self.phase_paths[idx]))

    def __len__(self):
        """Return the size of the dataset.

        Returns:
            int: size of the dataset
        """
        return len(self.logabs_paths)


def collate_fn_vm(batch, cfg):
    """Collate function for von Mises.

    Args:
        batch (Tuple): tuple of minibatch.
        cfg (DictConfig): configuration in YAML format.

    Returns:
        tuple: Batch of inputs, targets, and lengths.
    """
    lengths = [len(x[0]) for x in batch]
    hop_length = cfg.feature.hop_length
    sec_per_split = cfg.preprocess.sec_per_split
    required_frames = int(
        np.floor((cfg.feature.sample_rate * sec_per_split) // hop_length)
    )
    start_frames = np.array(
        [np.random.randint(0, frames - required_frames) for frames in lengths]
    )
    win_range = cfg.model.win_range
    win_width = 2 * win_range + 1

    batch_feats = {"logabs": None, "phase": None}
    for j, feat in enumerate(batch_feats.keys()):
        batch_temp = [
            x[j][start_frames[i] : start_frames[i] + required_frames]
            for i, x in enumerate(batch)
        ]
        batch_feats[feat] = torch.from_numpy(np.array(batch_temp))
        if feat == "logabs":
            batch_feats[feat] = batch_feats[feat].unfold(1, win_width, 1)
            n_batch, n_frame, _, _ = batch_feats[feat].shape
            batch_feats[feat] = batch_feats[feat].reshape(n_batch, n_frame, -1)
        else:
            batch_feats[feat] = batch_feats[feat][:, win_range:-win_range, :]

    return (batch_feats["logabs"], batch_feats["phase"])


def get_dataloader(cfg):
    """Get data loaders for training and validation.

    Args:
        cfg (DictConfig): configuration in YAML format.

    Returns:
        dict: Data loaders.
    """
    wav_list = os.listdir(
        os.path.join(
            cfg.vM.root_dir, cfg.vM.data_dir, cfg.vM.trainset_dir, cfg.vM.resample_dir
        )
    )
    utt_list = [
        os.path.splitext(os.path.basename(wav_file))[0] for wav_file in wav_list
    ]
    utt_list.sort()

    feat_dir = os.path.join(
        cfg.vM.root_dir, cfg.vM.feat_dir, cfg.vM.trainset_dir, cfg.feature.window
    )
    feat_paths = {"logabs": None, "phase": None}
    for feat in feat_paths:
        feat_paths[feat] = [
            os.path.join(feat_dir, f"{utt_id}-feats_{feat}.npy") for utt_id in utt_list
        ]

    dataset = VonMisesDataset(feat_paths)
    data_loaders = DataLoader(
        dataset,
        batch_size=cfg.training.n_batch,
        collate_fn=functools.partial(collate_fn_vm, cfg=cfg),
        pin_memory=True,
        num_workers=cfg.training.num_workers,
        shuffle=True,
    )
    return data_loaders
