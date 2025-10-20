# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import glob
import os

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MotionDataset(Dataset):
    def __init__(
        self,
        dataset,
        motion_clean,
        motion_with_orth,
        mean=0,
        std=1,
        input_motion_length=196,
        train_dataset_repeat_times=1,
        no_normalization=False,
    ):
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.motion_clean = motion_clean
        self.motion_with_orth = motion_with_orth
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization

        self.motion_clean = motion_clean
        self.motion_with_orth = motion_with_orth

        self.input_motion_length = input_motion_length

    def __len__(self):
        return len(self.motion_clean) * self.train_dataset_repeat_times

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, idx):
        motion = self.motion_clean[idx % len(self.motion_clean)]
        motion_w_o = self.motion_with_orth[idx % len(self.motion_clean)]
        seqlen = motion.shape[0]

        if seqlen <= self.input_motion_length:
            idx = 0
        else:
            idx = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]
        #TODO: pad motion and adjust to motion length
        """
        # Normalization
        if not self.no_normalization:
            motion = (motion - self.mean) / (self.std + 1e-8)
        """

        return motion.float(), motion_w_o.float()

    


def load_data(motion_clean_path, motion_w_o_path, **kwargs):
    """
    Load SMPLX keypoint .npy files from a folder into a list of PyTorch tensors.

    Args:
        dataset_path (str): Path to the folder containing .npy files.
        **kwargs: Optional keyword arguments for future extensions.

    Returns: 
        list[torch.Tensor]: A list of tensors, one per file, each of shape (frames, 135, 5).
    """
    motion_clean = []
    motion_w_o=[]
    #TODO: adjust input length based on frames
    # Iterate over all .npy files in the directory
    for file_name in sorted(os.listdir(motion_clean_path)):
        if file_name.endswith(".npy"):
            file_path = os.path.join(motion_clean_path, file_name)
            keypoints = np.load(file_path)  # shape (frames, 135, 5)
            
            # Validate shape
            if keypoints.ndim != 3 or keypoints.shape[1:] != (135, 5):
                raise ValueError(f"Invalid shape {keypoints.shape} in file: {file_name}")

            # Convert to torch tensor (float32)
            tensor = torch.tensor(keypoints, dtype=torch.float32)
            motion_clean.append(tensor)

    if not motion_clean:
        raise FileNotFoundError(f"No .npy files found in directory: {motion_clean_path}")

    return motion_clean, motion_w_o


def get_dataloader(
    dataset,
    split,
    batch_size,
    num_workers=32,
):

    if split == "train":
        shuffle = True
        drop_last = True
        num_workers = num_workers
    else:
        shuffle = False
        drop_last = False
        num_workers = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=False,
    )
    return loader