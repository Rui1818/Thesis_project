import glob
import os

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def drop_duplicate_frames(data):
    """
    Drop frames where all 25 rows are identical.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Array with shape (frames, 25, 5)
    
    Returns:
    --------
    numpy.ndarray
        Filtered array with duplicate frames removed
    """
    first_row = data[:, 0:1, :]  # Shape: (frames, 1, 5)
    all_rows_same = np.all(data == first_row, axis=(1,2))

    mask = ~all_rows_same
    return data[mask]


class MotionDataset(Dataset):
    def __init__(
        self,
        dataset,
        motion_clean,
        motion_without_orth,
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
        self.motion_without_orth = motion_without_orth
        self.train_dataset_repeat_times = train_dataset_repeat_times
        self.no_normalization = no_normalization

        self.motion_clean = motion_clean
        self.motion_without_orth = motion_without_orth

        self.input_motion_length = input_motion_length

    def __len__(self):
        return len(self.motion_clean) * self.train_dataset_repeat_times

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __getitem__(self, idx):
        motion = self.motion_clean[idx % len(self.motion_clean)]
        motion_w_o = self.motion_without_orth[idx % len(self.motion_clean)]
        seqlen = motion.shape[0]
        seqlen_wo = motion_w_o.shape[0]

        #sequence padding or random cropping to fit input length
        if seqlen <= self.input_motion_length:
            frames_to_add = self.input_motion_length - seqlen
            padding = torch.zeros(frames_to_add, motion.shape[1], motion.shape[2], dtype=motion.dtype)
            motion = torch.cat([motion, padding], dim=0)
        else:
            idx = torch.randint(0, int(seqlen - self.input_motion_length), (1,))[0]
            motion = motion[idx : idx + self.input_motion_length]
        
        if seqlen_wo <= self.input_motion_length:
            frames_to_add = self.input_motion_length - seqlen_wo
            padding = torch.zeros(frames_to_add, motion_w_o.shape[1], motion_w_o.shape[2], dtype=motion_w_o.dtype)
            motion_w_o = torch.cat([motion_w_o, padding], dim=0)
        else:
            idx = torch.randint(0, int(seqlen_wo - self.input_motion_length), (1,))[0]
            motion_w_o = motion_w_o[idx : idx + self.input_motion_length]

        """
        # Normalization
        if not self.no_normalization:
            motion = (motion - self.mean) / (self.std + 1e-8)
        """

        return motion.float(), motion_w_o.float()

    


def load_data(motion_path, split, **kwargs):
    """
    Load SMPL keypoint .npy files from a folder into a list of PyTorch tensors.

    Args:
        dataset_path (str): Path to the folder containing .npy files.
        **kwargs: Optional keyword arguments for future extensions.

    Returns: 
        list[torch.Tensor]: A list of tensors, one per file, each of shape (frames, 125, 5).
    """
    if split == "train":
        motion_clean =[]
        motion_w_o=[]
        #TODO: adjust to the dataset file structure
        for patient in sorted(os.listdir(motion_path)):
            for file in sorted(os.listdir(os.path.join(motion_path, patient))):
                take=file.split('_')
                if take[1]=='c1':
                    #motion with orthosis
                    file_name=file
                    file_path = os.path.join(motion_path, patient, file_name, "vitpose", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
                    clean_keypoints = np.load(file_path)  # shape (frames, 25, 5)
                    clean_keypoints = drop_duplicate_frames(clean_keypoints)
                    #reshape to (frame, 125)
                    clean_keypoints = clean_keypoints.reshape(-1, 125)
                    tensor = torch.tensor(clean_keypoints, dtype=torch.float32)
                    motion_clean.append(tensor)

                    #motion without orthosis
                    orth_path = take[0]+'_c2_'+"_".join(take[2:])
                    file_path= os.path.join(motion_path, patient, orth_path, "vitpose", "keypoints_3d", "smpl-keypoints-3d_cut.npy")
                    no_orth_keypoints = drop_duplicate_frames(np.load(file_path))  # shape (frames, 25, 5)
                    #reshape to (frame, 125)
                    no_orth_keypoints = no_orth_keypoints.reshape(-1, 125)
                    orth_tensor = torch.tensor(no_orth_keypoints, dtype=torch.float32)
                    motion_w_o.append(orth_tensor)
        if not motion_clean:
            raise FileNotFoundError(f"No files found in directory: {motion_path}")
    elif split == "test":
        raise NotImplementedError("Test split loading not implemented yet.")


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