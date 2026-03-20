"""
Data loader for 3D flux volumes (time, height, width).

Expected input: npy or torch file containing data shaped [B, T, H, W].
Output to model: [B, 1, T, H, W] float32.
"""

import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
from mpi4py import MPI


def load_volumetric_data(
    *,
    data_path,
    batch_size,
    sequence_length,
    height,
    width,
    deterministic=False,
    normalize=False,
):
    """
    Load 3D temporal-spatial flux data for diffusion training.

    :param data_path: path to .npy or .pt file containing flux data
    :param batch_size: the batch size of each returned pair
    :param sequence_length: expected temporal length T
    :param height: expected height H
    :param width: expected width W
    :param deterministic: if True, yield results in a deterministic order
    :param normalize: if True, scale data to [0, 1]
    :return: a generator over (volumes, kwargs) pairs
    """
    if not data_path:
        raise ValueError("unspecified data path")

    dataset = VolumetricDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        height=height,
        width=width,
        normalize=normalize,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=1,
        drop_last=True,
    )
    while True:
        yield from loader


class VolumetricDataset(Dataset):
    """
    Dataset for 3D flux volumes.

    Expects data in shape [B, T, H, W].
    Converts to [B, 1, T, H, W] (single channel).
    """

    def __init__(
        self,
        data_path,
        sequence_length,
        height,
        width,
        normalize=False,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.height = height
        self.width = width
        self.normalize = normalize

        print(f"Loading volumetric data from {data_path}")
        data = np.load(data_path) if data_path.endswith(".npy") else th.load(data_path)

        if isinstance(data, np.ndarray):
            data = th.from_numpy(data)
        elif not isinstance(data, th.Tensor):
            raise ValueError(f"Unsupported data format: {type(data)}")

        self.data = data.float()

        if self.data.dim() != 4:
            raise ValueError(f"Expected shape [B, T, H, W], got {self.data.shape}")

        # Validate dimensions
        if self.data.shape[1] != sequence_length:
            print(f"Warning: T mismatch ({self.data.shape[1]} vs {sequence_length}); will pad/trim.")
        if self.data.shape[2] != height or self.data.shape[3] != width:
            print(f"Warning: spatial mismatch ({self.data.shape[2:]}, expected {(height, width)}); will pad/trim.")

        # Pad/trim T
        if self.data.shape[1] < sequence_length:
            pad_t = sequence_length - self.data.shape[1]
            self.data = th.nn.functional.pad(self.data, (0, 0, 0, 0, 0, pad_t))
        elif self.data.shape[1] > sequence_length:
            self.data = self.data[:, :sequence_length]

        # Pad/trim H, W
        if self.data.shape[2] < height:
            pad_h = height - self.data.shape[2]
            self.data = th.nn.functional.pad(self.data, (0, 0, 0, pad_h))
        elif self.data.shape[2] > height:
            self.data = self.data[:, :, :height, :]
        if self.data.shape[3] < width:
            pad_w = width - self.data.shape[3]
            self.data = th.nn.functional.pad(self.data, (0, pad_w))
        elif self.data.shape[3] > width:
            self.data = self.data[:, :, :, :width]

        # Optional normalization to [0,1]
        if self.normalize:
            dmin = self.data.min()
            dmax = self.data.max()
            if dmax > dmin:
                self.data = (self.data - dmin) / (dmax - dmin)
                print(f"Normalized data from [{dmin:.4f}, {dmax:.4f}] to [0, 1]")
            else:
                print(f"Warning: constant data value {dmin}, skipping normalization")

        # Shard for distributed training
        self.local_data = self.data[shard::num_shards]
        print(f"Loaded {len(self.local_data)} volumes (shard {shard}/{num_shards})")
        print(f"Volume shape after prep: {self.local_data.shape}")

    def __len__(self):
        return len(self.local_data)

    def __getitem__(self, idx):
        """
        Returns:
            - volume: [C, T, H, W] tensor where C=1
            - dict: empty dict (for compatibility with training loop)
        """
        vol = self.local_data[idx]  # [T, H, W]
        vol = vol.unsqueeze(0)  # [1, T, H, W]
        return vol.float(), {}

