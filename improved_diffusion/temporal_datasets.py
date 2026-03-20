"""
Data loader for 1D temporal signals.
"""

import torch as th
from torch.utils.data import DataLoader, Dataset
import numpy as np
from mpi4py import MPI


def load_temporal_data(
    *,
    data_path,
    batch_size,
    sequence_length,
    deterministic=False,
    normalize=True,
):
    """
    Load 1D temporal data for diffusion training.

    :param data_path: path to .pt file containing temporal data
    :param batch_size: the batch size of each returned pair
    :param sequence_length: expected length of each sequence
    :param deterministic: if True, yield results in a deterministic order
    :param normalize: if True, normalize data to [-1, 1] range
    :return: a generator over (sequences, kwargs) pairs
    """
    if not data_path:
        raise ValueError("unspecified data path")
    
    dataset = TemporalDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        normalize=normalize,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    
    while True:
        yield from loader


class TemporalDataset(Dataset):
    """
    Dataset for 1D temporal signals.
    
    Expects data in shape [N, T] where N is number of samples and T is time steps.
    Converts to [N, C, T] where C=1 (single channel).
    """
    
    def __init__(
        self,
        data_path,
        sequence_length,
        normalize=True,
        shard=0,
        num_shards=1,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Load the data
        print(f"Loading temporal data from {data_path}")
        data = th.load(data_path)
        
        # Handle different input formats
        if isinstance(data, th.Tensor):
            self.data = data
        elif isinstance(data, np.ndarray):
            self.data = th.from_numpy(data).float()
        elif isinstance(data, dict) and 'data' in data:
            # Support dict format with 'data' key
            self.data = data['data']
            if isinstance(self.data, np.ndarray):
                self.data = th.from_numpy(self.data).float()
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        # Ensure data is 2D [N, T]
        if self.data.dim() == 1:
            self.data = self.data.unsqueeze(0)  # [T] -> [1, T]
        elif self.data.dim() > 2:
            raise ValueError(f"Expected 2D data [N, T], got shape {self.data.shape}")
        
        # Verify sequence length
        if self.data.shape[1] != sequence_length:
            print(f"Warning: Data has length {self.data.shape[1]}, expected {sequence_length}")
            if self.data.shape[1] < sequence_length:
                # Pad if too short
                padding = sequence_length - self.data.shape[1]
                self.data = th.nn.functional.pad(self.data, (0, padding), mode='constant', value=0)
            else:
                # Truncate if too long
                self.data = self.data[:, :sequence_length]
        
        # Normalize to [-1, 1] if requested
        if self.normalize:
            data_min = self.data.min()
            data_max = self.data.max()
            if data_max > data_min:
                # Normalize to [0, 1] then to [-1, 1]
                self.data = (self.data - data_min) / (data_max - data_min)
                self.data = self.data * 2 - 1
                print(f"Normalized data from [{data_min:.4f}, {data_max:.4f}] to [-1, 1]")
            else:
                print(f"Warning: Data has constant value {data_min}, skipping normalization")
        
        # Shard the data for distributed training
        self.local_data = self.data[shard::num_shards]
        print(f"Loaded {len(self.local_data)} temporal sequences (shard {shard}/{num_shards})")
        print(f"Sequence shape: {self.local_data.shape}")
    
    def __len__(self):
        return len(self.local_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            - sequence: [C, T] tensor where C=1
            - dict: empty dict (for compatibility with training loop)
        """
        sequence = self.local_data[idx]  # [T]
        sequence = sequence.unsqueeze(0)  # [1, T] - add channel dimension
        
        # Ensure it's float32
        sequence = sequence.float()
        
        return sequence, {}

