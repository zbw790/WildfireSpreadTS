import torch
import numpy as np
from tqdm import tqdm

class Standardizer:
    """
    A class to handle standardization of wildfire data.
    It calculates the mean and standard deviation from a training dataset
    and applies the transformation to new data.
    """
    def __init__(self):
        """
        Initializes the mean and std attributes to None.
        """
        self.mean = None
        self.std = None
        self.channel_indices = None

    def fit(self, dataset, ignore_channels=None):
        """
        Calculates the mean and standard deviation for each feature channel
        from the entire training dataset, ignoring specified channels.

        Args:
            dataset: An instance of WildfireDataset containing the training data.
            ignore_channels (list, optional): List of channel indices to ignore
                                              during calculation.
        """
        print("Calculating mean and std for standardization...")
        
        num_channels = dataset[0][0].shape[1]
        if ignore_channels is None:
            ignore_channels = []

        # Determine which channels to calculate statistics for
        self.channel_indices = [i for i in range(num_channels) if i not in ignore_channels]
        
        if not self.channel_indices:
            print("Warning: All channels are ignored. Standardizer will not be fitted.")
            return

        sums = torch.zeros(len(self.channel_indices))
        sq_sums = torch.zeros(len(self.channel_indices))
        num_pixels = 0

        for i in tqdm(range(len(dataset))):
            sample, _ = dataset[i] # Shape: (T, C, H, W)
            
            # Select only the channels we need to standardize
            sample_subset = sample[:, self.channel_indices, :, :]
            
            # Permute to (C, T, H, W) and flatten spatial and time dims
            sample_flat = sample_subset.permute(1, 0, 2, 3).flatten(1)
            
            sums += torch.sum(sample_flat, dim=1)
            sq_sums += torch.sum(sample_flat**2, dim=1)
            num_pixels += sample_flat.shape[1]

        mean_subset = sums / num_pixels
        variance = (sq_sums / num_pixels) - (mean_subset ** 2)
        std_subset = torch.sqrt(variance) + 1e-7

        # Create full mean/std tensors, with 0 for mean and 1 for std on ignored channels
        self.mean = torch.zeros(num_channels)
        self.std = torch.ones(num_channels)

        # Place the calculated values into the correct positions
        for i, channel_idx in enumerate(self.channel_indices):
            self.mean[channel_idx] = mean_subset[i]
            self.std[channel_idx] = std_subset[i]

        # Reshape for broadcasting: (1, C, 1, 1)
        self.mean = self.mean.view(1, -1, 1, 1)
        self.std = self.std.view(1, -1, 1, 1)

        print("Standardization parameters (mean/std) calculated.")
        print(f"Mean shape: {self.mean.shape}")
        print(f"Std shape: {self.std.shape}")


    def transform(self, sample):
        """
        Applies the standardization to a single sample.

        Args:
            sample (torch.Tensor): A single sample tensor of shape (T, C, H, W).

        Returns:
            torch.Tensor: The standardized sample tensor.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fitted. Call fit() first.")
        
        # The sample is (T, C, H, W), mean/std is (1, C, 1, 1)
        # Broadcasting will handle the T, H, W dimensions.
        return (sample - self.mean) / self.std

    def to(self, device):
        """
        Moves the mean and std tensors to the specified device.
        """
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self