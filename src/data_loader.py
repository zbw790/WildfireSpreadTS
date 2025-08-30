import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop
from tqdm import tqdm

# Assuming utils.py is in the same src directory
from .utils import Standardizer

class WildfireDataset(Dataset):
    """
    PyTorch Dataset for loading wildfire data from HDF5 files.
    Handles data indexing, preprocessing, standardization, and augmentation.
    """
    def __init__(self, data_dir, years, window_size, standardizer, augmentations=None, feature_subset=None, mode='train'):
        """
        Args:
            data_dir (str): Path to the processed data directory.
            years (list): List of years to include in the dataset.
            window_size (int): Number of time steps for input (w_t).
            standardizer (Standardizer): An already fitted Standardizer instance.
            augmentations (callable, optional): Augmentation function to apply.
            feature_subset (list, optional): List of indices for feature channels to use.
            mode (str): 'train', 'validation', or 'test'. Controls augmentation/cropping.
        """
        self.data_dir = data_dir
        self.years = years
        self.window_size = window_size
        self.standardizer = standardizer
        self.augmentations = augmentations
        self.feature_subset = feature_subset
        self.mode = mode

        self.file_paths = self._get_file_paths()
        self.index_map = self._create_index_map()

    def _get_file_paths(self):
        paths = []
        for year in self.years:
            year_path = os.path.join(self.data_dir, str(year), '*.hdf5')
            paths.extend(glob.glob(year_path))
        if not paths:
            raise FileNotFoundError(f"No HDF5 files found for years {self.years} in {self.data_dir}")
        return paths

    def _create_index_map(self):
        """
        Pre-computes a map of (file_index, start_time_step) for every valid sample.
        This avoids repeatedly opening files and checking lengths, speeding up access.
        """
        print("Creating index map for the dataset...")
        index_map = []
        for file_idx, file_path in enumerate(tqdm(self.file_paths)):
            with h5py.File(file_path, 'r') as f:
                # Total time steps available in the file
                total_timesteps = f['data'].shape[0]
                # A valid sample requires window_size input frames + 1 target frame
                num_valid_samples = total_timesteps - self.window_size
                
                if num_valid_samples > 0:
                    for start_step in range(num_valid_samples):
                        index_map.append((file_idx, start_step))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, start_step = self.index_map[idx]
        file_path = self.file_paths[file_idx]

        # 1. Data Reading
        with h5py.File(file_path, 'r') as f:
            # Input window: from start_step to start_step + window_size
            input_data = f['data'][start_step : start_step + self.window_size, :, :, :]
            # Target mask: Channel 22 (index 21) is the Active Fire mask
            target_mask = f['data'][start_step + self.window_size, 21, :, :]

        input_tensor = torch.from_numpy(input_data.astype(np.float32))
        target_tensor = torch.from_numpy(target_mask.astype(np.float32)).unsqueeze(0) # Add channel dim

        # --- PREPROCESSING STEPS based on the FINAL feature descriptions ---
        
        # Note: feature_subset logic is omitted for clarity in this final version.
        # It would require careful index management if re-introduced.

        # 2. Temperature Unit Unification (Celsius to Kelvin)
        # Channel 20 (index 19) is Forecast Temperature in Celsius.
        forecast_temp_idx = 19
        input_tensor[:, forecast_temp_idx, :, :] += 273.15

        # 3. Missing Value Imputation for Active Fire (Channel 22 / index 21)
        active_fire_idx = 21
        input_tensor[:, active_fire_idx, :, :] = torch.nan_to_num(input_tensor[:, active_fire_idx, :, :], nan=0.0)

        # 4. Cyclical Feature Encoding
        # Channel 7 (Wind Dir), Channel 13 (Aspect). Indices: 6, 12.
        # Channel 19 (Forecast Wind Dir) is a special case and not encoded here.
        cyclical_indices = [6, 12]
        for idx in cyclical_indices:
            input_tensor[:, idx, :, :] = torch.sin(input_tensor[:, idx, :, :])

        # 5. One-Hot Encoding for Land Cover (Channel 16 / index 15)
        land_cover_idx = 15
        land_cover_channel = input_tensor[:, land_cover_idx, :, :].long()
        
        # Landcover classes are 1-17, so 18 classes to be safe (for 0).
        num_land_cover_classes = 18
        land_cover_one_hot = torch.nn.functional.one_hot(land_cover_channel, num_classes=num_land_cover_classes)
        land_cover_one_hot = land_cover_one_hot.permute(0, 3, 1, 2).float()

        # Replace the original land cover channel with the one-hot encoded channels
        other_features = torch.cat([input_tensor[:, :land_cover_idx, :, :], input_tensor[:, land_cover_idx+1:, :, :]], dim=1)
        input_tensor = torch.cat([other_features, land_cover_one_hot], dim=1)

        # 6. Standardization
        # The standardizer should be fitted externally, ignoring static, cyclical,
        # and the original land cover channels.
        input_tensor = self.standardizer.transform(input_tensor)
        
        # Post-standardization NaN imputation for all other features
        input_tensor = torch.nan_to_num(input_tensor, nan=0.0)

        # 7. Data Augmentation (Train/Validation)
        if self.mode in ['train', 'validation'] and self.augmentations:
            # The complex oversampling is better handled in a custom sampler or wrapper.
            # Here we apply geometric augmentations.
            # Note: Augmentation library needs to handle multi-channel time-series data.
            # Permuting to (C, T, H, W) for some libraries might be necessary.
            pass # Placeholder for actual augmentation call

        # 8. Test-time Cropping
        if self.mode == 'test':
            h, w = input_tensor.shape[-2:]
            # Ensure size is a multiple of 32 for model compatibility
            h_new, w_new = (h // 32) * 32, (w // 32) * 32
            if (h, w) != (h_new, w_new):
                cropper = CenterCrop((h_new, w_new))
                input_tensor = cropper(input_tensor)
                target_tensor = cropper(target_tensor)

        return input_tensor, target_tensor