import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_features(file_path, output_dir='feature_analysis'):
    """
    Analyzes and visualizes the feature channels from an HDF5 file.
    """
    # --- 1. Setup and Validation ---
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    try:
        with h5py.File(file_path, 'r') as f:
            if 'data' not in f:
                print("Error: 'data' dataset not found in the file.")
                return
            
            dataset = f['data']
            
            # --- 2. Select a time step (e.g., the first one) ---
            # Shape: (time, features, height, width)
            time_step_data = dataset[0, :, :, :]
            num_features = time_step_data.shape[0]
            
            print(f"Analyzing features for file: {file_path}")
            print(f"Data shape for one time step: {time_step_data.shape}")
            print("-" * 30)

            # --- 3. Analyze and Visualize each feature ---
            for i in range(num_features):
                feature_channel = time_step_data[i, :, :]
                
                # Calculate statistics
                min_val = np.min(feature_channel)
                max_val = np.max(feature_channel)
                mean_val = np.mean(feature_channel)
                std_val = np.std(feature_channel)
                
                print(f"Feature Channel {i}:")
                print(f"  Stats: Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}, Std={std_val:.4f}")

                # Check for binary-like features (potential fire masks)
                unique_vals = np.unique(feature_channel)
                if len(unique_vals) < 5: # Heuristic to find simple/binary masks
                    print(f"  Unique values: {unique_vals}")

                # Visualize the feature channel
                plt.figure(figsize=(8, 6))
                plt.imshow(feature_channel, cmap='viridis')
                plt.title(f'Feature Channel {i}\nMin={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}')
                plt.colorbar()
                save_path = os.path.join(output_dir, f'feature_{i}.png')
                plt.savefig(save_path)
                plt.close()

            print("-" * 30)
            print(f"All feature visualizations have been saved to the '{output_dir}' directory.")
            
            # --- 4. Check suspicious channels at a later time step ---
            print("\n--- Checking suspicious channels at a later time step (t=10) ---")
            suspicious_channels = [5, 17, 22]
            later_time_step = 10
            if dataset.shape[0] > later_time_step:
                for channel_idx in suspicious_channels:
                    later_channel_data = dataset[later_time_step, channel_idx, :, :]
                    min_val = np.min(later_channel_data)
                    max_val = np.max(later_channel_data)
                    mean_val = np.mean(later_channel_data)
                    print(f"Channel {channel_idx} at t={later_time_step}: Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}")
                    unique_vals = np.unique(later_channel_data)
                    if len(unique_vals) < 10:
                         print(f"  Unique values: {unique_vals}")


    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_to_analyze = 'data/processed/2020/fire_24462610.hdf5'
    analyze_features(file_to_analyze)