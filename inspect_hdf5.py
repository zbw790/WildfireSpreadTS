import h5py
import os

def inspect_hdf5_file(file_path):
    """
    Inspects an HDF5 file and prints information about its datasets.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"Inspecting file: {file_path}")
            if not f.keys():
                print("  File is empty or contains no datasets.")
                return
            
            print("  Datasets found:")
            f.visititems(print_dataset_info)
            
            # Also, check for attributes on the main dataset, which might contain feature names
            if 'data' in f:
                data_attrs = f['data'].attrs
                if data_attrs:
                    print("\n  Attributes found for 'data' dataset:")
                    for key, value in data_attrs.items():
                        print(f"    - {key}: {value}")
                else:
                    print("\n  No attributes found for 'data' dataset.")


    except Exception as e:
        print(f"An error occurred: {e}")

def print_dataset_info(name, obj):
    """
    Callback function to print dataset name, shape, and dtype.
    """
    if isinstance(obj, h5py.Dataset):
        print(f"    - Name: {name}")
        print(f"      Shape: {obj.shape}")
        print(f"      Dtype: {obj.dtype}")

if __name__ == "__main__":
    # Path to the HDF5 file to inspect.
    # Pointing to a 2018 file to debug the metadata issue.
    file_to_inspect = 'data/processed/2018/fire_21458798.hdf5'
    if not os.path.exists(file_to_inspect):
        print(f"Error: Test file not found at {file_to_inspect}")
    else:
        inspect_hdf5_file(file_to_inspect)