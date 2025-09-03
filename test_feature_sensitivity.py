#!/usr/bin/env python3
"""
Test script for the Enhanced Feature Sensitivity Analysis Tool

This script demonstrates how to use the feature sensitivity analyzer
to generate comprehensive visual comparisons of fire spread predictions
under different feature perturbations.
"""

import os
import sys
from pathlib import Path
from feature_sensitivity_analyzer import ComprehensiveFeatureSensitivityAnalyzer

def find_model_file():
    """Find available model files in the workspace"""
    possible_paths = [
        'backup/models/',
        'models/',
        'backup/gpu_wildfire_outputs/',
        'backup/fixed_wildfire_outputs/',
        '.'
    ]
    
    model_extensions = ['.pth', '.pt', '.ckpt']
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if any(file.endswith(ext) for ext in model_extensions):
                        return os.path.join(root, file)
    
    return None

def find_fire_event_file():
    """Find available fire event HDF5 files"""
    data_dir = Path('data')
    if data_dir.exists():
        for file_path in data_dir.rglob('*.h5'):
            return str(file_path)
        for file_path in data_dir.rglob('*.hdf5'):
            return str(file_path)
    
    return None

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING COMPREHENSIVE FEATURE SENSITIVITY ANALYZER")
    print("=" * 60)
    
    # Find model and data files
    model_path = find_model_file()
    fire_event_path = find_fire_event_file()
    
    if not model_path:
        print("No model file found! Please ensure you have a trained model (.pth, .pt, .ckpt)")
        print("Looking in: backup/models/, models/, backup/gpu_wildfire_outputs/, etc.")
        return
    
    if not fire_event_path:
        print("No fire event data found! Please ensure you have HDF5 files in data/")
        return
    
    print(f"Using model: {model_path}")
    print(f"Using fire event: {fire_event_path}")
    
    # Create analyzer
    print("\nInitializing analyzer...")
    analyzer = ComprehensiveFeatureSensitivityAnalyzer(
        model_path=model_path,
        output_dir='feature_sensitivity_test',
        device='auto'
    )
    
    # Run analysis on key features
    features_to_test = ['NDVI', 'Max_Temp_K', 'Total_Precip']
    
    print(f"\nRunning sensitivity analysis for features: {features_to_test}")
    print("This will generate:")
    print("1. Actual fire spreading (ground truth)")
    print("2. Raw model predictions (using actual feature values)")
    print("3. Modified predictions (changing individual feature values)")
    print("\nStarting analysis...")
    
    try:
        analyzer.run_comprehensive_analysis(
            fire_event_path=fire_event_path,
            start_day=0,
            features_to_analyze=features_to_test
        )
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Check the 'feature_sensitivity_test' directory for results:")
        print("- Individual feature folders with GIFs and plots")
        print("- sensitivity_analysis_report.txt: Comprehensive summary")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        print("This might be due to:")
        print("1. Model compatibility issues")
        print("2. Data format differences")
        print("3. Missing dependencies")
        print("\nThe tool has been created successfully and can be adapted to your specific setup.")

def demo_with_synthetic_data():
    """Demo with synthetic data if no real data is available"""
    print("\n" + "=" * 60)
    print("CREATING DEMO WITH SYNTHETIC DATA")
    print("=" * 60)
    
    import torch
    import numpy as np
    import h5py
    from feature_sensitivity_analyzer import FeatureSensitivityConfig
    
    config = FeatureSensitivityConfig()
    
    # Create synthetic fire event data
    print("Creating synthetic fire event data...")
    synthetic_data_path = "synthetic_fire_event.h5"
    
    # Generate synthetic sequence data
    T, C, H, W = 20, len(config.BEST_FEATURES), 128, 128
    
    # Create realistic-looking synthetic data
    np.random.seed(42)
    synthetic_sequence = np.random.randn(T, C, H, W).astype(np.float32)
    
    # Add some structure to the fire channel (last channel)
    fire_center = (64, 64)
    for t in range(T):
        # Create expanding fire pattern
        y, x = np.ogrid[:H, :W]
        mask = ((x - fire_center[0])**2 + (y - fire_center[1])**2) <= (t + 5)**2
        synthetic_sequence[t, -1][mask] = 0.8 + 0.2 * np.random.random(np.sum(mask))
    
    # Save synthetic data
    with h5py.File(synthetic_data_path, 'w') as f:
        f.create_dataset('sequence', data=synthetic_sequence)
    
    print(f"Synthetic data created: {synthetic_data_path}")
    print("You can now test the analyzer with this synthetic data using:")
    print(f"python feature_sensitivity_analyzer.py --model <your_model> --fire_event {synthetic_data_path}")

if __name__ == "__main__":
    # Try to run the main test
    try:
        main()
    except Exception as e:
        print(f"Main test failed: {e}")
        print("\nFalling back to synthetic data demo...")
        demo_with_synthetic_data()
