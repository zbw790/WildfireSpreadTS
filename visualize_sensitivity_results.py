#!/usr/bin/env python3
"""
Quick visualization script for feature sensitivity results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_feature_sensitivity(feature_dir):
    """Visualize sensitivity results for a single feature"""
    
    feature_name = feature_dir.name
    data_file = feature_dir / f"{feature_name}_data.json"
    
    if not data_file.exists():
        print(f"No data file found for {feature_name}")
        return
    
    print(f"Loading data for {feature_name}...")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extract perturbations and fire areas
    perturbations = []
    avg_fire_areas = []
    
    for pert_str, results in data['perturbation_results'].items():
        perturbation = float(pert_str)
        fire_areas = results['fire_areas']
        avg_area = np.mean(fire_areas)
        
        perturbations.append(perturbation)
        avg_fire_areas.append(avg_area)
    
    # Sort by perturbation value
    sorted_data = sorted(zip(perturbations, avg_fire_areas))
    perturbations, avg_fire_areas = zip(*sorted_data)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(perturbations, avg_fire_areas, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=avg_fire_areas[perturbations.index(0.0)] if 0.0 in perturbations else np.mean(avg_fire_areas), 
                color='r', linestyle='--', alpha=0.7, label='Baseline')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    plt.xlabel(f'{feature_name} Perturbation (%)')
    plt.ylabel('Average Fire Area (pixels)')
    plt.title(f'{feature_name} Sensitivity Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Calculate correlation
    if len(perturbations) > 2:
        correlation = np.corrcoef(perturbations, avg_fire_areas)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.tight_layout()
    plt.savefig(feature_dir / f"{feature_name}_sensitivity_quick.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved: {feature_name}_sensitivity_quick.png")
    print(f"Perturbation range: {min(perturbations)}% to {max(perturbations)}%")
    print(f"Fire area range: {min(avg_fire_areas):.1f} to {max(avg_fire_areas):.1f} pixels")
    if len(perturbations) > 2:
        print(f"Correlation: {correlation:.3f}")
    print()

def main():
    """Main function to visualize all feature results"""
    
    results_dir = Path("feature_sensitivity_test")
    
    if not results_dir.exists():
        print("No results directory found. Run the sensitivity analysis first.")
        return
    
    print("Feature Sensitivity Analysis Results")
    print("=" * 40)
    
    # Find all feature directories
    feature_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    
    if not feature_dirs:
        print("No feature directories found.")
        return
    
    for feature_dir in sorted(feature_dirs):
        visualize_feature_sensitivity(feature_dir)
    
    print("All visualizations complete!")
    print(f"Check the {results_dir} directory for:")
    print("- Individual feature sensitivity plots")
    print("- JSON data files with detailed results")
    print("- sensitivity_analysis_report.txt")

if __name__ == "__main__":
    main()
