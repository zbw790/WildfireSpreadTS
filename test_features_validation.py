"""
Physics Validation Experiment Module
====================================

This module performs controlled experiments to validate that the deep learning model
has learned physically meaningful relationships between environmental variables and fire spread.

For each selected feature, it:
1. Systematically perturbs the variable across a range
2. Measures the model's response (fire spread area)
3. Generates multi-panel comparison visualizations
4. Creates quantitative analysis tables
5. Outputs detailed statistics and correlations

Usage:
    python physics_validation_experiment.py --model best_fire_model_official.pth --fire_event your_fire.hdf5
    python physics_validation_experiment.py --model best_fire_model_official.pth --demo
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy import ndimage
import argparse
from tqdm import tqdm
import json
import shutil
from pathlib import Path

# Import necessary components from the main simulation code
# Assuming these are available in the same directory
try:
    from test_simulation import (
        SimulationConfig, FixedFireEventLoader, FixedFireSpreadSimulator,
        load_model_with_compatibility, OfficialFireUNet, WildFireConfig, FirePredictionConfig
    )
except ImportError:
    print("Error: Cannot import from test_simulation.py")
    print("Please ensure test_simulation.py is in the same directory")
    sys.exit(1)

# Make compatibility classes available for model loading
import sys
sys.modules[__name__].WildFireConfig = WildFireConfig
sys.modules[__name__].FirePredictionConfig = FirePredictionConfig

# ============================================================================
# PHYSICS VALIDATION CONFIGURATION
# ============================================================================

class PhysicsValidationConfig:
    """Configuration for physics validation experiments"""
    
    def __init__(self):
        # Base configuration
        self.base_config = SimulationConfig()
        
        # Feature mapping for perturbation experiments
        # Maps original feature indices to their positions in BEST_FEATURES
        self.feature_mapping = self._create_feature_mapping()
        
        # Physics hypotheses based on scientific literature
        self.physics_hypotheses = {
            'Wind_Speed': {
                'expected_direction': 'positive',
                'expected_magnitude': 'strong',
                'physical_principle': 'Wind increases oxygen supply and flame spread rate',
                'perturbation_range': (-0.5, 1.0),
                'perturbation_steps': 8
            },
            'Max_Temp_K': {
                'expected_direction': 'positive',
                'expected_magnitude': 'moderate',
                'physical_principle': 'Higher temperature reduces fuel ignition threshold',
                'perturbation_range': (-0.3, 0.5),
                'perturbation_steps': 7
            },
            'Min_Temp_K': {
                'expected_direction': 'positive',
                'expected_magnitude': 'moderate',
                'physical_principle': 'Higher minimum temperature indicates drier conditions',
                'perturbation_range': (-0.3, 0.5),
                'perturbation_steps': 7
            },
            'Slope': {
                'expected_direction': 'positive',
                'expected_magnitude': 'strong',
                'physical_principle': 'Upslope terrain enhances fire spread via preheating',
                'perturbation_range': (-0.4, 0.8),
                'perturbation_steps': 7
            },
            'Aspect': {
                'expected_direction': 'variable',
                'expected_magnitude': 'moderate',
                'physical_principle': 'South-facing slopes receive more solar radiation',
                'perturbation_range': (-0.3, 0.3),
                'perturbation_steps': 6
            },
            'Elevation': {
                'expected_direction': 'negative',
                'expected_magnitude': 'weak',
                'physical_principle': 'Higher elevation often means cooler temperatures',
                'perturbation_range': (-0.2, 0.4),
                'perturbation_steps': 6
            },
            'Landcover': {
                'expected_direction': 'variable',
                'expected_magnitude': 'moderate',
                'physical_principle': 'Different vegetation types have varying flammability',
                'perturbation_range': (-0.3, 0.5),
                'perturbation_steps': 6
            },
            'NDVI': {
                'expected_direction': 'positive',
                'expected_magnitude': 'moderate',
                'physical_principle': 'High NDVI indicates dense vegetation providing fuel',
                'perturbation_range': (-0.4, 0.6),
                'perturbation_steps': 6
            },
            'EVI2': {
                'expected_direction': 'positive',
                'expected_magnitude': 'moderate',
                'physical_principle': 'EVI2 reflects vegetation vigor and biomass',
                'perturbation_range': (-0.4, 0.6),
                'perturbation_steps': 6
            },
            'Total_Precip': {
                'expected_direction': 'negative',
                'expected_magnitude': 'strong',
                'physical_principle': 'Precipitation increases fuel moisture content',
                'perturbation_range': (-0.8, 1.5),
                'perturbation_steps': 8
            },
            'VIIRS_M11': {
                'expected_direction': 'positive',
                'expected_magnitude': 'moderate',
                'physical_principle': 'Thermal infrared correlates with surface temperature',
                'perturbation_range': (-0.3, 0.7),
                'perturbation_steps': 6
            },
            'VIIRS_I2': {
                'expected_direction': 'positive',
                'expected_magnitude': 'moderate',
                'physical_principle': 'Near-infrared reflectance indicates dry vegetation',
                'perturbation_range': (-0.3, 0.7),
                'perturbation_steps': 6
            },
            'VIIRS_I1': {
                'expected_direction': 'positive',
                'expected_magnitude': 'moderate',
                'physical_principle': 'Red reflectance sensitive to vegetation stress',
                'perturbation_range': (-0.3, 0.7),
                'perturbation_steps': 6
            }
        }
        
        # Experiment parameters
        self.simulation_days = 5
        self.output_dir = "physics_validation"
        self.visualization_grid = (2, 4)  # 8-panel layout
        self.fire_threshold = 0.3
        
    def _create_feature_mapping(self):
        """Create mapping from feature names to indices in BEST_FEATURES"""
        feature_names = self.base_config.FEATURE_NAMES
        best_features = self.base_config.BEST_FEATURES
        
        mapping = {}
        for i, feature_idx in enumerate(best_features):
            if feature_idx < len(feature_names):
                feature_name = feature_names[feature_idx]
                mapping[feature_name] = i
        
        return mapping
    
    def get_testable_variables(self):
        """Get list of variables that can be tested"""
        testable = []
        for var_name in self.physics_hypotheses.keys():
            if var_name in self.feature_mapping:
                testable.append(var_name)
        return testable

# ============================================================================
# CONTROLLED PERTURBATION EXPERIMENT
# ============================================================================

class ControlledPerturbationExperiment:
    """Performs controlled perturbation experiments on physical variables"""
    
    def __init__(self, simulator, fire_loader, config):
        self.simulator = simulator
        self.fire_loader = fire_loader
        self.config = config
        self.feature_stats = fire_loader.feature_stats
        
    def run_variable_experiment(self, fire_event_data, variable_name, start_day=0):
        """Run perturbation experiment for a single variable"""
        
        if variable_name not in self.config.physics_hypotheses:
            raise ValueError(f"Variable {variable_name} not configured for testing")
        
        var_config = self.config.physics_hypotheses[variable_name]
        feature_idx = self.config.feature_mapping.get(variable_name)
        
        if feature_idx is None:
            raise ValueError(f"Variable {variable_name} not found in feature mapping")
        
        print(f"Testing variable: {variable_name} (feature index: {feature_idx})")
        
        # Get baseline data
        baseline_sequence, _, _ = self.fire_loader.prepare_simulation_data(
            fire_event_data, start_day=start_day, max_future_days=self.config.simulation_days
        )
        
        # Run baseline prediction
        baseline_prediction = self._run_simulation_sequence(baseline_sequence)
        baseline_area = self._calculate_total_fire_area(baseline_prediction)
        
        # Run perturbation experiments
        perturbation_range = var_config['perturbation_range']
        perturbation_steps = var_config['perturbation_steps']
        perturbations = np.linspace(perturbation_range[0], perturbation_range[1], perturbation_steps)
        
        results = {
            'variable_name': variable_name,
            'baseline_area': baseline_area,
            'perturbations': [],
            'predictions': [],
            'fire_areas': [],
            'area_changes': [],
            'baseline_prediction': baseline_prediction
        }
        
        for perturbation in tqdm(perturbations, desc=f"Perturbing {variable_name}"):
            # Apply perturbation
            perturbed_sequence = self._apply_perturbation(
                baseline_sequence, feature_idx, perturbation
            )
            
            # Run prediction
            perturbed_prediction = self._run_simulation_sequence(perturbed_sequence)
            perturbed_area = self._calculate_total_fire_area(perturbed_prediction)
            
            # Record results
            results['perturbations'].append(perturbation)
            results['predictions'].append(perturbed_prediction)
            results['fire_areas'].append(perturbed_area)
            results['area_changes'].append(
                (perturbed_area - baseline_area) / max(baseline_area, 1) * 100
            )
        
        # Calculate correlation and statistical analysis
        results['correlation'] = np.corrcoef(results['perturbations'], results['area_changes'])[0,1]
        results['slope'], results['intercept'], results['r_value'], results['p_value'], results['std_err'] = \
            stats.linregress(results['perturbations'], results['area_changes'])
        
        # Determine if results match physics expectations
        expected_direction = var_config['expected_direction']
        actual_direction = 'positive' if results['correlation'] > 0 else 'negative'
        results['physics_consistent'] = (expected_direction == actual_direction)
        
        return results
    
    def _apply_perturbation(self, sequence, feature_idx, perturbation_factor):
        """Apply perturbation to a specific feature"""
        perturbed_sequence = sequence.clone()
        
        # Get original feature indices for reverse mapping
        original_idx = self.config.base_config.BEST_FEATURES[feature_idx]
        
        # Get normalization statistics
        if original_idx < len(self.feature_stats['mean']):
            feature_mean = self.feature_stats['mean'][original_idx]
            feature_std = self.feature_stats['std'][original_idx]
        else:
            feature_mean = 0.0
            feature_std = 1.0
        
        # Apply perturbation in physical space
        for t in range(sequence.shape[0]):
            # Denormalize to physical space
            normalized_data = perturbed_sequence[t, feature_idx]
            physical_data = normalized_data * feature_std + feature_mean
            
            # Apply relative perturbation
            perturbed_physical = physical_data * (1 + perturbation_factor)
            
            # Renormalize
            perturbed_normalized = (perturbed_physical - feature_mean) / (feature_std + 1e-6)
            
            perturbed_sequence[t, feature_idx] = perturbed_normalized
        
        return perturbed_sequence
    
    def _run_simulation_sequence(self, initial_sequence):
        """Run fire simulation for multiple days"""
        predictions = []
        current_sequence = initial_sequence.clone()
        
        for day in range(self.config.simulation_days):
            pred_fire = self.simulator.predict_single_step(current_sequence.unsqueeze(0))
            pred_fire = self._apply_physics_constraints(pred_fire)
            predictions.append(pred_fire.cpu().numpy())
            
            # Update sequence for next day (autoregressive mode)
            if day < self.config.simulation_days - 1:
                new_frame = current_sequence[-1].clone()
                active_fire_idx = len(self.config.base_config.BEST_FEATURES) - 1
                new_frame[active_fire_idx] = pred_fire
                
                current_sequence = torch.cat([
                    current_sequence[1:],
                    new_frame.unsqueeze(0)
                ], dim=0)
        
        return predictions
    
    def _apply_physics_constraints(self, fire_prediction):
        """Apply basic physics constraints to fire prediction"""
        # Apply threshold
        fire_binary = (fire_prediction > self.config.fire_threshold).float()
        
        # Apply spatial smoothing
        fire_smoothed = torch.tensor(
            ndimage.gaussian_filter(fire_binary.numpy(), sigma=0.8)
        )
        
        return fire_smoothed
    
    def _calculate_total_fire_area(self, predictions):
        """Calculate total fire area across all prediction days"""
        total_area = 0
        for pred in predictions:
            binary_fire = (pred > self.config.fire_threshold).astype(float)
            total_area += np.sum(binary_fire)
        return total_area

# ============================================================================
# VISUALIZATION GENERATOR
# ============================================================================

class PhysicsVisualizationGenerator:
    """Generates comprehensive visualizations for physics validation"""
    
    def __init__(self, config):
        self.config = config
    
    def create_variable_comparison_gif(self, experiment_results, output_path):
        """Create multi-panel comparison GIF showing different perturbation levels"""
        
        variable_name = experiment_results['variable_name']
        baseline_pred = experiment_results['baseline_prediction']
        perturbations = experiment_results['perturbations']
        predictions = experiment_results['predictions']
        
        # Setup figure with multiple subplots
        nrows, ncols = self.config.visualization_grid
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
        axes = axes.flatten()
        
        # Colors for different perturbation levels
        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(perturbations)))
        
        def animate(frame_day):
            if frame_day >= len(baseline_pred):
                return
            
            for ax in axes:
                ax.clear()
            
            # Panel 0: Baseline (top-left)
            baseline_frame = baseline_pred[frame_day]
            im0 = axes[0].imshow(baseline_frame, cmap='Reds', vmin=0, vmax=1)
            axes[0].set_title(f'{variable_name}\nBaseline (No Perturbation)', fontsize=10)
            axes[0].axis('off')
            
            # Add colorbar to first panel
            cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            cbar0.set_label('Fire Probability', fontsize=8)
            
            # Remaining panels: Different perturbation levels
            selected_indices = np.linspace(0, len(perturbations)-1, nrows*ncols-1, dtype=int)
            
            for i, pert_idx in enumerate(selected_indices):
                if i+1 >= len(axes):
                    break
                
                ax = axes[i+1]
                perturbation = perturbations[pert_idx]
                prediction = predictions[pert_idx]
                
                if frame_day < len(prediction):
                    frame = prediction[frame_day]
                    im = ax.imshow(frame, cmap='Reds', vmin=0, vmax=1)
                    
                    # Calculate area difference from baseline
                    baseline_area = np.sum(baseline_frame > self.config.fire_threshold)
                    current_area = np.sum(frame > self.config.fire_threshold)
                    area_change = ((current_area - baseline_area) / max(baseline_area, 1)) * 100
                    
                    ax.set_title(f'Perturbation: {perturbation:+.1%}\nArea Change: {area_change:+.1f}%', 
                               fontsize=9)
                    ax.axis('off')
                else:
                    ax.axis('off')
            
            # Overall title
            correlation = experiment_results['correlation']
            physics_consistent = experiment_results['physics_consistent']
            consistency_text = "Physics Consistent" if physics_consistent else "Physics Inconsistent"
            
            fig.suptitle(
                f'{variable_name} Perturbation Experiment - Day {frame_day+1}\n'
                f'Correlation: {correlation:.3f} | {consistency_text}',
                fontsize=14, fontweight='bold'
            )
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
        
        # Create animation
        frames = min(len(baseline_pred), self.config.simulation_days)
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000, repeat=True)
        
        # Save animation
        try:
            anim.save(output_path, writer='pillow', fps=1)
            print(f"  Animation saved: {output_path}")
        except Exception as e:
            print(f"  Animation save failed: {e}")
            # Save individual frames as fallback
            for frame in range(min(frames, 3)):
                animate(frame)
                frame_path = output_path.replace('.gif', f'_frame_{frame}.png')
                plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        
        plt.close(fig)
        return anim
    
    def create_fire_evolution_analysis(self, experiment_results, output_path):
        """Create fire evolution analysis similar to test_simulation"""
        
        variable_name = experiment_results['variable_name']
        baseline_pred = experiment_results['baseline_prediction']
        perturbations = experiment_results['perturbations']
        predictions = experiment_results['predictions']
        
        # Create 4-panel analysis for baseline and selected perturbations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Select representative perturbations
        if len(perturbations) >= 4:
            selected_indices = [0, len(perturbations)//3, 2*len(perturbations)//3, len(perturbations)-1]
        else:
            selected_indices = list(range(len(perturbations)))
        
        # Calculate fire areas over time for each case
        baseline_areas = []
        perturbed_areas = [[] for _ in selected_indices]
        
        for day in range(len(baseline_pred)):
            baseline_fire = (baseline_pred[day] > self.config.fire_threshold).astype(float)
            baseline_areas.append(np.sum(baseline_fire))
            
            for i, pert_idx in enumerate(selected_indices):
                if day < len(predictions[pert_idx]):
                    pert_fire = (predictions[pert_idx][day] > self.config.fire_threshold).astype(float)
                    perturbed_areas[i].append(np.sum(pert_fire))
        
        days = range(1, len(baseline_areas) + 1)
        
        # Panel 1: Fire area evolution over time
        ax1 = axes[0, 0]
        ax1.plot(days, baseline_areas, 'k-', linewidth=3, label='Baseline', alpha=0.8)
        
        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(selected_indices)))
        for i, (pert_idx, color) in enumerate(zip(selected_indices, colors)):
            perturbation = perturbations[pert_idx]
            ax1.plot(days, perturbed_areas[i], color=color, linewidth=2, alpha=0.8,
                    label=f'{perturbation:+.1%}')
        
        ax1.set_xlabel('Day', fontsize=12)
        ax1.set_ylabel('Fire Area (pixels)', fontsize=12)
        ax1.set_title('Fire Area Evolution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Panel 2: Cumulative fire area comparison
        ax2 = axes[0, 1]
        cumulative_baseline = np.cumsum(baseline_areas)
        ax2.plot(days, cumulative_baseline, 'k-', linewidth=3, label='Baseline', alpha=0.8)
        
        for i, (pert_idx, color) in enumerate(zip(selected_indices, colors)):
            perturbation = perturbations[pert_idx]
            cumulative_pert = np.cumsum(perturbed_areas[i])
            ax2.plot(days, cumulative_pert, color=color, linewidth=2, alpha=0.8,
                    label=f'{perturbation:+.1%}')
        
        ax2.set_xlabel('Day', fontsize=12)
        ax2.set_ylabel('Cumulative Fire Area (pixels)', fontsize=12)
        ax2.set_title('Cumulative Fire Impact', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Panel 3: Response curve (perturbation vs total fire area)
        ax3 = axes[1, 0]
        total_areas = [np.sum(perturbed_areas[i]) for i in range(len(selected_indices))]
        selected_perts = [perturbations[i] for i in selected_indices]
        
        ax3.plot(selected_perts, total_areas, 'bo-', linewidth=2, markersize=8)
        ax3.axhline(y=np.sum(baseline_areas), color='red', linestyle='--', 
                   linewidth=2, label='Baseline Total')
        
        # Add trend line
        if len(selected_perts) > 2:
            z = np.polyfit(selected_perts, total_areas, 1)
            p = np.poly1d(z)
            ax3.plot(selected_perts, p(selected_perts), "r--", alpha=0.8, linewidth=2)
        
        ax3.set_xlabel(f'{variable_name} Perturbation Factor', fontsize=12)
        ax3.set_ylabel('Total Fire Area (pixels)', fontsize=12)
        ax3.set_title('Variable Response Curve', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Panel 4: Fire spread pattern comparison (last day)
        ax4 = axes[1, 1]
        
        if len(baseline_pred) > 0:
            # Show difference between most extreme perturbation and baseline
            extreme_idx = selected_indices[-1] if len(selected_indices) > 0 else 0
            last_day = len(baseline_pred) - 1
            
            baseline_last = baseline_pred[last_day]
            if extreme_idx < len(predictions) and last_day < len(predictions[extreme_idx]):
                extreme_last = predictions[extreme_idx][last_day]
                
                # Calculate difference
                difference = extreme_last - baseline_last
                
                im4 = ax4.imshow(difference, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
                ax4.set_title(f'Fire Pattern Difference\n(Extreme vs Baseline - Day {last_day+1})', 
                             fontweight='bold')
                ax4.axis('off')
                
                cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
                cbar4.set_label('Fire Probability Difference', fontsize=10)
            else:
                ax4.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                        transform=ax4.transAxes, fontsize=14)
                ax4.axis('off')
        
        # Overall title with physics information
        correlation = experiment_results['correlation']
        var_config = self.config.physics_hypotheses[variable_name]
        expected_dir = var_config['expected_direction']
        actual_dir = 'positive' if correlation > 0 else 'negative'
        consistent = expected_dir == actual_dir or expected_dir == 'variable'
        
        consistency_text = "Physics Consistent" if consistent else "Physics Inconsistent"
        
        plt.suptitle(
            f'{variable_name} Fire Evolution Analysis\n'
            f'Expected: {expected_dir.title()}, Observed: {actual_dir.title()} '
            f'(r={correlation:.3f}) - {consistency_text}',
            fontsize=16, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Fire evolution analysis saved: {output_path}")
    
    def create_response_curve_plot(self, experiment_results, output_path):
        """Create response curve plot showing variable vs fire area change"""
        
        variable_name = experiment_results['variable_name']
        perturbations = experiment_results['perturbations']
        area_changes = experiment_results['area_changes']
        correlation = experiment_results['correlation']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left panel: Response curve
        ax1.plot(perturbations, area_changes, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add trend line
        z = np.polyfit(perturbations, area_changes, 1)
        p = np.poly1d(z)
        ax1.plot(perturbations, p(perturbations), "r--", alpha=0.8, linewidth=2)
        
        ax1.set_xlabel(f'{variable_name} Perturbation Factor', fontsize=12)
        ax1.set_ylabel('Fire Area Change (%)', fontsize=12)
        ax1.set_title(f'{variable_name} Response Curve', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add correlation annotation
        ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        # Right panel: Fire area by day
        fire_areas_by_day = []
        baseline_areas_by_day = []
        
        for day in range(len(experiment_results['baseline_prediction'])):
            baseline_day_area = np.sum(
                experiment_results['baseline_prediction'][day] > self.config.fire_threshold
            )
            baseline_areas_by_day.append(baseline_day_area)
            
            day_areas = []
            for pred in experiment_results['predictions']:
                if day < len(pred):
                    day_area = np.sum(pred[day] > self.config.fire_threshold)
                    day_areas.append(day_area)
            fire_areas_by_day.append(day_areas)
        
        days = range(1, len(baseline_areas_by_day) + 1)
        
        # Plot baseline
        ax2.plot(days, baseline_areas_by_day, 'k-', linewidth=3, label='Baseline', alpha=0.8)
        
        # Plot perturbations with color gradient
        colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(perturbations)))
        for i, (pert, color) in enumerate(zip(perturbations, colors)):
            areas = [day_areas[i] if i < len(day_areas) else 0 for day_areas in fire_areas_by_day]
            ax2.plot(days, areas, color=color, alpha=0.7, linewidth=1.5, 
                    label=f'{pert:+.1%}' if i % 2 == 0 or len(perturbations) <= 5 else "")
        
        ax2.set_xlabel('Day', fontsize=12)
        ax2.set_ylabel('Fire Area (pixels)', fontsize=12)
        ax2.set_title('Fire Evolution under Different Perturbations', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Response curve saved: {output_path}")=0.7, linewidth=1.5, 
                    label=f'{pert:+.1%}' if i % 2 == 0 or len(perturbations) <= 5 else "")
        
        ax2.set_xlabel('天数', fontsize=12)
        ax2.set_ylabel('火灾面积 (像素)', fontsize=12)
        ax2.set_title('不同扰动下的火灾演化', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Response curve saved: {output_path}")

# ============================================================================
# RESULTS ANALYSIS AND REPORTING
# ============================================================================

class PhysicsValidationAnalyzer:
    """Analyzes and reports physics validation results"""
    
    def __init__(self, config):
        self.config = config
    
    def create_results_table(self, all_results, output_path):
        """Create comprehensive results table"""
        
        table_data = []
        
        for results in all_results:
            var_name = results['variable_name']
            var_config = self.config.physics_hypotheses[var_name]
            
            # Calculate key metrics
            max_area_change = max(results['area_changes'])
            min_area_change = min(results['area_changes'])
            area_range = max_area_change - min_area_change
            
            # Determine response strength
            if abs(results['correlation']) > 0.7:
                response_strength = "Strong"
            elif abs(results['correlation']) > 0.4:
                response_strength = "Moderate"
            else:
                response_strength = "Weak"
            
            table_data.append({
                'Physical_Variable': var_name,
                'Perturbation_Range': f"{var_config['perturbation_range'][0]:+.1%} to {var_config['perturbation_range'][1]:+.1%}",
                'Expected_Direction': var_config['expected_direction'],
                'Observed_Direction': 'positive' if results['correlation'] > 0 else 'negative',
                'Correlation_Coeff': f"{results['correlation']:.4f}",
                'P_Value': f"{results['p_value']:.4f}",
                'Max_Area_Change': f"{max_area_change:+.1f}%",
                'Min_Area_Change': f"{min_area_change:+.1f}%",
                'Response_Range': f"{area_range:.1f}%",
                'Response_Strength': response_strength,
                'Physics_Consistent': "Yes" if results['physics_consistent'] else "No",
                'Physical_Principle': var_config['physical_principle']
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_path = output_path.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Save as Excel if possible
        try:
            df.to_excel(output_path, index=False, sheet_name='Physics Validation Results')
            print(f"  Results table saved: {output_path}")
        except ImportError:
            print(f"  Results table saved as CSV: {csv_path}")
        
        return df
    
    def create_summary_statistics(self, all_results, output_path):
        """Create summary statistics file"""
        
        stats = {
            'experiment_info': {
                'total_variables_tested': len(all_results),
                'simulation_days': self.config.simulation_days,
                'fire_threshold': self.config.fire_threshold,
                'timestamp': str(np.datetime64('now'))
            },
            'overall_performance': {},
            'variable_details': {}
        }
        
        # Overall performance metrics
        correlations = [r['correlation'] for r in all_results]
        consistent_count = sum(1 for r in all_results if r['physics_consistent'])
        
        stats['overall_performance'] = {
            'physics_consistency_rate': f"{consistent_count}/{len(all_results)} ({consistent_count/len(all_results)*100:.1f}%)",
            'mean_correlation': np.mean(correlations),
            'median_correlation': np.median(correlations),
            'strong_responses': sum(1 for c in correlations if abs(c) > 0.7),
            'moderate_responses': sum(1 for c in correlations if 0.4 < abs(c) <= 0.7),
            'weak_responses': sum(1 for c in correlations if abs(c) <= 0.4)
        }
        
        # Variable-specific details
        for results in all_results:
            var_name = results['variable_name']
            stats['variable_details'][var_name] = {
                'correlation': results['correlation'],
                'p_value': results['p_value'],
                'r_squared': results['r_value']**2,
                'physics_consistent': results['physics_consistent'],
                'max_fire_area_change': max(results['area_changes']),
                'min_fire_area_change': min(results['area_changes']),
                'baseline_total_area': results['baseline_area']
            }
        
        # Save statistics
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"  Summary statistics saved: {output_path}")
        return stats
    
    def generate_physics_report(self, all_results, output_path):
        """Generate comprehensive physics validation report"""
        
        consistent_count = sum(1 for r in all_results if r['physics_consistent'])
        total_count = len(all_results)
        consistency_rate = consistent_count / total_count if total_count > 0 else 0
        
        report = f"""
野火蔓延模型物理一致性验证报告
===============================

实验概述:
--------
本报告通过受控实验验证深度学习模型是否学习到野火蔓延的物理规律。
实验时间: {np.datetime64('now')}
测试变量数: {total_count}
模拟天数: {self.config.simulation_days}
火灾阈值: {self.config.fire_threshold}

总体结果:
--------
物理一致性: {consistent_count}/{total_count} ({consistency_rate*100:.1f}%)
模型可信度: {'高' if consistency_rate > 0.8 else '中等' if consistency_rate > 0.6 else '低'}

"""
        
        # Add individual variable results
        report += "详细验证结果:\n"
        report += "=" * 50 + "\n\n"
        
        for results in all_results:
            var_name = results['variable_name']
            var_config = self.config.physics_hypotheses[var_name]
            
            status = "✓ 一致" if results['physics_consistent'] else "✗ 不一致"
            
            report += f"{var_name}:\n"
            report += f"  物理原理: {var_config['physical_principle']}\n"
            report += f"  预期方向: {var_config['expected_direction']}\n"
            report += f"  实际方向: {'positive' if results['correlation'] > 0 else 'negative'}\n"
            report += f"  物理一致性: {status}\n"
            report += f"  相关系数: {results['correlation']:.4f}\n"
            report += f"  显著性: p = {results['p_value']:.4f}\n"
            report += f"  最大响应: {max(results['area_changes']):+.1f}%\n"
            report += f"  响应范围: {max(results['area_changes']) - min(results['area_changes']):.1f}%\n"
            report += "\n"
        
        # Add conclusions
        report += "结论和建议:\n"
        report += "=" * 50 + "\n"
        
        if consistency_rate > 0.8:
            report += "模型成功学习了大部分物理规律，具有较高的科学可信度。\n"
            report += "建议: 模型可用于实际火灾预测和决策支持。\n"
        elif consistency_rate > 0.6:
            report += "模型学习了部分物理规律，但仍有改进空间。\n"
            report += "建议: 重点改进不一致的变量，增加相关物理约束。\n"
        else:
            report += "模型物理一致性较低，需要重新审视训练策略。\n"
            report += "建议: 考虑加入物理损失函数或使用物理引导的训练方法。\n"
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  Physics validation report saved: {output_path}")
        return report

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================

class PhysicsValidationRunner:
    """Main runner for physics validation experiments"""
    
    def __init__(self, model_path, config=None):
        self.config = config or PhysicsValidationConfig()
        
        # Initialize simulator using the existing FixedFireSpreadSimulator
        self.simulator = FixedFireSpreadSimulator(model_path, self.config.base_config)
        
        print(f"Physics validation simulator ready on {self.simulator.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.simulator.model.parameters()):,}")
        
        # Initialize fire loader
        self.fire_loader = FixedFireEventLoader(self.config.base_config)
        
        # Initialize experiment components
        self.experiment = ControlledPerturbationExperiment(
            self.simulator, self.fire_loader, self.config
        )
        self.visualizer = PhysicsVisualizationGenerator(self.config)
        self.analyzer = PhysicsValidationAnalyzer(self.config)
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def run_full_validation(self, fire_event_path, start_day=0):
        """Run complete physics validation experiment"""
        
        print("=" * 60)
        print("PHYSICS VALIDATION EXPERIMENT")
        print("=" * 60)
        
        # Load fire event
        print(f"Loading fire event: {fire_event_path}")
        fire_event_data = self.fire_loader.load_fire_event(fire_event_path)
        
        # Get testable variables
        testable_variables = self.config.get_testable_variables()
        print(f"Testing {len(testable_variables)} variables: {testable_variables}")
        
        # Run experiments for each variable
        all_results = []
        
        for variable_name in testable_variables:
            print(f"\n{'='*40}")
            print(f"TESTING: {variable_name}")
            print(f"{'='*40}")
            
            try:
                # Run controlled experiment
                results = self.experiment.run_variable_experiment(
                    fire_event_data, variable_name, start_day
                )
                all_results.append(results)
                
                # Create variable-specific output directory
                var_dir = self.output_dir / variable_name
                var_dir.mkdir(exist_ok=True)
                
                # Generate visualizations
                print(f"Generating visualizations for {variable_name}...")
                
                # Multi-panel comparison GIF
                gif_path = var_dir / f"{variable_name}_comparison.gif"
                self.visualizer.create_variable_comparison_gif(results, str(gif_path))
                
                # Fire evolution analysis (4-panel like test_simulation)
                evolution_path = var_dir / f"{variable_name}_fire_evolution.png"
                self.visualizer.create_fire_evolution_analysis(results, str(evolution_path))
                
                # Response curve plot
                curve_path = var_dir / f"{variable_name}_response_curve.png"
                self.visualizer.create_response_curve_plot(results, str(curve_path))
                
                # Save variable-specific data
                data_path = var_dir / f"{variable_name}_data.json"
                self._save_variable_data(results, str(data_path))
                
                print(f"Results for {variable_name}: correlation = {results['correlation']:.4f}, "
                      f"consistent = {results['physics_consistent']}")
                
            except Exception as e:
                print(f"Error testing {variable_name}: {e}")
                continue
        
        if not all_results:
            print("No successful experiments completed!")
            return
        
        # Generate comprehensive analysis
        print(f"\n{'='*40}")
        print("GENERATING COMPREHENSIVE ANALYSIS")
        print(f"{'='*40}")
        
        # Results table
        table_path = self.output_dir / "physics_validation_results.xlsx"
        self.analyzer.create_results_table(all_results, str(table_path))
        
        # Summary statistics
        stats_path = self.output_dir / "summary_statistics.json"
        summary_stats = self.analyzer.create_summary_statistics(all_results, str(stats_path))
        
        # Physics validation report
        report_path = self.output_dir / "physics_validation_report.txt"
        self.analyzer.generate_physics_report(all_results, str(report_path))
        
        # Create overview visualization
        self._create_overview_dashboard(all_results)
        
        # Print summary
        self._print_experiment_summary(all_results, summary_stats)
        
        print(f"\n{'='*60}")
        print("PHYSICS VALIDATION COMPLETED")
        print(f"{'='*60}")
        print(f"Results saved in: {self.output_dir}")
        print("Generated files:")
        print("- Individual variable folders with GIFs and plots")
        print("- physics_validation_results.xlsx: Comprehensive results table")
        print("- summary_statistics.json: Detailed statistics")
        print("- physics_validation_report.txt: Human-readable report")
        print("- overview_dashboard.png: Overall results visualization")
        
        return all_results
    
    def _save_variable_data(self, results, output_path):
        """Save detailed variable experiment data"""
        
        # Prepare serializable data
        data_to_save = {
            'variable_name': results['variable_name'],
            'baseline_area': float(results['baseline_area']),
            'perturbations': [float(p) for p in results['perturbations']],
            'fire_areas': [float(a) for a in results['fire_areas']],
            'area_changes': [float(a) for a in results['area_changes']],
            'correlation': float(results['correlation']),
            'slope': float(results['slope']),
            'intercept': float(results['intercept']),
            'r_value': float(results['r_value']),
            'p_value': float(results['p_value']),
            'std_err': float(results['std_err']),
            'physics_consistent': bool(results['physics_consistent']),
            'physics_hypothesis': self.config.physics_hypotheses[results['variable_name']],
            'experiment_config': {
                'simulation_days': self.config.simulation_days,
                'fire_threshold': self.config.fire_threshold,
                'perturbation_range': self.config.physics_hypotheses[results['variable_name']]['perturbation_range'],
                'perturbation_steps': self.config.physics_hypotheses[results['variable_name']]['perturbation_steps']
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    
    def _create_overview_dashboard(self, all_results):
        """Create overview dashboard showing all results"""
        
        n_vars = len(all_results)
        if n_vars == 0:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        var_names = [r['variable_name'] for r in all_results]
        correlations = [r['correlation'] for r in all_results]
        consistencies = [r['physics_consistent'] for r in all_results]
        max_changes = [max(r['area_changes']) for r in all_results]
        min_changes = [min(r['area_changes']) for r in all_results]
        
        # 1. Correlation strength plot
        colors = ['green' if c else 'red' for c in consistencies]
        bars1 = ax1.bar(range(n_vars), correlations, color=colors, alpha=0.7)
        ax1.set_xlabel('变量', fontsize=12)
        ax1.set_ylabel('相关系数', fontsize=12)
        ax1.set_title('物理变量与火灾响应相关性', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(n_vars))
        ax1.set_xticklabels(var_names, rotation=45, ha='right')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, corr) in enumerate(zip(bars1, correlations)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
                    f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # 2. Physics consistency pie chart
        consistent_count = sum(consistencies)
        inconsistent_count = n_vars - consistent_count
        
        if n_vars > 0:
            labels = ['物理一致', '物理不一致']
            sizes = [consistent_count, inconsistent_count]
            colors_pie = ['lightgreen', 'lightcoral']
            
            ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
                   startangle=90, textprops={'fontsize': 12})
            ax2.set_title(f'物理一致性统计\n({consistent_count}/{n_vars} 变量符合预期)', 
                         fontsize=14, fontweight='bold')
        
        # 3. Response magnitude comparison
        x_pos = np.arange(n_vars)
        width = 0.35
        
        bars3_max = ax3.bar(x_pos - width/2, max_changes, width, label='最大响应', 
                           alpha=0.7, color='orange')
        bars3_min = ax3.bar(x_pos + width/2, min_changes, width, label='最小响应', 
                           alpha=0.7, color='blue')
        
        ax3.set_xlabel('变量', fontsize=12)
        ax3.set_ylabel('火灾面积变化 (%)', fontsize=12)
        ax3.set_title('变量响应幅度对比', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(var_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # 4. Correlation vs Response scatter plot
        response_ranges = [max_changes[i] - min_changes[i] for i in range(n_vars)]
        
        scatter = ax4.scatter([abs(c) for c in correlations], response_ranges, 
                             c=colors, s=100, alpha=0.7)
        
        for i, var in enumerate(var_names):
            ax4.annotate(var, (abs(correlations[i]), response_ranges[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax4.set_xlabel('相关系数绝对值', fontsize=12)
        ax4.set_ylabel('响应范围 (%)', fontsize=12)
        ax4.set_title('相关性 vs 响应幅度', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add consistency legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='物理一致'),
                          Patch(facecolor='red', alpha=0.7, label='物理不一致')]
        ax4.legend(handles=legend_elements)
        
        plt.suptitle('物理验证实验总览', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.output_dir / "overview_dashboard.png"
        plt.savefig(str(dashboard_path), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Overview dashboard saved: {dashboard_path}")
    
    def _print_experiment_summary(self, all_results, summary_stats):
        """Print experiment summary to console"""
        
        if not all_results:
            return
        
        print(f"\n{'='*50}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*50}")
        
        # Overall statistics
        consistent_count = sum(1 for r in all_results if r['physics_consistent'])
        total_count = len(all_results)
        
        print(f"Variables tested: {total_count}")
        print(f"Physics consistent: {consistent_count}/{total_count} ({consistent_count/total_count*100:.1f}%)")
        print(f"Mean correlation: {summary_stats['overall_performance']['mean_correlation']:.4f}")
        print(f"Strong responses (|r| > 0.7): {summary_stats['overall_performance']['strong_responses']}")
        print(f"Moderate responses (0.4 < |r| ≤ 0.7): {summary_stats['overall_performance']['moderate_responses']}")
        print(f"Weak responses (|r| ≤ 0.4): {summary_stats['overall_performance']['weak_responses']}")
        
        print(f"\nVARIABLE-SPECIFIC RESULTS:")
        print("-" * 30)
        
        for results in sorted(all_results, key=lambda x: abs(x['correlation']), reverse=True):
            var_name = results['variable_name']
            corr = results['correlation']
            consistent = "✓" if results['physics_consistent'] else "✗"
            max_change = max(results['area_changes'])
            
            print(f"{var_name:15s}: r={corr:+.4f}, {consistent}, max_response={max_change:+.1f}%")
    
    def create_demo_data(self):
        """Create synthetic demonstration data"""
        print("Creating synthetic demonstration data...")
        
        # Create a synthetic fire event
        T, C, H, W = 10, len(self.config.base_config.FEATURE_NAMES), 128, 128
        synthetic_data = np.random.randn(T, C, H, W).astype(np.float32)
        
        # Add realistic values for key features
        feature_names = self.config.base_config.FEATURE_NAMES
        
        for i, name in enumerate(feature_names):
            if 'Temp' in name:
                synthetic_data[:, i] = np.random.uniform(280, 320, (T, H, W))
            elif 'Wind' in name and 'Direction' not in name:
                synthetic_data[:, i] = np.random.uniform(0, 20, (T, H, W))
            elif 'NDVI' in name or 'EVI' in name:
                synthetic_data[:, i] = np.random.uniform(0.1, 0.8, (T, H, W))
            elif 'Precip' in name:
                synthetic_data[:, i] = np.random.exponential(2, (T, H, W))
            elif 'Slope' in name:
                synthetic_data[:, i] = np.random.uniform(-30, 45, (T, H, W))
            elif 'Active_Fire' in name:
                # Create realistic fire progression
                center_h, center_w = H//2, W//2
                for t in range(T):
                    fire_size = 3 + t * 1.5
                    h_start = max(0, int(center_h - fire_size))
                    h_end = min(H, int(center_h + fire_size))
                    w_start = max(0, int(center_w - fire_size))
                    w_end = min(W, int(center_w + fire_size))
                    
                    fire_region = np.zeros((H, W))
                    fire_region[h_start:h_end, w_start:w_end] = np.random.rand(
                        h_end-h_start, w_end-w_start) * 0.7
                    synthetic_data[t, i] = fire_region
        
        # Save synthetic data
        demo_path = "demo_fire_event.hdf5"
        with h5py.File(demo_path, 'w') as f:
            f.create_dataset('data', data=synthetic_data)
        
        print(f"Demo data created: {demo_path}")
        return demo_path

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for physics validation experiments"""
    
    parser = argparse.ArgumentParser(description='Physics Validation Experiments for Wildfire Spread Model')
    parser.add_argument('--model', type=str, default='best_fire_model_official.pth',
                       help='Path to trained model')
    parser.add_argument('--fire_event', type=str, default=None,
                       help='Path to HDF5 fire event file')
    parser.add_argument('--start_day', type=int, default=0,
                       help='Start day in fire event')
    parser.add_argument('--demo', action='store_true',
                       help='Use synthetic demo data')
    parser.add_argument('--output_dir', type=str, default='physics_validation',
                       help='Output directory for results')
    parser.add_argument('--simulation_days', type=int, default=5,
                       help='Number of days to simulate')
    parser.add_argument('--variables', type=str, nargs='+', default=None,
                       help='Specific variables to test (default: all available)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return
    
    # Setup configuration
    config = PhysicsValidationConfig()
    config.output_dir = args.output_dir
    config.simulation_days = args.simulation_days
    
    if args.variables:
        # Filter to only requested variables
        available_vars = config.get_testable_variables()
        requested_vars = [v for v in args.variables if v in available_vars]
        if requested_vars != args.variables:
            missing = set(args.variables) - set(requested_vars)
            print(f"Warning: Variables not available for testing: {missing}")
            print(f"Available variables: {available_vars}")
        
        # Update config to only test requested variables
        filtered_hypotheses = {k: v for k, v in config.physics_hypotheses.items() 
                              if k in requested_vars}
        config.physics_hypotheses = filtered_hypotheses
    
    try:
        # Initialize experiment runner
        runner = PhysicsValidationRunner(args.model, config)
        
        # Determine fire event to use
        fire_event_path = None
        
        if args.demo:
            print("Using synthetic demonstration data")
            fire_event_path = runner.create_demo_data()
        elif args.fire_event and os.path.exists(args.fire_event):
            fire_event_path = args.fire_event
        else:
            # Try to find available fire events
            import glob
            patterns = [
                'data/processed/*/*.hdf5',
                'fire_*.hdf5',
                '*.hdf5'
            ]
            
            for pattern in patterns:
                found_files = glob.glob(pattern)
                if found_files:
                    fire_event_path = found_files[0]
                    print(f"Using found fire event: {fire_event_path}")
                    break
            
            if not fire_event_path:
                print("No fire event found. Creating demo data...")
                fire_event_path = runner.create_demo_data()
        
        # Run physics validation experiments
        results = runner.run_full_validation(fire_event_path, args.start_day)
        
        if results:
            print(f"\nPhysics validation completed successfully!")
            print(f"Results available in: {args.output_dir}/")
        else:
            print("No successful experiments completed.")
            
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
=== USAGE EXAMPLES ===

1. Basic usage with demo data:
   python physics_validation_experiment.py --demo

2. Test specific fire event:
   python physics_validation_experiment.py --fire_event data/processed/2020/fire_12345.hdf5

3. Test only specific variables:
   python physics_validation_experiment.py --demo --variables Wind_Speed Max_Temp_K Slope

4. Extended simulation with custom output:
   python physics_validation_experiment.py --demo --simulation_days 7 --output_dir my_physics_results

=== OUTPUT STRUCTURE ===

physics_validation/
├── Wind_Speed/
│   ├── Wind_Speed_comparison.gif      # 8-panel comparison animation
│   ├── Wind_Speed_response_curve.png  # Response curve and evolution
│   └── Wind_Speed_data.json          # Detailed experiment data
├── Max_Temp_K/
│   ├── Max_Temp_K_comparison.gif
│   ├── Max_Temp_K_response_curve.png
│   └── Max_Temp_K_data.json
├── ... (other variables)
├── physics_validation_results.xlsx    # Comprehensive results table
├── summary_statistics.json           # Detailed statistics
├── physics_validation_report.txt     # Human-readable report
└── overview_dashboard.png            # Overall results visualization

=== KEY FEATURES ===

1. **Multi-panel visualizations**: Each variable gets an 8-panel GIF showing baseline vs different perturbation levels
2. **Quantitative analysis**: Correlation coefficients, p-values, response curves
3. **Physics consistency validation**: Automatic checking against expected physical relationships
4. **Comprehensive reporting**: Excel tables, JSON statistics, text reports
5. **Flexible configuration**: Easily modify variables, perturbation ranges, simulation parameters
6. **Robust error handling**: Continues with other variables if one fails
7. **Demo mode**: Creates synthetic data for testing without real fire events
"""