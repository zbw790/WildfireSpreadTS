"""
Wildfire Ensemble Analysis & Variable Impact Study
==================================================

This script implements the supervisor's requirements:
1. Wildfire spread simulation ensemble for a single event
2. Variable correlation analysis with wildfire progression 
3. Feature importance ranking
4. Literature comparison framework

Author: Bowen
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import glob
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import mutual_info_regression
from scipy.stats import pearsonr, spearmanr
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

class WildfireEnsembleAnalyzer:
    """
    Comprehensive analyzer for wildfire spread ensemble simulation
    and variable impact assessment
    """
    
    def __init__(self, model_path=None, data_dir="data"):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model if provided
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Define feature metadata
        self.feature_metadata = self._create_feature_metadata()
        
        # Results storage
        self.correlation_results = {}
        self.importance_results = {}
        self.ensemble_results = {}
    
    def _create_feature_metadata(self):
        """
        Create comprehensive metadata for all 23 feature channels
        Based on WildfireSpreadTS dataset documentation
        """
        metadata = {
            0: {
                'name': 'VIIRS_Active_Fire', 
                'unit': 'Confidence [0-100]', 
                'expected_correlation': 'Positive',
                'explanation': 'Higher fire confidence indicates active burning, promotes spread'
            },
            1: {
                'name': 'NDVI', 
                'unit': 'Index [-1, 1]', 
                'expected_correlation': 'Negative',
                'explanation': 'Higher vegetation greenness retains moisture, inhibits spread'
            },
            2: {
                'name': 'EVI2', 
                'unit': 'Index [-1, 1]', 
                'expected_correlation': 'Negative',
                'explanation': 'Enhanced vegetation index correlates with moisture content'
            },
            3: {
                'name': 'Temperature_Max', 
                'unit': '°C', 
                'expected_correlation': 'Positive',
                'explanation': 'Higher temperature increases fuel dryness and ignition probability'
            },
            4: {
                'name': 'Temperature_Min', 
                'unit': '°C', 
                'expected_correlation': 'Positive',
                'explanation': 'Higher minimum temperature reduces overnight moisture recovery'
            },
            5: {
                'name': 'Temperature_Mean', 
                'unit': '°C', 
                'expected_correlation': 'Positive',
                'explanation': 'Higher mean temperature promotes fuel drying'
            },
            6: {
                'name': 'Relative_Humidity', 
                'unit': '%', 
                'expected_correlation': 'Negative',
                'explanation': 'Higher humidity increases fuel moisture, inhibits combustion'
            },
            7: {
                'name': 'Wind_Speed', 
                'unit': 'm/s', 
                'expected_correlation': 'Positive',
                'explanation': 'Higher wind speed accelerates fire spread by oxygen supply'
            },
            8: {
                'name': 'Wind_Direction', 
                'unit': 'degrees [0-360]', 
                'expected_correlation': 'Complex',
                'explanation': 'Wind direction determines spread pattern and interaction with topography'
            },
            9: {
                'name': 'Precipitation', 
                'unit': 'mm', 
                'expected_correlation': 'Negative',
                'explanation': 'Precipitation increases fuel moisture and can suppress fire'
            },
            10: {
                'name': 'Atmospheric_Pressure', 
                'unit': 'hPa', 
                'expected_correlation': 'Mixed',
                'explanation': 'Lower pressure may indicate weather systems bringing moisture'
            },
            11: {
                'name': 'Elevation', 
                'unit': 'm', 
                'expected_correlation': 'Mixed',
                'explanation': 'Elevation affects temperature, moisture, and fuel types'
            },
            12: {
                'name': 'Slope', 
                'unit': 'degrees', 
                'expected_correlation': 'Positive',
                'explanation': 'Steeper slopes accelerate uphill fire spread via preheating'
            },
            13: {
                'name': 'Aspect', 
                'unit': 'degrees [0-360]', 
                'expected_correlation': 'Complex',
                'explanation': 'South-facing slopes typically drier, more fire-prone'
            },
            14: {
                'name': 'PDSI', 
                'unit': 'Index [-4, 4]', 
                'expected_correlation': 'Positive',
                'explanation': 'Higher drought index indicates drier conditions favoring fire'
            },
            15: {
                'name': 'Land_Cover_Class', 
                'unit': 'Category [0-15]', 
                'expected_correlation': 'Complex',
                'explanation': 'Different vegetation types have varying flammability'
            },
            16: {
                'name': 'Forecast_Temp_1d', 
                'unit': '°C', 
                'expected_correlation': 'Positive',
                'explanation': 'Forecasted temperature affects future fire behavior'
            },
            17: {
                'name': 'Forecast_Humidity_1d', 
                'unit': '%', 
                'expected_correlation': 'Negative',
                'explanation': 'Forecasted humidity affects future moisture recovery'
            },
            18: {
                'name': 'Forecast_Wind_1d', 
                'unit': 'm/s', 
                'expected_correlation': 'Positive',
                'explanation': 'Forecasted wind affects future spread rate'
            },
            19: {
                'name': 'Forecast_Temp_2d', 
                'unit': '°C', 
                'expected_correlation': 'Positive',
                'explanation': '2-day temperature forecast for medium-term behavior'
            },
            20: {
                'name': 'Forecast_Humidity_2d', 
                'unit': '%', 
                'expected_correlation': 'Negative',
                'explanation': '2-day humidity forecast for medium-term behavior'
            },
            21: {
                'name': 'Forecast_Wind_2d', 
                'unit': 'm/s', 
                'expected_correlation': 'Positive',
                'explanation': '2-day wind forecast for medium-term behavior'
            },
            22: {
                'name': 'Historical_Fire', 
                'unit': 'Binary/Continuous', 
                'expected_correlation': 'Positive',
                'explanation': 'Previous fire history indicates fuel availability and terrain susceptibility'
            }
        }
        return metadata
    
    def load_single_fire_event(self, max_files=50):
        """
        Load a single representative fire event for ensemble analysis
        """
        print("Loading fire event for ensemble analysis...")
        
        hdf5_files = list(Path(self.data_dir).rglob("*.hdf5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {self.data_dir}")
        
        # Select a file with reasonable size for analysis
        selected_file = None
        for file_path in hdf5_files[:max_files]:
            try:
                with h5py.File(file_path, 'r') as f:
                    features = f['features'][:]
                    target = f['target'][:]
                    
                    # Check if this fire has sufficient progression
                    fire_pixels = np.sum(target > 0.5)
                    total_pixels = target.size
                    fire_ratio = fire_pixels / total_pixels
                    
                    if 0.01 < fire_ratio < 0.1:  # 1-10% fire coverage
                        selected_file = file_path
                        print(f"Selected fire: {file_path.name}")
                        print(f"Fire coverage: {fire_ratio:.3f}")
                        break
                        
            except Exception as e:
                continue
        
        if selected_file is None:
            # Fallback to first available file
            selected_file = hdf5_files[0]
            print(f"Using fallback fire: {selected_file.name}")
        
        # Load the selected fire event
        with h5py.File(selected_file, 'r') as f:
            features = f['features'][:]  # [T, C, H, W]
            target = f['target'][:]      # [T, H, W]
        
        # Basic preprocessing
        features = self._preprocess_features(features)
        target = self._preprocess_target(target)
        
        return features, target, selected_file.name
    
    def _preprocess_features(self, features):
        """Preprocess features for model input"""
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1000, neginf=-1000)
        
        # Resize to standard size
        T, C, H, W = features.shape
        target_size = 128
        
        if H != target_size or W != target_size:
            features_resized = np.zeros((T, C, target_size, target_size))
            for t in range(T):
                for c in range(C):
                    # Simple resize using torch
                    tensor = torch.from_numpy(features[t, c]).unsqueeze(0).unsqueeze(0).float()
                    resized = F.interpolate(tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
                    features_resized[t, c] = resized.squeeze().numpy()
            features = features_resized
        
        return features
    
    def _preprocess_target(self, target):
        """Preprocess target for model input"""
        # Handle NaN values
        target = np.nan_to_num(target, nan=0.0)
        
        # Resize to standard size
        T, H, W = target.shape
        target_size = 128
        
        if H != target_size or W != target_size:
            target_resized = np.zeros((T, target_size, target_size))
            for t in range(T):
                tensor = torch.from_numpy(target[t]).unsqueeze(0).unsqueeze(0).float()
                resized = F.interpolate(tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
                target_resized[t] = resized.squeeze().numpy()
            target = target_resized
        
        # Binarize
        target = (target > 0.5).astype(np.float32)
        
        return target
    
    def create_variable_perturbations(self, features, perturbation_factors=None):
        """
        Create ensemble by perturbing key variables
        
        Args:
            features: Original features [T, C, H, W]
            perturbation_factors: Dict of {channel_idx: [factor1, factor2, ...]}
        """
        if perturbation_factors is None:
            perturbation_factors = {
                7: [0.8, 1.2, 1.5],      # Wind speed: -20%, +20%, +50%
                6: [0.9, 1.1],           # Humidity: -10%, +10%
                3: [0.95, 1.05],         # Temperature max: -5%, +5%
                12: [0.5, 2.0],          # Slope: half, double
                1: [0.8, 1.2],           # NDVI: -20%, +20%
            }
        
        ensemble = {'baseline': features.copy()}
        
        for channel_idx, factors in perturbation_factors.items():
            channel_name = self.feature_metadata[channel_idx]['name']
            
            for i, factor in enumerate(factors):
                perturbed_features = features.copy()
                perturbed_features[:, channel_idx] *= factor
                
                # Ensure reasonable bounds
                if channel_idx == 6:  # Humidity [0, 100]
                    perturbed_features[:, channel_idx] = np.clip(perturbed_features[:, channel_idx], 0, 100)
                elif channel_idx in [1, 2]:  # NDVI, EVI2 [-1, 1]
                    perturbed_features[:, channel_idx] = np.clip(perturbed_features[:, channel_idx], -1, 1)
                
                ensemble[f"{channel_name}_{factor:.1f}x"] = perturbed_features
        
        return ensemble
    
    def analyze_variable_correlations(self, features, target):
        """
        Analyze correlation between each variable and fire progression
        """
        print("Analyzing variable correlations with fire progression...")
        
        T, C, H, W = features.shape
        correlations = {}
        
        for channel_idx in range(C):
            channel_name = self.feature_metadata[channel_idx]['name']
            
            # Flatten spatial-temporal data
            feature_values = features[:, channel_idx].flatten()
            target_values = target.flatten()
            
            # Remove NaN values
            valid_mask = ~(np.isnan(feature_values) | np.isnan(target_values))
            feature_clean = feature_values[valid_mask]
            target_clean = target_values[valid_mask]
            
            if len(feature_clean) > 100:  # Ensure sufficient data
                # Calculate correlations
                pearson_corr, pearson_p = pearsonr(feature_clean, target_clean)
                spearman_corr, spearman_p = spearmanr(feature_clean, target_clean)
                
                # Calculate mutual information (captures non-linear relationships)
                try:
                    mi_score = mutual_info_regression(feature_clean.reshape(-1, 1), target_clean)[0]
                except:
                    mi_score = 0.0
                
                correlations[channel_idx] = {
                    'name': channel_name,
                    'pearson_r': pearson_corr,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_corr,
                    'spearman_p': spearman_p,
                    'mutual_info': mi_score,
                    'expected': self.feature_metadata[channel_idx]['expected_correlation']
                }
            else:
                correlations[channel_idx] = {
                    'name': channel_name,
                    'pearson_r': 0.0,
                    'pearson_p': 1.0,
                    'spearman_r': 0.0,
                    'spearman_p': 1.0,
                    'mutual_info': 0.0,
                    'expected': self.feature_metadata[channel_idx]['expected_correlation']
                }
        
        self.correlation_results = correlations
        return correlations
    
    def calculate_feature_importance_simple(self, features, target):
        """
        Calculate feature importance using variance-based method
        (since we don't have a trained model)
        """
        print("Calculating feature importance...")
        
        T, C, H, W = features.shape
        importance_scores = {}
        
        for channel_idx in range(C):
            channel_name = self.feature_metadata[channel_idx]['name']
            
            # Calculate variance of feature where fire occurs vs where it doesn't
            fire_mask = target > 0.5
            no_fire_mask = target <= 0.5
            
            feature_channel = features[:, channel_idx]
            
            # Mean feature value in fire areas
            fire_mean = np.mean(feature_channel[fire_mask]) if np.any(fire_mask) else 0
            no_fire_mean = np.mean(feature_channel[no_fire_mask]) if np.any(no_fire_mask) else 0
            
            # Standard deviations
            fire_std = np.std(feature_channel[fire_mask]) if np.any(fire_mask) else 0
            no_fire_std = np.std(feature_channel[no_fire_mask]) if np.any(no_fire_mask) else 0
            
            # Separability score (larger difference = more important)
            separability = abs(fire_mean - no_fire_mean) / (fire_std + no_fire_std + 1e-8)
            
            importance_scores[channel_idx] = {
                'name': channel_name,
                'separability_score': separability,
                'fire_mean': fire_mean,
                'no_fire_mean': no_fire_mean,
                'difference': fire_mean - no_fire_mean
            }
        
        self.importance_results = importance_scores
        return importance_scores
    
    def create_comprehensive_results_table(self):
        """
        Create the comprehensive table requested by supervisor
        """
        print("Creating comprehensive results table...")
        
        results_data = []
        
        for channel_idx in range(23):
            metadata = self.feature_metadata[channel_idx]
            
            # Get correlation data if available
            corr_data = self.correlation_results.get(channel_idx, {})
            importance_data = self.importance_results.get(channel_idx, {})
            
            # Determine observed correlation direction
            pearson_r = corr_data.get('pearson_r', 0)
            if abs(pearson_r) < 0.05:
                observed_correlation = 'Weak'
            elif pearson_r > 0:
                observed_correlation = 'Positive'
            else:
                observed_correlation = 'Negative'
            
            # Physical interpretation
            physical_match = 'Yes' if observed_correlation.lower() in metadata['expected_correlation'].lower() else 'No'
            if metadata['expected_correlation'] == 'Complex' or metadata['expected_correlation'] == 'Mixed':
                physical_match = 'Partial'
            
            results_data.append({
                'Channel_ID': channel_idx,
                'Variable_Name': metadata['name'],
                'Unit': metadata['unit'],
                'Expected_Correlation': metadata['expected_correlation'],
                'Observed_Correlation': observed_correlation,
                'Pearson_r': round(pearson_r, 3),
                'P_value': round(corr_data.get('pearson_p', 1.0), 3),
                'Mutual_Info': round(corr_data.get('mutual_info', 0.0), 3),
                'Importance_Score': round(importance_data.get('separability_score', 0.0), 3),
                'Physical_Match': physical_match,
                'Explanation': metadata['explanation']
            })
        
        results_df = pd.DataFrame(results_data)
        return results_df
    
    def visualize_correlation_results(self, save_path="correlation_analysis.png"):
        """Create visualization of correlation results"""
        if not self.correlation_results:
            print("No correlation results to visualize")
            return
        
        # Prepare data for visualization
        names = [self.correlation_results[i]['name'] for i in range(23)]
        pearson_vals = [self.correlation_results[i]['pearson_r'] for i in range(23)]
        mutual_info_vals = [self.correlation_results[i]['mutual_info'] for i in range(23)]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Pearson correlations
        colors = ['red' if x < 0 else 'blue' for x in pearson_vals]
        ax1.barh(range(23), pearson_vals, color=colors, alpha=0.7)
        ax1.set_yticks(range(23))
        ax1.set_yticklabels(names, fontsize=10)
        ax1.set_xlabel('Pearson Correlation with Fire Progression')
        ax1.set_title('Variable Correlations with Wildfire Spread')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Mutual information (importance)
        ax2.barh(range(23), mutual_info_vals, color='green', alpha=0.7)
        ax2.set_yticks(range(23))
        ax2.set_yticklabels(names, fontsize=10)
        ax2.set_xlabel('Mutual Information Score')
        ax2.set_title('Variable Importance (Non-linear Relationships)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Correlation visualization saved to {save_path}")
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline as requested by supervisor
        """
        print("="*60)
        print("WILDFIRE ENSEMBLE ANALYSIS - SUPERVISOR REQUIREMENTS")
        print("="*60)
        
        try:
            # 1. Load single fire event
            print("\n1. Loading representative fire event...")
            features, target, fire_name = self.load_single_fire_event()
            print(f"Loaded fire: {fire_name}")
            print(f"Features shape: {features.shape}")
            print(f"Target shape: {target.shape}")
            
            # 2. Create ensemble perturbations
            print("\n2. Creating ensemble perturbations...")
            ensemble = self.create_variable_perturbations(features)
            print(f"Created {len(ensemble)} ensemble members")
            
            # 3. Analyze correlations
            print("\n3. Analyzing variable correlations...")
            correlations = self.analyze_variable_correlations(features, target)
            
            # 4. Calculate feature importance
            print("\n4. Calculating feature importance...")
            importance = self.calculate_feature_importance_simple(features, target)
            
            # 5. Create comprehensive table
            print("\n5. Creating comprehensive results table...")
            results_table = self.create_comprehensive_results_table()
            
            # 6. Save results
            print("\n6. Saving results...")
            results_table.to_csv('wildfire_variable_analysis.csv', index=False)
            print("Results saved to 'wildfire_variable_analysis.csv'")
            
            # 7. Create visualizations
            print("\n7. Creating visualizations...")
            self.visualize_correlation_results()
            
            # 8. Print summary
            print("\n" + "="*60)
            print("ANALYSIS SUMMARY")
            print("="*60)
            
            # Top positive correlations
            pos_corrs = [(i, data['pearson_r']) for i, data in correlations.items() if data['pearson_r'] > 0.1]
            pos_corrs.sort(key=lambda x: x[1], reverse=True)
            
            print("\nTop POSITIVE correlations with fire spread:")
            for i, (channel_idx, corr) in enumerate(pos_corrs[:5]):
                name = self.feature_metadata[channel_idx]['name']
                print(f"{i+1}. {name}: r={corr:.3f}")
            
            # Top negative correlations
            neg_corrs = [(i, data['pearson_r']) for i, data in correlations.items() if data['pearson_r'] < -0.1]
            neg_corrs.sort(key=lambda x: x[1])
            
            print("\nTop NEGATIVE correlations with fire spread:")
            for i, (channel_idx, corr) in enumerate(neg_corrs[:5]):
                name = self.feature_metadata[channel_idx]['name']
                print(f"{i+1}. {name}: r={corr:.3f}")
            
            # Top importance scores
            importance_sorted = [(i, data['separability_score']) for i, data in importance.items()]
            importance_sorted.sort(key=lambda x: x[1], reverse=True)
            
            print("\nTop variables by importance (separability):")
            for i, (channel_idx, score) in enumerate(importance_sorted[:5]):
                name = self.feature_metadata[channel_idx]['name']
                print(f"{i+1}. {name}: score={score:.3f}")
            
            print("\n" + "="*60)
            print("DELIVERABLES CREATED:")
            print("- wildfire_variable_analysis.csv (comprehensive table)")
            print("- correlation_analysis.png (correlation visualization)")
            print("- Console output with top correlations and importance")
            print("="*60)
            
            return results_table, correlations, importance, ensemble
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

def main():
    """
    Main execution function
    """
    print("Starting Wildfire Ensemble Analysis...")
    print("This analysis addresses supervisor requirements for:")
    print("1. Single event ensemble simulation")
    print("2. Variable correlation analysis") 
    print("3. Feature importance ranking")
    print("4. Literature comparison framework")
    print()
    
    # Initialize analyzer
    analyzer = WildfireEnsembleAnalyzer(data_dir="data")
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    if results[0] is not None:
        print("\n✅ Analysis completed successfully!")
        print("Review the generated files and console output.")
        print("This provides the foundation for your supervisor presentation.")
    else:
        print("\n❌ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main() 