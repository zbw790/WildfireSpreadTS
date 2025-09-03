# Enhanced Feature Sensitivity Analysis Tool

This tool creates comprehensive visual comparisons showing how changes in individual features affect wildfire spread predictions. It generates animated GIFs and detailed analyses that help understand model behavior and feature importance.

## What This Tool Does

The tool generates three types of visualizations for each feature:

1. **Actual Fire Spreading** (Ground Truth) - Shows the real fire evolution from your data
2. **Raw Model Predictions** - Shows predictions using actual feature values (baseline)
3. **Modified Predictions** - Shows how predictions change when individual feature values are perturbed while keeping all other features constant

This allows you to visually see how changing specific environmental conditions (like temperature, vegetation, precipitation) affects fire spread predictions.

## Generated Outputs

For each analyzed feature, the tool creates:

### 1. Evolution GIFs (`{feature}_evolution.gif`)
- **4-panel animated comparison** showing:
  - Top-left: Ground truth fire evolution
  - Top-right: Baseline model prediction
  - Bottom-left: Prediction with feature decreased by 20%
  - Bottom-right: Prediction with feature increased by 20%

### 2. Sensitivity Metrics (`{feature}_sensitivity_metrics.png`)
- **Response curve** showing how fire area changes with feature perturbations
- **Correlation coefficient** indicating sensitivity strength
- **Baseline reference** for comparison

### 3. Response Curves (`{feature}_response_curves.png`)
- **Multi-day analysis** showing how sensitivity changes over time
- **Daily response curves** for each simulation day
- **Color-coded progression** from day 1 to final day

### 4. Raw Data (`{feature}_data.json`)
- **Detailed numerical results** for further analysis
- **Fire areas, changes, and predictions** for all perturbation levels
- **Baseline comparisons** and metrics

## Usage

### Basic Usage
```bash
python feature_sensitivity_analyzer.py \
    --model path/to/your/model.pth \
    --fire_event path/to/fire_event.h5 \
    --features NDVI EVI2 Max_Temp_K Total_Precip
```

### Advanced Usage
```bash
python feature_sensitivity_analyzer.py \
    --model backup/models/best_model.pth \
    --fire_event data/2021/fire_event_001.h5 \
    --start_day 5 \
    --output_dir my_sensitivity_analysis \
    --features NDVI EVI2 Max_Temp_K Min_Temp_K Total_Precip Wind_Speed ERC \
    --device cuda
```

### Testing the Tool
```bash
python test_feature_sensitivity.py
```

## Parameters

- `--model`: Path to your trained PyTorch model file (.pth, .pt, .ckpt)
- `--fire_event`: Path to HDF5 fire event data file
- `--start_day`: Starting day in the fire event sequence (default: 0)
- `--output_dir`: Directory for output files (default: 'feature_sensitivity')
- `--features`: List of features to analyze (default: ['NDVI', 'EVI2', 'Max_Temp_K', 'Total_Precip'])
- `--device`: Computing device ('auto', 'cpu', 'cuda')

## Available Features for Analysis

The tool can analyze any of these features from your model:

**Vegetation Indices:**
- `NDVI` - Normalized Difference Vegetation Index
- `EVI2` - Enhanced Vegetation Index 2

**Temperature:**
- `Max_Temp_K` - Maximum Temperature
- `Min_Temp_K` - Minimum Temperature

**Weather:**
- `Total_Precip` - Total Precipitation
- `Wind_Speed` - Wind Speed
- `Spec_Hum` - Specific Humidity

**Fire Weather:**
- `ERC` - Energy Release Component
- `PDSI` - Palmer Drought Severity Index

**Topography:**
- `Elevation` - Terrain Elevation
- `Slope` - Terrain Slope

**Satellite Data:**
- `VIIRS_M11`, `VIIRS_I2`, `VIIRS_I1` - Satellite thermal/reflectance bands

## Understanding the Results

### Sensitivity Interpretation

**High Positive Correlation (>0.5):**
- Feature increases lead to more fire spread
- Strong model dependency on this feature
- Critical for fire prediction accuracy

**High Negative Correlation (<-0.5):**
- Feature increases lead to less fire spread
- Protective/suppressive effect
- Important for fire containment scenarios

**Low Correlation (-0.2 to 0.2):**
- Feature has minimal impact on fire spread
- May be redundant or context-dependent
- Consider for feature selection

### Visual Analysis Tips

1. **Compare GIF panels** to see immediate visual differences
2. **Check response curves** for non-linear relationships
3. **Look at daily progression** to understand temporal effects
4. **Compare multiple features** to identify most influential ones

## Example Workflow

1. **Prepare your data:**
   ```bash
   # Ensure you have:
   # - Trained model file (.pth)
   # - Fire event data (.h5)
   # - Feature statistics (feature_stats.npz)
   ```

2. **Run analysis:**
   ```bash
   python feature_sensitivity_analyzer.py \
       --model your_model.pth \
       --fire_event your_fire_event.h5 \
       --features NDVI Max_Temp_K Total_Precip
   ```

3. **Review results:**
   ```bash
   # Check the output directory:
   feature_sensitivity/
   ├── NDVI/
   │   ├── NDVI_evolution.gif
   │   ├── NDVI_sensitivity_metrics.png
   │   └── NDVI_data.json
   ├── Max_Temp_K/
   │   └── ...
   └── sensitivity_analysis_report.txt
   ```

4. **Analyze findings:**
   - Open GIFs to see visual differences
   - Read the summary report
   - Compare sensitivity metrics across features

## Integration with Existing Code

This tool is designed to work with your existing wildfire prediction pipeline:

- **Compatible with** `test_simulation.py` fire evolution functionality
- **Uses same configuration** as `test_with_Stats.py` feature handling
- **Extends existing** physics validation approaches
- **Maintains consistency** with your model architecture

## Troubleshooting

**Model Loading Issues:**
- Ensure model file is compatible
- Check input channel dimensions
- Verify feature configuration matches training

**Data Format Problems:**
- Confirm HDF5 files have 'sequence' dataset
- Check data shape: (Time, Channels, Height, Width)
- Verify feature indices align with model

**Memory Issues:**
- Reduce simulation days
- Use fewer perturbation levels
- Switch to CPU if GPU memory insufficient

**No Visual Differences:**
- Try larger perturbation ranges
- Check if feature is actually used by model
- Verify feature normalization is correct

## Output Structure

```
feature_sensitivity/
├── NDVI/
│   ├── NDVI_evolution.gif          # 4-panel evolution comparison
│   ├── NDVI_sensitivity_metrics.png # Response curve
│   ├── NDVI_response_curves.png     # Daily progression
│   └── NDVI_data.json              # Raw numerical data
├── Max_Temp_K/
│   └── ... (same structure)
└── sensitivity_analysis_report.txt  # Comprehensive summary
```

This tool provides the visual fire evolution comparisons you requested, showing actual fire changes and predictions with modified feature values, helping you understand how environmental factors influence your wildfire prediction model.
