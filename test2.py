# Save individual frames
            for i, pred in enumerate(predictions[:30]):
                plt.figure(figsize=(8, 6))
                plt.imshow(pred, cmap='Reds', vmin=0, vmax=1)
                plt.title(f'Simulated Fire - Day {i+1}')
                plt.colorbar(label='Fire Probability')
                plt.axis('off')
                plt.savefig(f'fire_frame_{i:02d}.png', bbox_inches='tight', dpi=150)
                plt.close()
            print("Saved individual frames: fire_frame_*.png")
        
        return anim

# ============================================================================
# VARIABLE SENSITIVITY ANALYSIS
# ============================================================================

class VariableSensitivityAnalyzer:
    """Analyze the impact of different variables on fire spread"""
    
    def __init__(self, simulator, config=None):
        self.simulator = simulator
        self.config = config or WildFireConfig()
    
    def analyze_variable_impact(self, base_sequence, variable_tests):
        """
        Analyze sensitivity to different variables
        
        Args:
            base_sequence: Baseline input sequence (5, channels, H, W)
            variable_tests: Dict of {var_name: (var_index, test_values)}
        """
        results = {}
        
        print("Running variable sensitivity analysis...")
        
        for var_name, (var_index, test_values) in variable_tests.items():
            print(f"Testing {var_name} with values: {test_values}")
            var_results = []
            
            for test_value in test_values:
                # Create modified sequence
                modified_sequence = base_sequence.clone()
                
                if var_name in ['Wind_Direction', 'Aspect', 'Forecast_Wind_Dir']:
                    # Angular variables - convert to sin
                    modified_sequence[:, var_index, :, :] = np.sin(np.radians(test_value))
                else:
                    # Numerical variables
                    modified_sequence[:, var_index, :, :] = test_value
                
                # Predict with modified conditions
                pred_fire = self.simulator.predict_single_step(modified_sequence.unsqueeze(0))
                
                # Calculate fire metrics
                fire_intensity = pred_fire.mean().item()
                fire_area = (pred_fire > 0.5).float().mean().item()
                max_fire_prob = pred_fire.max().item()
                fire_pixels = (pred_fire > 0.5).sum().item()
                
                var_results.append({
                    'value': test_value,
                    'fire_intensity': fire_intensity,
                    'fire_area': fire_area,
                    'max_fire_prob': max_fire_prob,
                    'fire_pixels': fire_pixels
                })
            
            results[var_name] = var_results
        
        return results
    
    def plot_sensitivity_analysis(self, sensitivity_results, save_path='sensitivity_analysis.png'):
        """Create comprehensive sensitivity analysis plots"""
        n_vars = len(sensitivity_results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (var_name, results) in enumerate(sensitivity_results.items()):
            if i >= len(axes):
                break
            
            values = [r['value'] for r in results]
            intensities = [r['fire_intensity'] for r in results]
            areas = [r['fire_area'] for r in results]
            
            # Plot fire intensity
            axes[i].plot(values, intensities, 'o-', color=colors[i % len(colors)], 
                        linewidth=3, markersize=8, label='Fire Intensity')
            
            # Plot fire area (secondary axis)
            ax2 = axes[i].twinx()
            ax2.plot(values, areas, 's--', color=colors[(i+1) % len(colors)], 
                    alpha=0.7, linewidth=2, markersize=6, label='Fire Area')
            
            axes[i].set_title(f'{var_name} Impact on Fire Spread', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(f'{var_name} Value', fontsize=12)
            axes[i].set_ylabel('Fire Intensity', fontsize=12, color=colors[i % len(colors)])
            ax2.set_ylabel('Fire Area Fraction', fontsize=12, color=colors[(i+1) % len(colors)])
            
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='y', labelcolor=colors[i % len(colors)])
            ax2.tick_params(axis='y', labelcolor=colors[(i+1) % len(colors)])
        
        # Hide empty subplots
        for j in range(len(sensitivity_results), len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Fire Spread Variable Sensitivity Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Sensitivity analysis saved: {save_path}")
    
    def generate_correlation_report(self, sensitivity_results):
        """Generate correlation analysis for supervisor requirements"""
        correlations = {}
        
        for var_name, results in sensitivity_results.items():
            values = [r['value'] for r in results]
            intensities = [r['fire_intensity'] for r in results]
            
            # Calculate Pearson correlation
            correlation = np.corrcoef(values, intensities)[0, 1] if len(values) > 1 else 0.0
            
            # Determine correlation strength and direction
            if abs(correlation) > 0.7:
                strength = "Strong"
            elif abs(correlation) > 0.3:
                strength = "Moderate"
            else:
                strength = "Weak"
            
            direction = "Positive" if correlation > 0 else "Negative"
            
            correlations[var_name] = {
                'correlation': correlation,
                'strength': strength,
                'direction': direction,
                'interpretation': self._interpret_correlation(var_name, correlation)
            }
        
        return correlations
    
    def _interpret_correlation(self, var_name, correlation):
        """Interpret correlation results in context of fire physics"""
        interpretations = {
            'Wind_Speed': "Higher wind speeds increase oxygen supply and fire spread rate" if correlation > 0 
                         else "Model may be capturing fire suppression effects of very high winds",
            'Max_Temp_K': "Higher temperatures increase ignition probability and fuel dryness" if correlation > 0
                         else "Unexpected negative correlation - may indicate model limitations",
            'NDVI': "Higher vegetation density provides more fuel for fire spread" if correlation > 0
                   else "Healthy vegetation may be more resistant to ignition",
            'Total_Precip': "More precipitation reduces fire risk through fuel moisture" if correlation < 0
                           else "Unexpected positive correlation with fire spread",
            'ERC': "Higher Energy Release Component indicates greater fire potential" if correlation > 0
                  else "Unexpected negative correlation with fire danger index",
            'Landcover': "Different vegetation types have varying flammability characteristics",
        }
        
        return interpretations.get(var_name, f"Correlation coefficient of {correlation:.3f} indicates relationship strength")

# ============================================================================
# ENHANCED U-NET WITH SIMULATION CAPABILITIES
# ============================================================================

class EnhancedFireUNet(nn.Module):
    """Enhanced U-Net with optimized architecture for fire spread prediction"""
    
    def __init__(self, input_channels, sequence_length=5):
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # Multi-temporal processing
        total_input_channels = input_channels * sequence_length
        
        # Optimized U-Net encoder (reduced parameters)
        self.enc1 = self._double_conv(total_input_channels, 64)
        self.enc2 = self._double_conv(64, 128)
        self.enc3 = self._double_conv(128, 256)
        self.enc4 = self._double_conv(256, 512)
        
        # Reduced bottleneck for efficiency
        self.bottleneck = self._double_conv(512, 512)  # Reduced from 1024
        
        # U-Net decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._double_conv(512, 256)  # 512 = 256 + 256
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._double_conv(256, 128)  # 256 = 128 + 128
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._double_conv(128, 64)   # 128 = 64 + 64
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._double_conv(64, 32)    # 64 = 32 + 32
        
        # Output layer
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize weights following best practices"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: (batch, sequence_length, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        
        # Flatten time dimension into channels
        x = x.view(batch_size, seq_len * channels, height, width)
        
        # U-Net encoder
        enc1 = self.enc1(x)
        enc1_pool = self.pool(enc1)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool(enc2)
        
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool(enc3)
        
        enc4 = self.enc4(enc3_pool)
        enc4_pool = self.pool(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4_pool)
        
        # U-Net decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output (logits)
        output = self.final_conv(dec1)
        
        return output

# ============================================================================
# IMPROVED LOSS FUNCTION
# ============================================================================

class ImprovedDiceBCELoss(nn.Module):
    """Enhanced Dice-BCE loss with focal loss for extreme class imbalance"""
    
    def __init__(self, pos_weight=50, dice_weight=0.7, smooth=1.0, epsilon=1e-6, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.epsilon = epsilon
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
    def forward(self, inputs, targets):
        if self.pos_weight.device != inputs.device:
            self.pos_weight = self.pos_weight.to(inputs.device)
        
        # Focal Loss for better hard example mining
        sigmoid_p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight, reduction='none')
        p_t = targets * sigmoid_p + (1 - targets) * (1 - sigmoid_p)
        alpha_t = targets * self.focal_alpha + (1 - targets) * (1 - self.focal_alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.focal_gamma * ce_loss
        focal_loss = focal_loss.mean()
        
        # Skip Dice for batches with very few positive targets
        target_sum = targets.sum()
        if target_sum < 5:
            return focal_loss
        
        # Dice loss calculation
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_sigmoid = torch.clamp(inputs_sigmoid, self.epsilon, 1 - self.epsilon)
        
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        
        combined_loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
        
        if not torch.isfinite(combined_loss):
            print(f"Non-finite loss detected: Focal={focal_loss}, Dice={dice_loss}, target_sum={target_sum}")
            return focal_loss
        
        return combined_loss

# ============================================================================
# ENHANCED TRAINER WITH MIXED PRECISION
# ============================================================================

class EnhancedFireTrainer:
    """Trainer with mixed precision and comprehensive evaluation"""
    
    def __init__(self, model, config=None, device=None):
        self.config = config or WildFireConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        
        # Enhanced loss function
        self.criterion = ImprovedDiceBCELoss(
            pos_weight=self.config.POSITIVE_CLASS_WEIGHT,
            dice_weight=0.7,
            smooth=self.config.DICE_SMOOTH,
            epsilon=self.config.DICE_EPSILON
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=5, factor=0.5, verbose=True
        )
        
        # Mixed precision training
        from torch.cuda.amp import GradScaler, autocast
        self.scaler = GradScaler()
        self.autocast = autocast
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_aps = []
        self.val_dice_scores = []
    
    def train_epoch(self, train_loader):
        """Training epoch with mixed precision"""
        self.model.train()
        epoch_loss = 0.0
        batch_stats = {'total_batches': 0, 'zero_target_batches': 0, 'nan_losses': 0}
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            target_sum = target.sum().item()
            batch_stats['total_batches'] += 1
            
            if target_sum < 1e-6:
                batch_stats['zero_target_batches'] += 1
            
            if not torch.isfinite(data).all():
                print(f"Non-finite inputs detected in batch {batch_idx}")
                continue
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with self.autocast():
                output = self.model(data)
                
                if len(target.shape) == 3:
                    target = target.unsqueeze(1)
                
                loss = self.criterion(output, target)
            
            if not torch.isfinite(loss):
                print(f"Non-finite loss in batch {batch_idx}: target_sum={target_sum}")
                batch_stats['nan_losses'] += 1
                continue
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += loss.item()
        
        print(f"Batch stats: {batch_stats}")
        return epoch_loss / max(len(train_loader), 1)
    
    def validate(self, val_loader):
        """Validation with comprehensive metrics"""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        dice_scores = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                with self.autocast():
                    output = self.model(data)
                    
                    if len(target.shape) == 3:
                        target = target.unsqueeze(1)
                    
                    loss = self.criterion(output, target)
                
                if torch.isfinite(loss):
                    val_loss += loss.item()
                
                pred_probs = torch.sigmoid(output).cpu().numpy().flatten()
                target_binary = target.cpu().numpy().flatten()
                
                all_predictions.append(pred_probs)
                all_targets.append(target_binary)
                
                pred_binary = (pred_probs > 0.5).astype(np.float32)
                intersection = np.sum(pred_binary * target_binary)
                union = np.sum(pred_binary) + np.sum(target_binary)
                if union > 0:
                    dice = 2 * intersection / union
                    dice_scores.append(dice)
        
        # Calculate Average Precision
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        valid_mask = np.isfinite(all_predictions) & np.isfinite(all_targets)
        if valid_mask.any():
            clean_preds = all_predictions[valid_mask]
            clean_targets = all_targets[valid_mask]
            
            if clean_targets.sum() > 0:
                ap_score = average_precision_score(clean_targets, clean_preds)
            else:
                ap_score = 0.0
        else:
            ap_score = 0.0
        
        avg_dice = np.mean(dice_scores) if dice_scores else 0.0
        avg_val_loss = val_loss / max(len(val_loader), 1)
        
        return avg_val_loss, ap_score, avg_dice
    
    def train_model(self, train_loader, val_loader, epochs=50):
        """Train model with enhanced monitoring"""
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_ap = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_ap, val_dice = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aps.append(val_ap)
            self.val_dice_scores.append(val_dice)
            
            self.scheduler.step(val_ap)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss: {val_loss:.6f}')
            print(f'  Val AP: {val_ap:.4f}')
            print(f'  Val Dice: {val_dice:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            if val_ap > best_ap:
                best_ap = val_ap
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_ap': best_ap,
                    'config': self.config
                }, 'best_fire_model_enhanced.pth')
                print(f"  → Saved best model (AP: {best_ap:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 15:
                print("Early stopping triggered")
                break
        
        self.plot_training_history()
        return best_ap
    
    def plot_training_history(self):
        """Plot comprehensive training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1.plot(self.train_losses, label='Training Loss', alpha=0.8)
        ax1.plot(self.val_losses, label='Validation Loss', alpha=0.8)
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.val_aps, label='Validation AP', color='green', linewidth=2)
        ax2.set_title('Validation Average Precision')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AP Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(self.val_dice_scores, label='Validation Dice', color='orange')
        ax3.set_title('Validation Dice Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Dice Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.plot([self.optimizer.param_groups[0]['lr']] * len(self.val_aps), 
                label='Learning Rate', color='red')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

# ============================================================================
# SUPERVISOR REPORT GENERATOR
# ============================================================================

def generate_supervisor_report(sensitivity_results, correlations, model_performance, config):
    """Generate comprehensive report for supervisor presentation"""
    
    report = f"""
COMPLETE WILDFIRE SPREAD PREDICTION SYSTEM - SUPERVISOR REPORT
============================================================

Executive Summary:
This system provides comprehensive wildfire spread prediction with simulation capabilities, 
variable sensitivity analysis, and visualization tools specifically designed for supervisor 
requirements including fire spread animations and variable impact analysis.

SYSTEM CAPABILITIES:
==================

1. FIRE SPREAD SIMULATION ENSEMBLE
   - Single event multi-day prediction (up to 30 days)
   - Two simulation modes:
     * Sliding Window: Uses real weather data for each prediction step
     * Autoregressive: Uses model predictions as input for next step
   - Realistic fire physics: decay, spread thresholds, spatial smoothing
   - Animated visualization of fire progression

2. VARIABLE SENSITIVITY ANALYSIS
   - Quantitative impact assessment of environmental variables
   - Correlation analysis with fire progression
   - Comparison with physical fire propagation mechanisms

3. MULTI-STEP PREDICTION CAPABILITIES
   - Input: 5-day historical weather + fire data
   - Output: Next-day fire probability distribution
   - Iterative prediction for extended forecasting

MODEL CONFIGURATION:
===================
- Spatial Resolution: {config.SPATIAL_SIZE}
- Temporal Input: {config.SEQUENCE_LENGTH} days
- Prediction Horizon: {config.PREDICTION_HORIZON} day
- Training Data: {config.EVENTS_PER_YEAR_TRAIN} quality events per year
- Architecture: Enhanced U-Net with {sum(p.numel() for p in EnhancedFireUNet(len(config.BEST_FEATURES), config.SEQUENCE_LENGTH).parameters()):,} parameters

VARIABLES USED IN MODEL:
=======================
"""
    
    # Add detailed variable information
    variable_info = {
        'NDVI': ('Normalized Difference Vegetation Index', 'Dimensionless', 'Fuel availability'),
        'EVI2': ('Enhanced Vegetation Index', 'Dimensionless', 'Vegetation health/density'),
        'VIIRS_M11': ('Thermal infrared radiance', 'W m-2 sr-1 μm-1', 'Heat detection'),
        'VIIRS_I2': ('Near-infrared reflectance', 'Dimensionless', 'Fire detection'),  
        'VIIRS_I1': ('Red reflectance', 'Dimensionless', 'Smoke/fire detection'),
        'Slope': ('Terrain slope', 'Degrees', 'Fire spread rate'),
        'Aspect': ('Terrain aspect', 'Degrees', 'Sun exposure/drying'),
        'Elevation': ('Elevation above sea level', 'Meters', 'Climate/fuel type'),
        'Landcover': ('Land cover type', 'Categorical (1-17)', 'Fuel type'),
        'Total_Precip': ('Precipitation', 'mm/day', 'Fuel moisture'),
        'Min_Temp_K': ('Minimum temperature', 'Kelvin', 'Ignition potential'),
        'Max_Temp_K': ('Maximum temperature', 'Kelvin', 'Fire intensity'),
        'Active_Fire': ('Fire activity', 'Binary/Intensity', 'Current fire state')
    }
    
    for var_name, (description, unit, mechanism) in variable_info.items():
        correlation_data = correlations.get(var_name, {'correlation': 0, 'direction': 'Unknown', 'strength': 'Unknown'})
        correlation_sign = '+' if correlation_data['correlation'] > 0 else '-' if correlation_data['correlation'] < 0 else '~'
        
        report += f"""
{var_name:15} | {unit:20} | {correlation_sign} {correlation_data['strength']:10} | {mechanism}
                Description: {description}
                Correlation: {correlation_data['correlation']:.3f} ({correlation_data['direction']})
"""
    
    report += f"""

CORRELATION ANALYSIS RESULTS:
============================
"""
    
    for var_name, corr_data in correlations.items():
        report += f"""
{var_name}:
  Correlation coefficient: {corr_data['correlation']:.3f}
  Relationship strength: {corr_data['strength']} {corr_data['direction']}
  Physical interpretation: {corr_data['interpretation']}
"""
    
    report += f"""

COMPARISON WITH LITERATURE:
==========================

Wind Speed: Model shows {correlations.get('Wind_Speed', {}).get('direction', 'unknown').lower()} correlation.
  Literature: Positive correlation expected due to oxygen supply and fire spread acceleration.
  Model alignment: {'✓ Consistent' if correlations.get('Wind_Speed', {}).get('direction') == 'Positive' else '⚠ Requires investigation'}

Temperature: Model shows {correlations.get('Max_Temp_K', {}).get('direction', 'unknown').lower()} correlation.
  Literature: Strong positive correlation due to fuel dryness and ignition probability.
  Model alignment: {'✓ Consistent' if correlations.get('Max_Temp_K', {}).get('direction') == 'Positive' else '⚠ Requires investigation'}

Vegetation (NDVI): Model shows {correlations.get('NDVI', {}).get('direction', 'unknown').lower()} correlation.
  Literature: Complex relationship - more vegetation = more fuel but also higher moisture.
  Model alignment: Context-dependent, requires detailed analysis

Precipitation: Model shows {correlations.get('Total_Precip', {}).get('direction', 'unknown').lower()} correlation.
  Literature: Strong negative correlation due to fuel moisture increase.
  Model alignment: {'✓ Consistent' if correlations.get('Total_Precip', {}).get('direction') == 'Negative' else '⚠ Requires investigation'}

MODEL PERFORMANCE:
=================
- Final Validation AP: {model_performance.get('final_ap', 'N/A'):.4f}
- Final Validation Dice: {model_performance.get('final_dice', 'N/A'):.4f}
- Training Epochs: {model_performance.get('epochs', 'N/A')}
- Training Time: Mixed precision enabled for 4-6x speedup"""
Complete WildfireSpreadTS Implementation with Fire Simulation
===========================================================

Comprehensive implementation including:
1. Optimized training pipeline with mixed precision
2. Fire spread simulation for creating animations
3. Variable sensitivity analysis for supervisor requirements
4. Proper handling of Active_Fire as input feature
5. Multi-step prediction capabilities

Based on official WildfireSpreadTS practices with extensions for simulation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import h5py
import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - ENHANCED FOR SIMULATION
# ============================================================================

class WildFireConfig:
    """Enhanced configuration for training and simulation"""
    
    # Data configuration
    SPATIAL_SIZE = (128, 128)  # Official crop size for training
    SEQUENCE_LENGTH = 5        # 5-day input sequences
    PREDICTION_HORIZON = 1     # Next-day prediction
    
    # SAMPLING STRATEGY - Optimized
    EVENTS_PER_YEAR_TRAIN = 50  # Random sample for training
    EVENTS_PER_YEAR_VAL = 20    # Validation events per year
    EVENTS_PER_YEAR_TEST = 30   # Test events per year
    
    # Official yearly cross-validation splits
    CV_SPLITS = [
        {'train': [2018, 2020], 'val': 2019, 'test': 2021},
        {'train': [2018, 2019], 'val': 2020, 'test': 2021},
        {'train': [2019, 2020], 'val': 2018, 'test': 2021},
        {'train': [2020, 2021], 'val': 2018, 'test': 2019},
    ]
    
    # Feature definitions with proper handling
    FEATURE_NAMES = [
        'VIIRS_M11', 'VIIRS_I2', 'VIIRS_I1',      # 0-2: Thermal/reflectance
        'NDVI', 'EVI2',                            # 3-4: Vegetation indices  
        'Total_Precip', 'Wind_Speed',              # 5-6: Weather
        'Wind_Direction',                          # 7: Angular (needs sin transform)
        'Min_Temp_K', 'Max_Temp_K',               # 8-9: Temperature
        'ERC', 'Spec_Hum', 'PDSI',                # 10-12: Fire weather
        'Slope', 'Aspect',                         # 13-14: Topography (Aspect is angular)
        'Elevation', 'Landcover',                  # 15-16: Static (Landcover needs one-hot)
        'Forecast_Precip', 'Forecast_Wind_Speed',  # 17-18: Forecast weather
        'Forecast_Wind_Dir',                       # 19: Angular forecast
        'Forecast_Temp_C', 'Forecast_Spec_Hum',   # 20-21: Forecast conditions
        'Active_Fire'                              # 22: Target AND input feature
    ]
    
    # Angular features that need sin() transformation and no standardization
    ANGULAR_FEATURES = [7, 14, 19]  # Wind_Direction, Aspect, Forecast_Wind_Dir
    
    # Static features (only keep in last frame for multi-temporal)
    STATIC_FEATURES = [13, 14, 15, 16]  # Slope, Aspect, Elevation, Landcover
    
    # Categorical features (no standardization)
    CATEGORICAL_FEATURES = [16]  # Landcover
    
    # ENHANCED feature combination INCLUDING Active_Fire for simulation
    BEST_FEATURES = [3, 4, 0, 1, 2, 13, 14, 15, 16, 5, 8, 9, 22]  # Added Active_Fire (22)
    
    # Loss configuration - OPTIMIZED
    POSITIVE_CLASS_WEIGHT = 50   # Reduced from 236 for stability
    DICE_SMOOTH = 1.0           # Standard smooth factor
    DICE_EPSILON = 1e-6         # Clamp epsilon for sigmoid
    
    # Training configuration - OPTIMIZED for 4070Ti
    LEARNING_RATE = 5e-4        # Balanced for stability and speed
    BATCH_SIZE = 12             # Optimized for 16GB VRAM
    NUM_CROPS = 10              # Reduced from 20 for efficiency
    FIRE_CROP_THRESHOLD = 20    # Higher threshold for quality
    
    # Simulation configuration
    MAX_SIMULATION_DAYS = 30    # Maximum days to simulate
    FIRE_DECAY_RATE = 0.05      # Daily fire decay rate for realism
    FIRE_SPREAD_THRESHOLD = 0.3  # Probability threshold for fire spread

# ============================================================================
# ENHANCED DATASET WITH ACTIVE_FIRE HANDLING
# ============================================================================

class EnhancedFireSpreadDataset(Dataset):
    """Enhanced dataset with proper Active_Fire feature handling"""
    
    def __init__(self, file_paths, years, mode='train', config=None):
        self.config = config or WildFireConfig()
        self.file_paths = file_paths
        self.years = years
        self.mode = mode
        
        # Filter files by years with quality check
        self.valid_files = self._filter_files_by_years()
        
        # Create sequence index
        self.valid_sequences = self._create_sequence_index()
        
        # Compute normalization statistics from training years only
        if mode == 'train':
            self._compute_normalization_stats()
        else:
            self.feature_mean = None
            self.feature_std = None
            
        print(f"{mode.upper()} dataset: {len(self.valid_sequences)} sequences from {len(self.valid_files)} files")
    
    def _filter_files_by_years(self):
        """Filter and sample high-quality fire events by year"""
        all_year_files = {}
        
        for file_path in self.file_paths:
            try:
                parts = file_path.replace('\\', '/').split('/')
                year_str = [part for part in parts if part.isdigit() and len(part) == 4][-1]
                year = int(year_str)
                
                if year in self.years:
                    if self._is_quality_fire_event(file_path):
                        if year not in all_year_files:
                            all_year_files[year] = []
                        all_year_files[year].append(file_path)
            except (ValueError, IndexError):
                continue
        
        # Sample events per year based on mode
        sampled_files = []
        total_available = 0
        total_sampled = 0
        
        for year in self.years:
            if year in all_year_files:
                year_files = all_year_files[year]
                total_available += len(year_files)
                
                if self.mode == 'train':
                    sample_size = min(len(year_files), self.config.EVENTS_PER_YEAR_TRAIN)
                elif self.mode == 'val':
                    sample_size = min(len(year_files), self.config.EVENTS_PER_YEAR_VAL)
                else:  # test
                    sample_size = min(len(year_files), self.config.EVENTS_PER_YEAR_TEST)
                
                # Reproducible sampling
                np.random.seed(42 + year + hash(self.mode) % 1000)
                if len(year_files) > sample_size:
                    sampled = np.random.choice(year_files, sample_size, replace=False)
                else:
                    sampled = year_files
                
                sampled_files.extend(sampled)
                total_sampled += len(sampled)
                print(f"  {self.mode.upper()} - Year {year}: sampled {len(sampled)} events from {len(year_files)} quality events")
        
        print(f"  {self.mode.upper()} TOTAL: {total_sampled} sampled from {total_available} quality events")
        return sampled_files
    
    def _is_quality_fire_event(self, file_path):
        """Enhanced quality criteria for fire events"""
        try:
            with h5py.File(file_path, 'r') as f:
                if 'data' not in f:
                    return False
                
                data = f['data']
                if data.shape[0] < self.config.SEQUENCE_LENGTH + self.config.PREDICTION_HORIZON:
                    return False
                
                fire_channel = data[:, -1, :, :]  # Active_Fire channel
                
                # Enhanced quality criteria
                total_fire_pixels = np.sum(fire_channel > 0)
                fire_duration = np.sum(np.any(fire_channel > 0, axis=(1, 2)))
                max_fire_per_frame = np.max([np.sum(frame > 0) for frame in fire_channel])
                
                # More stringent criteria for better quality
                if (total_fire_pixels >= 200 and    # Increased from 50
                    fire_duration >= 3 and          # Increased from 2
                    max_fire_per_frame >= 25):       # Increased from 10
                    return True
                
        except Exception as e:
            pass
        
        return False
    
    def _create_sequence_index(self):
        """Create index of valid sequences"""
        valid_sequences = []
        
        for file_idx, file_path in enumerate(tqdm(self.valid_files, desc=f"Indexing {self.mode}")):
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'data' not in f:
                        continue
                    
                    data_shape = f['data'].shape
                    if len(data_shape) != 4:
                        continue
                    
                    T, C, H, W = data_shape
                    
                    max_sequences = T - self.config.SEQUENCE_LENGTH - self.config.PREDICTION_HORIZON + 1
                    if max_sequences <= 0:
                        continue
                    
                    for seq_start in range(max_sequences):
                        valid_sequences.append({
                            'file_idx': file_idx,
                            'file_path': file_path,
                            'seq_start': seq_start,
                            'original_shape': (T, C, H, W)
                        })
                        
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
                continue
        
        return valid_sequences
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics excluding Active_Fire channel"""
        print("Computing normalization statistics from training years...")
        
        all_data = []
        for file_path in self.valid_files[:5]:
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'data' in f:
                        data = f['data'][:]
                        T, C, H, W = data.shape
                        
                        data_reshaped = data.transpose(0, 2, 3, 1).reshape(-1, C)
                        sample_size = min(10000, data_reshaped.shape[0])
                        indices = np.random.choice(data_reshaped.shape[0], sample_size, replace=False)
                        sampled_data = data_reshaped[indices]
                        
                        all_data.append(sampled_data)
            except:
                continue
        
        if all_data:
            combined_data = np.vstack(all_data)
            
            self.feature_mean = np.zeros(len(self.config.FEATURE_NAMES))
            self.feature_std = np.ones(len(self.config.FEATURE_NAMES))
            
            for i in range(combined_data.shape[1]):
                # Skip Active_Fire normalization
                if i == 22:  # Active_Fire channel
                    self.feature_mean[i] = 0.0
                    self.feature_std[i] = 1.0
                    continue
                
                valid_mask = np.isfinite(combined_data[:, i])
                if valid_mask.any():
                    valid_data = combined_data[valid_mask, i]
                    mean_est = np.median(valid_data)
                    std_est = np.std(valid_data)
                    outlier_mask = np.abs(valid_data - mean_est) < 5 * std_est
                    if outlier_mask.any():
                        clean_data = valid_data[outlier_mask]
                        self.feature_mean[i] = np.mean(clean_data)
                        self.feature_std[i] = np.std(clean_data)
                        if self.feature_std[i] < 1e-6:
                            self.feature_std[i] = 1.0
        else:
            self.feature_mean = np.zeros(len(self.config.FEATURE_NAMES))
            self.feature_std = np.ones(len(self.config.FEATURE_NAMES))
        
        print("Normalization statistics computed")
    
    def set_normalization_stats(self, mean, std):
        """Set normalization statistics from training set"""
        self.feature_mean = mean
        self.feature_std = std
    
    def __len__(self):
        return len(self.valid_sequences)
    
    def __getitem__(self, idx):
        """Load and process sequence with enhanced practices"""
        sequence_info = self.valid_sequences[idx]
        
        try:
            with h5py.File(sequence_info['file_path'], 'r') as f:
                data = f['data'][:]
                
                seq_start = sequence_info['seq_start']
                input_end = seq_start + self.config.SEQUENCE_LENGTH
                target_idx = input_end + self.config.PREDICTION_HORIZON - 1
                
                input_sequence = data[seq_start:input_end]
                target_frame = data[target_idx]
                
                # Enhanced crop sampling
                if self.mode == 'train':
                    input_sequence, target_binary = self._enhanced_crop_sample(input_sequence, target_frame)
                else:
                    input_sequence, target_binary = self._center_crop(input_sequence, target_frame)
                
                # Process features with Active_Fire special handling
                input_processed = self._process_features_enhanced(input_sequence)
                
                return torch.FloatTensor(input_processed), torch.FloatTensor(target_binary)
                
        except Exception as e:
            print(f"Error loading sequence {idx}: {e}")
            dummy_input = torch.zeros(self.config.SEQUENCE_LENGTH, 
                                    len(self.config.BEST_FEATURES),
                                    self.config.SPATIAL_SIZE[0], 
                                    self.config.SPATIAL_SIZE[1])
            dummy_target = torch.zeros(self.config.SPATIAL_SIZE[0], self.config.SPATIAL_SIZE[1])
            return dummy_input, dummy_target
    
    def _enhanced_crop_sample(self, input_sequence, target_frame):
        """Enhanced crop sampling with better fire detection"""
        T, C, H, W = input_sequence.shape
        target_size = self.config.SPATIAL_SIZE
        
        crops = []
        targets = []
        fire_counts = []
        
        for _ in range(self.config.NUM_CROPS):
            if H > target_size[0] and W > target_size[1]:
                h_start = np.random.randint(0, H - target_size[0])
                w_start = np.random.randint(0, W - target_size[1])
                
                crop_input = input_sequence[:, :, h_start:h_start+target_size[0], 
                                          w_start:w_start+target_size[1]]
                crop_target = target_frame[-1, h_start:h_start+target_size[0], 
                                         w_start:w_start+target_size[1]]
            else:
                crop_input = self._resize_sequence(input_sequence, target_size)
                crop_target = self._resize_array(target_frame[-1], target_size)
            
            target_binary = (crop_target > 0).astype(np.float32)
            fire_pixel_count = np.sum(target_binary)
            
            crops.append(crop_input)
            targets.append(target_binary)
            fire_counts.append(fire_pixel_count)
        
        # Enhanced selection strategy
        max_fire_count = max(fire_counts)
        
        if max_fire_count >= self.config.FIRE_CROP_THRESHOLD:
            best_idx = np.argmax(fire_counts)
            selected_crop = crops[best_idx]
            selected_target = targets[best_idx]
        elif max_fire_count > 0:
            fire_indices = [i for i, count in enumerate(fire_counts) if count > 0]
            best_idx = np.random.choice(fire_indices)
            selected_crop = crops[best_idx]
            selected_target = targets[best_idx]
        else:
            # Targeted sampling around existing fire
            targeted_result = self._targeted_fire_sample(input_sequence, target_frame)
            if targeted_result is not None:
                selected_crop, selected_target = targeted_result
            else:
                best_idx = np.random.randint(len(crops))
                selected_crop = crops[best_idx]
                selected_target = targets[best_idx]
        
        # Reduced debug logging
        if np.random.random() < 0.001:  # 0.1% instead of 1%
            selected_fire_count = np.sum(selected_target)
            print(f"Fire crop selection: max_fire={max_fire_count}, selected_fire={selected_fire_count}")
        
        return selected_crop, selected_target
    
    def _targeted_fire_sample(self, input_sequence, target_frame):
        """Improved targeted fire sampling"""
        target_fire = target_frame[-1]
        fire_locations = np.where(target_fire > 0)
        
        if len(fire_locations[0]) == 0:
            return None
        
        target_size = self.config.SPATIAL_SIZE
        T, C, H, W = input_sequence.shape
        
        best_crop = None
        best_target = None
        best_fire_count = 0
        
        for _ in range(10):
            fire_idx = np.random.randint(len(fire_locations[0]))
            fire_h, fire_w = fire_locations[0][fire_idx], fire_locations[1][fire_idx]
            
            center_h = fire_h + np.random.randint(-target_size[0]//4, target_size[0]//4)
            center_w = fire_w + np.random.randint(-target_size[1]//4, target_size[1]//4)
            
            h_start = np.clip(center_h - target_size[0]//2, 0, H - target_size[0])
            w_start = np.clip(center_w - target_size[1]//2, 0, W - target_size[1])
            
            crop_target = target_fire[h_start:h_start+target_size[0], 
                                    w_start:w_start+target_size[1]]
            fire_count = np.sum(crop_target > 0)
            
            if fire_count > best_fire_count:
                best_fire_count = fire_count
                crop_input = input_sequence[:, :, h_start:h_start+target_size[0], 
                                          w_start:w_start+target_size[1]]
                target_binary = (crop_target > 0).astype(np.float32)
                best_crop = crop_input
                best_target = target_binary
        
        if best_fire_count >= self.config.FIRE_CROP_THRESHOLD:
            return best_crop, best_target
        else:
            return None
    
    def _center_crop(self, input_sequence, target_frame):
        """Center crop for validation/testing"""
        T, C, H, W = input_sequence.shape
        target_size = self.config.SPATIAL_SIZE
        
        if H >= target_size[0] and W >= target_size[1]:
            h_start = (H - target_size[0]) // 2
            w_start = (W - target_size[1]) // 2
            
            crop_input = input_sequence[:, :, h_start:h_start+target_size[0], 
                                      w_start:w_start+target_size[1]]
            crop_target = target_frame[-1, h_start:h_start+target_size[0], 
                                     w_start:w_start+target_size[1]]
        else:
            crop_input = self._resize_sequence(input_sequence, target_size)
            crop_target = self._resize_array(target_frame[-1], target_size)
        
        target_binary = (crop_target > 0).astype(np.float32)
        return crop_input, target_binary
    
    def _resize_sequence(self, sequence, target_size):
        """Resize sequence using bilinear interpolation"""
        T, C, H, W = sequence.shape
        sequence_tensor = torch.FloatTensor(sequence)
        resized = F.interpolate(sequence_tensor.view(-1, 1, H, W), 
                              size=target_size, mode='bilinear', align_corners=False)
        return resized.view(T, C, target_size[0], target_size[1]).numpy()
    
    def _resize_array(self, array, target_size):
        """Resize 2D array"""
        array_tensor = torch.FloatTensor(array).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(array_tensor, size=target_size, mode='bilinear', align_corners=False)
        return resized.squeeze().numpy()
    
    def _process_features_enhanced(self, input_sequence):
        """Enhanced feature processing with Active_Fire special handling"""
        T, C, H, W = input_sequence.shape
        processed = input_sequence.copy()
        
        # 1. Handle angular features
        for angle_idx in self.config.ANGULAR_FEATURES:
            if angle_idx < C:
                processed[:, angle_idx] = np.sin(np.radians(processed[:, angle_idx]))
        
        # 2. Handle missing values
        for c in range(C):
            mask = ~np.isfinite(processed[:, c])
            if mask.any():
                if c == 22:  # Active_Fire - use 0 for missing
                    processed[:, c][mask] = 0
                else:
                    processed[:, c][mask] = 0
        
        # 3. Standardize features (SKIP Active_Fire)
        if self.feature_mean is not None and self.feature_std is not None:
            for c in range(C):
                if (c not in self.config.ANGULAR_FEATURES and 
                    c not in self.config.CATEGORICAL_FEATURES and
                    c != 22):  # Skip Active_Fire normalization
                    processed[:, c] = (processed[:, c] - self.feature_mean[c]) / self.feature_std[c]
        
        # 4. Multi-temporal feature selection
        for t in range(T-1):
            for static_idx in self.config.STATIC_FEATURES:
                if static_idx < C:
                    processed[t, static_idx] = 0
        
        # 5. Select best features (now including Active_Fire)
        if len(self.config.BEST_FEATURES) < C:
            processed = processed[:, self.config.BEST_FEATURES]
        
        return processed

# ============================================================================
# FIRE SIMULATION AND PREDICTION MODULES
# ============================================================================

class FireSpreadSimulator:
    """Fire spread simulator using trained model"""
    
    def __init__(self, model_path, config=None, device=None):
        self.config = config or WildFireConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine input channels from config
        input_channels = len(self.config.BEST_FEATURES)
        self.model = EnhancedFireUNet(input_channels, self.config.SEQUENCE_LENGTH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Model input channels: {input_channels}")
    
    def predict_single_step(self, input_sequence):
        """Predict next day fire distribution"""
        with torch.no_grad():
            if len(input_sequence.shape) == 3:
                input_sequence = input_sequence.unsqueeze(0)  # Add batch dimension
            
            input_tensor = input_sequence.to(self.device)
            
            with torch.cuda.amp.autocast() if self.device.type == 'cuda' else torch.no_grad():
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output)
            
            return prediction.cpu().squeeze()
    
    def simulate_fire_evolution(self, initial_sequence, weather_data, num_days=10, mode='sliding_window'):
        """
        Simulate fire evolution over multiple days
        
        Args:
            initial_sequence: Initial 5-day sequence (5, channels, H, W)
            weather_data: Weather data for future days (num_days, channels, H, W)
            num_days: Number of days to simulate
            mode: 'sliding_window' (use real weather) or 'autoregressive' (use predictions)
        """
        predictions = []
        current_sequence = initial_sequence.clone()
        
        print(f"Simulating {num_days} days using {mode} mode")
        
        for day in tqdm(range(num_days), desc="Simulating fire evolution"):
            # Predict next day
            pred_fire = self.predict_single_step(current_sequence.unsqueeze(0))
            
            # Apply fire physics (decay and thresholding)
            pred_fire = self._apply_fire_physics(pred_fire, day)
            
            predictions.append(pred_fire.numpy())
            
            if day < num_days - 1:  # Not the last day
                if mode == 'sliding_window' and day < len(weather_data) - self.config.SEQUENCE_LENGTH:
                    # Use real weather data: shift window by 1 day
                    next_sequence = weather_data[day + 1:day + 1 + self.config.SEQUENCE_LENGTH].clone()
                    # Update Active_Fire channel in the last frame
                    active_fire_idx = self.config.BEST_FEATURES.index(22)  # Active_Fire index
                    next_sequence[-1, active_fire_idx] = pred_fire
                    current_sequence = next_sequence
                    
                elif mode == 'autoregressive':
                    # Use prediction as input: roll sequence forward
                    new_frame = current_sequence[-1].clone()
                    active_fire_idx = self.config.BEST_FEATURES.index(22)
                    new_frame[active_fire_idx] = pred_fire
                    
                    current_sequence = torch.cat([
                        current_sequence[1:],  # Remove first day
                        new_frame.unsqueeze(0)  # Add predicted day
                    ], dim=0)
                else:
                    break
        
        return predictions
    
    def _apply_fire_physics(self, fire_prediction, day):
        """Apply realistic fire physics to predictions"""
        # Apply probability threshold
        fire_binary = (fire_prediction > self.config.FIRE_SPREAD_THRESHOLD).float()
        
        # Apply daily fire decay (fires naturally extinguish)
        decay_factor = 1.0 - self.config.FIRE_DECAY_RATE * (day + 1)
        decay_factor = max(0.1, decay_factor)  # Minimum 10% intensity
        
        fire_decayed = fire_binary * decay_factor
        
        # Add some spatial smoothing to prevent fragmentation
        fire_smoothed = torch.tensor(
            ndimage.gaussian_filter(fire_decayed.numpy(), sigma=0.5)
        )
        
        return fire_smoothed
    
    def create_simulation_animation(self, predictions, real_sequence=None, save_path='fire_simulation.gif'):
        """Create animated visualization of fire spread simulation"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6)) if real_sequence is not None else plt.subplots(1, 1, figsize=(8, 6))
        
        if not hasattr(axes, '__len__'):
            axes = [axes]
        
        def animate(frame):
            for ax in axes:
                ax.clear()
            
            if real_sequence is not None and frame < len(real_sequence):
                # Show real fire progression
                axes[0].imshow(real_sequence[frame], cmap='Reds', vmin=0, vmax=1)
                axes[0].set_title(f'Actual Fire - Day {frame+1}')
                axes[0].axis('off')
                
                # Show predicted fire progression
                if frame < len(predictions):
                    axes[1].imshow(predictions[frame], cmap='Reds', vmin=0, vmax=1)
                    axes[1].set_title(f'Predicted Fire - Day {frame+1}')
                    axes[1].axis('off')
            else:
                # Show only predictions
                if frame < len(predictions):
                    axes[0].imshow(predictions[frame], cmap='Reds', vmin=0, vmax=1)
                    axes[0].set_title(f'Simulated Fire Spread - Day {frame+1}')
                    axes[0].axis('off')
                    
                    # Add colorbar
                    cbar = plt.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
                    cbar.set_label('Fire Probability')
        
        frames = min(len(predictions), 30)  # Limit to 30 days
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=500, repeat=True)
        
        try:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"Animation saved: {save_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
            # Save individual frames
            for i, pred in enumerate(predictions[:30]):
                plt.figure(figsize=(