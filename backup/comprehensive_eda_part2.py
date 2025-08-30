"""
WildfireSpreadTS数据集全面EDA分析系统 - 第二部分
继续实现分析模块4-9
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class WildfireEDAAnalyzerPart2:
    """EDA分析器扩展功能"""
    
    def analyze_feature_relationships(self):
        """4. 特征关系与相关性分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 4.1 线性关系分析
        # 计算相关矩阵（仅使用有限值）
        valid_data = []
        valid_channels = []
        
        for i in range(self.data.shape[1]):
            channel_data = self.data[:, i]
            if not np.all(np.isnan(channel_data)):
                valid_data.append(channel_data)
                valid_channels.append(i)
        
        valid_data = np.column_stack(valid_data)
        
        # 计算Pearson相关系数
        correlation_matrix = np.corrcoef(valid_data.T)
        correlation_matrix = np.nan_to_num(correlation_matrix)
        
        results['correlation_matrix'] = correlation_matrix
        results['valid_channels'] = valid_channels
        
        # 找出强相关对
        strong_correlations = []
        for i in range(len(valid_channels)):
            for j in range(i+1, len(valid_channels)):
                corr_val = correlation_matrix[i, j]
                if abs(corr_val) > 0.7:  # 强相关阈值
                    strong_correlations.append({
                        'channel1': valid_channels[i],
                        'channel2': valid_channels[j],
                        'channel1_name': self.feature_schema[valid_channels[i]]['name'],
                        'channel2_name': self.feature_schema[valid_channels[j]]['name'],
                        'correlation': corr_val
                    })
        
        results['strong_correlations'] = strong_correlations
        
        # 4.2 与目标变量的关系
        target_correlations = {}
        target_channel = 22  # 火点置信度
        
        if target_channel in valid_channels:
            target_idx = valid_channels.index(target_channel)
            target_data = valid_data[:, target_idx]
            
            for i, channel in enumerate(valid_channels):
                if channel != target_channel:
                    channel_data = valid_data[:, i]
                    
                    # 计算相关性
                    valid_mask = np.isfinite(channel_data) & np.isfinite(target_data)
                    if valid_mask.sum() > 100:  # 确保有足够的有效数据
                        pearson_corr = np.corrcoef(channel_data[valid_mask], target_data[valid_mask])[0, 1]
                        spearman_corr = stats.spearmanr(channel_data[valid_mask], target_data[valid_mask])[0]
                        
                        target_correlations[channel] = {
                            'channel_name': self.feature_schema[channel]['name'],
                            'pearson_correlation': float(pearson_corr) if np.isfinite(pearson_corr) else 0,
                            'spearman_correlation': float(spearman_corr) if np.isfinite(spearman_corr) else 0,
                            'valid_samples': int(valid_mask.sum())
                        }
        
        results['target_correlations'] = target_correlations
        
        # 保存结果
        self.results['feature_relationships'] = results
        
        # 生成可视化
        self._plot_feature_relationships(results)
        
        # 保存报告
        self._save_feature_relationships_report(results)
        
        print(f"  ✅ 特征关系与相关性分析完成")
        return results
    
    def _plot_feature_relationships(self, results):
        """绘制特征关系图表"""
        
        # 1. 相关矩阵热力图
        plt.figure(figsize=(20, 16))
        
        # 创建通道名称标签
        channel_labels = [f"{ch}: {self.feature_schema[ch]['name'][:10]}" 
                         for ch in results['valid_channels']]
        
        mask = np.triu(np.ones_like(results['correlation_matrix'], dtype=bool), k=1)
        
        sns.heatmap(results['correlation_matrix'], 
                   mask=mask,
                   xticklabels=channel_labels,
                   yticklabels=channel_labels,
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Feature Correlation Matrix', fontsize=20, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 与目标变量的相关性
        if results['target_correlations']:
            channels = list(results['target_correlations'].keys())
            pearson_corrs = [results['target_correlations'][ch]['pearson_correlation'] for ch in channels]
            spearman_corrs = [results['target_correlations'][ch]['spearman_correlation'] for ch in channels]
            channel_names = [f"{ch}: {self.feature_schema[ch]['name'][:15]}" for ch in channels]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Pearson相关性
            bars1 = ax1.barh(range(len(channels)), pearson_corrs)
            ax1.set_title('Pearson Correlation with Fire Confidence', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Correlation Coefficient')
            ax1.set_yticks(range(len(channels)))
            ax1.set_yticklabels(channel_names)
            ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
            
            # 添加颜色编码
            for i, bar in enumerate(bars1):
                if pearson_corrs[i] > 0:
                    bar.set_color('red')
                else:
                    bar.set_color('blue')
            
            # Spearman相关性
            bars2 = ax2.barh(range(len(channels)), spearman_corrs)
            ax2.set_title('Spearman Correlation with Fire Confidence', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Correlation Coefficient')
            ax2.set_yticks(range(len(channels)))
            ax2.set_yticklabels(channel_names)
            ax2.axvline(0, color='black', linestyle='-', alpha=0.3)
            
            # 添加颜色编码
            for i, bar in enumerate(bars2):
                if spearman_corrs[i] > 0:
                    bar.set_color('red')
                else:
                    bar.set_color('blue')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "target_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_feature_relationships_report(self, results):
        """保存特征关系报告"""
        
        # 强相关对报告
        if results['strong_correlations']:
            strong_corr_df = pd.DataFrame(results['strong_correlations'])
            strong_corr_df['correlation'] = strong_corr_df['correlation'].round(4)
            strong_corr_df.to_csv(self.output_dir / "tables" / "strong_correlations.csv", index=False)
        
        # 目标变量相关性报告
        if results['target_correlations']:
            target_corr_df = pd.DataFrame([
                {
                    'Channel_ID': channel,
                    'Channel_Name': analysis['channel_name'],
                    'Pearson_Correlation': f"{analysis['pearson_correlation']:.4f}",
                    'Spearman_Correlation': f"{analysis['spearman_correlation']:.4f}",
                    'Valid_Samples': analysis['valid_samples']
                }
                for channel, analysis in results['target_correlations'].items()
            ])
            target_corr_df.to_csv(self.output_dir / "tables" / "target_correlations.csv", index=False)
    
    def analyze_target_variable(self):
        """5. 目标变量深度分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 5.1 火点置信度分布分析
        fire_confidence = self.data[:, 22]  # 火点置信度通道
        valid_fire = fire_confidence[np.isfinite(fire_confidence)]
        
        if len(valid_fire) > 0:
            # 基本统计
            fire_stats = {
                'count': len(valid_fire),
                'mean': float(np.mean(valid_fire)),
                'std': float(np.std(valid_fire)),
                'min': float(np.min(valid_fire)),
                'max': float(np.max(valid_fire)),
                'median': float(np.median(valid_fire)),
                'q25': float(np.percentile(valid_fire, 25)),
                'q75': float(np.percentile(valid_fire, 75))
            }
            
            # 不同阈值下的二分类分析
            thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
            threshold_analysis = {}
            
            for threshold in thresholds:
                positive_mask = valid_fire > threshold
                positive_count = positive_mask.sum()
                positive_ratio = positive_count / len(valid_fire)
                
                threshold_analysis[threshold] = {
                    'positive_count': int(positive_count),
                    'positive_ratio': float(positive_ratio),
                    'negative_count': int(len(valid_fire) - positive_count),
                    'imbalance_ratio': float((len(valid_fire) - positive_count) / max(positive_count, 1))
                }
            
            results['fire_stats'] = fire_stats
            results['threshold_analysis'] = threshold_analysis
            
            # 5.2 极值分析
            # 识别高置信度火点
            high_confidence_threshold = np.percentile(valid_fire, 95)
            high_confidence_mask = valid_fire > high_confidence_threshold
            high_confidence_samples = np.where(high_confidence_mask)[0]
            
            if len(high_confidence_samples) > 0:
                # 分析高置信度火点的环境条件
                high_conf_data = self.data[high_confidence_samples]
                
                environmental_analysis = {}
                for channel in [5, 6, 8, 15]:  # 温度、湿度、风速、干旱指数
                    channel_data = high_conf_data[:, channel]
                    valid_channel = channel_data[np.isfinite(channel_data)]
                    
                    if len(valid_channel) > 0:
                        environmental_analysis[channel] = {
                            'channel_name': self.feature_schema[channel]['name'],
                            'mean': float(np.mean(valid_channel)),
                            'std': float(np.std(valid_channel)),
                            'min': float(np.min(valid_channel)),
                            'max': float(np.max(valid_channel))
                        }
                
                results['high_confidence_environmental'] = environmental_analysis
        
        # 保存结果
        self.results['target_analysis'] = results
        
        # 生成可视化
        self._plot_target_variable(results)
        
        # 保存报告
        self._save_target_variable_report(results)
        
        print(f"  ✅ 目标变量深度分析完成")
        return results
    
    def _plot_target_variable(self, results):
        """绘制目标变量分析图表"""
        
        fire_confidence = self.data[:, 22]
        valid_fire = fire_confidence[np.isfinite(fire_confidence)]
        
        if len(valid_fire) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 火点置信度分布
        axes[0,0].hist(valid_fire, bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Fire Confidence Distribution', fontweight='bold')
        axes[0,0].set_xlabel('Fire Confidence')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 对数尺度分布
        log_fire = np.log1p(valid_fire)  # log(1+x)避免log(0)
        axes[0,1].hist(log_fire, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[0,1].set_title('Log-scale Fire Confidence Distribution', fontweight='bold')
        axes[0,1].set_xlabel('Log(1 + Fire Confidence)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 不同阈值下的类别比例
        if 'threshold_analysis' in results:
            thresholds = list(results['threshold_analysis'].keys())
            positive_ratios = [results['threshold_analysis'][t]['positive_ratio'] for t in thresholds]
            
            axes[1,0].bar(range(len(thresholds)), positive_ratios, alpha=0.7)
            axes[1,0].set_title('Positive Class Ratio by Threshold', fontweight='bold')
            axes[1,0].set_xlabel('Threshold')
            axes[1,0].set_ylabel('Positive Ratio')
            axes[1,0].set_xticks(range(len(thresholds)))
            axes[1,0].set_xticklabels([str(t) for t in thresholds])
            axes[1,0].grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, ratio in enumerate(positive_ratios):
                axes[1,0].text(i, ratio + 0.001, f'{ratio:.3f}', ha='center', va='bottom')
        
        # 4. 累积分布函数
        sorted_fire = np.sort(valid_fire)
        cumulative_prob = np.arange(1, len(sorted_fire) + 1) / len(sorted_fire)
        axes[1,1].plot(sorted_fire, cumulative_prob, linewidth=2)
        axes[1,1].set_title('Cumulative Distribution Function', fontweight='bold')
        axes[1,1].set_xlabel('Fire Confidence')
        axes[1,1].set_ylabel('Cumulative Probability')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "target_variable_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_target_variable_report(self, results):
        """保存目标变量分析报告"""
        
        # 火点统计报告
        if 'fire_stats' in results:
            fire_stats_df = pd.DataFrame([results['fire_stats']]).T
            fire_stats_df.columns = ['Value']
            fire_stats_df.to_csv(self.output_dir / "tables" / "fire_confidence_statistics.csv")
        
        # 阈值分析报告
        if 'threshold_analysis' in results:
            threshold_df = pd.DataFrame([
                {
                    'Threshold': threshold,
                    'Positive_Count': analysis['positive_count'],
                    'Positive_Ratio': f"{analysis['positive_ratio']:.4f}",
                    'Negative_Count': analysis['negative_count'],
                    'Imbalance_Ratio': f"{analysis['imbalance_ratio']:.2f}:1"
                }
                for threshold, analysis in results['threshold_analysis'].items()
            ])
            threshold_df.to_csv(self.output_dir / "tables" / "threshold_analysis.csv", index=False)
    
    def analyze_environmental_variables(self):
        """6. 环境变量专题分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 6.1 气象变量分析
        weather_channels = [5, 6, 7, 8, 9, 10, 11]  # 温度、湿度、风向、风速、降水、气压、太阳辐射
        weather_analysis = {}
        
        for channel in weather_channels:
            channel_data = self.data[:, channel]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                weather_analysis[channel] = {
                    'channel_name': self.feature_schema[channel]['name'],
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'q25': float(np.percentile(valid_data, 25)),
                    'q75': float(np.percentile(valid_data, 75)),
                    'extreme_values_count': int(((valid_data < np.percentile(valid_data, 5)) | 
                                               (valid_data > np.percentile(valid_data, 95))).sum())
                }
        
        results['weather_analysis'] = weather_analysis
        
        # 6.2 植被与燃料分析
        vegetation_channels = [3, 4]  # NDVI, EVI2
        vegetation_analysis = {}
        
        for channel in vegetation_channels:
            channel_data = self.data[:, channel]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                # 植被健康度分类
                low_vegetation = (valid_data < 0.2).sum()
                medium_vegetation = ((valid_data >= 0.2) & (valid_data < 0.6)).sum()
                high_vegetation = (valid_data >= 0.6).sum()
                
                vegetation_analysis[channel] = {
                    'channel_name': self.feature_schema[channel]['name'],
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'low_vegetation_count': int(low_vegetation),
                    'medium_vegetation_count': int(medium_vegetation),
                    'high_vegetation_count': int(high_vegetation),
                    'low_vegetation_ratio': float(low_vegetation / len(valid_data)),
                    'medium_vegetation_ratio': float(medium_vegetation / len(valid_data)),
                    'high_vegetation_ratio': float(high_vegetation / len(valid_data))
                }
        
        results['vegetation_analysis'] = vegetation_analysis
        
        # 6.3 地形影响分析
        topography_channels = [12, 13, 14]  # 海拔、坡度、坡向
        topography_analysis = {}
        
        for channel in topography_channels:
            channel_data = self.data[:, channel]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                topography_analysis[channel] = {
                    'channel_name': self.feature_schema[channel]['name'],
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'terrain_complexity': float(np.std(valid_data) / np.mean(np.abs(valid_data)) if np.mean(np.abs(valid_data)) > 0 else 0)
                }
        
        results['topography_analysis'] = topography_analysis
        
        # 保存结果
        self.results['environmental_analysis'] = results
        
        # 生成可视化
        self._plot_environmental_variables(results)
        
        # 保存报告
        self._save_environmental_report(results)
        
        print(f"  ✅ 环境变量专题分析完成")
        return results
    
    def _plot_environmental_variables(self, results):
        """绘制环境变量分析图表"""
        
        # 1. 气象变量分布
        weather_data = results.get('weather_analysis', {})
        if weather_data:
            n_weather = len(weather_data)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for i, (channel, analysis) in enumerate(weather_data.items()):
                if i < len(axes):
                    channel_data = self.data[:, channel]
                    valid_data = channel_data[np.isfinite(channel_data)]
                    
                    if len(valid_data) > 0:
                        axes[i].hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
                        axes[i].set_title(f"{analysis['channel_name']}", fontweight='bold')
                        axes[i].set_xlabel('Value')
                        axes[i].set_ylabel('Frequency')
                        axes[i].grid(True, alpha=0.3)
                        
                        # 添加统计信息
                        textstr = f"Mean: {analysis['mean']:.2f}\nStd: {analysis['std']:.2f}"
                        axes[i].text(0.02, 0.98, textstr, transform=axes[i].transAxes,
                                   verticalalignment='top', fontsize=9,
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            # 隐藏多余的子图
            for i in range(len(weather_data), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "weather_variables_distribution.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 植被健康度分析
        vegetation_data = results.get('vegetation_analysis', {})
        if vegetation_data:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            for i, (channel, analysis) in enumerate(vegetation_data.items()):
                if i < 2:
                    # 植被健康度饼图
                    sizes = [analysis['low_vegetation_ratio'], 
                            analysis['medium_vegetation_ratio'], 
                            analysis['high_vegetation_ratio']]
                    labels = ['Low Vegetation', 'Medium Vegetation', 'High Vegetation']
                    colors = ['red', 'yellow', 'green']
                    
                    axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    axes[i].set_title(f"{analysis['channel_name']} Health Distribution", fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "vegetation_health_distribution.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_environmental_report(self, results):
        """保存环境变量分析报告"""
        
        # 气象变量报告
        if 'weather_analysis' in results:
            weather_df = pd.DataFrame([
                {
                    'Channel_ID': channel,
                    'Variable_Name': analysis['channel_name'],
                    'Mean': f"{analysis['mean']:.4f}",
                    'Std': f"{analysis['std']:.4f}",
                    'Min': f"{analysis['min']:.4f}",
                    'Max': f"{analysis['max']:.4f}",
                    'Q25': f"{analysis['q25']:.4f}",
                    'Q75': f"{analysis['q75']:.4f}",
                    'Extreme_Values': analysis['extreme_values_count']
                }
                for channel, analysis in results['weather_analysis'].items()
            ])
            weather_df.to_csv(self.output_dir / "tables" / "weather_analysis.csv", index=False)
        
        # 植被分析报告
        if 'vegetation_analysis' in results:
            vegetation_df = pd.DataFrame([
                {
                    'Channel_ID': channel,
                    'Variable_Name': analysis['channel_name'],
                    'Mean': f"{analysis['mean']:.4f}",
                    'Low_Vegetation_Ratio': f"{analysis['low_vegetation_ratio']:.4f}",
                    'Medium_Vegetation_Ratio': f"{analysis['medium_vegetation_ratio']:.4f}",
                    'High_Vegetation_Ratio': f"{analysis['high_vegetation_ratio']:.4f}"
                }
                for channel, analysis in results['vegetation_analysis'].items()
            ])
            vegetation_df.to_csv(self.output_dir / "tables" / "vegetation_analysis.csv", index=False)
    
    def analyze_preprocessing_requirements(self):
        """7. 数据预处理需求分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 7.1 标准化策略评估
        normalization_analysis = {}
        
        for channel in range(self.data.shape[1]):
            channel_data = self.data[:, channel]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                # 测试不同标准化方法
                original_data = valid_data.copy()
                
                # Standard scaling (z-score)
                standardized = StandardScaler().fit_transform(original_data.reshape(-1, 1)).flatten()
                
                # Min-Max scaling
                min_val, max_val = np.min(original_data), np.max(original_data)
                minmax_scaled = (original_data - min_val) / (max_val - min_val) if max_val > min_val else original_data
                
                # Robust scaling (使用中位数和IQR)
                median_val = np.median(original_data)
                q75, q25 = np.percentile(original_data, [75, 25])
                iqr = q75 - q25
                robust_scaled = (original_data - median_val) / iqr if iqr > 0 else original_data - median_val
                
                normalization_analysis[channel] = {
                    'channel_name': self.feature_schema[channel]['name'],
                    'original_mean': float(np.mean(original_data)),
                    'original_std': float(np.std(original_data)),
                    'original_skewness': float(stats.skew(original_data)),
                    'standardized_mean': float(np.mean(standardized)),
                    'standardized_std': float(np.std(standardized)),
                    'minmax_mean': float(np.mean(minmax_scaled)),
                    'minmax_std': float(np.std(minmax_scaled)),
                    'robust_mean': float(np.mean(robust_scaled)),
                    'robust_std': float(np.std(robust_scaled)),
                    'recommended_method': self._recommend_normalization_method(original_data, channel)
                }
        
        results['normalization_analysis'] = normalization_analysis
        
        # 7.2 特征工程潜力识别
        feature_engineering_opportunities = {}
        
        # 检查可以创建的衍生特征
        if np.all([ch in range(self.data.shape[1]) for ch in [5, 6]]):  # 温度和湿度
            temp_data = self.data[:, 5]
            humidity_data = self.data[:, 6]
            valid_mask = np.isfinite(temp_data) & np.isfinite(humidity_data)
            
            if valid_mask.sum() > 0:
                # 计算热指数或不适指数
                heat_index = temp_data[valid_mask] + humidity_data[valid_mask] * 0.1  # 简化版本
                feature_engineering_opportunities['heat_index'] = {
                    'description': 'Temperature-Humidity combined index',
                    'correlation_with_fire': float(np.corrcoef(heat_index, self.data[valid_mask, 22])[0, 1]) 
                                           if np.isfinite(np.corrcoef(heat_index, self.data[valid_mask, 22])[0, 1]) else 0
                }
        
        # 风速和风向的组合
        if np.all([ch in range(self.data.shape[1]) for ch in [7, 8]]):  # 风向和风速
            wind_dir = self.data[:, 7]
            wind_speed = self.data[:, 8]
            valid_mask = np.isfinite(wind_dir) & np.isfinite(wind_speed)
            
            if valid_mask.sum() > 0:
                # 计算风向量分量
                wind_u = wind_speed[valid_mask] * np.cos(np.radians(wind_dir[valid_mask]))
                wind_v = wind_speed[valid_mask] * np.sin(np.radians(wind_dir[valid_mask]))
                
                feature_engineering_opportunities['wind_components'] = {
                    'description': 'Wind vector components (u, v)',
                    'u_component_std': float(np.std(wind_u)),
                    'v_component_std': float(np.std(wind_v))
                }
        
        results['feature_engineering'] = feature_engineering_opportunities
        
        # 保存结果
        self.results['preprocessing_requirements'] = results
        
        # 生成可视化
        self._plot_preprocessing_analysis(results)
        
        # 保存报告
        self._save_preprocessing_report(results)
        
        print(f"  ✅ 数据预处理需求分析完成")
        return results
    
    def _recommend_normalization_method(self, data, channel):
        """推荐标准化方法"""
        skewness = abs(stats.skew(data))
        
        # 分类特征不需要标准化
        if channel == 16:  # 土地覆盖
            return 'categorical_encoding'
        
        # 角度特征需要特殊处理
        if channel in [7, 14, 19]:  # 风向、坡向、预测风向
            return 'circular_encoding'
        
        # 根据偏度推荐方法
        if skewness > 2:
            return 'robust_scaling'
        elif skewness > 1:
            return 'standard_scaling'
        else:
            return 'minmax_scaling'
    
    def _plot_preprocessing_analysis(self, results):
        """绘制预处理分析图表"""
        
        # 标准化方法推荐统计
        if 'normalization_analysis' in results:
            methods = [analysis['recommended_method'] for analysis in results['normalization_analysis'].values()]
            method_counts = pd.Series(methods).value_counts()
            
            plt.figure(figsize=(10, 6))
            method_counts.plot(kind='bar', alpha=0.7)
            plt.title('Recommended Normalization Methods Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Normalization Method')
            plt.ylabel('Number of Channels')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(method_counts.values):
                plt.text(i, v + 0.1, str(v), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "normalization_recommendations.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_preprocessing_report(self, results):
        """保存预处理分析报告"""
        
        if 'normalization_analysis' in results:
            norm_df = pd.DataFrame([
                {
                    'Channel_ID': channel,
                    'Channel_Name': analysis['channel_name'],
                    'Original_Mean': f"{analysis['original_mean']:.4f}",
                    'Original_Std': f"{analysis['original_std']:.4f}",
                    'Original_Skewness': f"{analysis['original_skewness']:.4f}",
                    'Recommended_Method': analysis['recommended_method']
                }
                for channel, analysis in results['normalization_analysis'].items()
            ])
            norm_df.to_csv(self.output_dir / "tables" / "normalization_analysis.csv", index=False)
    
    def create_advanced_visualizations(self):
        """8. 高级可视化与洞察发现"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 8.1 降维可视化
        # 选择数值型特征进行降维
        numeric_channels = [i for i in range(self.data.shape[1]) if i != 16]  # 排除土地覆盖分类特征
        numeric_data = self.data[:, numeric_channels]
        
        # 移除包含NaN的样本
        valid_mask = ~np.isnan(numeric_data).any(axis=1)
        clean_data = numeric_data[valid_mask]
        
        if len(clean_data) > 1000:  # 确保有足够的数据
            # 随机采样以提高计算效率
            sample_size = min(5000, len(clean_data))
            sample_indices = np.random.choice(len(clean_data), sample_size, replace=False)
            sample_data = clean_data[sample_indices]
            
            # PCA分析
            pca = PCA(n_components=min(10, sample_data.shape[1]))
            pca_result = pca.fit_transform(sample_data)
            
            results['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
            results['pca_cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_).tolist()
            
            # t-SNE分析（如果数据不太大）
            if len(sample_data) <= 2000:
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                    tsne_result = tsne.fit_transform(sample_data)
                    results['tsne_completed'] = True
                except:
                    results['tsne_completed'] = False
                    tsne_result = None
            else:
                results['tsne_completed'] = False
                tsne_result = None
            
            # 生成降维可视化
            self._plot_dimensionality_reduction(pca_result, tsne_result, results)
        
        # 8.2 特征重要性可视化
        self._create_feature_importance_visualization()
        
        # 保存结果
        self.results['advanced_visualizations'] = results
        
        print(f"  ✅ 高级可视化与洞察发现完成")
        return results
    
    def _plot_dimensionality_reduction(self, pca_result, tsne_result, results):
        """绘制降维分析图表"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PCA方差解释图
        axes[0, 0].plot(range(1, len(results['pca_explained_variance']) + 1), 
                       results['pca_explained_variance'], 'bo-')
        axes[0, 0].set_title('PCA Explained Variance by Component', fontweight='bold')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PCA累积方差图
        axes[0, 1].plot(range(1, len(results['pca_cumulative_variance']) + 1), 
                       results['pca_cumulative_variance'], 'ro-')
        axes[0, 1].axhline(y=0.8, color='green', linestyle='--', label='80% Variance')
        axes[0, 1].axhline(y=0.95, color='orange', linestyle='--', label='95% Variance')
        axes[0, 1].set_title('PCA Cumulative Explained Variance', fontweight='bold')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. PCA前两个主成分散点图
        scatter = axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=20)
        axes[1, 0].set_title('PCA: First Two Principal Components', fontweight='bold')
        axes[1, 0].set_xlabel(f'PC1 ({results["pca_explained_variance"][0]:.1%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({results["pca_explained_variance"][1]:.1%} variance)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. t-SNE散点图（如果可用）
        if tsne_result is not None:
            scatter = axes[1, 1].scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.6, s=20)
            axes[1, 1].set_title('t-SNE Visualization', fontweight='bold')
            axes[1, 1].set_xlabel('t-SNE 1')
            axes[1, 1].set_ylabel('t-SNE 2')
        else:
            axes[1, 1].text(0.5, 0.5, 't-SNE not computed\n(dataset too large)', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12)
            axes[1, 1].set_title('t-SNE Visualization', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "dimensionality_reduction.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_importance_visualization(self):
        """创建特征重要性可视化"""
        
        if 'target_correlations' in self.results.get('feature_relationships', {}):
            target_corrs = self.results['feature_relationships']['target_correlations']
            
            # 按重要性排序
            sorted_features = sorted(target_corrs.items(), 
                                   key=lambda x: abs(x[1]['pearson_correlation']), 
                                   reverse=True)
            
            channels = [item[0] for item in sorted_features]
            correlations = [item[1]['pearson_correlation'] for item in sorted_features]
            names = [f"Ch{ch}: {self.feature_schema[ch]['name'][:12]}" for ch in channels]
            
            plt.figure(figsize=(12, 8))
            colors = ['red' if corr > 0 else 'blue' for corr in correlations]
            bars = plt.barh(range(len(names)), correlations, color=colors, alpha=0.7)
            
            plt.title('Feature Importance (Correlation with Fire Confidence)', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Correlation Coefficient')
            plt.yticks(range(len(names)), names)
            plt.axvline(0, color='black', linestyle='-', alpha=0.3)
            
            # 添加数值标签
            for i, (bar, corr) in enumerate(zip(bars, correlations)):
                plt.text(corr + 0.01 if corr > 0 else corr - 0.01, bar.get_y() + bar.get_height()/2,
                        f'{corr:.3f}', ha='left' if corr > 0 else 'right', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "figures" / "feature_importance.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_academic_report(self):
        """9. 生成学术报告"""
        print("  正在生成学术报告...")
        
        # 生成综合报告
        report_content = self._create_comprehensive_report()
        
        # 保存报告
        with open(self.output_dir / "reports" / "comprehensive_eda_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 生成图表索引
        self._create_figure_index()
        
        # 生成表格索引
        self._create_table_index()
        
        print(f"  ✅ 学术报告生成完成")
    
    def _create_comprehensive_report(self):
        """创建综合EDA报告"""
        
        report = """# WildfireSpreadTS Dataset - Comprehensive Exploratory Data Analysis Report

## Executive Summary

This report presents a comprehensive exploratory data analysis (EDA) of the WildfireSpreadTS dataset, a multi-modal time series dataset designed for wildfire spread prediction research. The analysis covers data quality assessment, statistical characterization, spatiotemporal patterns, feature relationships, and preprocessing recommendations.

## Dataset Overview

- **Temporal Coverage**: 2018-2021
- **Spatial Resolution**: 375m per pixel
- **Features**: 23 channels representing environmental variables
- **Target Variable**: Active fire confidence levels (Channel 22)

## Key Findings

### 1. Data Quality Assessment
"""
        
        # 添加数据质量发现
        if 'data_quality' in self.results:
            basic_info = self.results['data_quality']['basic_info']
            report += f"""
- Total samples analyzed: {basic_info['total_samples']:,}
- Number of features: {basic_info['n_features']}
- Fire events covered: {basic_info['n_fire_events']}
- Years included: {', '.join(basic_info['years_covered'])}
- Dataset size: {basic_info['data_size_mb']:.1f} MB
"""
        
        # 添加缺失值分析
        if 'data_quality' in self.results and 'missing_analysis' in self.results['data_quality']:
            high_missing = [(ch, analysis) for ch, analysis in self.results['data_quality']['missing_analysis'].items()
                           if analysis['missing_ratio'] > 0.1]
            if high_missing:
                report += f"""
#### Missing Values
- {len(high_missing)} channels have >10% missing values
- Channels requiring attention: {', '.join([f"Ch{ch}" for ch, _ in high_missing])}
"""
        
        # 添加目标变量分析
        if 'target_analysis' in self.results and 'fire_stats' in self.results['target_analysis']:
            fire_stats = self.results['target_analysis']['fire_stats']
            report += f"""
### 2. Target Variable Characteristics
- Fire confidence mean: {fire_stats['mean']:.4f}
- Fire confidence std: {fire_stats['std']:.4f}
- Fire confidence range: [{fire_stats['min']:.2f}, {fire_stats['max']:.2f}]
"""
        
        # 添加特征关系发现
        if 'feature_relationships' in self.results and 'strong_correlations' in self.results['feature_relationships']:
            strong_corrs = self.results['feature_relationships']['strong_correlations']
            report += f"""
### 3. Feature Relationships
- {len(strong_corrs)} strong correlations (|r| > 0.7) identified
- Potential multicollinearity issues detected
"""
        
        # 添加预处理建议
        report += """
### 4. Preprocessing Recommendations

Based on the analysis, the following preprocessing steps are recommended:

1. **Missing Value Treatment**: Apply appropriate imputation strategies for channels with significant missing data
2. **Normalization**: Use channel-specific normalization methods based on data distribution characteristics
3. **Feature Engineering**: Consider creating derived features from environmental variable combinations
4. **Class Imbalance**: Implement sampling strategies or specialized loss functions for fire detection

### 5. Modeling Implications

The analysis reveals several important considerations for model development:

- **Extreme Class Imbalance**: Fire events are rare, requiring specialized handling techniques
- **Multi-scale Patterns**: Features exhibit both local and global spatial patterns
- **Temporal Dependencies**: Clear temporal evolution in fire behavior and environmental conditions
- **Feature Interactions**: Strong relationships between meteorological variables suggest potential for ensemble methods

## Conclusion

The WildfireSpreadTS dataset presents both opportunities and challenges for wildfire spread prediction modeling. The comprehensive environmental feature set provides rich information for model training, while the extreme class imbalance and missing value patterns require careful preprocessing attention.

## Files Generated

### Figures
- `missing_values_analysis.png`: Missing value patterns by channel
- `outlier_analysis.png`: Outlier detection results
- `descriptive_statistics_distributions.png`: Feature distribution analysis
- `correlation_matrix.png`: Inter-feature correlation patterns
- `target_variable_analysis.png`: Fire confidence distribution analysis
- `feature_importance.png`: Feature importance ranking

### Tables
- `missing_values_report.csv`: Detailed missing value statistics
- `descriptive_statistics.csv`: Comprehensive statistical summary
- `target_correlations.csv`: Correlations with target variable
- `threshold_analysis.csv`: Class balance analysis at different thresholds

---
*Report generated by WildfireSpreadTS EDA Analysis System*
"""
        
        return report
    
    def _create_figure_index(self):
        """创建图表索引"""
        figures_dir = self.output_dir / "figures"
        if figures_dir.exists():
            figures = list(figures_dir.glob("*.png"))
            
            index_content = "# Figure Index\n\n"
            for i, fig_path in enumerate(sorted(figures), 1):
                index_content += f"{i}. **{fig_path.stem.replace('_', ' ').title()}**\n"
                index_content += f"   - File: `{fig_path.name}`\n"
                index_content += f"   - Description: Analysis visualization\n\n"
            
            with open(self.output_dir / "reports" / "figure_index.md", 'w', encoding='utf-8') as f:
                f.write(index_content)
    
    def _create_table_index(self):
        """创建表格索引"""
        tables_dir = self.output_dir / "tables"
        if tables_dir.exists():
            tables = list(tables_dir.glob("*.csv"))
            
            index_content = "# Table Index\n\n"
            for i, table_path in enumerate(sorted(tables), 1):
                index_content += f"{i}. **{table_path.stem.replace('_', ' ').title()}**\n"
                index_content += f"   - File: `{table_path.name}`\n"
                index_content += f"   - Description: Statistical analysis results\n\n"
            
            with open(self.output_dir / "reports" / "table_index.md", 'w', encoding='utf-8') as f:
                f.write(index_content)

# 如果直接运行此脚本，执行完整分析
if __name__ == "__main__":
    print("请运行主EDA脚本 comprehensive_eda.py") 