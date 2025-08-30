"""
WildfireSpreadTS数据集全面EDA分析系统
包含9个主要分析模块，为模型开发和论文写作提供支撑
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import glob
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 科学计算和统计
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

# 可视化增强
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# 时间序列分析
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

# 空间分析（如果可用）
try:
    from pysal.lib import weights
    from pysal.explore import esda
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    print("PyPAL not available, spatial analysis will be limited")

class WildfireEDAAnalyzer:
    """WildfireSpreadTS数据集EDA分析器"""
    
    def __init__(self, data_dir="data/processed", output_dir="eda_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # 特征定义
        self.feature_schema = self._define_feature_schema()
        self.channel_groups = self._define_channel_groups()
        
        # 存储分析结果
        self.results = {}
        self.summary_stats = {}
        
        print(f"EDA分析器初始化完成")
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")
    
    def _define_feature_schema(self):
        """定义23通道特征模式"""
        return {
            0: {'name': 'VIIRS_I4', 'category': 'Remote Sensing', 'unit': 'Brightness Temperature (K)', 'range': [200, 400]},
            1: {'name': 'VIIRS_I5', 'category': 'Remote Sensing', 'unit': 'Brightness Temperature (K)', 'range': [200, 400]},
            2: {'name': 'VIIRS_M13', 'category': 'Remote Sensing', 'unit': 'Brightness Temperature (K)', 'range': [200, 400]},
            3: {'name': 'NDVI', 'category': 'Vegetation', 'unit': 'Index', 'range': [-1, 1]},
            4: {'name': 'EVI2', 'category': 'Vegetation', 'unit': 'Index', 'range': [-1, 1]},
            5: {'name': 'Temperature', 'category': 'Weather', 'unit': 'Celsius', 'range': [-50, 60]},
            6: {'name': 'Humidity', 'category': 'Weather', 'unit': 'Percentage', 'range': [0, 100]},
            7: {'name': 'Wind_Direction', 'category': 'Weather', 'unit': 'Degrees', 'range': [0, 360]},
            8: {'name': 'Wind_Speed', 'category': 'Weather', 'unit': 'm/s', 'range': [0, 50]},
            9: {'name': 'Precipitation', 'category': 'Weather', 'unit': 'mm', 'range': [0, 500]},
            10: {'name': 'Surface_Pressure', 'category': 'Weather', 'unit': 'hPa', 'range': [800, 1100]},
            11: {'name': 'Solar_Radiation', 'category': 'Weather', 'unit': 'W/m²', 'range': [0, 1500]},
            12: {'name': 'Elevation', 'category': 'Topography', 'unit': 'meters', 'range': [-500, 9000]},
            13: {'name': 'Slope', 'category': 'Topography', 'unit': 'degrees', 'range': [0, 90]},
            14: {'name': 'Aspect', 'category': 'Topography', 'unit': 'degrees', 'range': [0, 360]},
            15: {'name': 'PDSI', 'category': 'Drought', 'unit': 'Index', 'range': [-10, 10]},
            16: {'name': 'Land_Cover', 'category': 'Land Cover', 'unit': 'Class (1-16)', 'range': [1, 16]},
            17: {'name': 'Forecast_Temperature', 'category': 'Weather Forecast', 'unit': 'Celsius', 'range': [-50, 60]},
            18: {'name': 'Forecast_Humidity', 'category': 'Weather Forecast', 'unit': 'Percentage', 'range': [0, 100]},
            19: {'name': 'Forecast_Wind_Direction', 'category': 'Weather Forecast', 'unit': 'Degrees', 'range': [0, 360]},
            20: {'name': 'Forecast_Wind_Speed', 'category': 'Weather Forecast', 'unit': 'm/s', 'range': [0, 50]},
            21: {'name': 'Forecast_Precipitation', 'category': 'Weather Forecast', 'unit': 'mm', 'range': [0, 500]},
            22: {'name': 'Active_Fire_Confidence', 'category': 'Target', 'unit': 'Confidence (0-100)', 'range': [0, 100]}
        }
    
    def _define_channel_groups(self):
        """定义通道分组"""
        return {
            'remote_sensing': [0, 1, 2],
            'vegetation': [3, 4],
            'weather_historical': [5, 6, 7, 8, 9, 10, 11],
            'topography': [12, 13, 14],
            'drought': [15],
            'land_cover': [16],
            'weather_forecast': [17, 18, 19, 20, 21],
            'target': [22]
        }
    
    def run_complete_analysis(self):
        """运行完整的EDA分析"""
        print("🚀 开始全面EDA分析...")
        
        # 1. 数据质量与完整性分析
        print("\n📋 1. 数据质量与完整性分析")
        self.analyze_data_quality()
        
        # 2. 描述性统计分析
        print("\n📊 2. 描述性统计分析")
        self.analyze_descriptive_statistics()
        
        # 3. 时空分布特征分析
        print("\n🌍 3. 时空分布特征分析")
        self.analyze_spatiotemporal_patterns()
        
        # 4. 特征关系与相关性分析
        print("\n🔗 4. 特征关系与相关性分析")
        self.analyze_feature_relationships()
        
        # 5. 目标变量深度分析
        print("\n🎯 5. 目标变量深度分析")
        self.analyze_target_variable()
        
        # 6. 环境变量专题分析
        print("\n🌡️ 6. 环境变量专题分析")
        self.analyze_environmental_variables()
        
        # 7. 数据预处理需求分析
        print("\n📊 7. 数据预处理需求分析")
        self.analyze_preprocessing_requirements()
        
        # 8. 高级可视化与洞察发现
        print("\n🎨 8. 高级可视化与洞察发现")
        self.create_advanced_visualizations()
        
        # 9. 生成学术报告
        print("\n📝 9. 生成学术报告")
        self.generate_academic_report()
        
        print(f"\n✅ EDA分析完成！结果保存在 {self.output_dir}")
    
    def load_sample_data(self, max_files=10, sample_ratio=0.1):
        """加载样本数据进行分析"""
        print(f"正在加载数据样本...")
        
        # 按年份分组查找HDF5文件
        hdf5_files_by_year = {}
        for year in ['2018', '2019', '2020', '2021']:
            year_dir = self.data_dir / year
            if year_dir.exists():
                year_files = list(year_dir.glob("*.hdf5"))
                hdf5_files_by_year[year] = year_files
                print(f"  {year}年: {len(year_files)} 个文件")
        
        total_files = sum(len(files) for files in hdf5_files_by_year.values())
        if total_files == 0:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到HDF5文件")
        
        print(f"总计找到 {total_files} 个HDF5文件")
        
        # 从每年均匀采样文件
        files_to_process = []
        files_per_year = max_files // len(hdf5_files_by_year)
        remainder = max_files % len(hdf5_files_by_year)
        
        for i, (year, year_files) in enumerate(hdf5_files_by_year.items()):
            if year_files:
                # 为前几年分配额外的文件
                n_files = files_per_year + (1 if i < remainder else 0)
                n_files = min(n_files, len(year_files))
                
                # 随机采样以获得多样性
                import random
                sampled_files = random.sample(year_files, n_files) if n_files < len(year_files) else year_files
                files_to_process.extend(sampled_files)
                print(f"  从{year}年采样 {len(sampled_files)} 个文件")
        
        print(f"总共将处理 {len(files_to_process)} 个文件")
        
        all_data = []
        fire_events = []
        metadata = []
        
        for file_path in files_to_process:
            try:
                with h5py.File(file_path, 'r') as f:
                    year = file_path.parent.name
                    
                    # 直接读取数据（HDF5文件结构：根目录只有'data'键）
                    data = f['data'][:]  # Shape: (T, C, H, W)
                    
                    # 从属性中获取火灾事件信息
                    fire_name = f['data'].attrs.get('fire_name', file_path.stem)
                    if isinstance(fire_name, bytes):
                        fire_name = fire_name.decode('utf-8')
                    elif isinstance(fire_name, np.ndarray):
                        fire_name = str(fire_name)
                    
                    # 采样数据以减少内存占用
                    T, C, H, W = data.shape
                    sample_size = int(H * W * sample_ratio)
                    
                    # 随机采样像素
                    pixel_indices = np.random.choice(H*W, size=min(sample_size, H*W), replace=False)
                    h_indices = pixel_indices // W
                    w_indices = pixel_indices % W
                    
                    # 提取采样数据
                    sampled_data = data[:, :, h_indices, w_indices]  # (T, C, N_samples)
                    sampled_data = sampled_data.transpose(2, 0, 1)  # (N_samples, T, C)
                    sampled_data = sampled_data.reshape(-1, C)  # (N_samples*T, C)
                    
                    all_data.append(sampled_data)
                    
                    # 记录元数据
                    for i in range(len(sampled_data)):
                        metadata.append({
                            'year': year,
                            'fire_event': fire_name,
                            'file_path': str(file_path),
                            'sample_id': i
                        })
                    
                    fire_events.append(fire_name)
                        
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        if not all_data:
            raise ValueError("未能成功加载任何数据")
        
        # 合并所有数据
        self.data = np.vstack(all_data)
        self.metadata_df = pd.DataFrame(metadata)
        self.fire_events = fire_events
        
        print(f"数据加载完成:")
        print(f"  - 数据形状: {self.data.shape}")
        print(f"  - 火灾事件数: {len(set(fire_events))}")
        print(f"  - 时间跨度: {sorted(set(self.metadata_df['year']))}")
        
        return self.data, self.metadata_df
    
    def analyze_data_quality(self):
        """1. 数据质量与完整性分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 1.1 基础数据审查
        results['basic_info'] = {
            'total_samples': len(self.data),
            'n_features': self.data.shape[1],
            'n_fire_events': len(set(self.fire_events)),
            'years_covered': sorted(set(self.metadata_df['year'])),
            'data_size_mb': self.data.nbytes / (1024**2)
        }
        
        # 1.2 缺失值分析
        missing_analysis = {}
        for i in range(self.data.shape[1]):
            channel_data = self.data[:, i]
            missing_count = np.isnan(channel_data).sum()
            missing_ratio = missing_count / len(channel_data)
            
            missing_analysis[i] = {
                'channel_name': self.feature_schema[i]['name'],
                'missing_count': int(missing_count),
                'missing_ratio': float(missing_ratio),
                'total_samples': len(channel_data)
            }
        
        results['missing_analysis'] = missing_analysis
        
        # 1.3 异常值检测
        outlier_analysis = {}
        for i in range(self.data.shape[1]):
            channel_data = self.data[:, i]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                # 使用IQR方法检测异常值
                Q1 = np.percentile(valid_data, 25)
                Q3 = np.percentile(valid_data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (valid_data < lower_bound) | (valid_data > upper_bound)
                
                # 检查是否超出物理合理范围
                expected_range = self.feature_schema[i]['range']
                physical_outliers = (valid_data < expected_range[0]) | (valid_data > expected_range[1])
                
                outlier_analysis[i] = {
                    'channel_name': self.feature_schema[i]['name'],
                    'iqr_outliers': int(outliers.sum()),
                    'iqr_outlier_ratio': float(outliers.sum() / len(valid_data)),
                    'physical_outliers': int(physical_outliers.sum()),
                    'physical_outlier_ratio': float(physical_outliers.sum() / len(valid_data)),
                    'value_range': [float(valid_data.min()), float(valid_data.max())],
                    'expected_range': expected_range
                }
        
        results['outlier_analysis'] = outlier_analysis
        
        # 保存结果
        self.results['data_quality'] = results
        
        # 生成可视化
        self._plot_data_quality(results)
        
        # 保存报告
        self._save_data_quality_report(results)
        
        print(f"  ✅ 数据质量分析完成")
        return results
    
    def _plot_data_quality(self, results):
        """绘制数据质量相关图表"""
        
        # 1. 缺失值分析图
        missing_data = []
        for i, analysis in results['missing_analysis'].items():
            missing_data.append({
                'Channel': f"{i}: {analysis['channel_name']}",
                'Missing_Ratio': analysis['missing_ratio']
            })
        
        missing_df = pd.DataFrame(missing_data)
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(missing_df)), missing_df['Missing_Ratio'])
        plt.title('Missing Values by Channel', fontsize=16, fontweight='bold')
        plt.xlabel('Channel Index', fontsize=12)
        plt.ylabel('Missing Ratio', fontsize=12)
        plt.xticks(range(len(missing_df)), [f"{i}" for i in range(len(missing_df))], rotation=45)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0.01:  # 只显示超过1%的缺失率
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.1%}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "missing_values_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 异常值分析图
        outlier_data = []
        for i, analysis in results['outlier_analysis'].items():
            outlier_data.append({
                'Channel': f"{i}: {analysis['channel_name']}",
                'IQR_Outliers': analysis['iqr_outlier_ratio'],
                'Physical_Outliers': analysis['physical_outlier_ratio']
            })
        
        outlier_df = pd.DataFrame(outlier_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # IQR异常值
        ax1.bar(range(len(outlier_df)), outlier_df['IQR_Outliers'])
        ax1.set_title('IQR-based Outliers by Channel', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Outlier Ratio', fontsize=12)
        ax1.set_xticks(range(len(outlier_df)))
        ax1.set_xticklabels([f"{i}" for i in range(len(outlier_df))], rotation=45)
        
        # 物理异常值
        ax2.bar(range(len(outlier_df)), outlier_df['Physical_Outliers'], color='red', alpha=0.7)
        ax2.set_title('Physical Range Outliers by Channel', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Channel Index', fontsize=12)
        ax2.set_ylabel('Outlier Ratio', fontsize=12)
        ax2.set_xticks(range(len(outlier_df)))
        ax2.set_xticklabels([f"{i}" for i in range(len(outlier_df))], rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "outlier_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_data_quality_report(self, results):
        """保存数据质量报告"""
        
        # 创建缺失值报告
        missing_df = pd.DataFrame([
            {
                'Channel_ID': i,
                'Channel_Name': analysis['channel_name'],
                'Missing_Count': analysis['missing_count'],
                'Missing_Ratio': f"{analysis['missing_ratio']:.2%}",
                'Total_Samples': analysis['total_samples']
            }
            for i, analysis in results['missing_analysis'].items()
        ])
        missing_df.to_csv(self.output_dir / "tables" / "missing_values_report.csv", index=False)
        
        # 创建异常值报告
        outlier_df = pd.DataFrame([
            {
                'Channel_ID': i,
                'Channel_Name': analysis['channel_name'],
                'IQR_Outliers': analysis['iqr_outliers'],
                'IQR_Outlier_Ratio': f"{analysis['iqr_outlier_ratio']:.2%}",
                'Physical_Outliers': analysis['physical_outliers'],
                'Physical_Outlier_Ratio': f"{analysis['physical_outlier_ratio']:.2%}",
                'Value_Range': f"[{analysis['value_range'][0]:.2f}, {analysis['value_range'][1]:.2f}]",
                'Expected_Range': f"[{analysis['expected_range'][0]}, {analysis['expected_range'][1]}]"
            }
            for i, analysis in results['outlier_analysis'].items()
        ])
        outlier_df.to_csv(self.output_dir / "tables" / "outlier_analysis_report.csv", index=False)
    
    def analyze_descriptive_statistics(self):
        """2. 描述性统计分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 2.1 单变量统计特征
        univariate_stats = {}
        for i in range(self.data.shape[1]):
            channel_data = self.data[:, i]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                # 基本统计量
                stats_dict = {
                    'channel_name': self.feature_schema[i]['name'],
                    'count': len(valid_data),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'q25': float(np.percentile(valid_data, 25)),
                    'median': float(np.percentile(valid_data, 50)),
                    'q75': float(np.percentile(valid_data, 75)),
                    'skewness': float(stats.skew(valid_data)),
                    'kurtosis': float(stats.kurtosis(valid_data))
                }
                
                # 正态性检验
                if len(valid_data) > 5000:  # 大样本
                    # Shapiro-Wilk test for smaller samples
                    sample_indices = np.random.choice(len(valid_data), 5000, replace=False)
                    test_data = valid_data[sample_indices]
                else:
                    test_data = valid_data
                
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(test_data)
                    stats_dict['shapiro_stat'] = float(shapiro_stat)
                    stats_dict['shapiro_p'] = float(shapiro_p)
                    stats_dict['is_normal'] = shapiro_p > 0.05
                except:
                    stats_dict['shapiro_stat'] = None
                    stats_dict['shapiro_p'] = None
                    stats_dict['is_normal'] = None
                
                univariate_stats[i] = stats_dict
        
        results['univariate_stats'] = univariate_stats
        
        # 保存结果
        self.results['descriptive_stats'] = results
        
        # 生成可视化
        self._plot_descriptive_statistics(results)
        
        # 保存报告
        self._save_descriptive_statistics_report(results)
        
        print(f"  ✅ 描述性统计分析完成")
        return results
    
    def _plot_descriptive_statistics(self, results):
        """绘制描述性统计图表"""
        
        # 创建大型子图布局
        n_channels = len(results['univariate_stats'])
        n_cols = 4
        n_rows = (n_channels + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, (channel_id, stats_dict) in enumerate(results['univariate_stats'].items()):
            if i >= len(axes):
                break
                
            # 获取有效数据
            channel_data = self.data[:, channel_id]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                # 绘制直方图
                axes[i].hist(valid_data, bins=50, alpha=0.7, density=True)
                axes[i].set_title(f"Ch{channel_id}: {stats_dict['channel_name']}", fontsize=10)
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Density')
                
                # 添加统计信息
                textstr = f"Mean: {stats_dict['mean']:.2f}\nStd: {stats_dict['std']:.2f}\nSkew: {stats_dict['skewness']:.2f}"
                axes[i].text(0.02, 0.98, textstr, transform=axes[i].transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 隐藏多余的子图
        for i in range(len(results['univariate_stats']), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "descriptive_statistics_distributions.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建统计摘要热力图
        stats_matrix = []
        channel_names = []
        
        for channel_id, stats_dict in results['univariate_stats'].items():
            stats_matrix.append([
                stats_dict['mean'],
                stats_dict['std'],
                stats_dict['skewness'],
                stats_dict['kurtosis']
            ])
            channel_names.append(f"Ch{channel_id}: {stats_dict['channel_name'][:15]}")
        
        stats_matrix = np.array(stats_matrix)
        
        # 标准化以便可视化
        stats_matrix_norm = StandardScaler().fit_transform(stats_matrix)
        
        plt.figure(figsize=(8, 15))
        sns.heatmap(stats_matrix_norm, 
                   xticklabels=['Mean', 'Std', 'Skewness', 'Kurtosis'],
                   yticklabels=channel_names,
                   cmap='RdBu_r', center=0, annot=False)
        plt.title('Normalized Statistical Summary Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "statistical_summary_heatmap.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_descriptive_statistics_report(self, results):
        """保存描述性统计报告"""
        
        stats_df = pd.DataFrame([
            {
                'Channel_ID': channel_id,
                'Channel_Name': stats_dict['channel_name'],
                'Count': stats_dict['count'],
                'Mean': f"{stats_dict['mean']:.4f}",
                'Std': f"{stats_dict['std']:.4f}",
                'Min': f"{stats_dict['min']:.4f}",
                'Q25': f"{stats_dict['q25']:.4f}",
                'Median': f"{stats_dict['median']:.4f}",
                'Q75': f"{stats_dict['q75']:.4f}",
                'Max': f"{stats_dict['max']:.4f}",
                'Skewness': f"{stats_dict['skewness']:.4f}",
                'Kurtosis': f"{stats_dict['kurtosis']:.4f}",
                'Is_Normal': stats_dict['is_normal'],
                'Shapiro_P': f"{stats_dict['shapiro_p']:.4f}" if stats_dict['shapiro_p'] else "N/A"
            }
            for channel_id, stats_dict in results['univariate_stats'].items()
        ])
        
        stats_df.to_csv(self.output_dir / "tables" / "descriptive_statistics.csv", index=False)
    
    def analyze_spatiotemporal_patterns(self):
        """3. 时空分布特征分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 3.1 时间维度分析
        temporal_analysis = {}
        
        # 按年份分组分析
        year_groups = self.metadata_df.groupby('year')
        
        for year, group in year_groups:
            year_indices = group.index
            year_data = self.data[year_indices]
            
            # 计算年度统计
            fire_confidence = year_data[:, 22]  # 火点置信度通道
            valid_fire = fire_confidence[np.isfinite(fire_confidence)]
            
            temporal_analysis[year] = {
                'n_samples': len(year_data),
                'fire_mean_confidence': float(np.mean(valid_fire)) if len(valid_fire) > 0 else 0,
                'fire_positive_ratio': float((valid_fire > 0.5).sum() / len(valid_fire)) if len(valid_fire) > 0 else 0,
                'temperature_mean': float(np.nanmean(year_data[:, 5])),  # 温度通道
                'humidity_mean': float(np.nanmean(year_data[:, 6])),     # 湿度通道
                'wind_speed_mean': float(np.nanmean(year_data[:, 8]))    # 风速通道
            }
        
        results['temporal_analysis'] = temporal_analysis
        
        # 3.2 火灾事件分析
        fire_event_analysis = {}
        
        for fire_event in set(self.metadata_df['fire_event']):
            event_mask = self.metadata_df['fire_event'] == fire_event
            event_indices = self.metadata_df[event_mask].index
            event_data = self.data[event_indices]
            
            # 火灾强度分析
            fire_confidence = event_data[:, 22]
            valid_fire = fire_confidence[np.isfinite(fire_confidence)]
            
            if len(valid_fire) > 0:
                fire_event_analysis[fire_event] = {
                    'duration': len(event_data),
                    'max_confidence': float(np.max(valid_fire)),
                    'mean_confidence': float(np.mean(valid_fire)),
                    'fire_area_ratio': float((valid_fire > 0.5).sum() / len(valid_fire)),
                    'intensity_std': float(np.std(valid_fire))
                }
        
        results['fire_event_analysis'] = fire_event_analysis
        
        # 保存结果
        self.results['spatiotemporal'] = results
        
        # 生成可视化
        self._plot_spatiotemporal_patterns(results)
        
        # 保存报告
        self._save_spatiotemporal_report(results)
        
        print(f"  ✅ 时空分布特征分析完成")
        return results
    
    def _plot_spatiotemporal_patterns(self, results):
        """绘制时空分布图表"""
        
        # 1. 年度趋势分析
        years = sorted(results['temporal_analysis'].keys())
        metrics = ['fire_mean_confidence', 'fire_positive_ratio', 'temperature_mean', 'humidity_mean']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results['temporal_analysis'][year][metric] for year in years]
            axes[i].plot(years, values, marker='o', linewidth=2, markersize=8)
            axes[i].set_title(f'Annual {metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "annual_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 火灾事件强度分布
        event_intensities = [analysis['mean_confidence'] 
                           for analysis in results['fire_event_analysis'].values()]
        
        plt.figure(figsize=(12, 6))
        plt.hist(event_intensities, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Fire Event Mean Confidence', fontsize=16, fontweight='bold')
        plt.xlabel('Mean Fire Confidence')
        plt.ylabel('Number of Events')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_intensity = np.mean(event_intensities)
        plt.axvline(mean_intensity, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_intensity:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "fire_event_intensity_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_spatiotemporal_report(self, results):
        """保存时空分析报告"""
        
        # 年度分析报告
        temporal_df = pd.DataFrame([
            {
                'Year': year,
                'Samples': analysis['n_samples'],
                'Fire_Mean_Confidence': f"{analysis['fire_mean_confidence']:.4f}",
                'Fire_Positive_Ratio': f"{analysis['fire_positive_ratio']:.4f}",
                'Temperature_Mean': f"{analysis['temperature_mean']:.2f}",
                'Humidity_Mean': f"{analysis['humidity_mean']:.2f}",
                'Wind_Speed_Mean': f"{analysis['wind_speed_mean']:.2f}"
            }
            for year, analysis in results['temporal_analysis'].items()
        ])
        temporal_df.to_csv(self.output_dir / "tables" / "annual_analysis.csv", index=False)
        
        # 火灾事件分析报告
        fire_events_df = pd.DataFrame([
            {
                'Fire_Event': event,
                'Duration': analysis['duration'],
                'Max_Confidence': f"{analysis['max_confidence']:.4f}",
                'Mean_Confidence': f"{analysis['mean_confidence']:.4f}",
                'Fire_Area_Ratio': f"{analysis['fire_area_ratio']:.4f}",
                'Intensity_Std': f"{analysis['intensity_std']:.4f}"
            }
            for event, analysis in results['fire_event_analysis'].items()
        ])
        fire_events_df.to_csv(self.output_dir / "tables" / "fire_events_analysis.csv", index=False)

# 继续实现其他分析模块...
# (为了避免单个文件过长，我将继续在下一个文件中实现剩余功能) 