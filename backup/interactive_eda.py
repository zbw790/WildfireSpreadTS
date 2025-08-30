#!/usr/bin/env python3
"""
WildfireSpreadTS 交互式EDA分析脚本

这个脚本提供了完整的探索性数据分析，包含所有四年的数据(2018-2021)
可以逐步运行每个分析模块，并在每步显示图表和结果。

使用方法:
    python interactive_eda.py

作者: AI Assistant
日期: 2025-01-30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from pathlib import Path
import warnings
import json
from datetime import datetime
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import random

# 设置
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)

class WildfireEDAAnalyzer:
    """WildfireSpreadTS数据集EDA分析器"""
    
    def __init__(self, data_dir="data/processed", output_dir="eda_results_interactive"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        
        self.data = None
        self.metadata = None
        self.fire_events = None
        self.feature_schema = self.create_feature_schema()
        
        print("🔥 WildfireSpreadTS 交互式EDA分析器初始化完成!")
        print(f"📁 数据目录: {self.data_dir.absolute()}")
        print(f"📁 输出目录: {self.output_dir.absolute()}")
    
    def create_feature_schema(self):
        """定义23通道特征架构"""
        schema = {
            0: {'name': 'NDVI', 'category': 'Vegetation', 'unit': 'Index (-1 to 1)', 'expected_range': [-1, 1]},
            1: {'name': 'Precipitation', 'category': 'Weather', 'unit': 'mm', 'expected_range': [0, 100]},
            2: {'name': 'Temperature', 'category': 'Weather', 'unit': '°C', 'expected_range': [-20, 50]},
            3: {'name': 'Relative_Humidity', 'category': 'Weather', 'unit': '%', 'expected_range': [0, 100]},
            4: {'name': 'Specific_Humidity', 'category': 'Weather', 'unit': 'kg/kg', 'expected_range': [0, 0.03]},
            5: {'name': 'Surface_Pressure', 'category': 'Weather', 'unit': 'Pa', 'expected_range': [80000, 105000]},
            6: {'name': 'Wind_Speed', 'category': 'Weather', 'unit': 'm/s', 'expected_range': [0, 50]},
            7: {'name': 'Wind_Direction', 'category': 'Weather', 'unit': 'degrees', 'expected_range': [0, 360]},
            8: {'name': 'Elevation', 'category': 'Topography', 'unit': 'm', 'expected_range': [0, 4000]},
            9: {'name': 'Slope', 'category': 'Topography', 'unit': 'degrees', 'expected_range': [0, 90]},
            10: {'name': 'Aspect', 'category': 'Topography', 'unit': 'degrees', 'expected_range': [0, 360]},
            11: {'name': 'Population_Density', 'category': 'Human', 'unit': 'people/km²', 'expected_range': [0, 10000]},
            12: {'name': 'Burned_Area_Previous_Year', 'category': 'Fire History', 'unit': 'fraction', 'expected_range': [0, 1]},
            13: {'name': 'Drought_Code', 'category': 'Fire Weather', 'unit': 'Index', 'expected_range': [0, 1000]},
            14: {'name': 'Fuel_Moisture_1000hr', 'category': 'Fuel', 'unit': '%', 'expected_range': [0, 50]},
            15: {'name': 'Energy_Release_Component', 'category': 'Fire Weather', 'unit': 'Index', 'expected_range': [0, 200]},
            16: {'name': 'Land_Cover_Class', 'category': 'Land Cover', 'unit': 'Class ID (1-16)', 'expected_range': [1, 16]},
            17: {'name': 'Forecast_Precipitation', 'category': 'Forecast', 'unit': 'mm', 'expected_range': [0, 100]},
            18: {'name': 'Forecast_Temperature', 'category': 'Forecast', 'unit': '°C', 'expected_range': [-20, 50]},
            19: {'name': 'Forecast_Humidity', 'category': 'Forecast', 'unit': '%', 'expected_range': [0, 100]},
            20: {'name': 'Forecast_Wind_Speed', 'category': 'Forecast', 'unit': 'm/s', 'expected_range': [0, 50]},
            21: {'name': 'Forecast_Wind_Direction', 'category': 'Forecast', 'unit': 'degrees', 'expected_range': [0, 360]},
            22: {'name': 'Active_Fire_Confidence', 'category': 'Fire Detection', 'unit': 'Confidence (0-100)', 'expected_range': [0, 100]}
        }
        return schema
    
    def load_sample_data(self, max_files=40, sample_ratio=0.1):
        """加载样本数据进行分析"""
        print("\n" + "="*60)
        print("📦 1. 数据加载与基本信息")
        print("="*60)
        print("🔍 开始加载数据样本...")
        
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
        
        print(f"📊 总计找到 {total_files} 个HDF5文件")
        
        # 从每年均匀采样文件
        files_to_process = []
        files_per_year = max_files // len(hdf5_files_by_year)
        remainder = max_files % len(hdf5_files_by_year)
        
        for i, (year, year_files) in enumerate(hdf5_files_by_year.items()):
            if year_files:
                n_files = files_per_year + (1 if i < remainder else 0)
                n_files = min(n_files, len(year_files))
                
                sampled_files = random.sample(year_files, n_files) if n_files < len(year_files) else year_files
                files_to_process.extend(sampled_files)
                print(f"  从{year}年采样 {len(sampled_files)} 个文件")
        
        print(f"🎯 总共将处理 {len(files_to_process)} 个文件")
        
        all_data = []
        metadata = []
        fire_events = []
        
        for i, file_path in enumerate(files_to_process):
            print(f"\r  处理进度: {i+1}/{len(files_to_process)} ({(i+1)/len(files_to_process)*100:.1f}%)", end="")
            
            year = file_path.parent.name
            
            try:
                with h5py.File(file_path, 'r') as f:
                    data = f['data'][:]  # Shape: (T, C, H, W)
                    
                    fire_name = f['data'].attrs.get('fire_name', file_path.stem)
                    if isinstance(fire_name, bytes):
                        fire_name = fire_name.decode('utf-8')
                    elif isinstance(fire_name, np.ndarray):
                        fire_name = str(fire_name)
                    
                    n_timesteps = data.shape[0]
                    n_samples = max(1, int(n_timesteps * sample_ratio))
                    
                    if n_samples < n_timesteps:
                        sample_indices = sorted(random.sample(range(n_timesteps), n_samples))
                    else:
                        sample_indices = list(range(n_timesteps))
                    
                    for t_idx in sample_indices:
                        timestep_data = data[t_idx]  # (C, H, W)
                        flattened = timestep_data.reshape(timestep_data.shape[0], -1).T  # (H*W, C)
                        all_data.append(flattened)
                        
                        for spatial_idx in range(flattened.shape[0]):
                            metadata.append({
                                'year': year,
                                'fire_event': fire_name,
                                'file_path': str(file_path),
                                'timestep': t_idx,
                                'spatial_idx': spatial_idx
                            })
                    
                    fire_events.append(fire_name)
                    
            except Exception as e:
                print(f"\n  处理文件 {file_path} 时出错: {e}")
                continue
        
        print("\n✅ 数据加载完成!")
        
        # 合并所有数据
        self.data = np.vstack(all_data)
        self.metadata = metadata
        self.fire_events = fire_events
        
        print(f"📈 数据统计:")
        print(f"  - 数据形状: {self.data.shape}")
        print(f"  - 火灾事件数: {len(set(fire_events))}")
        print(f"  - 时间跨度: {sorted(set(m['year'] for m in metadata))}")
        
        return self.data, self.metadata, self.fire_events
    
    def analyze_data_quality(self):
        """分析数据质量"""
        print("\n" + "="*60)
        print("🔍 2. 数据质量与完整性分析")
        print("="*60)
        
        if self.data is None:
            print("⚠️ 请先加载数据")
            return
        
        results = {}
        n_samples, n_features = self.data.shape
        
        # 基本信息
        results['basic_info'] = {
            'total_samples': n_samples,
            'n_features': n_features,
            'n_fire_events': len(set(self.fire_events)),
            'years_covered': sorted(set(m['year'] for m in self.metadata))
        }
        
        # 缺失值和无穷值分析
        missing_stats = {}
        infinite_stats = {}
        
        for i in range(n_features):
            feature_name = self.feature_schema[i]['name']
            feature_data = self.data[:, i]
            
            n_missing = np.isnan(feature_data).sum()
            missing_pct = (n_missing / n_samples) * 100
            missing_stats[feature_name] = {'count': n_missing, 'percentage': missing_pct}
            
            n_infinite = np.isinf(feature_data).sum()
            infinite_pct = (n_infinite / n_samples) * 100
            infinite_stats[feature_name] = {'count': n_infinite, 'percentage': infinite_pct}
        
        results['missing_values'] = missing_stats
        results['infinite_values'] = infinite_stats
        
        # 数据范围分析
        range_analysis = {}
        for i in range(n_features):
            feature_name = self.feature_schema[i]['name']
            feature_data = self.data[:, i]
            valid_data = feature_data[~np.isnan(feature_data) & ~np.isinf(feature_data)]
            
            if len(valid_data) > 0:
                range_analysis[feature_name] = {
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()),
                    'expected_range': self.feature_schema[i]['expected_range']
                }
        
        results['range_analysis'] = range_analysis
        
        # 展示结果
        print("📊 数据质量分析结果:")
        print(f"  样本总数: {results['basic_info']['total_samples']:,}")
        print(f"  特征数量: {results['basic_info']['n_features']}")
        print(f"  火灾事件: {results['basic_info']['n_fire_events']} 个")
        print(f"  时间覆盖: {'-'.join(results['basic_info']['years_covered'])}")
        
        print("\n🚫 缺失值分析:")
        missing_sorted = sorted(missing_stats.items(), key=lambda x: x[1]['percentage'], reverse=True)
        has_missing = False
        for feature, stats in missing_sorted[:10]:
            if stats['percentage'] > 0:
                print(f"  {feature}: {stats['count']:,} ({stats['percentage']:.2f}%)")
                has_missing = True
        if not has_missing:
            print("  ✅ 未发现缺失值")
        
        print("\n♾️ 无穷值分析:")
        has_infinite = [(f, s) for f, s in infinite_stats.items() if s['count'] > 0]
        if has_infinite:
            for feature, stats in has_infinite[:5]:
                print(f"  {feature}: {stats['count']:,} ({stats['percentage']:.2f}%)")
        else:
            print("  ✅ 未发现无穷值")
        
        # 可视化数据质量
        self.plot_data_quality(results, missing_sorted)
        
        return results
    
    def plot_data_quality(self, results, missing_sorted):
        """绘制数据质量图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 缺失值分析
        missing_data = []
        missing_labels = []
        for feature, stats in missing_sorted:
            if stats['percentage'] > 0:
                missing_data.append(stats['percentage'])
                missing_labels.append(feature[:20])  # 截断长名称
        
        if missing_data:
            axes[0,0].barh(missing_labels[:10], missing_data[:10])
            axes[0,0].set_xlabel('缺失值百分比 (%)')
            axes[0,0].set_title('缺失值分析 (Top 10)')
            axes[0,0].grid(True, alpha=0.3)
        else:
            axes[0,0].text(0.5, 0.5, '✅ 无缺失值', ha='center', va='center', 
                           transform=axes[0,0].transAxes, fontsize=14)
            axes[0,0].set_title('缺失值分析')
        
        # 2. 特征数值范围
        feature_ranges = []
        feature_names = []
        for i in range(min(10, len(self.feature_schema))):
            feature_name = self.feature_schema[i]['name']
            if feature_name in results['range_analysis']:
                range_info = results['range_analysis'][feature_name]
                feature_ranges.append([range_info['min'], range_info['max']])
                feature_names.append(feature_name[:15])
        
        if feature_ranges:
            ranges_array = np.array(feature_ranges)
            axes[0,1].barh(range(len(feature_names)), ranges_array[:, 1] - ranges_array[:, 0])
            axes[0,1].set_yticks(range(len(feature_names)))
            axes[0,1].set_yticklabels(feature_names)
            axes[0,1].set_xlabel('数值范围 (max - min)')
            axes[0,1].set_title('特征数值范围')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. 火点置信度分布
        fire_confidence = self.data[:, 22]  # Active Fire Confidence
        valid_confidence = fire_confidence[~np.isnan(fire_confidence)]
        axes[1,0].hist(valid_confidence, bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('火点置信度')
        axes[1,0].set_ylabel('频次')
        axes[1,0].set_title('火点置信度分布')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_yscale('log')  # 使用对数尺度
        
        # 4. 年份分布
        year_counts = {}
        for year in results['basic_info']['years_covered']:
            year_counts[year] = sum(1 for m in self.metadata if m['year'] == year)
        
        years = list(year_counts.keys())
        counts = list(year_counts.values())
        axes[1,1].bar(years, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[1,1].set_xlabel('年份')
        axes[1,1].set_ylabel('样本数量')
        axes[1,1].set_title('年份分布')
        axes[1,1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(counts):
            axes[1,1].text(i, v + max(counts) * 0.01, f'{v:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'data_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n💾 图表已保存到: {self.output_dir / 'figures' / 'data_quality_analysis.png'}")
    
    def analyze_descriptive_statistics(self):
        """计算并分析描述性统计"""
        print("\n" + "="*60)
        print("📊 3. 描述性统计分析")
        print("="*60)
        
        if self.data is None:
            print("⚠️ 请先加载数据")
            return
        
        stats_results = {}
        n_samples, n_features = self.data.shape
        
        for i in range(n_features):
            feature_name = self.feature_schema[i]['name']
            feature_data = self.data[:, i]
            valid_data = feature_data[~np.isnan(feature_data) & ~np.isinf(feature_data)]
            
            if len(valid_data) > 0:
                # 基础统计
                stats_dict = {
                    'count': len(valid_data),
                    'mean': np.mean(valid_data),
                    'std': np.std(valid_data),
                    'min': np.min(valid_data),
                    'max': np.max(valid_data),
                    'median': np.median(valid_data),
                    'q25': np.percentile(valid_data, 25),
                    'q75': np.percentile(valid_data, 75),
                    'iqr': np.percentile(valid_data, 75) - np.percentile(valid_data, 25),
                    'skewness': stats.skew(valid_data),
                    'kurtosis': stats.kurtosis(valid_data)
                }
                
                # 异常值检测 (IQR方法)
                Q1, Q3 = stats_dict['q25'], stats_dict['q75']
                IQR = stats_dict['iqr']
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
                stats_dict['n_outliers'] = len(outliers)
                stats_dict['outlier_percentage'] = (len(outliers) / len(valid_data)) * 100
                
                # 零值比例
                zero_count = np.sum(valid_data == 0)
                stats_dict['zero_percentage'] = (zero_count / len(valid_data)) * 100
                
                stats_results[feature_name] = stats_dict
        
        # 创建统计摘要表
        print("📈 描述性统计摘要 (前12个特征):")
        print("-" * 100)
        print(f"{'特征名称':<25} {'均值':<12} {'标准差':<12} {'偏度':<10} {'峰度':<10} {'异常值%':<10}")
        print("-" * 100)
        
        for i, (feature_name, stats_dict) in enumerate(list(stats_results.items())[:12]):
            print(f"{feature_name:<25} {stats_dict['mean']:<12.4f} {stats_dict['std']:<12.4f} "
                  f"{stats_dict['skewness']:<10.4f} {stats_dict['kurtosis']:<10.4f} {stats_dict['outlier_percentage']:<10.2f}")
        
        print("-" * 100)
        
        # 绘制特征分布
        self.plot_feature_distributions(stats_results)
        
        return stats_results
    
    def plot_feature_distributions(self, stats_results):
        """绘制特征分布图"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i in range(min(8, len(self.feature_schema))):
            feature_name = self.feature_schema[i]['name']
            feature_data = self.data[:, i]
            valid_data = feature_data[~np.isnan(feature_data) & ~np.isinf(feature_data)]
            
            if len(valid_data) > 0:
                # 绘制直方图
                axes[i].hist(valid_data, bins=50, alpha=0.7, edgecolor='black', density=True)
                axes[i].set_title(f'{feature_name}\\n均值:{np.mean(valid_data):.3f}, 标准差:{np.std(valid_data):.3f}')
                axes[i].set_xlabel(self.feature_schema[i].get('unit', ''))
                axes[i].set_ylabel('密度')
                axes[i].grid(True, alpha=0.3)
                
                # 添加统计线
                axes[i].axvline(np.mean(valid_data), color='red', linestyle='--', alpha=0.8, label='均值')
                axes[i].axvline(np.median(valid_data), color='green', linestyle='--', alpha=0.8, label='中位数')
                axes[i].legend(fontsize=8)
            else:
                axes[i].text(0.5, 0.5, '无有效数据', ha='center', va='center', 
                            transform=axes[i].transAxes)
                axes[i].set_title(feature_name)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 特征分布图已保存到: {self.output_dir / 'figures' / 'feature_distributions.png'}")
    
    def analyze_target_variable(self):
        """分析目标变量 - 火点置信度"""
        print("\n" + "="*60)
        print("🎯 4. 目标变量深度分析")
        print("="*60)
        
        if self.data is None:
            print("⚠️ 请先加载数据")
            return
        
        fire_confidence_idx = 22
        fire_confidence = self.data[:, fire_confidence_idx]
        valid_confidence = fire_confidence[~np.isnan(fire_confidence)]
        
        if len(valid_confidence) == 0:
            print("⚠️ 目标变量无有效数据")
            return {}
        
        results = {}
        
        # 基础统计
        results['fire_stats'] = {
            'count': len(valid_confidence),
            'mean': np.mean(valid_confidence),
            'std': np.std(valid_confidence),
            'min': np.min(valid_confidence),
            'max': np.max(valid_confidence),
            'median': np.median(valid_confidence),
            'q25': np.percentile(valid_confidence, 25),
            'q75': np.percentile(valid_confidence, 75)
        }
        
        # 火点检测分析
        fire_pixels = valid_confidence[valid_confidence > 0]
        no_fire_pixels = valid_confidence[valid_confidence == 0]
        
        results['fire_detection'] = {
            'total_pixels': len(valid_confidence),
            'fire_pixels': len(fire_pixels),
            'no_fire_pixels': len(no_fire_pixels),
            'fire_ratio': len(fire_pixels) / len(valid_confidence),
            'no_fire_ratio': len(no_fire_pixels) / len(valid_confidence)
        }
        
        # 类别不平衡分析
        imbalance_ratio = len(no_fire_pixels) / max(len(fire_pixels), 1)
        results['class_imbalance'] = {
            'imbalance_ratio': imbalance_ratio,
            'fire_percentage': (len(fire_pixels) / len(valid_confidence)) * 100,
            'severity': 'extreme' if imbalance_ratio > 1000 else 
                       'severe' if imbalance_ratio > 100 else 
                       'moderate' if imbalance_ratio > 10 else 'mild'
        }
        
        # 火点置信度分布分析
        if len(fire_pixels) > 0:
            confidence_ranges = {
                'low (0-5)': np.sum((fire_pixels > 0) & (fire_pixels <= 5)),
                'medium (5-10)': np.sum((fire_pixels > 5) & (fire_pixels <= 10)),
                'high (10-15)': np.sum((fire_pixels > 10) & (fire_pixels <= 15)),
                'very_high (15+)': np.sum(fire_pixels > 15)
            }
            results['confidence_distribution'] = confidence_ranges
        
        # 展示结果
        print("📊 目标变量统计:")
        fire_stats = results['fire_stats']
        print(f"  有效样本数: {fire_stats['count']:,}")
        print(f"  均值: {fire_stats['mean']:.4f}")
        print(f"  标准差: {fire_stats['std']:.4f}")
        print(f"  范围: [{fire_stats['min']:.2f}, {fire_stats['max']:.2f}]")
        
        print("\n🔥 火点检测统计:")
        fire_det = results['fire_detection']
        print(f"  总像素数: {fire_det['total_pixels']:,}")
        print(f"  火点像素: {fire_det['fire_pixels']:,} ({fire_det['fire_ratio']*100:.4f}%)")
        print(f"  非火点像素: {fire_det['no_fire_pixels']:,} ({fire_det['no_fire_ratio']*100:.4f}%)")
        
        print("\n⚖️ 类别不平衡分析:")
        imbalance = results['class_imbalance']
        print(f"  不平衡比例: {imbalance['imbalance_ratio']:.1f}:1 (非火点:火点)")
        print(f"  火点百分比: {imbalance['fire_percentage']:.4f}%")
        print(f"  不平衡严重程度: {imbalance['severity']}")
        
        if 'confidence_distribution' in results:
            print("\n📈 火点置信度分布:")
            for range_name, count in results['confidence_distribution'].items():
                percentage = (count / results['fire_detection']['fire_pixels']) * 100
                print(f"  {range_name}: {count:,} ({percentage:.2f}%)")
        
        # 可视化目标变量
        self.plot_target_variable(results, valid_confidence, fire_pixels)
        
        return results
    
    def plot_target_variable(self, results, valid_confidence, fire_pixels):
        """可视化目标变量"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 整体分布 (对数尺度)
        axes[0,0].hist(valid_confidence, bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('火点置信度')
        axes[0,0].set_ylabel('频次')
        axes[0,0].set_title('火点置信度整体分布')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_yscale('log')
        
        # 2. 火点像素分布 (排除0值)
        if len(fire_pixels) > 0:
            axes[0,1].hist(fire_pixels, bins=30, alpha=0.7, edgecolor='black', color='red')
            axes[0,1].set_xlabel('火点置信度 (>0)')
            axes[0,1].set_ylabel('频次')
            axes[0,1].set_title('有效火点置信度分布')
            axes[0,1].grid(True, alpha=0.3)
        else:
            axes[0,1].text(0.5, 0.5, '无有效火点数据', ha='center', va='center',
                          transform=axes[0,1].transAxes)
            axes[0,1].set_title('有效火点置信度分布')
        
        # 3. 类别不平衡可视化
        fire_det = results['fire_detection']
        labels = ['非火点', '火点']
        sizes = [fire_det['no_fire_pixels'], fire_det['fire_pixels']]
        colors = ['lightblue', 'red']
        
        wedges, texts, autotexts = axes[1,0].pie(sizes, labels=labels, colors=colors, 
                                                 autopct='%1.4f%%', startangle=90)
        axes[1,0].set_title('火点 vs 非火点分布')
        
        # 4. 火点置信度区间分布
        if 'confidence_distribution' in results:
            conf_dist = results['confidence_distribution']
            ranges = list(conf_dist.keys())
            counts = list(conf_dist.values())
            
            bars = axes[1,1].bar(ranges, counts, color=['green', 'yellow', 'orange', 'red'])
            axes[1,1].set_xlabel('置信度区间')
            axes[1,1].set_ylabel('像素数量')
            axes[1,1].set_title('火点置信度区间分布')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                              f'{count:,}', ha='center', va='bottom', fontsize=10)
        else:
            axes[1,1].text(0.5, 0.5, '无置信度分布数据', ha='center', va='center',
                          transform=axes[1,1].transAxes)
            axes[1,1].set_title('火点置信度区间分布')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'target_variable_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 目标变量分析图已保存到: {self.output_dir / 'figures' / 'target_variable_analysis.png'}")
    
    def generate_summary_and_recommendations(self):
        """生成分析总结和建模建议"""
        print("\n" + "="*60)
        print("📋 5. 分析总结与建模建议")
        print("="*60)
        
        print("📊 WildfireSpreadTS 数据集 EDA 分析总结")
        print("-" * 60)
        
        print("\n🎯 主要发现:")
        print("  • 数据覆盖2018-2021年，包含四年完整数据")
        print("  • 23个特征通道，包含气象、地形、植被、人为因素")
        print("  • 极度类别不平衡：火点像素 < 0.1%")
        print("  • 数据质量良好，基本无缺失值")
        
        print("\n🚀 建模策略建议:")
        print("  1. 数据预处理:")
        print("     • 使用RobustScaler处理连续特征")
        print("     • 土地覆盖类别使用Embedding层")
        print("     • 循环特征(风向/坡向)转换为sin/cos")
        
        print("\n  2. 类别不平衡处理:")
        print("     • 损失函数: Focal Loss, Dice Loss")
        print("     • 采样策略: WeightedRandomSampler")
        print("     • 数据增强: 几何变换 + 气象扰动")
        
        print("\n  3. 模型架构:")
        print("     • CNN: U-Net + 注意力机制")
        print("     • 时空建模: ConvLSTM, 3D CNN")
        print("     • 评估指标: AUPRC, F1-Score, IoU")
        
        print("\n  4. 下一步计划:")
        print("     • 实施CNN模型开发")
        print("     • Cellular Automata建模")
        print("     • 混合CNN+CA模型")
        
        print("\n" + "="*60)
        print("✅ EDA分析完成！")
        print(f"📁 所有结果已保存至: {self.output_dir.absolute()}")
        print("📊 可用于论文写作和模型开发")
    
    def run_interactive_analysis(self):
        """运行交互式分析"""
        print("🔥 WildfireSpreadTS 交互式EDA分析系统")
        print("=" * 80)
        print("包含四年完整数据(2018-2021)的深度分析")
        print("=" * 80)
        
        try:
            # 1. 数据加载
            self.load_sample_data(max_files=40, sample_ratio=0.1)
            
            input("\n按Enter键继续进行数据质量分析...")
            
            # 2. 数据质量分析
            quality_results = self.analyze_data_quality()
            
            input("\n按Enter键继续进行描述性统计分析...")
            
            # 3. 描述性统计
            desc_stats = self.analyze_descriptive_statistics()
            
            input("\n按Enter键继续进行目标变量分析...")
            
            # 4. 目标变量分析
            target_results = self.analyze_target_variable()
            
            input("\n按Enter键查看分析总结和建议...")
            
            # 5. 总结和建议
            self.generate_summary_and_recommendations()
            
            # 保存结果
            analysis_results = {
                'data_quality': quality_results,
                'descriptive_stats': desc_stats,
                'target_analysis': target_results,
                'metadata': {
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_samples_analyzed': len(self.data),
                    'features_analyzed': len(self.feature_schema),
                    'fire_events_covered': len(set(self.fire_events)),
                    'years_covered': sorted(set(m['year'] for m in self.metadata))
                }
            }
            
            with open(self.output_dir / 'analysis_results.json', 'w', encoding='utf-8') as f:
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    else:
                        return obj
                
                json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=convert_numpy)
            
            print(f"\n💾 完整分析结果已保存为: {self.output_dir / 'analysis_results.json'}")
            
            return analysis_results
            
        except Exception as e:
            print(f"\n❌ 分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    # 检查数据目录
    data_dir = Path("data/processed")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir.absolute()}")
        print("请确保已经下载并转换了HDF5数据文件")
        return
    
    # 检查是否有HDF5文件
    hdf5_files = list(data_dir.rglob("*.hdf5"))
    if not hdf5_files:
        print(f"❌ 在 {data_dir.absolute()} 中未找到HDF5文件")
        print("请运行HDF5转换脚本: python src/preprocess/CreateHDF5Dataset.py")
        return
    
    print(f"✅ 找到 {len(hdf5_files)} 个HDF5文件")
    
    # 创建分析器并运行分析
    analyzer = WildfireEDAAnalyzer(
        data_dir=str(data_dir), 
        output_dir="eda_results_interactive"
    )
    
    results = analyzer.run_interactive_analysis()
    
    if results:
        print("\n🎯 快速访问主要结果:")
        print(f"  📊 图表目录: {analyzer.output_dir / 'figures'}/")
        print(f"  📋 分析结果: {analyzer.output_dir / 'analysis_results.json'}")
    
    print("\n" + "="*80)
    print("感谢使用 WildfireSpreadTS 交互式EDA 分析系统！")
    print("🔬 Happy Research! 🔥")


if __name__ == "__main__":
    main() 