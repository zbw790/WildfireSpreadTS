"""
WildfireSpreadTS数据集全面EDA分析系统 - 完整版
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

# 时间序列分析
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

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
    
    def load_sample_data(self, max_files=200, sample_ratio=0.05):
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
    
    def analyze_feature_relationships(self):
        """4. 特征关系与相关性分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 4.1 相关性矩阵分析
        # 计算所有特征的相关性
        valid_data_mask = np.all(np.isfinite(self.data), axis=1)
        clean_data = self.data[valid_data_mask]
        
        if len(clean_data) < 1000:
            # 如果清洁数据太少，使用插值
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            clean_data = imputer.fit_transform(self.data)
        
        # 计算Pearson相关系数
        correlation_matrix = np.corrcoef(clean_data.T)
        
        # 计算Spearman相关系数（对非线性关系更敏感）
        spearman_corr = []
        for i in range(clean_data.shape[1]):
            row_corr = []
            for j in range(clean_data.shape[1]):
                if i == j:
                    row_corr.append(1.0)
                else:
                    corr, _ = stats.spearmanr(clean_data[:, i], clean_data[:, j])
                    row_corr.append(corr if not np.isnan(corr) else 0.0)
            spearman_corr.append(row_corr)
        spearman_matrix = np.array(spearman_corr)
        
        results['correlation_analysis'] = {
            'pearson_matrix': correlation_matrix,
            'spearman_matrix': spearman_matrix,
            'feature_names': [self.feature_schema[i]['name'] for i in range(len(self.feature_schema))]
        }
        
        # 4.2 与目标变量的关系分析
        target_correlations = {}
        target_data = clean_data[:, 22]  # 火点置信度
        
        for i in range(22):  # 排除目标变量本身
            feature_data = clean_data[:, i]
            
            # Pearson相关
            pearson_corr, pearson_p = stats.pearsonr(feature_data, target_data)
            
            # Spearman相关
            spearman_corr, spearman_p = stats.spearmanr(feature_data, target_data)
            
            # 互信息（非线性关系）
            from sklearn.feature_selection import mutual_info_regression
            mi_score = mutual_info_regression(feature_data.reshape(-1, 1), target_data)[0]
            
            target_correlations[i] = {
                'feature_name': self.feature_schema[i]['name'],
                'pearson_corr': float(pearson_corr),
                'pearson_p': float(pearson_p),
                'spearman_corr': float(spearman_corr),
                'spearman_p': float(spearman_p),
                'mutual_info': float(mi_score)
            }
        
        results['target_correlations'] = target_correlations
        
        # 4.3 特征重要性分析
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        
        # 使用随机森林评估特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        X = clean_data[:, :22]  # 特征
        y = target_data  # 目标
        
        rf.fit(X, y)
        
        # 特征重要性
        feature_importance = rf.feature_importances_
        
        # 排列重要性
        perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        
        importance_analysis = {}
        for i in range(22):
            importance_analysis[i] = {
                'feature_name': self.feature_schema[i]['name'],
                'rf_importance': float(feature_importance[i]),
                'perm_importance_mean': float(perm_importance.importances_mean[i]),
                'perm_importance_std': float(perm_importance.importances_std[i])
            }
        
        results['feature_importance'] = importance_analysis
        
        # 保存结果
        self.results['feature_relationships'] = results
        
        # 生成可视化
        self._plot_feature_relationships(results)
        
        # 保存报告
        self._save_feature_relationships_report(results)
        
        print(f"  ✅ 特征关系与相关性分析完成")
        return results
    
    def analyze_target_variable(self):
        """5. 目标变量深度分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 获取目标变量数据
        target_data = self.data[:, 22]  # 火点置信度
        valid_target = target_data[np.isfinite(target_data)]
        
        # 5.1 目标变量分布特征
        distribution_analysis = {
            'total_samples': len(target_data),
            'valid_samples': len(valid_target),
            'missing_ratio': (len(target_data) - len(valid_target)) / len(target_data),
            'mean': float(np.mean(valid_target)),
            'std': float(np.std(valid_target)),
            'min': float(np.min(valid_target)),
            'max': float(np.max(valid_target)),
            'median': float(np.median(valid_target)),
            'q25': float(np.percentile(valid_target, 25)),
            'q75': float(np.percentile(valid_target, 75))
        }
        
        # 5.2 类别不平衡分析
        # 定义火点阈值
        fire_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        imbalance_analysis = {}
        
        for threshold in fire_thresholds:
            fire_pixels = (valid_target >= threshold).sum()
            no_fire_pixels = (valid_target < threshold).sum()
            fire_ratio = fire_pixels / len(valid_target)
            imbalance_ratio = no_fire_pixels / max(fire_pixels, 1)  # 避免除零
            
            imbalance_analysis[threshold] = {
                'fire_pixels': int(fire_pixels),
                'no_fire_pixels': int(no_fire_pixels),
                'fire_ratio': float(fire_ratio),
                'imbalance_ratio': float(imbalance_ratio)
            }
        
        # 5.3 时间序列特征分析
        # 按火灾事件分析目标变量的时间演化
        temporal_patterns = {}
        
        for fire_event in set(self.metadata_df['fire_event']):
            event_mask = self.metadata_df['fire_event'] == fire_event
            event_indices = self.metadata_df[event_mask].index
            event_target = target_data[event_indices]
            valid_event_target = event_target[np.isfinite(event_target)]
            
            if len(valid_event_target) > 0:
                temporal_patterns[fire_event] = {
                    'duration': len(event_target),
                    'max_confidence': float(np.max(valid_event_target)),
                    'mean_confidence': float(np.mean(valid_event_target)),
                    'evolution_trend': self._calculate_trend(valid_event_target),
                    'peak_timing': float(np.argmax(valid_event_target) / len(valid_event_target))
                }
        
        results['distribution_analysis'] = distribution_analysis
        results['imbalance_analysis'] = imbalance_analysis
        results['temporal_patterns'] = temporal_patterns
        
        # 保存结果
        self.results['target_analysis'] = results
        
        # 生成可视化
        self._plot_target_analysis(results)
        
        # 保存报告
        self._save_target_analysis_report(results)
        
        print(f"  ✅ 目标变量深度分析完成")
        return results
    
    def analyze_environmental_variables(self):
        """6. 环境变量专题分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 6.1 气象变量综合分析
        weather_channels = self.channel_groups['weather_historical']
        weather_analysis = {}
        
        for channel_id in weather_channels:
            channel_data = self.data[:, channel_id]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                # 季节性分析（简化版，基于数据分布）
                seasonal_stats = self._analyze_seasonality(valid_data)
                
                weather_analysis[channel_id] = {
                    'channel_name': self.feature_schema[channel_id]['name'],
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'cv': float(np.std(valid_data) / np.mean(valid_data)) if np.mean(valid_data) != 0 else 0,
                    'seasonal_stats': seasonal_stats
                }
        
        # 6.2 植被指数分析
        vegetation_channels = self.channel_groups['vegetation']
        vegetation_analysis = {}
        
        for channel_id in vegetation_channels:
            channel_data = self.data[:, channel_id]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                # 植被健康状态分类
                if 'NDVI' in self.feature_schema[channel_id]['name']:
                    health_categories = self._categorize_vegetation_health(valid_data, 'NDVI')
                else:
                    health_categories = self._categorize_vegetation_health(valid_data, 'EVI2')
                
                vegetation_analysis[channel_id] = {
                    'channel_name': self.feature_schema[channel_id]['name'],
                    'health_categories': health_categories,
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data))
                }
        
        # 6.3 地形因子影响分析
        topography_channels = self.channel_groups['topography']
        topography_analysis = {}
        
        for channel_id in topography_channels:
            channel_data = self.data[:, channel_id]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                # 地形分类分析
                if 'Elevation' in self.feature_schema[channel_id]['name']:
                    topo_categories = self._categorize_elevation(valid_data)
                elif 'Slope' in self.feature_schema[channel_id]['name']:
                    topo_categories = self._categorize_slope(valid_data)
                else:  # Aspect
                    topo_categories = self._categorize_aspect(valid_data)
                
                topography_analysis[channel_id] = {
                    'channel_name': self.feature_schema[channel_id]['name'],
                    'categories': topo_categories,
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data))
                }
        
        results['weather_analysis'] = weather_analysis
        results['vegetation_analysis'] = vegetation_analysis
        results['topography_analysis'] = topography_analysis
        
        # 保存结果
        self.results['environmental_analysis'] = results
        
        # 生成可视化
        self._plot_environmental_analysis(results)
        
        # 保存报告
        self._save_environmental_analysis_report(results)
        
        print(f"  ✅ 环境变量专题分析完成")
        return results
    
    def analyze_preprocessing_requirements(self):
        """7. 数据预处理需求分析"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 7.1 数据标准化需求分析
        normalization_analysis = {}
        
        for i in range(self.data.shape[1]):
            channel_data = self.data[:, i]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                # 分析数据分布特征
                skewness = stats.skew(valid_data)
                kurtosis = stats.kurtosis(valid_data)
                
                # 推荐标准化方法
                if abs(skewness) > 2:
                    if skewness > 0:
                        recommended_transform = "Log transform or Box-Cox"
                    else:
                        recommended_transform = "Square or inverse transform"
                elif abs(skewness) > 1:
                    recommended_transform = "StandardScaler or RobustScaler"
                else:
                    recommended_transform = "StandardScaler or MinMaxScaler"
                
                normalization_analysis[i] = {
                    'channel_name': self.feature_schema[i]['name'],
                    'skewness': float(skewness),
                    'kurtosis': float(kurtosis),
                    'recommended_transform': recommended_transform,
                    'range': [float(np.min(valid_data)), float(np.max(valid_data))],
                    'needs_log_transform': abs(skewness) > 2 and skewness > 0
                }
        
        # 7.2 异常值处理建议
        outlier_handling = {}
        
        for i in range(self.data.shape[1]):
            channel_data = self.data[:, i]
            valid_data = channel_data[np.isfinite(channel_data)]
            
            if len(valid_data) > 0:
                # 计算异常值比例
                Q1 = np.percentile(valid_data, 25)
                Q3 = np.percentile(valid_data, 75)
                IQR = Q3 - Q1
                outliers = ((valid_data < Q1 - 1.5 * IQR) | 
                           (valid_data > Q3 + 1.5 * IQR)).sum()
                outlier_ratio = outliers / len(valid_data)
                
                # 建议处理方法
                if outlier_ratio > 0.1:
                    handling_method = "Robust methods (clip or transform)"
                elif outlier_ratio > 0.05:
                    handling_method = "Careful inspection and selective removal"
                else:
                    handling_method = "Standard methods acceptable"
                
                outlier_handling[i] = {
                    'channel_name': self.feature_schema[i]['name'],
                    'outlier_ratio': float(outlier_ratio),
                    'handling_method': handling_method
                }
        
        # 7.3 类别不平衡处理建议
        target_data = self.data[:, 22]
        valid_target = target_data[np.isfinite(target_data)]
        
        # 分析不同阈值下的不平衡程度
        imbalance_strategies = {}
        for threshold in [0.1, 0.3, 0.5, 0.7]:
            positive_ratio = (valid_target >= threshold).sum() / len(valid_target)
            imbalance_ratio = (1 - positive_ratio) / max(positive_ratio, 0.001)
            
            if imbalance_ratio > 100:
                strategy = "SMOTE + Undersampling + Focal Loss"
            elif imbalance_ratio > 50:
                strategy = "Weighted sampling + Focal Loss"
            elif imbalance_ratio > 10:
                strategy = "Weighted loss functions"
            else:
                strategy = "Standard methods"
            
            imbalance_strategies[threshold] = {
                'positive_ratio': float(positive_ratio),
                'imbalance_ratio': float(imbalance_ratio),
                'recommended_strategy': strategy
            }
        
        results['normalization_analysis'] = normalization_analysis
        results['outlier_handling'] = outlier_handling
        results['imbalance_strategies'] = imbalance_strategies
        
        # 保存结果
        self.results['preprocessing_requirements'] = results
        
        # 生成报告
        self._save_preprocessing_report(results)
        
        print(f"  ✅ 数据预处理需求分析完成")
        return results
    
    def create_advanced_visualizations(self):
        """8. 高级可视化与洞察发现"""
        if not hasattr(self, 'data'):
            self.load_sample_data()
        
        results = {}
        
        # 8.1 多维特征空间可视化
        print("    生成PCA和t-SNE可视化...")
        self._create_dimensionality_reduction_viz()
        
        # 8.2 特征交互作用可视化
        print("    生成特征交互作用图...")
        self._create_feature_interaction_viz()
        
        # 8.3 时空模式可视化
        print("    生成时空模式图...")
        self._create_spatiotemporal_viz()
        
        # 8.4 环境条件vs火灾强度热力图
        print("    生成环境条件热力图...")
        self._create_environmental_heatmaps()
        
        results['visualizations_created'] = [
            'pca_tsne_scatter.png',
            'feature_interactions.png',
            'spatiotemporal_patterns.png',
            'environmental_heatmaps.png'
        ]
        
        self.results['advanced_visualizations'] = results
        
        print(f"  ✅ 高级可视化与洞察发现完成")
        return results
    
    def generate_academic_report(self):
        """9. 生成学术报告"""
        if not hasattr(self, 'results') or not self.results:
            print("请先运行完整的EDA分析")
            return
        
        report_content = self._generate_comprehensive_report()
        
        # 保存报告
        report_path = self.output_dir / "reports" / "comprehensive_eda_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"  ✅ 学术报告生成完成: {report_path}")
        return report_content
    
    def run_complete_analysis(self):
        """运行完整的EDA分析"""
        print("🚀 开始全面EDA分析...")
        
        try:
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
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()


# 主函数
def main():
    """主函数 - 运行完整的EDA分析"""
    print("🔥 WildfireSpreadTS数据集全面EDA分析系统")
    print("=" * 80)
    print("专为解决NaN值问题和论文写作设计的专业级分析系统")
    print("=" * 80)
    
    try:
        # 创建分析器
        analyzer = WildfireEDAAnalyzer(
            data_dir="data/processed",
            output_dir="professional_eda_results"
        )
        
        # 用户选择处理数据量
        print("\n📋 选择数据处理量:")
        print("1. 快速分析 (50个文件)")
        print("2. 中等分析 (200个文件)")  
        print("3. 完整分析 (全部607个文件)")
        print("4. 自定义数量")
        
        try:
            choice = input("请选择 (1-4, 默认2): ").strip()
            if choice == "1":
                max_files, sample_ratio = 50, 0.08
            elif choice == "3":
                max_files, sample_ratio = 607, 0.02  # 全部文件，降低采样比例
            elif choice == "4":
                max_files = int(input("请输入文件数量: "))
                sample_ratio = float(input("请输入采样比例 (0.01-0.1): "))
            else:  # 默认选择2
                max_files, sample_ratio = 200, 0.05
        except:
            max_files, sample_ratio = 200, 0.05  # 默认值
        
        print(f"\n🔄 将处理 {max_files} 个文件，每文件采样比例 {sample_ratio}")
        
        # 手动加载数据（覆盖默认参数）
        analyzer.load_sample_data(max_files=max_files, sample_ratio=sample_ratio)
        
        # 运行完整分析
        print("\n🚀 开始全面EDA分析...")
        analyzer.run_complete_analysis()
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()