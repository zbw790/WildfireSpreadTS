import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import glob
from multiprocessing import Pool, cpu_count
import gc
import psutil
import time
from datetime import datetime
import json

# --- 流式处理全局设置 ---
OUTPUT_DIR = 'eda_outputs_streaming'
YEARS_TO_ANALYZE = [2018, 2019, 2020, 2021]  

# 文件处理设置
FILE_LIMIT_PER_YEAR = None        # 处理所有文件
BATCH_SIZE_PER_YEAR = 5           # 每批文件数

# 进程设置
MAX_WORKERS = min(12, max(1, cpu_count() - 2))  # 保持你的设置

# 流式处理设置 - 关键改变！
MAX_SAMPLE_SIZE = 100000          # 只保留10万行样本用于可视化
STATS_UPDATE_FREQUENCY = 1000     # 每1000行更新一次统计

FEATURE_NAMES = [
    'VIIRS_M11', 'VIIRS_I2', 'VIIRS_I1', 'NDVI', 'EVI2',
    'Total_Precip', 'Wind_Speed', 'Wind_Direction', 'Min_Temp_K', 'Max_Temp_K',
    'ERC', 'Spec_Hum', 'PDSI', 'Slope', 'Aspect',
    'Elevation', 'Landcover', 'Forecast_Precip', 'Forecast_Wind_Speed',
    'Forecast_Wind_Dir', 'Forecast_Temp_C', 'Forecast_Spec_Hum', 'Active_Fire'
]

def get_memory_usage_gb():
    """获取当前内存使用量（GB）"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024**3)
    except:
        return 0

def log_status(message, output_dir=None):
    """记录状态信息"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    memory_gb = get_memory_usage_gb()
    status_msg = f"[{timestamp}] {message} | Memory: {memory_gb:.1f}GB"
    print(status_msg)
    
    if output_dir:
        log_file = os.path.join(output_dir, 'processing_log.txt')
        os.makedirs(output_dir, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(status_msg + '\n')

class OnlineStatistics:
    """在线统计计算器 - 不需要存储所有数据"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.n = 0  # 总样本数
        self.feature_stats = {}
        
        for feature in FEATURE_NAMES:
            self.feature_stats[feature] = {
                'count': 0,
                'sum': 0.0,
                'sum_sq': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'missing': 0,
                'value_counts': {}  # 只对类别变量
            }
    
    def update_batch(self, df):
        """批量更新统计信息"""
        batch_size = len(df)
        self.n += batch_size
        
        for feature in FEATURE_NAMES:
            if feature not in df.columns:
                continue
            
            feature_data = df[feature]
            stats = self.feature_stats[feature]
            
            # 缺失值统计
            missing_count = feature_data.isnull().sum()
            stats['missing'] += missing_count
            
            # 非缺失值
            valid_data = feature_data.dropna()
            if len(valid_data) == 0:
                continue
            
            stats['count'] += len(valid_data)
            
            # 数值统计
            if pd.api.types.is_numeric_dtype(valid_data):
                stats['sum'] += valid_data.sum()
                stats['sum_sq'] += (valid_data ** 2).sum()
                stats['min'] = min(stats['min'], valid_data.min())
                stats['max'] = max(stats['max'], valid_data.max())
            
            # 类别统计（只对特定特征或唯一值较少的特征）
            if feature in ['Landcover', 'Active_Fire'] or valid_data.nunique() <= 100:
                value_counts = valid_data.value_counts()
                for value, count in value_counts.items():
                    if value not in stats['value_counts']:
                        stats['value_counts'][value] = 0
                    stats['value_counts'][value] += count
    
    def get_summary(self):
        """获取统计摘要"""
        summary = {}
        
        for feature in FEATURE_NAMES:
            stats = self.feature_stats[feature]
            
            if stats['count'] == 0:
                summary[feature] = {
                    'count': 0,
                    'missing': stats['missing'],
                    'missing_rate': stats['missing'] / self.n if self.n > 0 else 0
                }
                continue
            
            # 计算均值和标准差
            mean = stats['sum'] / stats['count'] if stats['count'] > 0 else 0
            variance = (stats['sum_sq'] / stats['count'] - mean ** 2) if stats['count'] > 0 else 0
            std = np.sqrt(max(0, variance))
            
            feature_summary = {
                'count': int(stats['count']),
                'missing': int(stats['missing']),
                'missing_rate': stats['missing'] / self.n if self.n > 0 else 0,
                'mean': float(mean),
                'std': float(std),
                'min': float(stats['min']) if stats['min'] != float('inf') else None,
                'max': float(stats['max']) if stats['max'] != float('-inf') else None
            }
            
            # 添加类别统计
            if stats['value_counts']:
                feature_summary['unique_count'] = len(stats['value_counts'])
                # 只保存前10个最常见的值
                top_values = dict(sorted(stats['value_counts'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10])
                feature_summary['top_values'] = top_values
            
            summary[feature] = feature_summary
        
        summary['total_samples'] = self.n
        return summary

class ReservoirSampler:
    """蓄水池采样器 - 保持代表性样本"""
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.reservoir = []
        self.samples_seen = 0
    
    def add_batch(self, df):
        """添加一批数据"""
        for idx, row in df.iterrows():
            self.samples_seen += 1
            
            if len(self.reservoir) < self.max_size:
                # 蓄水池还没满，直接添加
                self.reservoir.append(row.to_dict())
            else:
                # 蓄水池满了，随机替换
                j = np.random.randint(1, self.samples_seen + 1)
                if j <= self.max_size:
                    self.reservoir[j - 1] = row.to_dict()
    
    def get_sample_df(self):
        """获取样本DataFrame"""
        if not self.reservoir:
            return pd.DataFrame()
        return pd.DataFrame(self.reservoir)

def process_file_chunk_streaming(file_paths):
    """流式处理文件块"""
    chunk_stats = OnlineStatistics()
    chunk_sampler = ReservoirSampler(MAX_SAMPLE_SIZE // MAX_WORKERS)
    
    successful_files = 0
    total_rows = 0
    
    for file_path in file_paths:
        try:
            with h5py.File(file_path, 'r') as f:
                if 'data' not in f:
                    continue
                
                data = f['data'][:]
                if data is None or len(data.shape) != 4:
                    continue
                
                t, c, h, w = data.shape
                if any(dim is None or dim <= 0 for dim in [t, c, h, w]):
                    continue
                
                if c != len(FEATURE_NAMES):
                    print(f"Warning: feature count mismatch in {os.path.basename(file_path)}")
                    continue
                
                # 重塑数据
                data_reshaped = data.transpose(0, 2, 3, 1).reshape(-1, c)
                df = pd.DataFrame(data_reshaped, columns=FEATURE_NAMES)
                
                # 更新统计信息（关键：不保存原始数据！）
                chunk_stats.update_batch(df)
                
                # 添加到样本池
                chunk_sampler.add_batch(df)
                
                successful_files += 1
                total_rows += len(df)
                
                # 立即释放内存
                del df, data, data_reshaped
                
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            continue
    
    return chunk_stats, chunk_sampler, successful_files, total_rows

def load_year_streaming(year):
    """流式加载一年的数据"""
    log_status(f"Starting streaming processing for {year}")
    
    # 获取文件路径
    file_paths = sorted(glob.glob(f'data/processed/{year}/*.hdf5'))
    
    if not file_paths:
        log_status(f"No files found for {year}")
        return None, None, 0, 0
    
    if FILE_LIMIT_PER_YEAR:
        file_paths = file_paths[:FILE_LIMIT_PER_YEAR]
        log_status(f"Limited to {len(file_paths)} files")
    
    # 初始化年度统计器和采样器
    year_stats = OnlineStatistics()
    year_sampler = ReservoirSampler(MAX_SAMPLE_SIZE)
    
    total_files_processed = 0
    total_rows_processed = 0
    
    # 分批处理文件
    for batch_start in range(0, len(file_paths), BATCH_SIZE_PER_YEAR):
        batch_end = min(batch_start + BATCH_SIZE_PER_YEAR, len(file_paths))
        batch_files = file_paths[batch_start:batch_end]
        
        log_status(f"Processing batch {batch_start//BATCH_SIZE_PER_YEAR + 1}/{int(np.ceil(len(file_paths)/BATCH_SIZE_PER_YEAR))}: files {batch_start+1}-{batch_end}")
        
        # 并行处理当前批次
        try:
            # 将批次文件分配给不同的进程
            files_per_worker = max(1, len(batch_files) // MAX_WORKERS)
            file_chunks = [batch_files[i:i + files_per_worker] 
                          for i in range(0, len(batch_files), files_per_worker)]
            
            with Pool(processes=min(len(file_chunks), MAX_WORKERS)) as pool:
                chunk_results = pool.map(process_file_chunk_streaming, file_chunks)
            
            # 合并结果
            for chunk_stats, chunk_sampler, files_count, rows_count in chunk_results:
                if chunk_stats and files_count > 0:
                    # 合并统计信息
                    merge_statistics(year_stats, chunk_stats)
                    
                    # 合并样本
                    chunk_sample_df = chunk_sampler.get_sample_df()
                    if not chunk_sample_df.empty:
                        year_sampler.add_batch(chunk_sample_df)
                    
                    total_files_processed += files_count
                    total_rows_processed += rows_count
            
            log_status(f"Batch completed. Total: {total_files_processed} files, {total_rows_processed:,} rows")
            
            # 强制内存清理
            gc.collect()
            
        except Exception as e:
            log_status(f"Error in batch processing: {e}")
            continue
    
    log_status(f"Year {year} streaming completed: {total_files_processed} files, {total_rows_processed:,} rows")
    
    return year_stats, year_sampler, total_files_processed, total_rows_processed

def merge_statistics(target_stats, source_stats):
    """合并两个统计对象"""
    target_stats.n += source_stats.n
    
    for feature in FEATURE_NAMES:
        if feature not in source_stats.feature_stats:
            continue
        
        target_feature = target_stats.feature_stats[feature]
        source_feature = source_stats.feature_stats[feature]
        
        # 合并基础统计
        target_feature['count'] += source_feature['count']
        target_feature['sum'] += source_feature['sum']
        target_feature['sum_sq'] += source_feature['sum_sq']
        target_feature['missing'] += source_feature['missing']
        
        # 更新最值
        if source_feature['min'] < target_feature['min']:
            target_feature['min'] = source_feature['min']
        if source_feature['max'] > target_feature['max']:
            target_feature['max'] = source_feature['max']
        
        # 合并类别计数
        for value, count in source_feature['value_counts'].items():
            if value not in target_feature['value_counts']:
                target_feature['value_counts'][value] = 0
            target_feature['value_counts'][value] += count

def section_1_initial_assessment_streaming(stats_summary, sample_df, output_dir, year_label=""):
    """基于统计摘要的初始评估"""
    log_status(f"Running Section 1: Initial Assessment {year_label}", output_dir)
    
    # 1. 基本信息
    with open(os.path.join(output_dir, 's1_1_basic_info.txt'), 'w') as f:
        f.write(f"Total samples processed: {stats_summary.get('total_samples', 0):,}\n")
        f.write(f"Sample dataset shape: {sample_df.shape}\n")
        f.write(f"Memory usage: {get_memory_usage_gb():.1f}GB\n")
        f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        f.write(f"\nMissing values summary:\n")
        for feature, stats in stats_summary.items():
            if isinstance(stats, dict) and 'missing' in stats:
                f.write(f"{feature}: {stats['missing']:,} ({stats.get('missing_rate', 0)*100:.2f}%)\n")
    
    # 2. 数据类型信息
    with open(os.path.join(output_dir, 's1_2_data_types.txt'), 'w') as f:
        if not sample_df.empty:
            sample_df.info(verbose=False, buf=f)
        else:
            f.write("No sample data available\n")
    
    # 3. 数据健康检查
    with open(os.path.join(output_dir, 's1_3_data_health.txt'), 'w') as f:
        total_samples = stats_summary.get('total_samples', 0)
        f.write(f"Total samples processed: {total_samples:,}\n")
        
        if not sample_df.empty:
            duplicates = sample_df.duplicated().sum()
            f.write(f"Duplicate rows in sample: {duplicates:,} ({duplicates/len(sample_df)*100:.2f}%)\n")
        
        f.write(f"\nMemory usage: {get_memory_usage_gb():.1f}GB\n")
        
        # 土地覆盖唯一值
        if 'Landcover' in stats_summary and 'top_values' in stats_summary['Landcover']:
            f.write(f"\nTop Landcover classes:\n")
            for value, count in stats_summary['Landcover']['top_values'].items():
                f.write(f"  {value}: {count:,}\n")
    
    # 4. 数值摘要
    try:
        summary_data = []
        for feature, stats in stats_summary.items():
            if isinstance(stats, dict) and 'mean' in stats:
                summary_data.append({
                    'Feature': feature,
                    'Count': stats['count'],
                    'Mean': stats['mean'],
                    'Std': stats['std'],
                    'Min': stats['min'],
                    'Max': stats['max'],
                    'Missing_Rate': f"{stats['missing_rate']*100:.2f}%"
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(output_dir, 's1_4_numerical_summary.csv'), index=False)
        
        log_status(f"Numerical summary saved", output_dir)
    except Exception as e:
        log_status(f"Error in numerical summary: {e}", output_dir)
    
    # 5. 数据字典
    try:
        create_data_dictionary_streaming(stats_summary, output_dir)
        log_status(f"Data dictionary saved", output_dir)
    except Exception as e:
        log_status(f"Error in data dictionary: {e}", output_dir)

def create_data_dictionary_streaming(stats_summary, output_dir):
    """基于统计摘要创建数据字典"""
    dict_data = []
    
    analysis_types = [
        'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Continuous',
        'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Cyclical', 'Numerical-Continuous (K)', 'Numerical-Continuous (K)',
        'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Cyclical',
        'Numerical-Continuous', 'Categorical-Nominal', 'Numerical-Continuous', 'Numerical-Continuous',
        'Numerical-Special', 'Numerical-Continuous (C)', 'Numerical-Continuous', 'Categorical-Binary'
    ]
    
    for i, feature in enumerate(FEATURE_NAMES):
        if feature in stats_summary:
            stats = stats_summary[feature]
            dict_data.append({
                'Variable Name': feature,
                'Analysis Type': analysis_types[i] if i < len(analysis_types) else 'Unknown',
                'Count': stats.get('count', 0),
                'Unique Values': stats.get('unique_count', 'N/A'),
                'Missing %': f"{stats.get('missing_rate', 0)*100:.2f}%",
                'Mean': round(stats.get('mean', 0), 4) if stats.get('mean') is not None else 'N/A',
                'Std': round(stats.get('std', 0), 4) if stats.get('std') is not None else 'N/A'
            })
    
    data_dict_df = pd.DataFrame(dict_data)
    data_dict_df.to_csv(os.path.join(output_dir, 's1_4_data_dictionary.csv'), index=False)

def section_3_univariate_analysis_streaming(sample_df, output_dir, year_label=""):
    """基于样本的单变量分析"""
    log_status(f"Running Section 3: Univariate Analysis {year_label}", output_dir)
    
    if sample_df.empty:
        log_status("No sample data available for visualization", output_dir)
        return
    
    try:
        # 获取连续变量
        continuous_vars = sample_df.select_dtypes(include=np.number).columns
        if 'Landcover' in continuous_vars:
            continuous_vars = continuous_vars.drop('Landcover')
        if 'Active_Fire' in continuous_vars:
            continuous_vars = continuous_vars.drop('Active_Fire')
        
        log_status(f"Using sample of {len(sample_df):,} rows for plotting", output_dir)
        
        # 分批绘制分布图
        vars_per_plot = 12
        n_plots = int(np.ceil(len(continuous_vars) / vars_per_plot))
        
        for plot_num in range(n_plots):
            start_idx = plot_num * vars_per_plot
            end_idx = min(start_idx + vars_per_plot, len(continuous_vars))
            batch_vars = continuous_vars[start_idx:end_idx]
            
            if len(batch_vars) == 0:
                continue
                
            n_cols = min(4, len(batch_vars))
            n_rows = int(np.ceil(len(batch_vars) / n_cols))
            
            plt.figure(figsize=(4 * n_cols, 3 * n_rows))
            
            for i, var in enumerate(batch_vars):
                plt.subplot(n_rows, n_cols, i + 1)
                try:
                    data_to_plot = sample_df[var].dropna()
                    if len(data_to_plot) > 0:
                        sns.histplot(data_to_plot, kde=True)
                        plt.title(var, fontsize=10)
                        plt.xlabel('')
                        plt.ylabel('')
                except Exception as e:
                    plt.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title(var, fontsize=10)
            
            plt.tight_layout()
            filename = f's3_1_continuous_distributions_batch_{plot_num + 1}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()
            log_status(f"Saved distribution plots batch {plot_num + 1}/{n_plots}", output_dir)
        
        # 土地覆盖分布
        if 'Landcover' in sample_df.columns:
            plt.figure(figsize=(10, 6))
            landcover_counts = sample_df['Landcover'].value_counts().head(20)
            sns.barplot(x=landcover_counts.values, y=landcover_counts.index)
            plt.title(f'Top 20 Landcover Classes Frequency {year_label} (Sample)')
            plt.xlabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 's3_2_landcover_distribution.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            log_status("Saved landcover distribution", output_dir)
        
    except Exception as e:
        log_status(f"Error in univariate analysis: {e}", output_dir)

def section_4_bivariate_analysis_streaming(sample_df, output_dir, year_label=""):
    """基于样本的双变量分析"""
    log_status(f"Running Section 4: Bivariate Analysis {year_label}", output_dir)
    
    if sample_df.empty:
        log_status("No sample data available for bivariate analysis", output_dir)
        return
    
    try:
        # 选择关键变量
        continuous_vars = ['Max_Temp_K', 'Wind_Speed', 'NDVI', 'Elevation', 'Slope', 'ERC']
        available_vars = [var for var in continuous_vars if var in sample_df.columns]
        
        if len(available_vars) > 1:
            corr_matrix = sample_df[available_vars].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
            plt.title(f'Correlation Matrix of Key Continuous Variables {year_label} (Sample)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 's4_1_correlation_heatmap.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            log_status("Saved correlation heatmap", output_dir)
        
        # 温度与火灾关系
        if 'Active_Fire' in sample_df.columns and 'Max_Temp_K' in sample_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=sample_df, x='Active_Fire', y='Max_Temp_K')
            plt.title(f'Max Temperature vs. Active Fire Presence {year_label} (Sample)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 's4_3_temp_vs_fire.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            log_status("Saved temperature vs. fire boxplot", output_dir)
            
    except Exception as e:
        log_status(f"Error in bivariate analysis: {e}", output_dir)

def main():
    """流式处理主函数"""
    start_time = time.time()
    
    print("="*60)
    print("流式处理EDA脚本启动 - 基于你的原始代码")
    print(f"目标年份: {YEARS_TO_ANALYZE}")
    print(f"每年文件限制: {FILE_LIMIT_PER_YEAR}")
    print(f"最大工作进程: {MAX_WORKERS}")
    print(f"样本大小限制: {MAX_SAMPLE_SIZE:,}")
    print(f"当前内存使用: {get_memory_usage_gb():.1f}GB")
    print("="*60)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    log_status("Streaming EDA script started", OUTPUT_DIR)
    
    # 用于存储各年份的统计摘要（只是摘要，不是原始数据！）
    all_year_summaries = []
    all_year_samples = []
    successful_years = []

    for year in YEARS_TO_ANALYZE:
        year_output_dir = os.path.join(OUTPUT_DIR, str(year))
        if not os.path.exists(year_output_dir):
            os.makedirs(year_output_dir)
        
        log_status(f"开始流式处理年份: {year}")
        
        try:
            # 流式处理年份数据
            year_stats, year_sampler, files_processed, rows_processed = load_year_streaming(year)
            
            if year_stats is None or files_processed == 0:
                log_status(f"{year}年无数据加载，跳过")
                continue
            
            log_status(f"{year}年流式处理完成: {files_processed} 文件, {rows_processed:,} 行")
            
            # 获取统计摘要和样本数据
            stats_summary = year_stats.get_summary()
            sample_df = year_sampler.get_sample_df()
            
            # 保存统计摘要到JSON
            with open(os.path.join(year_output_dir, 'statistics_summary.json'), 'w') as f:
                json.dump(stats_summary, f, indent=2, default=str)
            
            # 运行分析
            section_1_initial_assessment_streaming(stats_summary, sample_df, year_output_dir, f"({year})")
            section_3_univariate_analysis_streaming(sample_df, year_output_dir, f"({year})")
            section_4_bivariate_analysis_streaming(sample_df, year_output_dir, f"({year})")
            
            # 保存用于聚合分析（只保存摘要，不是原始数据）
            all_year_summaries.append({
                'year': year,
                'summary': stats_summary,
                'files_processed': files_processed,
                'rows_processed': rows_processed
            })
            
            # 保存样本的一小部分用于跨年分析
            if not sample_df.empty:
                # 只保留一小部分样本
                cross_year_sample = sample_df.sample(n=min(10000, len(sample_df)), random_state=42)
                cross_year_sample['Year'] = year
                all_year_samples.append(cross_year_sample)
            
            successful_years.append(year)
            log_status(f"{year}年处理完成")
            
            # 清理内存
            del year_stats, year_sampler, sample_df, stats_summary
            gc.collect()
                
        except Exception as e:
            log_status(f"处理{year}年时出错: {e}")
            continue

    # 跨年份分析
    if all_year_summaries:
        log_status("开始跨年份分析")
        try:
            # 合并少量样本数据进行可视化
            if all_year_samples:
                df_all_sample = pd.concat(all_year_samples, ignore_index=True)
                log_status(f"跨年份样本数据形状: {df_all_sample.shape}")
                
                # 生成跨年份统计
                aggregate_stats = create_aggregate_summary(all_year_summaries)
                
                section_1_initial_assessment_streaming(aggregate_stats, df_all_sample, OUTPUT_DIR, "(所有年份)")
                section_3_univariate_analysis_streaming(df_all_sample, OUTPUT_DIR, "(所有年份)")
                section_4_bivariate_analysis_streaming(df_all_sample, OUTPUT_DIR, "(所有年份)")
                
                # 生成年份对比图
                if len(successful_years) > 1:
                    try:
                        create_year_comparison_plots(df_all_sample, OUTPUT_DIR, successful_years)
                        log_status("年份对比分析完成")
                    except Exception as e:
                        log_status(f"年份对比分析出错: {e}")
            
        except Exception as e:
            log_status(f"跨年份分析出错: {e}")
    else:
        log_status("没有成功处理的年份数据")

    # 生成最终报告
    final_cleanup_and_summary_streaming(start_time, successful_years, all_year_summaries)

def create_aggregate_summary(all_year_summaries):
    """创建跨年份聚合统计摘要"""
    aggregate_stats = {}
    total_samples = 0
    
    # 初始化聚合统计
    for feature in FEATURE_NAMES:
        aggregate_stats[feature] = {
            'count': 0,
            'missing': 0,
            'sum': 0.0,
            'sum_sq': 0.0,
            'min': float('inf'),
            'max': float('-inf'),
            'value_counts': {}
        }
    
    # 合并所有年份的统计
    for year_data in all_year_summaries:
        year_summary = year_data['summary']
        total_samples += year_summary.get('total_samples', 0)
        
        for feature in FEATURE_NAMES:
            if feature not in year_summary:
                continue
            
            feature_stats = year_summary[feature]
            if not isinstance(feature_stats, dict):
                continue
            
            agg_feature = aggregate_stats[feature]
            
            # 合并基础统计
            agg_feature['count'] += feature_stats.get('count', 0)
            agg_feature['missing'] += feature_stats.get('missing', 0)
            
            if 'mean' in feature_stats and 'count' in feature_stats:
                # 重新计算加权均值需要的信息
                count = feature_stats['count']
                mean = feature_stats['mean']
                std = feature_stats.get('std', 0)
                
                agg_feature['sum'] += mean * count
                agg_feature['sum_sq'] += (std * std + mean * mean) * count
            
            # 更新最值
            if feature_stats.get('min') is not None:
                agg_feature['min'] = min(agg_feature['min'], feature_stats['min'])
            if feature_stats.get('max') is not None:
                agg_feature['max'] = max(agg_feature['max'], feature_stats['max'])
            
            # 合并类别统计
            if 'top_values' in feature_stats:
                for value, count in feature_stats['top_values'].items():
                    if value not in agg_feature['value_counts']:
                        agg_feature['value_counts'][value] = 0
                    agg_feature['value_counts'][value] += count
    
    # 计算最终统计
    final_stats = {}
    for feature in FEATURE_NAMES:
        agg_feature = aggregate_stats[feature]
        
        if agg_feature['count'] == 0:
            final_stats[feature] = {
                'count': 0,
                'missing': agg_feature['missing'],
                'missing_rate': agg_feature['missing'] / total_samples if total_samples > 0 else 0
            }
            continue
        
        # 计算聚合均值和标准差
        count = agg_feature['count']
        mean = agg_feature['sum'] / count if count > 0 else 0
        variance = (agg_feature['sum_sq'] / count - mean * mean) if count > 0 else 0
        std = np.sqrt(max(0, variance))
        
        final_stats[feature] = {
            'count': int(count),
            'missing': int(agg_feature['missing']),
            'missing_rate': agg_feature['missing'] / total_samples if total_samples > 0 else 0,
            'mean': float(mean),
            'std': float(std),
            'min': float(agg_feature['min']) if agg_feature['min'] != float('inf') else None,
            'max': float(agg_feature['max']) if agg_feature['max'] != float('-inf') else None
        }
        
        if agg_feature['value_counts']:
            final_stats[feature]['unique_count'] = len(agg_feature['value_counts'])
            final_stats[feature]['top_values'] = dict(
                sorted(agg_feature['value_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
            )
    
    final_stats['total_samples'] = total_samples
    return final_stats

def create_year_comparison_plots(df_all_sample, output_dir, successful_years):
    """创建年份对比图表"""
    plt.figure(figsize=(15, 8))
    
    # 1. 火灾活动年度对比
    if 'Active_Fire' in df_all_sample.columns and 'Year' in df_all_sample.columns:
        plt.subplot(2, 3, 1)
        fire_by_year = df_all_sample.groupby('Year')['Active_Fire'].mean()
        fire_by_year.plot(kind='bar', color='red', alpha=0.7)
        plt.title('Average Fire Activity by Year')
        plt.ylabel('Fire Activity Rate')
        plt.xticks(rotation=45)
    
    # 2. 温度分布年度对比
    if 'Max_Temp_K' in df_all_sample.columns and 'Year' in df_all_sample.columns:
        plt.subplot(2, 3, 2)
        df_all_sample.boxplot(column='Max_Temp_K', by='Year', ax=plt.gca())
        plt.title('Temperature Distribution by Year')
        plt.suptitle('')  # 移除默认标题
    
    # 3. NDVI年度对比
    if 'NDVI' in df_all_sample.columns and 'Year' in df_all_sample.columns:
        plt.subplot(2, 3, 3)
        ndvi_by_year = df_all_sample.groupby('Year')['NDVI'].mean()
        ndvi_by_year.plot(kind='line', marker='o', color='green')
        plt.title('Average NDVI by Year')
        plt.ylabel('NDVI')
    
    # 4. 降水年度对比
    if 'Total_Precip' in df_all_sample.columns and 'Year' in df_all_sample.columns:
        plt.subplot(2, 3, 4)
        precip_by_year = df_all_sample.groupby('Year')['Total_Precip'].mean()
        precip_by_year.plot(kind='bar', color='blue', alpha=0.7)
        plt.title('Average Precipitation by Year')
        plt.ylabel('Precipitation')
        plt.xticks(rotation=45)
    
    # 5. 风速年度对比
    if 'Wind_Speed' in df_all_sample.columns and 'Year' in df_all_sample.columns:
        plt.subplot(2, 3, 5)
        wind_by_year = df_all_sample.groupby('Year')['Wind_Speed'].mean()
        wind_by_year.plot(kind='line', marker='s', color='purple')
        plt.title('Average Wind Speed by Year')
        plt.ylabel('Wind Speed')
    
    # 6. 数据量年度对比
    plt.subplot(2, 3, 6)
    sample_counts = df_all_sample['Year'].value_counts().sort_index()
    sample_counts.plot(kind='bar', color='orange', alpha=0.7)
    plt.title('Sample Size by Year')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_year_comparison.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 单独创建火灾-温度关系的年度对比
    if 'Active_Fire' in df_all_sample.columns and 'Max_Temp_K' in df_all_sample.columns:
        plt.figure(figsize=(12, 6))
        for i, year in enumerate(successful_years):
            year_data = df_all_sample[df_all_sample['Year'] == year]
            plt.subplot(2, 2, i + 1)
            sns.boxplot(data=year_data, x='Active_Fire', y='Max_Temp_K')
            plt.title(f'{year} - Temperature vs Fire')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fire_temp_by_year.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

def final_cleanup_and_summary_streaming(start_time, successful_years, all_year_summaries):
    """流式处理的最终清理和总结"""
    gc.collect()
    
    elapsed_time = time.time() - start_time
    final_memory = get_memory_usage_gb()
    
    # 计算总体统计
    total_files = sum(year_data['files_processed'] for year_data in all_year_summaries)
    total_rows = sum(year_data['rows_processed'] for year_data in all_year_summaries)
    
    summary = f"""
{'='*60}
流式EDA处理完成总结
{'='*60}
成功处理年份: {successful_years}
总处理时间: {elapsed_time/60:.1f}分钟
总处理文件: {total_files:,}
总处理数据行: {total_rows:,}
最终内存使用: {final_memory:.1f}GB
输出目录: {OUTPUT_DIR}

各年份处理详情:"""
    
    for year_data in all_year_summaries:
        year = year_data['year']
        files = year_data['files_processed']
        rows = year_data['rows_processed']
        summary += f"\n  {year}: {files:,} 文件, {rows:,} 行"
    
    summary += f"\n\n内存效率对比:"
    summary += f"\n  传统方法预估内存需求: {total_rows * 23 * 8 / (1024**3):.1f}GB"
    summary += f"\n  流式处理实际内存使用: {final_memory:.1f}GB"
    summary += f"\n  内存节省: {((total_rows * 23 * 8 / (1024**3)) - final_memory) / (total_rows * 23 * 8 / (1024**3)) * 100:.1f}%"
    summary += f"\n{'='*60}"
    
    print(summary)
    log_status("Streaming EDA script completed", OUTPUT_DIR)
    
    # 保存详细总结
    with open(os.path.join(OUTPUT_DIR, 'streaming_processing_summary.txt'), 'w') as f:
        f.write(summary)
    
    # 保存各年份统计摘要的合并版本
    combined_summary = {
        'processing_info': {
            'successful_years': successful_years,
            'total_files': total_files,
            'total_rows': total_rows,
            'processing_time_minutes': elapsed_time / 60,
            'final_memory_gb': final_memory
        },
        'yearly_summaries': all_year_summaries
    }
    
    with open(os.path.join(OUTPUT_DIR, 'combined_statistics_summary.json'), 'w') as f:
        json.dump(combined_summary, f, indent=2, default=str)

if __name__ == '__main__':
    main()