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

# --- Global Setup ---
OUTPUT_DIR = 'eda_outputs'
YEARS_TO_ANALYZE = [2018, 2019, 2020, 2021]  

# 文件处理设置
FILE_LIMIT_PER_YEAR = None        # 处理所有文件
BATCH_SIZE_PER_YEAR = 5           # 每批文件数，根据你的内存调整

# 进程和内存设置
MAX_WORKERS = min(2, max(1, cpu_count() - 2))  # 更保守，留更多资源
MEMORY_LIMIT_GB = 24              # 根据你的实际内存调整

# 数据量设置
MAX_ROWS_PER_YEAR = None          # 设为None来处理所有数据
MAX_TOTAL_ROWS = None             # 设为None来处理所有数据

# 绘图设置
SAMPLE_SIZE_FOR_PLOTS = 100000    # 绘图采样

# 内存管理设置
AGGRESSIVE_MEMORY_CLEANUP = True
MEMORY_CHECK_FREQUENCY = 5

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
        with open(log_file, 'a') as f:
            f.write(status_msg + '\n')

def load_chunk_worker(file_path_chunk):
    """改进的worker函数，增加内存监控和错误处理"""
    all_dfs_in_chunk = []
    successful_files = 0
    
    for file_path in file_path_chunk:
        try:
            current_memory = get_memory_usage_gb()
            if current_memory > MEMORY_LIMIT_GB * 0.8:
                print(f"Worker: Memory limit approaching ({current_memory:.1f}GB), stopping chunk processing")
                break
                
            with h5py.File(file_path, 'r') as f:
                data = f['data'][:]
                t, c, h, w = data.shape
                data_reshaped = data.transpose(0, 2, 3, 1).reshape(-1, c)
                df = pd.DataFrame(data_reshaped, columns=FEATURE_NAMES)
                
                # Correctly placed sampling logic
                if MAX_ROWS_PER_YEAR is not None and len(df) > MAX_ROWS_PER_YEAR // 2:
                    df = df.sample(n=MAX_ROWS_PER_YEAR // 2, random_state=42)
                
                all_dfs_in_chunk.append(df)
                successful_files += 1
                
        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            continue
    
    if not all_dfs_in_chunk:
        return None, 0
        
    result = pd.concat(all_dfs_in_chunk, ignore_index=True)
    
    del all_dfs_in_chunk
    gc.collect()
    
    return result, successful_files

def load_and_flatten_h5_parallel_batched(file_paths, year):
    """分批处理所有文件，避免内存爆炸"""
    total_files = len(file_paths)
    log_status(f"Starting batched loading for {year} with {total_files} files")
    
    if FILE_LIMIT_PER_YEAR is not None:
        file_paths = file_paths[:FILE_LIMIT_PER_YEAR]
        log_status(f"Limited to first {FILE_LIMIT_PER_YEAR} files")
    
    all_year_data = []
    total_processed = 0
    
    for batch_start in range(0, len(file_paths), BATCH_SIZE_PER_YEAR):
        batch_end = min(batch_start + BATCH_SIZE_PER_YEAR, len(file_paths))
        batch_files = file_paths[batch_start:batch_end]
        
        log_status(f"Processing file batch {batch_start//BATCH_SIZE_PER_YEAR + 1}/{int(np.ceil(len(file_paths)/BATCH_SIZE_PER_YEAR))}: files {batch_start+1}-{batch_end}")
        
        current_memory = get_memory_usage_gb()
        if current_memory > MEMORY_LIMIT_GB * 0.6:
            log_status(f"Memory approaching limit ({current_memory:.1f}GB), stopping batch processing")
            break
        
        batch_df = load_batch_parallel(batch_files, year, batch_start//BATCH_SIZE_PER_YEAR + 1)
        
        if not batch_df.empty:
            all_year_data.append(batch_df)
            total_processed += len(batch_files)
            log_status(f"Batch completed. Total files processed: {total_processed}")
            
            total_rows = sum(len(df) for df in all_year_data)
            if MAX_ROWS_PER_YEAR is not None and total_rows > MAX_ROWS_PER_YEAR:
                log_status(f"Reached row limit ({total_rows:,}), stopping file processing")
                break
        
        del batch_df
        gc.collect()
    
    if not all_year_data:
        log_status("No data loaded for any batch")
        return pd.DataFrame()
    
    log_status(f"Concatenating {len(all_year_data)} batches")
    full_df = pd.concat(all_year_data, ignore_index=True)
    
    if MAX_ROWS_PER_YEAR is not None and len(full_df) > MAX_ROWS_PER_YEAR:
        log_status(f"Final sampling from {len(full_df):,} to {MAX_ROWS_PER_YEAR:,} rows")
        full_df = full_df.sample(n=MAX_ROWS_PER_YEAR, random_state=42).reset_index(drop=True)
    
    del all_year_data
    gc.collect()
    
    log_status(f"Year {year} completed: {len(full_df):,} rows from {total_processed} files")
    return full_df

def load_batch_parallel(file_paths, year, batch_num):
    """并行加载一个批次的文件"""
    log_status(f"Loading batch {batch_num} with {len(file_paths)} files")
    
    chunk_size = max(1, len(file_paths) // MAX_WORKERS)
    if chunk_size == 0:
        chunk_size = 1
    
    file_chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]
    
    all_results = []
    total_files_processed = 0
    
    with Pool(processes=min(len(file_chunks), MAX_WORKERS)) as pool:
        batch_results = pool.map(load_chunk_worker, file_chunks)
    
    for result, files_count in batch_results:
        if result is not None:
            all_results.append(result)
            total_files_processed += files_count
    
    if not all_results:
        log_status(f"No data loaded from batch {batch_num}")
        return pd.DataFrame()
    
    batch_df = pd.concat(all_results, ignore_index=True)
    
    del all_results
    gc.collect()
    
    log_status(f"Batch {batch_num} loaded: {len(batch_df):,} rows from {total_files_processed} files")
    return batch_df

def section_1_initial_assessment(df, output_dir, year_label=""):
    log_status(f"Running Section 1: Initial Assessment {year_label}", output_dir)
    
    with open(os.path.join(output_dir, 's1_1_basic_info.txt'), 'w') as f:
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Memory usage: {get_memory_usage_gb():.1f}GB\n")
        f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Missing values per column:\n{df.isnull().sum()}\n")
    
    with open(os.path.join(output_dir, 's1_2_data_types.txt'), 'w') as f:
        df.info(verbose=False, buf=f)
    log_status(f"Data type info saved", output_dir)
    
    duplicates = df.duplicated().sum()
    with open(os.path.join(output_dir, 's1_3_data_health.txt'), 'w') as f:
        f.write(f"Number of duplicate rows: {duplicates:,}\n")
        f.write(f"Percentage of duplicates: {duplicates/len(df)*100:.2f}%\n")
        f.write(f"\nDataset shape: {df.shape}\n")
        f.write(f"Memory usage: {get_memory_usage_gb():.1f}GB\n")
        f.write("\nUnique values in 'Landcover':\n")
        if 'Landcover' in df.columns:
            f.write(str(sorted(df['Landcover'].unique())))
    log_status(f"Data health check saved", output_dir)
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary = df[numeric_cols].describe()
        summary.T.to_csv(os.path.join(output_dir, 's1_4_numerical_summary.csv'))
        log_status(f"Numerical summary saved", output_dir)
    except Exception as e:
        log_status(f"Error in numerical summary: {e}", output_dir)
    
    try:
        data_dict = create_data_dictionary(df)
        data_dict.to_csv(os.path.join(output_dir, 's1_4_data_dictionary.csv'), index=False)
        log_status(f"Data dictionary saved", output_dir)
    except Exception as e:
        log_status(f"Error in data dictionary: {e}", output_dir)

def create_data_dictionary(df):
    dict_data = {
        'Variable Name': df.columns,
        'Original Dtype': [str(t) for t in df.dtypes],
        'Non-null Count': [df[col].count() for col in df.columns],
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Missing %': [(df[col].isnull().sum() / len(df) * 100) for col in df.columns]
    }
    data_dict_df = pd.DataFrame(dict_data)
    
    analysis_types = [
        'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Continuous',
        'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Cyclical', 'Numerical-Continuous (K)', 'Numerical-Continuous (K)',
        'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Continuous', 'Numerical-Cyclical',
        'Numerical-Continuous', 'Categorical-Nominal', 'Numerical-Continuous', 'Numerical-Continuous',
        'Numerical-Special', 'Numerical-Continuous (C)', 'Numerical-Continuous', 'Categorical-Binary'
    ]
    
    if len(analysis_types) == len(df.columns):
        data_dict_df['Analysis Type'] = analysis_types
    
    return data_dict_df

def section_3_univariate_analysis(df, output_dir, year_label=""):
    log_status(f"Running Section 3: Univariate Analysis {year_label}", output_dir)
    
    try:
        continuous_vars = df.select_dtypes(include=np.number).columns
        if 'Landcover' in continuous_vars:
            continuous_vars = continuous_vars.drop('Landcover')
        if 'Active_Fire' in continuous_vars:
            continuous_vars = continuous_vars.drop('Active_Fire')
        
        plot_df = df
        if len(df) > SAMPLE_SIZE_FOR_PLOTS:
            plot_df = df.sample(n=SAMPLE_SIZE_FOR_PLOTS, random_state=42)
            log_status(f"Using sample of {SAMPLE_SIZE_FOR_PLOTS:,} rows for plotting", output_dir)
        
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
                    data_to_plot = plot_df[var].dropna()
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
        
        if 'Landcover' in df.columns:
            plt.figure(figsize=(10, 6))
            landcover_counts = plot_df['Landcover'].value_counts().head(20)
            sns.barplot(x=landcover_counts.values, y=landcover_counts.index)
            plt.title(f'Top 20 Landcover Classes Frequency {year_label}')
            plt.xlabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 's3_2_landcover_distribution.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            log_status("Saved landcover distribution", output_dir)
        
    except Exception as e:
        log_status(f"Error in univariate analysis: {e}", output_dir)

def section_4_bivariate_analysis(df, output_dir, year_label=""):
    log_status(f"Running Section 4: Bivariate Analysis {year_label}", output_dir)
    
    try:
        continuous_vars = ['Max_Temp_K', 'Wind_Speed', 'NDVI', 'Elevation', 'Slope', 'ERC']
        available_vars = [var for var in continuous_vars if var in df.columns]
        
        if len(available_vars) > 1:
            sample_df = df
            if len(df) > SAMPLE_SIZE_FOR_PLOTS:
                sample_df = df.sample(n=SAMPLE_SIZE_FOR_PLOTS, random_state=42)
            
            corr_matrix = sample_df[available_vars].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
            plt.title(f'Correlation Matrix of Key Continuous Variables {year_label}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 's4_1_correlation_heatmap.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            log_status("Saved correlation heatmap", output_dir)
        
        if 'Active_Fire' in df.columns and 'Max_Temp_K' in df.columns:
            sample_df = df
            if len(df) > SAMPLE_SIZE_FOR_PLOTS:
                sample_df = df.sample(n=SAMPLE_SIZE_FOR_PLOTS, random_state=42)
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=sample_df, x='Active_Fire', y='Max_Temp_K')
            plt.title(f'Max Temperature vs. Active Fire Presence {year_label}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 's4_3_temp_vs_fire.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            log_status("Saved temperature vs. fire boxplot", output_dir)
            
    except Exception as e:
        log_status(f"Error in bivariate analysis: {e}", output_dir)

def cleanup_and_monitor():
    """清理内存并监控"""
    gc.collect()
    current_memory = get_memory_usage_gb()
    if current_memory > MEMORY_LIMIT_GB * 0.8:
        log_status(f"WARNING: High memory usage detected: {current_memory:.1f}GB")
        return False
    return True

def main():
    """主函数 - 改进的多年份EDA"""
    start_time = time.time()
    
    print("="*60)
    print("多年份安全EDA脚本启动")
    print(f"目标年份: {YEARS_TO_ANALYZE}")
    print(f"每年文件限制: {FILE_LIMIT_PER_YEAR}")
    print(f"最大工作进程: {MAX_WORKERS}")
    print(f"内存限制: {MEMORY_LIMIT_GB}GB")
    print(f"当前内存使用: {get_memory_usage_gb():.1f}GB")
    print("="*60)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    log_status("EDA script started", OUTPUT_DIR)
    
    all_yearly_dfs = []
    successful_years = []

    for year in YEARS_TO_ANALYZE:
        year_output_dir = os.path.join(OUTPUT_DIR, str(year))
        if not os.path.exists(year_output_dir):
            os.makedirs(year_output_dir)
        
        log_status(f"开始处理年份: {year}")
        
        file_paths = sorted(glob.glob(f'data/processed/{year}/*.hdf5'))
        
        if not file_paths:
            log_status(f"未找到{year}年的文件，跳过")
            continue
            
        if FILE_LIMIT_PER_YEAR:
            log_status(f"限制处理前{FILE_LIMIT_PER_YEAR}个文件")
            file_paths = file_paths[:FILE_LIMIT_PER_YEAR]

        if not cleanup_and_monitor():
            log_status("内存不足，停止处理更多年份")
            break

        try:
            df_year = load_and_flatten_h5_parallel_batched(file_paths, year)
            
            if df_year.empty:
                log_status(f"{year}年无数据加载，跳过")
                continue
            
            log_status(f"{year}年数据加载完成: {df_year.shape}")
            
            section_1_initial_assessment(df_year, year_output_dir, f"({year})")
            section_3_univariate_analysis(df_year, year_output_dir, f"({year})")
            section_4_bivariate_analysis(df_year, year_output_dir, f"({year})")
            
            if MAX_ROWS_PER_YEAR is not None and len(df_year) > MAX_ROWS_PER_YEAR // len(YEARS_TO_ANALYZE):
                sample_size = MAX_ROWS_PER_YEAR // len(YEARS_TO_ANALYZE)
                df_for_aggregate = df_year.sample(n=sample_size, random_state=42)
                log_status(f"为聚合分析采样{sample_size:,}行")
            else:
                df_for_aggregate = df_year.copy()
            
            df_for_aggregate['Year'] = year
            all_yearly_dfs.append(df_for_aggregate)
            successful_years.append(year)
            
            log_status(f"{year}年处理完成")
            
            del df_year, df_for_aggregate
            if not cleanup_and_monitor():
                log_status("内存压力过大，停止处理更多年份")
                break
                
        except Exception as e:
            log_status(f"处理{year}年时出错: {e}")
            continue

    if all_yearly_dfs:
        log_status("开始聚合分析")
        try:
            df_all = pd.concat(all_yearly_dfs, ignore_index=True)
            log_status(f"聚合数据集形状: {df_all.shape}")
            
            section_1_initial_assessment(df_all, OUTPUT_DIR, "(所有年份)")
            section_3_univariate_analysis(df_all, OUTPUT_DIR, "(所有年份)")
            section_4_bivariate_analysis(df_all, OUTPUT_DIR, "(所有年份)")
            
            if len(successful_years) > 1 and 'Year' in df_all.columns:
                try:
                    plt.figure(figsize=(12, 6))
                    if 'Active_Fire' in df_all.columns:
                        fire_by_year = df_all.groupby('Year')['Active_Fire'].mean()
                        plt.subplot(1, 2, 1)
                        fire_by_year.plot(kind='bar')
                        plt.title('Average Fire Activity by Year')
                        plt.ylabel('Fire Activity Rate')
                    
                    if 'Max_Temp_K' in df_all.columns:
                        plt.subplot(1, 2, 2)
                        df_all.boxplot(column='Max_Temp_K', by='Year', ax=plt.gca())
                        plt.title('Temperature Distribution by Year')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR, 'year_comparison.png'), 
                               dpi=150, bbox_inches='tight')
                    plt.close()
                    log_status("年份对比分析完成")
                except Exception as e:
                    log_status(f"年份对比分析出错: {e}")
            
        except Exception as e:
            log_status(f"聚合分析出错: {e}")
    else:
        log_status("没有成功处理的年份数据")

    final_cleanup_and_summary(start_time, successful_years)

def final_cleanup_and_summary(start_time, successful_years):
    """最终清理和总结"""
    gc.collect()
    
    elapsed_time = time.time() - start_time
    final_memory = get_memory_usage_gb()
    
    summary = f"""
{'='*60}
EDA处理完成总结
{'='*60}
成功处理年份: {successful_years}
总处理时间: {elapsed_time/60:.1f}分钟
最终内存使用: {final_memory:.1f}GB
输出目录: {OUTPUT_DIR}
{'='*60}
    """
    
    print(summary)
    log_status("EDA script completed", OUTPUT_DIR)
    
    with open(os.path.join(OUTPUT_DIR, 'processing_summary.txt'), 'w') as f:
        f.write(summary)

if __name__ == '__main__':
    main()
