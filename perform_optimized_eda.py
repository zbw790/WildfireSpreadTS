import os, glob, time, gc, json, h5py, psutil
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from scipy.stats import ttest_ind, pointbiserialr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

# -------------------- 全局配置 --------------------
OUTPUT_DIR = 'eda_comprehensive_report'   # 总输出目录
YEARS_TO_ANALYZE = [2018, 2019, 2020, 2021]
FILE_LIMIT_PER_YEAR = None                  # 设为 None 处理该年的所有文件
MAX_WORKERS = max(1, cpu_count() - 1)     # 进程数
MAX_SAMPLE_SIZE = 250_000                 # 绘图用“蓄水池”样本上限

FEATURE_NAMES = [
    'VIIRS_M11', 'VIIRS_I2', 'VIIRS_I1', 'NDVI', 'EVI2',
    'Total_Precip', 'Wind_Speed', 'Wind_Direction',
    'Min_Temp_K', 'Max_Temp_K', 'ERC', 'Spec_Hum', 'PDSI',
    'Slope', 'Aspect', 'Elevation', 'Landcover',
    'Forecast_Precip', 'Forecast_Wind_Speed', 'Forecast_Wind_Dir',
    'Forecast_Temp_C', 'Forecast_Spec_Hum', 'Active_Fire'
]
CATEGORICAL_FEATURES = ['Landcover', 'Active_Fire']
KEY_VARS_VS_FIRE = ['Wind_Speed', 'Slope', 'Max_Temp_K', 'Spec_Hum', 'NDVI', 'Elevation']
WEATHER_FEATURES = ['Total_Precip', 'Wind_Speed', 'Min_Temp_K', 'Max_Temp_K', 'ERC', 'Spec_Hum', 'PDSI']

# -------------------- 基础工具 --------------------
def log_status(msg, outdir=None):
    ts = datetime.now().strftime("%H:%M:%S")
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    line = f"[{ts}] {msg} | Memory: {mem:.1f}GB"
    print(line)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'log.txt'), 'a', encoding='utf-8') as f:
            f.write(line + '\n')

# -------------------- 在线统计 + 采样 --------------------
class OnlineStatisticsNumPy:
    """按列累计：均值/方差/极值/缺失率/类别计数"""
    def __init__(self):
        self.n = 0
        self.stats = {
            f: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0,
                'min': np.inf, 'max': -np.inf, 'missing': 0, 'value_counts': {}}
            for f in FEATURE_NAMES
        }

    def update_batch_numpy(self, data_array):
        rows = data_array.shape[0]
        self.n += rows
        for i, feature in enumerate(FEATURE_NAMES):
            if i >= data_array.shape[1]:
                continue
            col = data_array[:, i]
            # 缺失统计
            miss = np.isnan(col).sum()
            s = self.stats[feature]
            s['missing'] += int(miss)
            # 有效值
            valid = col[~np.isnan(col)]
            if valid.size == 0:
                continue
            s['count']  += valid.size
            s['sum']    += float(np.sum(valid))
            s['sum_sq'] += float(np.sum(valid**2))
            s['min']     = min(s['min'], float(np.min(valid)))
            s['max']     = max(s['max'], float(np.max(valid)))
            # 类别计数
            if feature in CATEGORICAL_FEATURES:
                u, c = np.unique(valid, return_counts=True)
                for val, cnt in zip(u, c):
                    key = int(val) if float(val).is_integer() else float(val)
                    s['value_counts'][key] = s['value_counts'].get(key, 0) + int(cnt)

    def merge(self, other):
        self.n += other.n
        for f in FEATURE_NAMES:
            a, b = self.stats[f], other.stats[f]
            a['sum']    += b['sum']
            a['sum_sq'] += b['sum_sq']
            a['count']  += b['count']
            a['missing']+= b['missing']
            a['min']     = min(a['min'], b['min'])
            a['max']     = max(a['max'], b['max'])
            for k, v in b['value_counts'].items():
                a['value_counts'][k] = a['value_counts'].get(k, 0) + v

    def get_summary(self):
        summary = {}
        for f in FEATURE_NAMES:
            s = self.stats[f]
            cnt = s['count']
            mean = s['sum']/cnt if cnt>0 else 0.0
            var  = s['sum_sq']/cnt - mean**2 if cnt>0 else 0.0
            summary[f] = {
                'mean': mean,
                'std':  np.sqrt(max(0.0, var)),
                'min': (None if s['min']==np.inf else s['min']),
                'max': (None if s['max']==-np.inf else s['max']),
                'count': cnt,
                'missing': s['missing'],
                'missing_rate': s['missing']/max(1, self.n),
                'value_counts': s['value_counts']
            }
        summary['total_samples'] = self.n
        return summary

class BlockReservoirSampler:
    """蓄水池采样：限制绘图样本规模"""
    def __init__(self, max_size):
        self.max_size = max_size
        self.reservoir = np.empty((0, len(FEATURE_NAMES)))
        self.samples_seen = 0

    def add_batch(self, data_batch):
        if data_batch.size == 0:
            return
        if self.reservoir.size == 0:
            self.reservoir = np.empty((0, data_batch.shape[1]))
        n_new = data_batch.shape[0]
        if self.samples_seen + n_new <= self.max_size:
            self.reservoir = np.vstack([self.reservoir, data_batch])
        else:
            if self.samples_seen < self.max_size:
                space = self.max_size - self.samples_seen
                self.reservoir = np.vstack([self.reservoir, data_batch[:space]])
                data_batch = data_batch[space:]
            for i in range(data_batch.shape[0]):
                j = np.random.randint(0, self.samples_seen + i + 1)
                if j < self.max_size:
                    self.reservoir[j, :] = data_batch[i, :]
        self.samples_seen += n_new

    def get_sample_df(self):
        return pd.DataFrame(self.reservoir, columns=FEATURE_NAMES)

def process_file_chunk_optimized(file_paths):
    stats = OnlineStatisticsNumPy()
    sampler = BlockReservoirSampler(MAX_SAMPLE_SIZE // max(1, MAX_WORKERS))
    for file_path in file_paths:
        try:
            with h5py.File(file_path, 'r') as f:
                data = f['data'][:]                              # (T,C,H,W)
                arr = data.transpose(0, 2, 3, 1).reshape(-1, data.shape[1])
                stats.update_batch_numpy(arr)
                sampler.add_batch(arr)
        except Exception:
            continue
    return stats, sampler

# -------------------- 数据清洗（关键稳健层） --------------------
def prepare_sample_df(df: pd.DataFrame) -> pd.DataFrame:
    """确保所有数值列为 float，Active_Fire 为 0/1 整型，移除非有限值，且只保留已知列。"""
    if df is None or df.empty:
        return df
    # 仅保留脚本期望的列（避免外来列类型奇怪）
    df = df[[c for c in FEATURE_NAMES if c in df.columns]].copy()

    # 强制数值列为 float
    for c in df.columns:
        if c != 'Landcover':  # Landcover 也可能是数值编码，这里先不强转分类类型，仍保留数值
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Active_Fire 归一化为 0/1
    if 'Active_Fire' in df.columns:
        af = df['Active_Fire'].values
        # 允许 >0 当成 1
        af = np.where(np.isfinite(af) & (af > 0), 1, 0)
        df['Active_Fire'] = af.astype(int)

    # 移除包含非有限值的行（只要气象/连续特征）
    cont_cols = [c for c in df.columns if c not in ['Landcover']]
    # 只清洗用于绘图/统计的主列；允许 Landcover 存在 NaN
    mask_finite = np.isfinite(df[cont_cols]).all(axis=1)
    df = df[mask_finite].reset_index(drop=True)

    return df

# -------------------- 分析模块（必须放在综合报告函数之前） --------------------
def plot_fire_imbalance(sample_df, output_dir, year_label=""):
    if 'Active_Fire' not in sample_df:
        return
    rate = float((sample_df['Active_Fire'] == 1).mean())
    with open(os.path.join(output_dir, 's2_label_imbalance.txt'), 'w') as f:
        f.write(f'Active_Fire positive rate {year_label}: {rate:.6%}\n')
    plt.figure(figsize=(4, 3))
    sns.barplot(x=['Non-fire', 'Fire'], y=[1 - rate, rate])
    plt.title(f'Active_Fire Imbalance {year_label}'); plt.ylabel('Proportion')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 's2_label_imbalance.png')); plt.close()

def data_health_checks(sample_df, output_dir):
    checks = []
    def add(name, cond, note): checks.append((name, int(cond.sum()), note))
    if 'Wind_Speed'   in sample_df: add('Wind_Speed>70',   sample_df['Wind_Speed']>70,   'unusually high')
    if 'Total_Precip' in sample_df: add('Total_Precip<0',  sample_df['Total_Precip']<0,  'negative precip')
    if 'Max_Temp_K'   in sample_df: add('Max_Temp_K<230',  sample_df['Max_Temp_K']<230,  'too low (K)')
    if 'Spec_Hum'     in sample_df: add('Spec_Hum<0',      sample_df['Spec_Hum']<0,      'negative humidity')
    pd.DataFrame(checks, columns=['rule','count','note']).to_csv(
        os.path.join(output_dir, 's0_anomaly_checks.csv'), index=False)

def plot_wide_correlation(sample_df, output_dir, year_label=""):
    cont = [c for c in sample_df.columns
            if c not in ['Landcover','Active_Fire'] and
            np.issubdtype(sample_df[c].dtype, np.number)]
    if len(cont) >= 2:
        corr = sample_df[cont].corr().clip(-1, 1)
        plt.figure(figsize=(min(0.4*len(cont)+4, 18), min(0.4*len(cont)+4, 18)))
        sns.heatmap(corr, cmap='coolwarm', center=0)
        plt.title(f'Correlation (continuous) {year_label}')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 's4_corr_wide.png')); plt.close()

    met = [c for c in WEATHER_FEATURES if c in sample_df.columns]
    if len(met) >= 2:
        corr = sample_df[met].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title(f'Correlation (meteorology) {year_label}')
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 's4_corr_meteorology.png')); plt.close()

def landcover_fire_rate(sample_df, output_dir, year_label=""):
    if 'Landcover' not in sample_df or 'Active_Fire' not in sample_df: return
    df = sample_df[['Landcover', 'Active_Fire']].dropna()
    if df.empty: return
    rate = df.groupby('Landcover')['Active_Fire'].mean().sort_values(ascending=False)
    rate.head(30).to_csv(os.path.join(output_dir, 's5_landcover_fire_rate.csv'))
    plt.figure(figsize=(10, 8))
    sns.barplot(x=rate.head(20).values, y=rate.head(20).index)
    plt.xlabel('Fire rate'); plt.title(f'Fire rate by Landcover {year_label}')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 's5_landcover_fire_rate.png')); plt.close()

def generate_literature_comparison_table(sample_df, output_dir):
    if 'Active_Fire' not in sample_df: return
    fire_col = sample_df['Active_Fire']
    if fire_col.nunique() < 2: return
    expected = {'Wind_Speed': '+', 'Slope': '+', 'Max_Temp_K': '+',
                'Spec_Hum': '-', 'NDVI': '-', 'Total_Precip': '-'}
    rows = []
    for var, exp in expected.items():
        if var not in sample_df.columns: continue
        v = sample_df[var]
        mask = v.notna() & fire_col.notna()
        if mask.sum() < 100: continue
        # point-biserial 前确保有限
        x = v[mask].astype(float)
        y = fire_col[mask].astype(int)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 100: continue
        r, _ = pointbiserialr(x[finite], y[finite])
        rows.append({'Feature': var, 'Expected': exp, 'Observed_r': round(float(r), 3)})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, 's4_literature_comparison.csv'), index=False)

def generate_grouped_boxplots_with_ttest(sample_df, output_dir):
    if 'Active_Fire' not in sample_df: return
    fire_pixels = sample_df[sample_df['Active_Fire'] == 1]
    non_fire   = sample_df[sample_df['Active_Fire'] == 0]
    if len(fire_pixels) < 2 or len(non_fire) < 2: return
    plt.figure(figsize=(20, 12))
    for i, var in enumerate(KEY_VARS_VS_FIRE):
        if var not in sample_df.columns: continue
        plt.subplot(2, 3, i + 1)
        sns.boxplot(x='Active_Fire', y=var, data=sample_df, showfliers=False)
        g1 = fire_pixels[var].dropna(); g2 = non_fire[var].dropna()
        g1 = g1[np.isfinite(g1)]; g2 = g2[np.isfinite(g2)]
        if len(g1) > 1 and len(g2) > 1:
            try:
                _, p = ttest_ind(g1, g2, equal_var=False)
                plt.title(f'{var} (p={p:.2e})')
            except Exception:
                plt.title(var)
        else:
            plt.title(var)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 's4_grouped_boxplots.png')); plt.close()

def fire_pointbiserial_full(sample_df, output_dir, year_label=""):
    if 'Active_Fire' not in sample_df: return
    y = sample_df['Active_Fire'].astype(int)
    rows = []
    for col in sample_df.columns:
        if col in ['Active_Fire', 'Landcover']: continue
        if not np.issubdtype(sample_df[col].dtype, np.number): continue
        v = sample_df[col]
        mask = v.notna() & y.notna()
        if mask.sum() < 100: continue
        x = v[mask].astype(float)
        yy = y[mask]
        finite = np.isfinite(x) & np.isfinite(yy)
        if finite.sum() < 100: continue
        try:
            r, p = pointbiserialr(x[finite], yy[finite])
            rows.append((col, float(r), float(p)))
        except Exception:
            continue
    if rows:
        pd.DataFrame(rows, columns=['feature', 'r_pointbiserial', 'p_value']) \
          .sort_values('r_pointbiserial', ascending=False) \
          .to_csv(os.path.join(output_dir, 's4_pointbiserial_all.csv'), index=False)

def generate_feature_importance_ranking(sample_df, output_dir):
    if 'Active_Fire' not in sample_df: return
    y = sample_df['Active_Fire'].astype(int)
    if y.nunique() < 2: return
    X = sample_df.drop(columns=['Active_Fire'])
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]
    if not num_cols: return
    X = X[num_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    try:
        mi = mutual_info_classif(X, y, random_state=42)
    except Exception:
        return
    mi_df = pd.DataFrame({'Feature': X.columns, 'MutualInformation': mi}) \
            .sort_values('MutualInformation', ascending=False)
    plt.figure(figsize=(12, 10))
    sns.barplot(x='MutualInformation', y='Feature', data=mi_df)
    plt.title('Feature Importance (Mutual Information)'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 's6_feature_importance.png')); plt.close()
    mi_df.to_csv(os.path.join(output_dir, 's6_feature_importance.csv'), index=False)

def generate_pca_plot(sample_df, output_dir):
    cols = [c for c in WEATHER_FEATURES if c in sample_df.columns]
    if len(cols) < 2: return
    sub = sample_df[cols + ['Active_Fire']].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 2: return
    X = StandardScaler().fit_transform(sub[cols])
    pca = PCA(n_components=2).fit(X)
    X2 = pca.transform(X)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=sub['Active_Fire'], alpha=0.5, s=10)
    plt.title('PCA of Weather Features (colored by Active_Fire)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.legend(title='Active_Fire')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 's7_pca_plot.png')); plt.close()

def save_variable_catalog(output_dir):
    catalog = [
        ('Wind_Speed','m/s','+','Higher speed → faster spread'),
        ('Wind_Direction','deg','0','Use with speed; directionality'),
        ('Max_Temp_K','K','+','Hotter → easier ignition/spread'),
        ('Min_Temp_K','K','+','Proxy of hot/dry periods'),
        ('Spec_Hum','kg/kg','-','Moisture suppresses fire'),
        ('Total_Precip','mm','-','Rain suppresses'),
        ('PDSI','index','-','Wetter → less fire'),
        ('ERC','index','+','Fuel dryness → more fire'),
        ('NDVI','index','-','Greenness/moisture → less fire'),
        ('Slope','deg','+','Uphill promotes spread'),
        ('Aspect','deg','0','Aspect interacts with sun/wind'),
        ('Elevation','m','±','Microclimate effects'),
    ]
    pd.DataFrame(catalog, columns=['feature','unit','expected_sign','note']) \
      .to_csv(os.path.join(output_dir, 's2_variable_catalog.csv'), index=False)

# -------------------- 报告整合（调用在所有函数之后） --------------------
def generate_comprehensive_report(stats_summary, sample_df, output_dir, year_label=""):
    os.makedirs(output_dir, exist_ok=True)
    log_status(f"Generating comprehensive report {year_label}", output_dir)

    # 样本数据清洗（关键）
    sample_df = prepare_sample_df(sample_df)

    # 缺失率条形图
    missing_rates = {f: s['missing_rate'] for f, s in stats_summary.items()
                     if isinstance(s, dict) and s.get('missing_rate', 0) > 0}
    if missing_rates:
        miss_df = pd.DataFrame(list(missing_rates.items()), columns=['Feature', 'MissingRate']) \
                 .sort_values('MissingRate', ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='MissingRate', y='Feature', data=miss_df)
        plt.title('Features by Missing Value Rate')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, 's3_missing_value_rates.png')); plt.close()

    # 标签不平衡 & 数据体检
    plot_fire_imbalance(sample_df, output_dir, year_label)
    data_health_checks(sample_df, output_dir)

    # 变量-标签关系
    generate_literature_comparison_table(sample_df, output_dir)
    generate_grouped_boxplots_with_ttest(sample_df, output_dir)
    fire_pointbiserial_full(sample_df, output_dir, year_label)

    # 相关性 + 重要性 + 降维
    plot_wide_correlation(sample_df, output_dir, year_label)
    generate_feature_importance_ranking(sample_df, output_dir)
    generate_pca_plot(sample_df, output_dir)

    # Landcover × 火点比例 & 变量目录
    landcover_fire_rate(sample_df, output_dir, year_label)
    save_variable_catalog(output_dir)

    # 保存统计摘要
    with open(os.path.join(output_dir, 's1_stats_summary.json'), 'w') as f:
        json.dump(stats_summary, f, indent=2)

    log_status(f"Report done {year_label}", output_dir)

# -------------------- 主流程 --------------------
def main():
    start = time.time()
    log_status("Comprehensive EDA started", OUTPUT_DIR)

    all_year_stats = OnlineStatisticsNumPy()
    all_year_sampler = BlockReservoirSampler(MAX_SAMPLE_SIZE)

    for year in YEARS_TO_ANALYZE:
        year_dir = os.path.join(OUTPUT_DIR, str(year))
        file_paths = sorted(glob.glob(f'data/processed/{year}/*.hdf5'))
        if not file_paths:
            continue
        if FILE_LIMIT_PER_YEAR:
            file_paths = file_paths[:FILE_LIMIT_PER_YEAR]

        log_status(f"Processing Year {year} | {len(file_paths)} files", year_dir)

        # 数据流式统计 + 采样
        chunk_size = int(np.ceil(len(file_paths) / MAX_WORKERS))
        chunks = [file_paths[i:i+chunk_size] for i in range(0, len(file_paths), chunk_size)]
        year_stats = OnlineStatisticsNumPy()
        year_sampler = BlockReservoirSampler(MAX_SAMPLE_SIZE)

        with Pool(processes=MAX_WORKERS) as pool:
            results = list(tqdm(pool.imap(process_file_chunk_optimized, chunks),
                                total=len(chunks), desc=f"Streaming Data {year}"))

        for s, samp in results:
            if s.n > 0:
                year_stats.merge(s)
                year_sampler.add_batch(samp.reservoir)

        # 生成年度报告
        if year_stats.n > 0:
            all_year_stats.merge(year_stats)
            all_year_sampler.add_batch(year_sampler.reservoir)
            sample_df = year_sampler.get_sample_df()
            stats_summary = year_stats.get_summary()
            generate_comprehensive_report(stats_summary, sample_df, year_dir, f"({year})")

        # 释放
        del year_stats, year_sampler, results
        gc.collect()

    # 聚合报告
    if all_year_stats.n > 0:
        agg_dir = os.path.join(OUTPUT_DIR, 'aggregate')
        os.makedirs(agg_dir, exist_ok=True)
        sample_df = all_year_sampler.get_sample_df()
        stats_summary = all_year_stats.get_summary()
        generate_comprehensive_report(stats_summary, sample_df, agg_dir, "(All Years)")

    log_status(f"Total EDA finished in {(time.time()-start)/60:.2f} minutes.", OUTPUT_DIR)

if __name__ == '__main__':
    main()
