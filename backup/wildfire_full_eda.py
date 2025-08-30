"""
WildfireSpreadTS数据集全面EDA分析系统 - 增强版
处理全部607个HDF5文件，专门解决NaN值问题
"""

import numpy as np
import pandas as pd
import h5py
import glob
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class WildfireFullEDA:
    """处理全部607个文件的强化EDA系统"""
    
    def __init__(self, data_dir="data/processed", output_dir="full_eda_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 23通道特征定义
        self.feature_names = [
            'VIIRS_I4', 'VIIRS_I5', 'VIIRS_M13', 'NDVI', 'EVI2', 
            'Temperature', 'Humidity', 'Wind_Direction', 'Wind_Speed', 'Precipitation',
            'Surface_Pressure', 'Solar_Radiation', 'Elevation', 'Slope', 'Aspect',
            'PDSI', 'Land_Cover', 'Forecast_Temperature', 'Forecast_Humidity', 
            'Forecast_Wind_Direction', 'Forecast_Wind_Speed', 'Forecast_Precipitation',
            'Active_Fire_Confidence'
        ]
        
    def analyze_all_files(self, max_files=607):
        """分析所有或指定数量的文件"""
        print(f"🔥 开始全面数据质量分析 - 最多处理 {max_files} 个文件")
        
        # 收集所有HDF5文件
        all_files = []
        for year in ['2018', '2019', '2020', '2021']:
            year_dir = self.data_dir / year
            if year_dir.exists():
                year_files = list(year_dir.glob("*.hdf5"))
                all_files.extend(year_files)
        
        print(f"找到总计 {len(all_files)} 个HDF5文件")
        
        # 限制文件数量
        files_to_process = all_files[:max_files] if max_files < len(all_files) else all_files
        print(f"将处理 {len(files_to_process)} 个文件")
        
        # 初始化统计
        total_samples = 0
        nan_counts = np.zeros(23)
        value_ranges = {'min': np.full(23, np.inf), 'max': np.full(23, -np.inf)}
        all_means = []
        fire_events = []
        
        print("\n📊 逐个分析文件...")
        
        for i, file_path in enumerate(files_to_process):
            try:
                with h5py.File(file_path, 'r') as f:
                    data = f['data'][:]  # Shape: (T, C, H, W)
                    
                    # 获取火灾事件名称
                    fire_name = f['data'].attrs.get('fire_name', file_path.stem)
                    if isinstance(fire_name, bytes):
                        fire_name = fire_name.decode('utf-8')
                    fire_events.append(fire_name)
                    
                    # 重塑数据为 (N, 23)
                    T, C, H, W = data.shape
                    reshaped_data = data.transpose(0, 2, 3, 1).reshape(-1, C)
                    
                    # 统计
                    total_samples += len(reshaped_data)
                    
                    # NaN计数
                    nan_counts += np.isnan(reshaped_data).sum(axis=0)
                    
                    # 值范围统计
                    for ch in range(23):
                        channel_data = reshaped_data[:, ch]
                        valid_data = channel_data[np.isfinite(channel_data)]
                        if len(valid_data) > 0:
                            value_ranges['min'][ch] = min(value_ranges['min'][ch], valid_data.min())
                            value_ranges['max'][ch] = max(value_ranges['max'][ch], valid_data.max())
                    
                    # 每个文件的均值
                    file_means = []
                    for ch in range(23):
                        channel_data = reshaped_data[:, ch]
                        valid_data = channel_data[np.isfinite(channel_data)]
                        file_means.append(np.mean(valid_data) if len(valid_data) > 0 else np.nan)
                    all_means.append(file_means)
                    
                if (i + 1) % 50 == 0:
                    print(f"  已处理 {i + 1}/{len(files_to_process)} 个文件...")
                    
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        # 生成报告
        self.generate_quality_report(total_samples, nan_counts, value_ranges, all_means, fire_events)
        return total_samples, nan_counts, value_ranges
    
    def generate_quality_report(self, total_samples, nan_counts, value_ranges, all_means, fire_events):
        """生成数据质量报告"""
        print(f"\n📋 生成数据质量报告...")
        
        # 计算NaN比例
        nan_ratios = nan_counts / total_samples
        
        # 创建报告DataFrame
        quality_df = pd.DataFrame({
            'Channel_ID': range(23),
            'Feature_Name': self.feature_names,
            'Total_Samples': [total_samples] * 23,
            'NaN_Count': nan_counts.astype(int),
            'NaN_Ratio': nan_ratios,
            'Min_Value': value_ranges['min'],
            'Max_Value': value_ranges['max'],
            'Has_NaN_Problem': nan_ratios > 0.01,  # 超过1%认为有问题
        })
        
        # 保存详细报告
        quality_df.to_csv(self.output_dir / "data_quality_full_analysis.csv", index=False)
        
        # 生成汇总报告
        summary_report = f"""
# WildfireSpreadTS 全面数据质量分析报告

## 数据概览
- **总样本数**: {total_samples:,}
- **总文件数**: {len(fire_events)}
- **独特火灾事件**: {len(set(fire_events))}
- **特征数**: 23

## NaN值问题分析

### 严重NaN问题 (>5%):
"""
        
        severe_nan = quality_df[quality_df['NaN_Ratio'] > 0.05]
        if len(severe_nan) > 0:
            for _, row in severe_nan.iterrows():
                summary_report += f"- **{row['Feature_Name']}**: {row['NaN_Ratio']:.1%} NaN值\n"
        else:
            summary_report += "无严重NaN问题\n"
        
        summary_report += f"""
### 中等NaN问题 (1-5%):
"""
        
        moderate_nan = quality_df[(quality_df['NaN_Ratio'] > 0.01) & (quality_df['NaN_Ratio'] <= 0.05)]
        if len(moderate_nan) > 0:
            for _, row in moderate_nan.iterrows():
                summary_report += f"- **{row['Feature_Name']}**: {row['NaN_Ratio']:.1%} NaN值\n"
        else:
            summary_report += "无中等NaN问题\n"
        
        summary_report += f"""
### 轻微NaN问题 (<1%):
"""
        
        light_nan = quality_df[(quality_df['NaN_Ratio'] > 0) & (quality_df['NaN_Ratio'] <= 0.01)]
        if len(light_nan) > 0:
            for _, row in light_nan.iterrows():
                summary_report += f"- **{row['Feature_Name']}**: {row['NaN_Ratio']:.1%} NaN值\n"
        else:
            summary_report += "无轻微NaN问题\n"
        
        # NaN处理建议
        summary_report += f"""
## NaN值处理建议

### 推荐策略:
1. **严重NaN通道**: 考虑特征工程或删除
2. **中等NaN通道**: 使用插值或基于邻近像素的方法
3. **轻微NaN通道**: 简单均值或中位数插值

### 针对CNN训练的具体建议:
- 使用 `sklearn.impute.SimpleImputer` 进行中位数插值
- 对于土地覆盖等分类特征，使用最频繁值填充
- 考虑添加NaN指示特征来保留缺失信息

## 数据范围分析
"""
        
        # 值范围异常检测
        for i, name in enumerate(self.feature_names):
            min_val = value_ranges['min'][i]
            max_val = value_ranges['max'][i]
            if np.isfinite(min_val) and np.isfinite(max_val):
                summary_report += f"- **{name}**: [{min_val:.2f}, {max_val:.2f}]\n"
        
        # 保存汇总报告
        with open(self.output_dir / "quality_summary_report.md", 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        # 打印关键发现
        print(f"\n✅ 分析完成！关键发现:")
        print(f"📊 总样本数: {total_samples:,}")
        print(f"🔥 火灾事件数: {len(set(fire_events))}")
        
        problematic_channels = quality_df[quality_df['NaN_Ratio'] > 0.01]
        if len(problematic_channels) > 0:
            print(f"⚠️  有{len(problematic_channels)}个通道存在>1%的NaN值:")
            for _, row in problematic_channels.iterrows():
                print(f"   - {row['Feature_Name']}: {row['NaN_Ratio']:.1%}")
        else:
            print(f"✅ 所有通道的NaN值比例都<1%")
        
        print(f"\n📁 详细报告保存在: {self.output_dir}")


def main():
    """主函数"""
    print("🔥 WildfireSpreadTS全面数据质量分析系统")
    print("=" * 60)
    print("专门解决NaN值问题，分析全部607个文件")
    print("=" * 60)
    
    try:
        analyzer = WildfireFullEDA()
        
        # 用户选择
        print("\n📋 选择分析范围:")
        print("1. 完整分析 (全部607个文件)")
        print("2. 大样本分析 (400个文件)")
        print("3. 中样本分析 (200个文件)")
        print("4. 快速分析 (100个文件)")
        
        choice = input("请选择 (1-4, 默认1): ").strip()
        
        if choice == "2":
            max_files = 400
        elif choice == "3":
            max_files = 200
        elif choice == "4":
            max_files = 100
        else:
            max_files = 607  # 默认全部
        
        # 运行分析
        analyzer.analyze_all_files(max_files=max_files)
        
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 