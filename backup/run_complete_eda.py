"""
WildfireSpreadTS数据集全面EDA分析 - 主运行脚本
整合所有分析功能，一键完成全面EDA
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# 导入分析类
from comprehensive_eda import WildfireEDAAnalyzer
from comprehensive_eda_part2 import WildfireEDAAnalyzerPart2

class CompleteWildfireEDA(WildfireEDAAnalyzer, WildfireEDAAnalyzerPart2):
    """完整的EDA分析器，整合所有功能"""
    
    def __init__(self, data_dir="data/processed", output_dir="eda_results"):
        super().__init__(data_dir, output_dir)
        print("🔥 WildfireSpreadTS Complete EDA Analyzer 初始化完成")
        print("=" * 60)
    
    def run_complete_analysis(self):
        """运行完整的EDA分析流程"""
        try:
            print("🚀 开始全面EDA分析...")
            print("=" * 60)
            
            # 首先加载数据
            print("\n📦 正在加载数据样本...")
            self.load_sample_data(max_files=40, sample_ratio=0.1)
            
            # 1. 数据质量与完整性分析
            print("\n" + "="*60)
            print("📋 1. 数据质量与完整性分析")
            print("="*60)
            self.analyze_data_quality()
            
            # 2. 描述性统计分析
            print("\n" + "="*60)
            print("📊 2. 描述性统计分析")
            print("="*60)
            self.analyze_descriptive_statistics()
            
            # 3. 时空分布特征分析
            print("\n" + "="*60)
            print("🌍 3. 时空分布特征分析")
            print("="*60)
            self.analyze_spatiotemporal_patterns()
            
            # 4. 特征关系与相关性分析
            print("\n" + "="*60)
            print("🔗 4. 特征关系与相关性分析")
            print("="*60)
            self.analyze_feature_relationships()
            
            # 5. 目标变量深度分析
            print("\n" + "="*60)
            print("🎯 5. 目标变量深度分析")
            print("="*60)
            self.analyze_target_variable()
            
            # 6. 环境变量专题分析
            print("\n" + "="*60)
            print("🌡️ 6. 环境变量专题分析")
            print("="*60)
            self.analyze_environmental_variables()
            
            # 7. 数据预处理需求分析
            print("\n" + "="*60)
            print("📊 7. 数据预处理需求分析")
            print("="*60)
            self.analyze_preprocessing_requirements()
            
            # 8. 高级可视化与洞察发现
            print("\n" + "="*60)
            print("🎨 8. 高级可视化与洞察发现")
            print("="*60)
            self.create_advanced_visualizations()
            
            # 9. 生成学术报告
            print("\n" + "="*60)
            print("📝 9. 生成学术报告")
            print("="*60)
            self.generate_academic_report()
            
            # 生成总结
            print("\n" + "="*60)
            print("📋 生成分析总结")
            print("="*60)
            self.generate_analysis_summary()
            
            print("\n" + "🎉"*20)
            print("✅ 全面EDA分析完成！")
            print(f"📁 结果保存在: {self.output_dir.absolute()}")
            print("🎉"*20)
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ 分析过程中出现错误: {str(e)}")
            print("🔍 请检查数据路径和文件完整性")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_analysis_summary(self):
        """生成分析总结"""
        print("  正在生成分析总结...")
        
        summary = {
            'analysis_modules_completed': [],
            'key_findings': {},
            'recommendations': [],
            'files_generated': {
                'figures': [],
                'tables': [],
                'reports': []
            }
        }
        
        # 检查完成的分析模块
        completed_modules = []
        if 'data_quality' in self.results:
            completed_modules.append("数据质量分析")
            # 提取关键发现
            basic_info = self.results['data_quality']['basic_info']
            summary['key_findings']['data_quality'] = {
                'total_samples': basic_info['total_samples'],
                'n_fire_events': basic_info['n_fire_events'],
                'years_covered': basic_info['years_covered']
            }
        
        if 'descriptive_stats' in self.results:
            completed_modules.append("描述性统计分析")
        
        if 'spatiotemporal' in self.results:
            completed_modules.append("时空分布分析")
        
        if 'feature_relationships' in self.results:
            completed_modules.append("特征关系分析")
            if 'strong_correlations' in self.results['feature_relationships']:
                summary['key_findings']['strong_correlations'] = len(
                    self.results['feature_relationships']['strong_correlations']
                )
        
        if 'target_analysis' in self.results:
            completed_modules.append("目标变量分析")
            if 'fire_stats' in self.results['target_analysis']:
                fire_stats = self.results['target_analysis']['fire_stats']
                summary['key_findings']['fire_characteristics'] = {
                    'mean_confidence': fire_stats['mean'],
                    'max_confidence': fire_stats['max'],
                    'fire_data_points': fire_stats['count']
                }
        
        if 'environmental_analysis' in self.results:
            completed_modules.append("环境变量分析")
        
        if 'preprocessing_requirements' in self.results:
            completed_modules.append("预处理需求分析")
        
        if 'advanced_visualizations' in self.results:
            completed_modules.append("高级可视化分析")
        
        summary['analysis_modules_completed'] = completed_modules
        
        # 生成建议
        recommendations = [
            "基于极端类别不平衡，建议使用专门的损失函数（如Focal Loss）",
            "考虑实施数据增强策略以增加火点样本的多样性",
            "对不同类型的环境变量采用适当的标准化方法",
            "利用强相关特征进行特征选择和降维",
            "考虑时空建模方法来捕获火灾传播的动态特征"
        ]
        summary['recommendations'] = recommendations
        
        # 统计生成的文件
        figures_dir = self.output_dir / "figures"
        if figures_dir.exists():
            summary['files_generated']['figures'] = [f.name for f in figures_dir.glob("*.png")]
        
        tables_dir = self.output_dir / "tables"
        if tables_dir.exists():
            summary['files_generated']['tables'] = [f.name for f in tables_dir.glob("*.csv")]
        
        reports_dir = self.output_dir / "reports"
        if reports_dir.exists():
            summary['files_generated']['reports'] = [f.name for f in reports_dir.glob("*.md")]
        
        # 保存总结
        import json
        with open(self.output_dir / "analysis_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        print(f"\n📊 分析总结:")
        print(f"  ✅ 完成的分析模块: {len(completed_modules)}/8")
        print(f"  📈 生成图表数量: {len(summary['files_generated']['figures'])}")
        print(f"  📋 生成表格数量: {len(summary['files_generated']['tables'])}")
        print(f"  📝 生成报告数量: {len(summary['files_generated']['reports'])}")
        
        if 'fire_characteristics' in summary['key_findings']:
            fire_info = summary['key_findings']['fire_characteristics']
            print(f"  🔥 火点数据特征:")
            print(f"     - 平均置信度: {fire_info['mean_confidence']:.4f}")
            print(f"     - 最大置信度: {fire_info['max_confidence']:.2f}")
            print(f"     - 有效数据点: {fire_info['fire_data_points']:,}")
        
        print(f"  📁 详细结果保存在: analysis_summary.json")

def main():
    """主函数"""
    print("🔥 WildfireSpreadTS 数据集全面EDA分析系统")
    print("📊 覆盖9个主要分析模块，生成论文级别的分析结果")
    print("=" * 80)
    
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
    print(f"📁 数据目录: {data_dir.absolute()}")
    
    # 创建分析器并运行分析
    analyzer = CompleteWildfireEDA(
        data_dir=str(data_dir), 
        output_dir="eda_results_complete"
    )
    
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n🎯 快速访问主要结果:")
        print("  📊 图表目录: eda_results_complete/figures/")
        print("  📋 表格目录: eda_results_complete/tables/")
        print("  📝 报告目录: eda_results_complete/reports/")
        print("  📈 分析总结: eda_results_complete/analysis_summary.json")
    
    print("\n" + "="*80)
    print("感谢使用 WildfireSpreadTS EDA 分析系统！")
    print("🔬 Happy Research! 🔥")

if __name__ == "__main__":
    main() 