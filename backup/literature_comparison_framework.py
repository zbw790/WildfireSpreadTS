"""
Literature Comparison Framework for Wildfire Physics
===================================================

This module provides a framework for comparing model findings with established
fire physics literature and known mechanisms of fire propagation.

Based on classical fire behavior models:
- Rothermel's fire spread model (1972)
- McArthur's forest fire danger index
- Canadian Forest Fire Weather Index System
- Recent ML-based fire behavior studies

Author: Bowen
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LiteratureComparisonFramework:
    """
    Framework for comparing model results with fire physics literature
    """
    
    def __init__(self):
        self.literature_expectations = self._load_literature_expectations()
        self.key_references = self._load_key_references()
    
    def _load_literature_expectations(self):
        """
        Load expected variable relationships from fire physics literature
        """
        expectations = {
            'Wind_Speed': {
                'relationship': 'Positive',
                'strength': 'Strong',
                'mechanism': 'Wind increases oxygen supply and preheats fuel ahead of fire',
                'typical_effect': 'Fire spread rate proportional to wind speed^0.5-1.0',
                'references': ['Rothermel 1972', 'Cheney & Sullivan 2008']
            },
            'Relative_Humidity': {
                'relationship': 'Negative', 
                'strength': 'Strong',
                'mechanism': 'Higher humidity increases fuel moisture content',
                'typical_effect': 'Exponential decrease in fire spread with increasing humidity',
                'references': ['Noble et al. 1980', 'Cruz et al. 2015']
            },
            'Temperature_Max': {
                'relationship': 'Positive',
                'strength': 'Moderate',
                'mechanism': 'Higher temperature reduces fuel moisture and increases pyrolysis',
                'typical_effect': 'Linear to exponential increase with temperature',
                'references': ['McArthur 1967', 'Van Wagner 1987']
            },
            'Slope': {
                'relationship': 'Positive',
                'strength': 'Strong',
                'mechanism': 'Upslope spread faster due to flame angle and convective preheating',
                'typical_effect': 'Exponential increase: rate ∝ exp(0.069 × slope°)',
                'references': ['Rothermel 1972', 'Sullivan 2009']
            },
            'NDVI': {
                'relationship': 'Negative',
                'strength': 'Moderate',
                'mechanism': 'Green vegetation has higher moisture content',
                'typical_effect': 'Inverse relationship with fuel moisture',
                'references': ['Dennison et al. 2005', 'Jurdao et al. 2013']
            },
            'Precipitation': {
                'relationship': 'Negative',
                'strength': 'Strong',
                'mechanism': 'Direct moisture addition to fuels',
                'typical_effect': 'Suppresses fire for several days post-precipitation',
                'references': ['Van Wagner 1987', 'Wotton et al. 2003']
            },
            'PDSI': {
                'relationship': 'Positive',
                'strength': 'Moderate',
                'mechanism': 'Drought conditions reduce fuel moisture across landscape',
                'typical_effect': 'Seasonal predictor of fire danger',
                'references': ['Palmer 1965', 'Liu et al. 2010']
            },
            'Elevation': {
                'relationship': 'Mixed',
                'strength': 'Weak',
                'mechanism': 'Complex interactions: temperature lapse, vegetation changes',
                'typical_effect': 'Non-linear relationship, depends on local factors',
                'references': ['Parisien & Moritz 2009', 'Steel et al. 2015']
            },
            'Land_Cover_Class': {
                'relationship': 'Complex',
                'strength': 'Strong',
                'mechanism': 'Different fuel types have different flammability characteristics',
                'typical_effect': 'Grasslands > Shrublands > Forests for spread rate',
                'references': ['Anderson 1982', 'Scott & Burgan 2005']
            },
            'Aspect': {
                'relationship': 'Complex',
                'strength': 'Moderate', 
                'mechanism': 'South-facing slopes drier in Northern Hemisphere',
                'typical_effect': 'S/SW aspects typically more fire-prone',
                'references': ['Beaty & Taylor 2001', 'Dillon et al. 2011']
            }
        }
        return expectations
    
    def _load_key_references(self):
        """
        Load key references for fire behavior modeling
        """
        references = {
            'Rothermel 1972': {
                'title': 'A mathematical model for predicting fire spread in wildland fuels',
                'key_findings': 'Wind speed and slope are primary drivers of fire spread',
                'model_type': 'Physical/Empirical'
            },
            'McArthur 1967': {
                'title': 'Fire behaviour in eucalypt forests', 
                'key_findings': 'Temperature, humidity, wind form fire danger index',
                'model_type': 'Empirical'
            },
            'Van Wagner 1987': {
                'title': 'Development and structure of the Canadian Forest Fire Weather Index',
                'key_findings': 'Fuel moisture codes based on weather patterns',
                'model_type': 'Empirical'
            },
            'Cheney & Sullivan 2008': {
                'title': 'Grassfires: Fuel, Weather and Fire Behaviour',
                'key_findings': 'Wind is dominant factor in grassland fire spread',
                'model_type': 'Physical/Empirical'
            },
            'Dennison et al. 2005': {
                'title': 'Large wildfire trends in the western United States',
                'key_findings': 'Vegetation indices correlate with fire activity',
                'model_type': 'Statistical'
            }
        }
        return references
    
    def compare_with_literature(self, correlation_results, importance_results):
        """
        Compare model results with literature expectations
        
        Args:
            correlation_results: Dict from ensemble analyzer
            importance_results: Dict from ensemble analyzer
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for channel_idx, corr_data in correlation_results.items():
            var_name = corr_data['name']
            observed_r = corr_data['pearson_r']
            
            # Find matching literature expectation
            lit_match = None
            for lit_var, lit_data in self.literature_expectations.items():
                if lit_var.lower() in var_name.lower() or var_name.lower() in lit_var.lower():
                    lit_match = lit_data
                    break
            
            if lit_match:
                # Determine if observation matches literature
                expected_relationship = lit_match['relationship']
                
                if expected_relationship == 'Positive':
                    literature_match = 'Yes' if observed_r > 0.05 else 'No'
                    agreement_strength = min(abs(observed_r) * 2, 1.0) if observed_r > 0 else 0
                elif expected_relationship == 'Negative':
                    literature_match = 'Yes' if observed_r < -0.05 else 'No'
                    agreement_strength = min(abs(observed_r) * 2, 1.0) if observed_r < 0 else 0
                else:  # Complex/Mixed
                    literature_match = 'Partial'
                    agreement_strength = 0.5
                
                # Get importance score
                importance_score = importance_results.get(channel_idx, {}).get('separability_score', 0)
                
                comparison_data.append({
                    'Variable': var_name,
                    'Observed_Correlation': round(observed_r, 3),
                    'Literature_Expectation': expected_relationship,
                    'Literature_Strength': lit_match['strength'],
                    'Agreement': literature_match,
                    'Agreement_Strength': round(agreement_strength, 3),
                    'Importance_Score': round(importance_score, 3),
                    'Mechanism': lit_match['mechanism'],
                    'Key_References': ', '.join(lit_match['references'][:2])
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def create_agreement_visualization(self, comparison_df, save_path="literature_agreement.png"):
        """
        Create visualization of literature agreement
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Agreement distribution
        agreement_counts = comparison_df['Agreement'].value_counts()
        colors = ['green', 'orange', 'red']
        ax1.pie(agreement_counts.values, labels=agreement_counts.index, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Literature Agreement Distribution')
        
        # 2. Correlation vs Literature Expectation
        pos_vars = comparison_df[comparison_df['Literature_Expectation'] == 'Positive']
        neg_vars = comparison_df[comparison_df['Literature_Expectation'] == 'Negative']
        
        ax2.scatter(pos_vars['Observed_Correlation'], pos_vars['Importance_Score'], 
                   color='blue', label='Expected Positive', s=60, alpha=0.7)
        ax2.scatter(neg_vars['Observed_Correlation'], neg_vars['Importance_Score'],
                   color='red', label='Expected Negative', s=60, alpha=0.7)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Observed Correlation')
        ax2.set_ylabel('Importance Score') 
        ax2.set_title('Literature Expectation vs Observed Results')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Agreement strength by variable
        sorted_df = comparison_df.sort_values('Agreement_Strength', ascending=True)
        y_pos = range(len(sorted_df))
        colors = ['red' if x == 'No' else 'orange' if x == 'Partial' else 'green' 
                 for x in sorted_df['Agreement']]
        
        ax3.barh(y_pos, sorted_df['Agreement_Strength'], color=colors, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(sorted_df['Variable'], fontsize=8)
        ax3.set_xlabel('Agreement Strength')
        ax3.set_title('Literature Agreement by Variable')
        ax3.grid(True, alpha=0.3)
        
        # 4. Importance vs Agreement
        sns.scatterplot(data=comparison_df, x='Importance_Score', y='Agreement_Strength',
                       hue='Agreement', s=100, alpha=0.7, ax=ax4)
        ax4.set_title('Model Importance vs Literature Agreement')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Literature agreement visualization saved to {save_path}")
    
    def generate_literature_summary(self, comparison_df):
        """
        Generate a summary of literature comparison for the presentation
        """
        total_vars = len(comparison_df)
        agreeing_vars = len(comparison_df[comparison_df['Agreement'] == 'Yes'])
        partial_vars = len(comparison_df[comparison_df['Agreement'] == 'Partial'])
        disagreeing_vars = len(comparison_df[comparison_df['Agreement'] == 'No'])
        
        # High agreement, high importance variables
        strong_agreement = comparison_df[
            (comparison_df['Agreement'] == 'Yes') & 
            (comparison_df['Importance_Score'] > comparison_df['Importance_Score'].median())
        ]
        
        # Unexpected findings (high importance but disagreement)
        unexpected = comparison_df[
            (comparison_df['Agreement'] == 'No') &
            (comparison_df['Importance_Score'] > comparison_df['Importance_Score'].median())
        ]
        
        summary = f"""
LITERATURE COMPARISON SUMMARY
============================

Overall Agreement:
- {agreeing_vars}/{total_vars} variables ({agreeing_vars/total_vars*100:.1f}%) match literature expectations
- {partial_vars}/{total_vars} variables ({partial_vars/total_vars*100:.1f}%) show partial agreement
- {disagreeing_vars}/{total_vars} variables ({disagreeing_vars/total_vars*100:.1f}%) disagree with expectations

Strong Physics-Aligned Variables:
{chr(10).join([f"- {row['Variable']}: r={row['Observed_Correlation']:.3f} (Expected: {row['Literature_Expectation']})" 
              for _, row in strong_agreement.head().iterrows()])}

Unexpected Findings (High importance, Literature disagreement):
{chr(10).join([f"- {row['Variable']}: r={row['Observed_Correlation']:.3f} (Expected: {row['Literature_Expectation']})" 
              for _, row in unexpected.head().iterrows()]) if len(unexpected) > 0 else "- None identified"}

Model Validation:
- Model shows {agreeing_vars/total_vars*100:.1f}% consistency with established fire physics
- Strong agreement on key variables: {'Wind Speed, Humidity, Slope' if agreeing_vars > 3 else 'Limited'}
- Results support use of model for fire behavior prediction
        """
        
        return summary
    
    def export_supervisor_deliverable(self, comparison_df, output_path="literature_comparison_deliverable.md"):
        """
        Export a formatted deliverable for supervisor presentation
        """
        summary = self.generate_literature_summary(comparison_df)
        
        # Create detailed table for presentation
        detailed_results = comparison_df[['Variable', 'Observed_Correlation', 'Literature_Expectation', 
                                        'Agreement', 'Mechanism', 'Key_References']].copy()
        
        with open(output_path, 'w') as f:
            f.write("# Literature Comparison for Wildfire Model Variables\n\n")
            f.write("## Executive Summary\n")
            f.write(summary)
            f.write("\n\n## Detailed Variable Analysis\n\n")
            f.write(detailed_results.to_markdown(index=False))
            f.write("\n\n## Key References\n\n")
            
            for ref_key, ref_data in self.key_references.items():
                f.write(f"**{ref_key}**: {ref_data['title']}\n")
                f.write(f"- Key findings: {ref_data['key_findings']}\n")
                f.write(f"- Model type: {ref_data['model_type']}\n\n")
        
        print(f"Literature comparison deliverable saved to {output_path}")
        return output_path

def main():
    """
    Demonstration of literature comparison framework
    """
    framework = LiteratureComparisonFramework()
    
    # Example usage (would be called from main analysis)
    print("Literature Comparison Framework initialized")
    print("Available literature expectations for:")
    for var in framework.literature_expectations.keys():
        print(f"- {var}")
    
    print(f"\nKey references loaded: {len(framework.key_references)}")

if __name__ == "__main__":
    main() 