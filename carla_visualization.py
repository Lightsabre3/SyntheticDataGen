#!/usr/bin/env python3
"""
CARLA Autonomous Driving Test Results Visualization

This script visualizes the performance data from CARLA autonomous driving tests 
across different model comparisons using Seaborn and Matplotlib.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for NeurIPS publication-quality visualizations
plt.style.use('default')

# Define consistent color palette for models
MODEL_COLORS = {
    'baseline': '#1f77b4',    # Blue
    'synthetic': '#ff7f0e',   # Orange  
    'BDD100K': '#2ca02c'      # Green
}

# Set NeurIPS-style font configuration
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern', 'Times New Roman', 'DejaVu Serif', 'serif']
plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is available
plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern math fonts

def clean_scenario_names(df):
    """Convert scenario names from snake case to plain English"""
    scenario_mapping = {
        'Rural Highway Consistency': 'Highway Consistency',
        'Rural Curves & Hills': 'Curves & Hills',
        'Rural Wet Weather': 'Wet Weather',
        'Rural Night Driving': 'Night Driving',
        'Rural Dawn Conditions': 'Dawn Conditions'
    }
    
    df['scenario_clean'] = df['scenario'].map(scenario_mapping).fillna(df['scenario'])
    return df

def clean_comparison_names(df):
    """Convert comparison folder names to descriptive training data labels"""
    comparison_mapping = {
        '10k_vs_5k': '10K Synthetic vs 5K BDD100K',
        '20k_vs_10k': '20K Synthetic vs 10K BDD100K', 
        '5k_vs_10k': '5K Synthetic vs 10K BDD100K'
    }
    
    df['comparison_clean'] = df['comparison'].map(comparison_mapping).fillna(df['comparison'])
    return df

def clean_model_names(df):
    """Add more descriptive model names with training data context"""
    # Create a combined label that shows model type and training context
    model_context = []
    for _, row in df.iterrows():
        model = row['model']
        comparison = row['comparison']
        
        if model == 'baseline':
            model_context.append('Baseline (Rule-based)')
        elif model == 'synthetic':
            if comparison == '5k_vs_10k':
                model_context.append('Synthetic (5K Images)')
            elif comparison == '10k_vs_5k':
                model_context.append('Synthetic (10K Images)')
            elif comparison == '20k_vs_10k':
                model_context.append('Synthetic (20K Images)')
            else:
                model_context.append('Synthetic')
        elif model == 'BDD100K':
            if comparison == '5k_vs_10k':
                model_context.append('BDD100K (10K Images)')
            elif comparison == '10k_vs_5k':
                model_context.append('BDD100K (5K Images)')
            elif comparison == '20k_vs_10k':
                model_context.append('BDD100K (10K Images)')
            else:
                model_context.append('BDD100K')
        else:
            model_context.append(model)
    
    df['model_with_context'] = model_context
    return df

def load_carla_data():
    """Load data from all JSON files"""
    data_files = {
        '10k_vs_5k': '10k vs 5k/carla_realistic_results.json',
        '20k_vs_10k': '20k-vs-10k/carla_realistic_results-8-23.json',
        '5k_vs_10k': '5k-vs-10k/carla_realistic_results.json'
    }
    
    all_data = []
    
    for comparison, file_path in data_files.items():
        try:
            print(f"Loading {file_path}...")
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract test results
            if 'test_results' in data:
                for test in data['test_results']:
                    test['comparison'] = comparison
                    all_data.append(test)
                print(f"  Loaded {len(data['test_results'])} tests")
            else:
                print(f"  No 'test_results' found in {file_path}")
                
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    
    df = pd.DataFrame(all_data)
    # Capitalize bdd100k to BDD100K for consistency
    df['model'] = df['model'].replace('bdd100k', 'BDD100K')
    df = clean_scenario_names(df)
    df = clean_comparison_names(df)
    df = clean_model_names(df)
    return df

def create_performance_plots(df):
    """Create performance comparison plots"""
    model_palette = [MODEL_COLORS[model] for model in df['model'].unique()]
    
    # Chart 1: Actual Performance by Model
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='model', y='actual_performance', ax=ax, 
                palette=model_palette, linewidth=1.5)
    ax.set_title('Performance by Model')
    ax.set_ylabel('Performance Score')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.savefig('fig1_performance_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 2: Performance Score Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df, x='model', y='performance_score', ax=ax, 
                   palette=model_palette, linewidth=1.5)
    ax.set_title('Performance Score Distribution')
    ax.set_ylabel('Performance Score')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.savefig('fig2_performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 3: Expected vs Actual Performance
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='expected_performance', y='actual_performance', 
                    hue='model', style='comparison_clean', s=60, ax=ax,
                    palette=MODEL_COLORS)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, linewidth=1, label='Perfect Match')
    ax.set_title('Expected vs Actual Performance')
    ax.set_xlabel('Expected Performance')
    ax.set_ylabel('Actual Performance')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('fig3_expected_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 4: Performance by Training Data Size
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x='comparison_clean', y='actual_performance', hue='model', 
                ax=ax, palette=MODEL_COLORS)
    ax.set_title('Performance by Training Data Size')
    ax.set_ylabel('Average Performance')
    ax.set_xlabel('Training Data Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig4_performance_by_training_size.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_scenario_plots(df):
    """Create scenario-based analysis plots"""
    
    # Chart 5: Performance by Scenario (Horizontal)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, y='scenario_clean', x='actual_performance', hue='model', 
                ax=ax, palette=MODEL_COLORS, linewidth=1.5, orient='h')
    ax.set_title('Performance by Scenario')
    ax.set_xlabel('Performance Score')
    ax.set_ylabel('Scenario')
    ax.legend(title='Model', loc='lower right')
    
    # Add lighter colored labels
    for i, scenario in enumerate(df['scenario_clean'].unique()):
        scenario_data = df[df['scenario_clean'] == scenario]
        mean_perf = scenario_data['actual_performance'].mean()
        ax.text(mean_perf + 2, i, f'{mean_perf:.1f}', 
                ha='left', va='center', fontsize=9, 
                color='lightgray', alpha=0.8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig5_performance_by_scenario.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 6: Performance Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    scenario_performance = df.groupby(['scenario_clean', 'model'])['actual_performance'].mean().unstack()
    # Transpose the data to swap x and y axes
    scenario_performance_transposed = scenario_performance.T
    sns.heatmap(scenario_performance_transposed, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, 
                annot_kws={'fontsize': 10, 'color': 'black', 'alpha': 1.0}, 
                cbar_kws={'label': 'Performance Score'})
    ax.set_title('Performance Heatmap by Scenario and Model')
    ax.set_ylabel('Model')
    ax.set_xlabel('Scenario')
    
    plt.tight_layout()
    plt.savefig('fig6_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_safety_plots(df):
    """Create safety and collision analysis plots"""
    model_palette = [MODEL_COLORS[model] for model in df['model'].unique()]
    
    # Chart 7: Collisions by Model
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='model', y='collisions', ax=ax, 
                palette=model_palette, linewidth=1.5)
    ax.set_title('Collisions by Model')
    ax.set_ylabel('Number of Collisions')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.savefig('fig7_collisions_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 8: Lane Invasions by Model
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='model', y='lane_invasions', ax=ax, 
                palette=model_palette, linewidth=1.5)
    ax.set_title('Lane Invasions by Model')
    ax.set_ylabel('Number of Lane Invasions')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.savefig('fig8_lane_invasions_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 9: Collisions vs Performance
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x='collisions', y='actual_performance', 
                    hue='model', style='scenario_clean', alpha=0.8, s=60, ax=ax,
                    palette=MODEL_COLORS)
    ax.set_title('Collisions vs Performance')
    ax.set_xlabel('Number of Collisions')
    ax.set_ylabel('Performance Score')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('fig9_collisions_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 10: Safety Score Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x='safety_score', hue='model', multiple='stack', 
                 ax=ax, palette=MODEL_COLORS)
    ax.set_title('Safety Score Distribution')
    ax.set_xlabel('Safety Score')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig10_safety_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_behavior_plots(df):
    """Create driving behavior analysis plots"""
    model_palette = [MODEL_COLORS[model] for model in df['model'].unique()]
    
    # Chart 11: Distance Traveled by Model
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='model', y='distance_traveled', ax=ax, 
                palette=model_palette, linewidth=1.5)
    ax.set_title('Distance Traveled by Model')
    ax.set_ylabel('Distance (units)')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.savefig('fig11_distance_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 12: Average Speed by Model
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='model', y='average_speed', ax=ax, 
                palette=model_palette, linewidth=1.5)
    ax.set_title('Average Speed by Model')
    ax.set_ylabel('Speed (units/s)')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.savefig('fig12_average_speed_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 13: Maximum Speed by Model
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='model', y='max_speed', ax=ax, 
                palette=model_palette, linewidth=1.5)
    ax.set_title('Maximum Speed by Model')
    ax.set_ylabel('Max Speed (units/s)')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.savefig('fig13_max_speed_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 14: Steering Smoothness by Model
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='model', y='steering_smoothness', ax=ax, 
                palette=model_palette, linewidth=1.5)
    ax.set_title('Steering Smoothness by Model')
    ax.set_ylabel('Smoothness Score')
    ax.set_xlabel('Model')
    plt.tight_layout()
    plt.savefig('fig14_steering_smoothness_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_environmental_plots(df):
    """Create weather and environmental condition plots"""
    
    # Chart 15: Performance by Weather Condition
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x='weather', y='actual_performance', hue='model', 
                ax=ax, palette=MODEL_COLORS, linewidth=1.5)
    ax.set_title('Performance by Weather Condition')
    ax.set_ylabel('Performance Score')
    ax.set_xlabel('Weather Condition')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig15_performance_by_weather.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 16: Collisions by Weather Condition
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x='weather', y='collisions', hue='model', 
                ax=ax, palette=MODEL_COLORS, linewidth=1.5)
    ax.set_title('Collisions by Weather Condition')
    ax.set_ylabel('Number of Collisions')
    ax.set_xlabel('Weather Condition')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig16_collisions_by_weather.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 17: Performance by Complexity Level
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='complexity', y='actual_performance', hue='model', 
                ax=ax, palette=MODEL_COLORS, linewidth=1.5)
    ax.set_title('Performance by Scenario Complexity')
    ax.set_ylabel('Performance Score')
    ax.set_xlabel('Complexity Level')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig17_performance_by_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 18: Performance by Map
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='map', y='actual_performance', hue='model', 
                ax=ax, palette=MODEL_COLORS, linewidth=1.5)
    ax.set_title('Performance by Map')
    ax.set_ylabel('Performance Score')
    ax.set_xlabel('Map')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig18_performance_by_map.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_plot(df):
    """Create correlation matrix plot"""
    numerical_cols = ['expected_performance', 'actual_performance', 'distance_traveled', 
                      'average_speed', 'max_speed', 'steering_smoothness', 'safety_score',
                      'collisions', 'lane_invasions', 'performance_score']
    
    correlation_matrix = df[numerical_cols].corr()
    
    # Chart 19: Correlation Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', annot_kws={'fontsize': 8},
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax)
    ax.set_title('Correlation Matrix of Performance Metrics')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('fig19_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_plots(df):
    """Create model comparison summary plots"""
    
    # Chart 20: Average Performance by Training Data Size
    fig, ax = plt.subplots(figsize=(8, 4))
    perf_by_comp = df.groupby(['comparison_clean', 'model'])['actual_performance'].mean().unstack()
    perf_by_comp.plot(kind='bar', ax=ax, width=0.8, color=[MODEL_COLORS[col] for col in perf_by_comp.columns])
    ax.set_title('Performance by Training Data Size')
    ax.set_ylabel('Performance Score')
    ax.set_xlabel('Training Data Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Model')
    plt.tight_layout()
    plt.savefig('fig20_performance_by_training_data.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 21: Model Ranking
    fig, ax = plt.subplots(figsize=(6, 4))
    model_ranking = df.groupby('model')['actual_performance'].mean().sort_values(ascending=False)
    ranking_colors = [MODEL_COLORS[model] for model in model_ranking.index]
    model_ranking.plot(kind='bar', ax=ax, color=ranking_colors, width=0.6)
    ax.set_title('Overall Model Ranking')
    ax.set_ylabel('Average Performance')
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig('fig21_model_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 22: Performance Consistency
    fig, ax = plt.subplots(figsize=(6, 4))
    consistency = df.groupby('model')['actual_performance'].agg(['mean', 'std'])
    consistency['cv'] = consistency['std'] / consistency['mean']
    consistency_colors = [MODEL_COLORS[model] for model in consistency.index]
    consistency['cv'].plot(kind='bar', ax=ax, width=0.6, color=consistency_colors)
    ax.set_title('Performance Consistency')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig('fig22_performance_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_training_data_plots(df):
    """Create plots specifically showing training data impact"""
    
    # Create consistent colors for model_with_context
    unique_contexts = df['model_with_context'].unique()
    context_colors = []
    for context in unique_contexts:
        if 'Baseline' in context:
            context_colors.append(MODEL_COLORS['baseline'])
        elif 'Synthetic' in context:
            context_colors.append(MODEL_COLORS['synthetic'])
        elif 'BDD100K' in context:
            context_colors.append(MODEL_COLORS['BDD100K'])
        else:
            context_colors.append('#gray')
    
    # Chart 23: Performance by Model and Training Data Size
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df, x='model_with_context', y='actual_performance', 
                ax=ax, palette=context_colors, linewidth=1.5)
    ax.set_title('Performance by Model and Training Data Size')
    ax.set_ylabel('Performance Score')
    ax.set_xlabel('Model (Training Data)')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('fig23_performance_by_model_training_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 24: Collisions by Model and Training Data Size
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df, x='model_with_context', y='collisions', 
                ax=ax, palette=context_colors, linewidth=1.5)
    ax.set_title('Collisions by Model and Training Data Size')
    ax.set_ylabel('Number of Collisions')
    ax.set_xlabel('Model (Training Data)')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('fig24_collisions_by_model_training_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 25: Performance Distribution by Training Data
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(data=df, x='comparison_clean', y='actual_performance', hue='model', 
                   ax=ax, palette=MODEL_COLORS)
    ax.set_title('Performance Distribution by Training Data')
    ax.set_ylabel('Performance Score')
    ax.set_xlabel('Training Data Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig25_performance_distribution_by_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Chart 26: Training Data Impact on Challenging Scenarios
    fig, ax = plt.subplots(figsize=(8, 4))
    scenario_subset = df[df['scenario_clean'].isin(['Night Driving', 'Wet Weather'])]
    sns.barplot(data=scenario_subset, x='scenario_clean', y='actual_performance', 
                hue='comparison_clean', ax=ax)
    ax.set_title('Training Data Impact on Challenging Scenarios')
    ax.set_ylabel('Average Performance')
    ax.set_xlabel('Challenging Scenarios')
    ax.legend(title='Training Data', fontsize=8)
    plt.tight_layout()
    plt.savefig('fig26_training_impact_challenging_scenarios.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_data_tables(df):
    """Create publication-quality tables showing raw data"""
    
    models = ['baseline', 'synthetic', 'BDD100K']
    
    # Table 1: Performance Summary by Model
    print("Creating performance summary table...")
    
    # Performance metrics by model - simplified
    perf_data = []
    for model in models:
        model_data = df[df['model'] == model]
        perf_data.append([
            model.title(),
            f"{model_data['actual_performance'].mean():.1f}",
            f"{model_data['actual_performance'].std():.1f}",
            f"{model_data['collisions'].mean():.0f}",
            f"{model_data['lane_invasions'].mean():.1f}",
            f"{model_data['steering_smoothness'].mean():.1f}",
            len(model_data)
        ])
    
    # Create minimal figure with just the table
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    table1 = ax.table(cellText=perf_data,
                      colLabels=['Model', 'Mean Perf', 'Std Perf', 'Mean Coll', 
                               'Mean Lane Inv', 'Mean Steering', 'Tests'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 1.5)
    
    # Style header row
    for i in range(len(perf_data[0])):
        table1[(0, i)].set_facecolor('#E6E6FA')
        table1[(0, i)].set_text_props(weight='bold')
    
    plt.savefig('table1_performance_summary.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    # Table 2: Performance by Scenario
    print("Creating scenario performance table...")
    
    scenario_data = []
    scenarios = df['scenario_clean'].unique()
    
    for scenario in scenarios:
        scenario_df = df[df['scenario_clean'] == scenario]
        row = [scenario]
        for model in models:
            model_scenario = scenario_df[scenario_df['model'] == model]
            if len(model_scenario) > 0:
                row.append(f"{model_scenario['actual_performance'].mean():.1f}")
            else:
                row.append("N/A")
        scenario_data.append(row)
    
    # Create minimal figure with just the table
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    table2 = ax.table(cellText=scenario_data,
                      colLabels=['Scenario', 'Baseline', 'Synthetic', 'BDD100K'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 1.5)
    
    # Style header row
    for i in range(4):
        table2[(0, i)].set_facecolor('#E6E6FA')
        table2[(0, i)].set_text_props(weight='bold')
    
    plt.savefig('table2_scenario_performance.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    # Table 3: Training Data Impact Summary
    print("Creating training data impact table...")
    
    # Create summary by comparison and model
    training_data = []
    comparisons = df['comparison_clean'].unique()
    
    for comparison in comparisons:
        comp_df = df[df['comparison_clean'] == comparison]
        for model in models:
            model_comp = comp_df[comp_df['model'] == model]
            if len(model_comp) > 0:
                training_data.append([
                    comparison,
                    model.title(),
                    f"{model_comp['actual_performance'].mean():.1f}",
                    f"{model_comp['actual_performance'].std():.1f}",
                    f"{model_comp['collisions'].mean():.0f}",
                    f"{model_comp['distance_traveled'].mean():.1f}",
                    len(model_comp)
                ])
    
    # Create minimal figure with just the table
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    table3 = ax.table(cellText=training_data,
                     colLabels=['Training Data', 'Model', 'Mean Perf', 'Std Perf', 
                               'Mean Coll', 'Mean Dist', 'Tests'],
                     cellLoc='center',
                     loc='center')
    
    table3.auto_set_font_size(False)
    table3.set_fontsize(8)
    table3.scale(1.5, 1.5)
    
    # Manually adjust column widths
    cellDict = table3.get_celld()
    for i in range(len(training_data) + 1):  # +1 for header
        # Make first column (Training Data) wider
        cellDict[(i, 0)].set_width(0.25)
        # Make other columns proportionally smaller
        for j in range(1, 7):
            cellDict[(i, j)].set_width(0.125)
    
    # Style header row
    for i in range(7):
        table3[(0, i)].set_facecolor('#E6E6FA')
        table3[(0, i)].set_text_props(weight='bold')
    
    # Color code rows by model
    for i, row in enumerate(training_data, 1):
        model = row[1].lower()
        if model == 'baseline':
            color = '#E6F3FF'  # Light blue
        elif model == 'synthetic':
            color = '#FFF2E6'  # Light orange
        elif model == 'bdd100k':
            color = '#E6FFE6'  # Light green
        else:
            color = '#FFFFFF'
            
        for j in range(7):
            table3[(i, j)].set_facecolor(color)
    
    plt.savefig('table3_training_data_impact.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    
    # Export raw data to CSV for reference
    print("Exporting raw data to CSV files...")
    
    # Export main dataset
    export_df = df[['model', 'scenario_clean', 'comparison_clean', 'weather', 
                   'actual_performance', 'expected_performance', 'collisions', 
                   'lane_invasions', 'distance_traveled', 'average_speed', 
                   'max_speed', 'steering_smoothness']].copy()
    export_df.to_csv('carla_test_results.csv', index=False)
    
    # Export summary statistics
    summary_stats = df.groupby(['model', 'comparison_clean']).agg({
        'actual_performance': ['count', 'mean', 'std', 'min', 'max'],
        'collisions': ['mean', 'std'],
        'lane_invasions': ['mean', 'std'],
        'distance_traveled': ['mean', 'std'],
        'steering_smoothness': ['mean', 'std']
    }).round(3)
    
    summary_stats.to_csv('carla_summary_statistics.csv')
    
    print("  - carla_test_results.csv (raw data)")
    print("  - carla_summary_statistics.csv (summary statistics)")

def generate_insights(df):
    """Generate key insights from the data"""
    print("=" * 50)
    print("KEY INSIGHTS FROM CARLA TEST DATA")
    print("=" * 50)
    print()
    
    # Best performing model overall
    best_model = df.groupby('model')['actual_performance'].mean().idxmax()
    best_performance = df.groupby('model')['actual_performance'].mean().max()
    print(f"1. Best Overall Model: {best_model} (avg performance: {best_performance:.1f})")
    
    # Most challenging scenario
    scenario_difficulty = df.groupby('scenario_clean')['actual_performance'].mean().sort_values()
    hardest_scenario = scenario_difficulty.index[0]
    print(f"2. Most Challenging Scenario: {hardest_scenario}")
    print(f"   (avg performance: {scenario_difficulty.iloc[0]:.1f})")
    
    # Safest model (fewest collisions)
    safest_model = df.groupby('model')['collisions'].mean().idxmin()
    avg_collisions = df.groupby('model')['collisions'].mean().min()
    print(f"3. Safest Model: {safest_model} (avg collisions: {avg_collisions:.0f})")
    
    # Weather impact
    weather_impact = df.groupby('weather')['actual_performance'].mean().sort_values()
    worst_weather = weather_impact.index[0]
    print(f"4. Most Challenging Weather: {worst_weather}")
    print(f"   (avg performance: {weather_impact.iloc[0]:.1f})")
    
    # Model consistency
    model_std = df.groupby('model')['actual_performance'].std()
    most_consistent = model_std.idxmin()
    print(f"5. Most Consistent Model: {most_consistent} (std dev: {model_std.min():.1f})")
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    print(f"• Focus on improving performance in challenging scenarios like '{hardest_scenario}'")
    print(f"• Investigate why '{best_model}' performs better and apply learnings to other models")
    print("• Address safety concerns, particularly collision rates across all models")
    print(f"• Improve weather adaptation capabilities, especially for '{worst_weather}' conditions")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS BY MODEL")
    print("=" * 50)
    summary_stats = df.groupby('model').agg({
        'actual_performance': ['mean', 'std', 'min', 'max'],
        'collisions': ['mean', 'std'],
        'lane_invasions': ['mean', 'std'],
        'distance_traveled': ['mean', 'std'],
        'steering_smoothness': ['mean', 'std']
    }).round(2)
    
    print(summary_stats)

def main():
    """Main function to run all visualizations"""
    print("CARLA Autonomous Driving Test Results Visualization")
    print("=" * 55)
    
    # Load data
    df = load_carla_data()
    print(f"\nLoaded {len(df)} test results")
    print(f"Comparisons: {df['comparison_clean'].unique()}")
    print(f"Models: {df['model'].unique()}")
    print(f"Scenarios: {df['scenario_clean'].unique()}")
    print(f"Weather conditions: {df['weather'].unique()}")
    
    # Create all visualizations
    print("\nGenerating individual chart visualizations...")
    
    print("Creating performance comparison charts...")
    create_performance_plots(df)
    
    print("Creating scenario analysis charts...")
    create_scenario_plots(df)
    
    print("Creating safety analysis charts...")
    create_safety_plots(df)
    
    print("Creating driving behavior charts...")
    create_behavior_plots(df)
    
    print("Creating environmental analysis charts...")
    create_environmental_plots(df)
    
    print("Creating correlation matrix...")
    create_correlation_plot(df)
    
    print("Creating model summary charts...")
    create_summary_plots(df)
    
    print("Creating training data impact charts...")
    create_training_data_plots(df)
    
    print("Creating data tables...")
    create_data_tables(df)
    
    # Generate insights
    print("\nGenerating insights...")
    generate_insights(df)
    
    print(f"\nAll 26 individual charts and 3 data tables saved as PNG files!")
    print("Chart files created (fig1 through fig26):")

if __name__ == "__main__":
    main()