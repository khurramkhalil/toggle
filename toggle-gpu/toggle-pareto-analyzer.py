"""
TOGGLE Pareto Front Analyzer
This script analyzes the Pareto front of compression configurations
across different models and relaxation levels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class ParetoAnalyzer:
    """
    Analyzes and visualizes the Pareto front for TOGGLE compression results
    """
    
    def __init__(self, results_dir="results"):
        """
        Initialize the Pareto analyzer.
        
        Args:
            results_dir: Base directory for results
        """
        self.results_dir = results_dir
        self.pareto_dir = os.path.join(results_dir, "pareto_analysis")
        self.pareto_data_path = os.path.join(self.pareto_dir, "pareto_data.csv")
        
        # Check if directories exist
        if not os.path.exists(self.pareto_dir):
            os.makedirs(self.pareto_dir)
            print(f"Created directory: {self.pareto_dir}")
    
    def load_data(self):
        """Load Pareto data from CSV file"""
        if not os.path.exists(self.pareto_data_path):
            print(f"No Pareto data found at {self.pareto_data_path}")
            return None
        
        return pd.read_csv(self.pareto_data_path)
    
    def identify_pareto_front(self, df, x_col='model_size', y_col='stl_score'):
        """
        Identify Pareto optimal points.
        
        Args:
            df: DataFrame with results
            x_col: Column to minimize (e.g., model_size)
            y_col: Column to maximize (e.g., stl_score)
            
        Returns:
            DataFrame with Pareto front points identified
        """
        # Filter for satisfied configurations
        satisfied_df = df[df['stl_satisfied'] == True].copy()
        
        if len(satisfied_df) == 0:
            print("No STL-satisfied configurations found")
            return df
        
        # Identify Pareto front
        is_pareto = np.ones(len(satisfied_df), dtype=bool)
        for i, (x_i, y_i) in enumerate(zip(satisfied_df[x_col], satisfied_df[y_col])):
            for j, (x_j, y_j) in enumerate(zip(satisfied_df[x_col], satisfied_df[y_col])):
                if i != j:
                    # Check if point j dominates point i
                    if x_j <= x_i and y_j >= y_i and (x_j < x_i or y_j > y_i):
                        is_pareto[i] = False
                        break
        
        # Add is_pareto column
        satisfied_df['is_pareto'] = is_pareto
        
        # Merge back with original dataframe
        merged_df = df.copy()
        merged_df['is_pareto'] = False
        
        # Update is_pareto for satisfied points
        for idx, row in satisfied_df.iterrows():
            if row['is_pareto']:
                # Find matching row in merged_df
                mask = (merged_df['model_name'] == row['model_name']) & \
                       (merged_df['stl_score'] == row['stl_score']) & \
                       (merged_df['model_size'] == row['model_size'])
                merged_df.loc[mask, 'is_pareto'] = True
        
        return merged_df
    
    def plot_comprehensive_pareto(self, models=None, relaxation_levels=None):
        """
        Create comprehensive Pareto front visualization.
        
        Args:
            models: List of model names to include (None for all)
            relaxation_levels: List of relaxation levels (None for all)
            
        Returns:
            Path to saved plot
        """
        # Load data
        df = self.load_data()
        if df is None or len(df) == 0:
            print("No data available for Pareto analysis")
            return None
        
        # Filter by models if specified
        if models is not None:
            df = df[df['model_name'].isin(models)]
        
        # Filter by relaxation levels if specified
        if relaxation_levels is not None:
            df = df[df['threshold_relaxation'].isin(relaxation_levels)]
        
        # Identify Pareto front
        df = self.identify_pareto_front(df)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create color map for relaxation levels
        unique_relaxations = df['threshold_relaxation'].unique()
        unique_relaxations = sorted(unique_relaxations)
        relaxation_cmap = cm.get_cmap('viridis', max(len(unique_relaxations), 2))
        
        # Create marker map for models
        unique_models = df['model_name'].unique()
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
        marker_map = {model: markers[i % len(markers)] for i, model in enumerate(unique_models)}
        
        # Plot all points
        for relaxation in unique_relaxations:
            for model in unique_models:
                subset = df[(df['model_name'] == model) & 
                           (df['threshold_relaxation'] == relaxation)]
                
                if len(subset) == 0:
                    continue
                
                # Plot all points for this model and relaxation
                ax.scatter(
                    subset['model_size'],
                    subset['stl_score'],
                    label=f"{model}, relax={relaxation}",
                    marker=marker_map[model],
                    color=relaxation_cmap(np.where(unique_relaxations == relaxation)[0][0] / max(1, len(unique_relaxations) - 1)),
                    alpha=0.6,
                    s=70
                )
        
        # Highlight Pareto front points
        pareto_points = df[df['is_pareto'] == True]
        if len(pareto_points) > 0:
            ax.scatter(
                pareto_points['model_size'],
                pareto_points['stl_score'],
                facecolors='none',
                edgecolors='red',
                s=150,
                linewidth=2,
                label='Pareto Front'
            )
            
            # Connect Pareto front with line
            pareto_sorted = pareto_points.sort_values('model_size')
            ax.plot(
                pareto_sorted['model_size'],
                pareto_sorted['stl_score'],
                'r--',
                alpha=0.7
            )
        
        # Add satisfaction threshold line
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='STL Satisfaction Threshold')
        
        # Configure plot
        ax.set_xlabel('Model Size (MB)', fontsize=12)
        ax.set_ylabel('STL Score', fontsize=12)
        ax.set_title('Pareto Front: STL Score vs Model Size Across Models and Relaxation Levels', fontsize=14)
        ax.grid(alpha=0.3)
        
        # Add legend with smaller font to accommodate many entries
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                 ncol=3, frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        
        # Save plot
        plot_path = os.path.join(self.pareto_dir, 'comprehensive_pareto_front.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"Comprehensive Pareto front saved to {plot_path}")
        return plot_path
    
    def plot_bit_width_analysis(self):
        """
        Create visualization of STL score vs bit-width with model size as color.
        
        Returns:
            Path to saved plot
        """
        # Load data
        df = self.load_data()
        if df is None or len(df) == 0:
            print("No data available for bit-width analysis")
            return None
        
        # Filter to only satisfied configurations
        satisfied_df = df[df['stl_satisfied'] == True]
        
        if len(satisfied_df) == 0:
            print("No STL-satisfied configurations found for bit-width analysis")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique models
        unique_models = satisfied_df['model_name'].unique()
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
        marker_map = {model: markers[i % len(markers)] for i, model in enumerate(unique_models)}
        
        # Plot points with color based on model size
        scatter = ax.scatter(
            satisfied_df['avg_bits'],
            satisfied_df['stl_score'],
            c=satisfied_df['model_size'],
            cmap='plasma',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Model Size (MB)', fontsize=12)
        
        # Add model indicators
        for model in unique_models:
            model_df = satisfied_df[satisfied_df['model_name'] == model]
            ax.scatter(
                model_df['avg_bits'],
                model_df['stl_score'],
                marker=marker_map[model],
                s=150,
                facecolors='none',
                edgecolors='black',
                linewidth=1.5,
                label=model
            )
        
        # Configure plot
        ax.set_xlabel('Average Bit-width', fontsize=12)
        ax.set_ylabel('STL Score', fontsize=12)
        ax.set_title('STL Score vs Bit-width (Color: Model Size)', fontsize=14)
        ax.grid(alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                 ncol=3, frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        # Save plot
        plot_path = os.path.join(self.pareto_dir, 'bit_width_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"Bit-width analysis saved to {plot_path}")
        return plot_path
    
    def plot_pruning_vs_bits(self):
        """
        Create visualization of pruning vs bit-width with STL score as color.
        
        Returns:
            Path to saved plot
        """
        # Load data
        df = self.load_data()
        if df is None or len(df) == 0:
            print("No data available for pruning analysis")
            return None
        
        # Filter to only satisfied configurations
        satisfied_df = df[df['stl_satisfied'] == True]
        
        if len(satisfied_df) == 0:
            print("No STL-satisfied configurations found for pruning analysis")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot points with color based on STL score
        scatter = ax.scatter(
            satisfied_df['avg_bits'],
            satisfied_df['avg_pruning'] * 100,  # Convert to percentage
            c=satisfied_df['stl_score'],
            cmap='viridis',
            s=100,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('STL Score', fontsize=12)
        
        # Get unique models
        unique_models = satisfied_df['model_name'].unique()
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
        marker_map = {model: markers[i % len(markers)] for i, model in enumerate(unique_models)}
        
        # Add model indicators
        for model in unique_models:
            model_df = satisfied_df[satisfied_df['model_name'] == model]
            ax.scatter(
                model_df['avg_bits'],
                model_df['avg_pruning'] * 100,  # Convert to percentage
                marker=marker_map[model],
                s=150,
                facecolors='none',
                edgecolors='black',
                linewidth=1.5,
                label=model
            )
        
        # Configure plot
        ax.set_xlabel('Average Bit-width', fontsize=12)
        ax.set_ylabel('Average Pruning (%)', fontsize=12)
        ax.set_title('Pruning vs Bit-width (Color: STL Score)', fontsize=14)
        ax.grid(alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                 ncol=3, frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        # Save plot
        plot_path = os.path.join(self.pareto_dir, 'pruning_vs_bits.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"Pruning vs bit-width analysis saved to {plot_path}")
        return plot_path
    
    def generate_pareto_report(self):
        """
        Generate a comprehensive report on Pareto analysis.
        
        Returns:
            Path to saved report
        """
        # Load data
        df = self.load_data()
        if df is None or len(df) == 0:
            print("No data available for Pareto report")
            return None
        
        # Identify Pareto front
        df = self.identify_pareto_front(df)
        pareto_points = df[df['is_pareto'] == True]
        
        # Create report content
        report = []
        report.append("# TOGGLE Pareto Analysis Report")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append(f"- Total configurations analyzed: {len(df)}")
        report.append(f"- STL-satisfied configurations: {len(df[df['stl_satisfied'] == True])}")
        report.append(f"- Pareto optimal configurations: {len(pareto_points)}")
        report.append(f"- Models analyzed: {', '.join(df['model_name'].unique())}")
        report.append(f"- Relaxation levels: {', '.join(map(str, sorted(df['threshold_relaxation'].unique())))}\n")
        
        # Pareto front analysis
        if len(pareto_points) > 0:
            report.append("## Pareto Front Configurations")
            
            # Sort by model size
            pareto_sorted = pareto_points.sort_values('model_size')
            
            for i, row in pareto_sorted.iterrows():
                report.append(f"### Configuration {i+1}")
                report.append(f"- Model: {row['model_name']}")
                report.append(f"- STL Score: {row['stl_score']:.4f}")
                report.append(f"- Model Size: {row['model_size']:.2f} MB")
                report.append(f"- Average Bit-width: {row['avg_bits']:.2f}")
                report.append(f"- Average Pruning: {row['avg_pruning']*100:.2f}%")
                report.append(f"- Threshold Relaxation: {row['threshold_relaxation']:.2f}")
                
                # Add robustness values if available
                robustness_cols = [col for col in row.index if col.startswith('robustness_')]
                if robustness_cols:
                    report.append("- Robustness Values:")
                    for col in robustness_cols:
                        property_name = col.replace('robustness_', '')
                        report.append(f"  - {property_name}: {row[col]:.4f}")
                
                report.append("")
        
        # Model-specific analysis
        report.append("## Model-Specific Analysis")
        
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            satisfied_model_df = model_df[model_df['stl_satisfied'] == True]
            
            report.append(f"### {model}")
            report.append(f"- Total configurations: {len(model_df)}")
            report.append(f"- STL-satisfied configurations: {len(satisfied_model_df)}")
            
            if len(satisfied_model_df) > 0:
                # Find best compression (smallest size)
                best_compression = satisfied_model_df.loc[satisfied_model_df['model_size'].idxmin()]
                
                # Find best STL score
                best_stl = satisfied_model_df.loc[satisfied_model_df['stl_score'].idxmax()]
                
                report.append("- Best Compression (STL-satisfied):")
                report.append(f"  - STL Score: {best_compression['stl_score']:.4f}")
                report.append(f"  - Model Size: {best_compression['model_size']:.2f} MB")
                report.append(f"  - Average Bit-width: {best_compression['avg_bits']:.2f}")
                report.append(f"  - Average Pruning: {best_compression['avg_pruning']*100:.2f}%")
                
                report.append("- Best STL Score:")
                report.append(f"  - STL Score: {best_stl['stl_score']:.4f}")
                report.append(f"  - Model Size: {best_stl['model_size']:.2f} MB")
                report.append(f"  - Average Bit-width: {best_stl['avg_bits']:.2f}")
                report.append(f"  - Average Pruning: {best_stl['avg_pruning']*100:.2f}%")
            
            report.append("")
        
        # Relaxation level analysis
        report.append("## Relaxation Level Analysis")
        
        for relax in sorted(df['threshold_relaxation'].unique()):
            relax_df = df[df['threshold_relaxation'] == relax]
            satisfied_relax_df = relax_df[relax_df['stl_satisfied'] == True]
            
            report.append(f"### Relaxation Level: {relax:.2f}")
            report.append(f"- Total configurations: {len(relax_df)}")
            report.append(f"- STL-satisfied configurations: {len(satisfied_relax_df)}")
            
            if len(satisfied_relax_df) > 0:
                report.append(f"- Average model size: {satisfied_relax_df['model_size'].mean():.2f} MB")
                report.append(f"- Average bit-width: {satisfied_relax_df['avg_bits'].mean():.2f}")
                report.append(f"- Average pruning: {satisfied_relax_df['avg_pruning'].mean()*100:.2f}%")
                report.append(f"- Average STL score: {satisfied_relax_df['stl_score'].mean():.4f}")
            
            report.append("")
        
        # Write report to file
        report_path = os.path.join(self.pareto_dir, 'pareto_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Pareto analysis report saved to {report_path}")
        return report_path
    
    def run_full_analysis(self):
        """Run all analyses and generate comprehensive visualizations and reports"""
        print("Running full Pareto analysis...")
        
        # Generate all plots
        self.plot_comprehensive_pareto()
        self.plot_bit_width_analysis()
        self.plot_pruning_vs_bits()
        
        # Generate report
        self.generate_pareto_report()
        
        print("Pareto analysis complete.")

def main():
    parser = argparse.ArgumentParser(description="TOGGLE Pareto Front Analyzer")
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Base directory for results')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Models to include in analysis (default: all)')
    parser.add_argument('--relaxations', type=float, nargs='+',
                       help='Relaxation levels to include (default: all)')
    parser.add_argument('--full', action='store_true',
                       help='Run full analysis')
    
    args = parser.parse_args()
    
    analyzer = ParetoAnalyzer(results_dir=args.results_dir)
    
    if args.full:
        analyzer.run_full_analysis()
    else:
        analyzer.plot_comprehensive_pareto(models=args.models, relaxation_levels=args.relaxations)
        analyzer.generate_pareto_report()

if __name__ == "__main__":
    main()