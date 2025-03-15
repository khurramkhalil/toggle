"""
Utilities for organizing TOGGLE results in proper folders
and preparing data for Pareto front analysis.
"""

import os
import json
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class ResultsManager:
    """
    Manages result storage, organization and preparation for analysis.
    """
    
    def __init__(self, base_dir="results", create_dirs=True):
        """
        Initialize the results manager.
        
        Args:
            base_dir: Base directory for storing results
            create_dirs: Whether to create necessary directories
        """
        self.base_dir = base_dir
        
        # Define directories
        self.model_results_dir = os.path.join(base_dir, "models")
        self.configs_dir = os.path.join(base_dir, "configs")
        self.plots_dir = os.path.join(base_dir, "plots")
        self.pareto_dir = os.path.join(base_dir, "pareto_analysis")
        
        # Create directories if needed
        if create_dirs:
            self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.base_dir,
            self.model_results_dir,
            self.configs_dir,
            self.plots_dir,
            self.pareto_dir
        ]
        
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def get_model_dir(self, model_name):
        """Get directory for specific model results"""
        # Convert model name to a safe directory name
        safe_name = model_name.replace('/', '-').replace('\\', '-')
        model_dir = os.path.join(self.model_results_dir, safe_name)
        
        # Create if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created model directory: {model_dir}")
            
        return model_dir
    
    def save_optimization_results(self, model_name, best_config, best_results, 
                                 optimization_summary, threshold_relaxation):
        """
        Save optimization results to appropriate locations.
        
        Args:
            model_name: Name of the model
            best_config: Best configuration found
            best_results: Results for the best configuration
            optimization_summary: Summary of the optimization process
            threshold_relaxation: Amount of threshold relaxation used
        """
        # Get safe model name for filenames
        safe_name = model_name.replace('/', '-').replace('\\', '-')
        
        # Get timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add model name to results
        optimization_summary['model_name'] = model_name
        
        # Create results object
        results = {
            'model_name': model_name,
            'optimization_summary': optimization_summary,
            'best_results': best_results,
            'threshold_relaxation': threshold_relaxation,
            'timestamp': timestamp
        }
        
        # Save to model-specific directory
        model_dir = self.get_model_dir(model_name)
        results_filename = f"{safe_name}_results_{timestamp}.json"
        results_path = os.path.join(model_dir, results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save config to configs directory
        config_filename = f"{safe_name}_config_{timestamp}.json"
        config_path = os.path.join(self.configs_dir, config_filename)
        
        with open(config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        # Add to pareto front data
        self.update_pareto_data(model_name, best_results, threshold_relaxation)
        
        print(f"Results saved to {results_path}")
        print(f"Configuration saved to {config_path}")
        
        return results_path, config_path
    
    def save_plots(self, model_name, plot_files, timestamp=None):
        """
        Save plot files to plots directory with model name prefix.
        
        Args:
            model_name: Name of the model
            plot_files: Dictionary of plot files {original_path: description}
            timestamp: Optional timestamp for filenames
        """
        # Get safe model name for filenames
        safe_name = model_name.replace('/', '-').replace('\\', '-')
        
        # Get timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model plot directory
        model_plot_dir = os.path.join(self.plots_dir, safe_name)
        if not os.path.exists(model_plot_dir):
            os.makedirs(model_plot_dir)
        
        saved_paths = {}
        
        # Copy plots to the directory with model prefix
        for original_path, description in plot_files.items():
            if os.path.exists(original_path):
                # Get base filename
                base_name = os.path.basename(original_path)
                
                # Create new filename with model prefix
                new_filename = f"{safe_name}_{description}_{timestamp}{os.path.splitext(base_name)[1]}"
                new_path = os.path.join(model_plot_dir, new_filename)
                
                # Copy the file
                shutil.copy2(original_path, new_path)
                saved_paths[original_path] = new_path
                
                print(f"Plot saved to {new_path}")
        
        return saved_paths
    
    def update_pareto_data(self, model_name, results, threshold_relaxation):
        """
        Update Pareto front data with new results.
        
        Args:
            model_name: Name of the model
            results: Results from optimization
            threshold_relaxation: Amount of threshold relaxation used
        """
        # Create data point for Pareto analysis
        data_point = {
            'model_name': model_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stl_score': float(results['stl_score']),
            'model_size': float(results['model_size']),
            'avg_bits': float(results['avg_bits']),
            'avg_pruning': float(results['avg_pruning']),
            'stl_satisfied': bool(results['stl_satisfied']),
            'threshold_relaxation': float(threshold_relaxation)
        }
        
        # Add STL component scores
        if 'robustness' in results:
            for metric, value in results['robustness'].items():
                data_point[f'robustness_{metric}'] = float(value)
        
        # Path to Pareto data file
        pareto_data_path = os.path.join(self.pareto_dir, 'pareto_data.csv')
        
        # Check if file exists
        if os.path.exists(pareto_data_path):
            # Read existing data
            try:
                df = pd.read_csv(pareto_data_path)
                # Append new data
                df = pd.concat([df, pd.DataFrame([data_point])], ignore_index=True)
            except:
                # Create new DataFrame if reading fails
                df = pd.DataFrame([data_point])
        else:
            # Create new DataFrame
            df = pd.DataFrame([data_point])
        
        # Save updated data
        df.to_csv(pareto_data_path, index=False)
        
        # Generate Pareto front plot
        self.generate_pareto_plots()
        
        print(f"Pareto data updated at {pareto_data_path}")
    
    def generate_pareto_plots(self):
        """Generate Pareto front plots for visualization"""
        # Path to Pareto data file
        pareto_data_path = os.path.join(self.pareto_dir, 'pareto_data.csv')
        
        if not os.path.exists(pareto_data_path):
            print("No Pareto data found to plot")
            return
        
        # Read data
        df = pd.read_csv(pareto_data_path)
        
        # Filter for satisfied configs only for Pareto front
        satisfied_df = df[df['stl_satisfied'] == True].copy()
        
        if len(satisfied_df) == 0:
            print("No STL-satisfied configurations found for Pareto front")
            return
        
        # Create Pareto front plot: STL Score vs Model Size
        plt.figure(figsize=(10, 6))
        
        # Plot all points, color by model
        models = df['model_name'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            model_df = df[df['model_name'] == model]
            plt.scatter(
                model_df['model_size'], 
                model_df['stl_score'],
                label=model,
                color=colors[i],
                alpha=0.7,
                s=50
            )
        
        # Highlight Pareto optimal points
        if len(satisfied_df) > 0:
            # Identify Pareto front
            is_pareto = np.ones(len(satisfied_df), dtype=bool)
            for i, (size_i, score_i) in enumerate(zip(satisfied_df['model_size'], satisfied_df['stl_score'])):
                for j, (size_j, score_j) in enumerate(zip(satisfied_df['model_size'], satisfied_df['stl_score'])):
                    if i != j:
                        # Check if point j dominates point i
                        if size_j <= size_i and score_j >= score_i and (size_j < size_i or score_j > score_i):
                            is_pareto[i] = False
                            break
            
            # Plot Pareto front
            pareto_df = satisfied_df[is_pareto]
            plt.scatter(
                pareto_df['model_size'], 
                pareto_df['stl_score'],
                facecolors='none', 
                edgecolors='red',
                s=100,
                linewidth=2,
                label='Pareto Front'
            )
            
            # Connect Pareto front points with line
            if len(pareto_df) > 1:
                # Sort by model size
                pareto_df = pareto_df.sort_values('model_size')
                plt.plot(
                    pareto_df['model_size'],
                    pareto_df['stl_score'],
                    'r--',
                    alpha=0.7
                )
        
        # Add satisfaction threshold line
        plt.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='STL Satisfaction Threshold')
        
        plt.xlabel('Model Size (MB)')
        plt.ylabel('STL Score')
        plt.title('Pareto Front: STL Score vs Model Size')
        plt.grid(alpha=0.3)
        plt.legend()
        
        pareto_plot_path = os.path.join(self.pareto_dir, 'pareto_front_plot.png')
        plt.savefig(pareto_plot_path)
        
        # Create bit-width vs STL score plot
        plt.figure(figsize=(10, 6))
        
        for i, model in enumerate(models):
            model_df = df[df['model_name'] == model]
            plt.scatter(
                model_df['avg_bits'], 
                model_df['stl_score'],
                label=model,
                color=colors[i],
                alpha=0.7,
                s=50
            )
        
        plt.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='STL Satisfaction Threshold')
        
        plt.xlabel('Average Bit-width')
        plt.ylabel('STL Score')
        plt.title('STL Score vs Average Bit-width')
        plt.grid(alpha=0.3)
        plt.legend()
        
        bitwidth_plot_path = os.path.join(self.pareto_dir, 'bitwidth_vs_stl_plot.png')
        plt.savefig(bitwidth_plot_path)
        
        print(f"Pareto front plots updated: {pareto_plot_path}, {bitwidth_plot_path}")