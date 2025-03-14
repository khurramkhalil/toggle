"""
Script to run Bayesian Optimization on TOGGLE compression configurations
"""

import argparse
import json
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from toggle_dynamic_poc import DynamicPrecisionTransformer
from toggle_optimization import BayesianOptimization

def run_optimization(model_name, max_iterations=20, exploration_weight=0.1, 
                    output_prefix="bo_result", random_seed=42, init_config=None):
    """
    Run Bayesian Optimization to find the best compression configuration
    
    Args:
        model_name: Name of the pre-trained model to use
        max_iterations: Maximum number of optimization iterations
        exploration_weight: Weight for exploration in acquisition function
        output_prefix: Prefix for output files
        random_seed: Random seed for reproducibility
        init_config: Path to initial configuration file (optional)
    """
    print(f"Initializing TOGGLE framework with model: {model_name}")
    
    # Initialize TOGGLE framework
    toggle = DynamicPrecisionTransformer(model_name=model_name)
    
    # Load initial configuration if provided
    init_configs = []
    if init_config:
        try:
            with open(init_config, 'r') as f:
                config = json.load(f)
                init_configs.append(config)
                print(f"Loaded initial configuration from {init_config}")
        except Exception as e:
            print(f"Error loading initial configuration: {e}")
    
    # Initialize the optimizer
    optimizer = BayesianOptimization(
        toggle_framework=toggle,
        num_layers=toggle.num_layers,
        components=toggle.components,
        init_configs=init_configs,
        random_state=random_seed
    )
    
    # Run optimization
    start_time = time.time()
    best_config, best_results = optimizer.optimize(
        max_iterations=max_iterations,
        exploration_weight=exploration_weight
    )
    optimization_time = time.time() - start_time
    
    # Save results
    results_path = f"{output_prefix}_results.json"
    config_path = f"{output_prefix}_config.json"
    
    # Create summary of optimization process
    optimization_summary = {
        'model_name': model_name,
        'num_iterations': max_iterations,
        'exploration_weight': exploration_weight,
        'optimization_time': optimization_time,
        'best_stl_score': best_results['stl_score'],
        'best_model_size': best_results['model_size'],
        'best_stl_satisfied': best_results['stl_satisfied'],
        'best_metrics': best_results['metrics'],
        'best_robustness': best_results['robustness'],
        'optimization_history': {
            'iterations': len(optimizer.y),
            'obj_values': optimizer.y
        }
    }
    
    # Save best configuration
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
        print(f"Best configuration saved to {config_path}")
    
    # Save optimization results
    with open(results_path, 'w') as f:
        result_dict = {
            'optimization_summary': optimization_summary,
            'best_results': best_results
        }
        json.dump(result_dict, f, indent=2)
        print(f"Optimization results saved to {results_path}")
    
    # Visualize optimization process
    _visualize_optimization(optimizer, output_prefix)
    
    # Visualize best configuration
    toggle.visualize_bit_width_config(best_config)
    
    return best_config, best_results

def _visualize_optimization(optimizer, output_prefix):
    """
    Create visualizations of the optimization process
    
    Args:
        optimizer: BayesianOptimization instance
        output_prefix: Prefix for output files
    """
    # Plot objective value over iterations
    plt.figure(figsize=(10, 6))
    y_values = np.array(optimizer.y)
    iterations = np.arange(len(y_values))
    
    # Mark if obj value > 0 (STL satisfied)
    is_satisfied = y_values > 0
    
    plt.scatter(iterations[is_satisfied], y_values[is_satisfied], 
                label='STL Satisfied', color='green', marker='o')
    plt.scatter(iterations[~is_satisfied], y_values[~is_satisfied], 
                label='STL Violated', color='red', marker='x')
    
    plt.plot(iterations, y_values, 'k--', alpha=0.3)
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Bayesian Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{output_prefix}_progress.png")
    print(f"Optimization progress saved to {output_prefix}_progress.png")
    
    # Plot best value found so far
    plt.figure(figsize=(10, 6))
    best_so_far = np.maximum.accumulate(y_values)
    plt.plot(iterations, best_so_far, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Objective Value')
    plt.title('Best Value Found')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f"{output_prefix}_best_value.png")
    print(f"Best value plot saved to {output_prefix}_best_value.png")

def main():
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization for TOGGLE compression")
    parser.add_argument('--model', type=str, default='gpt2', help='Model name/path')
    parser.add_argument('--iterations', type=int, default=20, help='Maximum optimization iterations')
    parser.add_argument('--exploration', type=float, default=0.1, help='Exploration weight')
    parser.add_argument('--output', type=str, default='bo_result', help='Output prefix')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--init-config', type=str, help='Path to initial configuration')
    
    args = parser.parse_args()
    
    run_optimization(
        model_name=args.model,
        max_iterations=args.iterations,
        exploration_weight=args.exploration,
        output_prefix=args.output,
        random_seed=args.seed,
        init_config=args.init_config
    )

if __name__ == "__main__":
    main()