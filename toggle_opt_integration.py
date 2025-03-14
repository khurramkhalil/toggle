"""
Integration script for running enhanced TOGGLE optimization
"""

import argparse
import json
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from toggle_dynamic_poc import DynamicPrecisionTransformer
from toggle_sensitivity_analysis import LayerSensitivityAnalyzer
from toggle_enhanced_optimization import EnhancedOptimizer

def run_optimization(model_name, max_iterations=20, exploration_weight=0.1, 
                    output_prefix="enhanced_opt", random_seed=42, 
                    skip_sensitivity=False, threshold_relax=0.0):
    """
    Run enhanced optimization to find the best compression configuration
    
    Args:
        model_name: Name of the pre-trained model to use
        max_iterations: Maximum number of optimization iterations
        exploration_weight: Weight for exploration in acquisition function
        output_prefix: Prefix for output files
        random_seed: Random seed for reproducibility
        skip_sensitivity: Skip sensitivity analysis if True
        threshold_relax: Amount to relax STL thresholds (0.0-0.2)
    """
    print(f"Initializing TOGGLE framework with model: {model_name}")
    
    # Initialize TOGGLE framework
    toggle = DynamicPrecisionTransformer(model_name=model_name)
    
    # Optionally relax STL thresholds
    if threshold_relax > 0:
        original_thresholds = toggle.stl_thresholds.copy()
        relaxed_thresholds = {
            'coherence': min(1.0, original_thresholds['coherence'] * (1 + threshold_relax)),
            'attention': max(0.0, original_thresholds['attention'] * (1 - threshold_relax)),
            'context': max(0.0, original_thresholds['context'] * (1 - threshold_relax)),
            'factual': max(0.0, original_thresholds['factual'] * (1 - threshold_relax))
        }
        toggle.stl_thresholds = relaxed_thresholds
        
        print("Using relaxed STL thresholds:")
        for k, v in relaxed_thresholds.items():
            print(f"  {k}: {v:.4f} (original: {original_thresholds[k]:.4f})")
    
    # Step 1: Perform sensitivity analysis (unless skipped)
    sensitivity_map = None
    if not skip_sensitivity:
        print("\n=== Performing Layer Sensitivity Analysis ===")
        analyzer = LayerSensitivityAnalyzer(toggle)
        sensitivity_map = analyzer.analyze_layer_sensitivity()
        
        # Save sensitivity map
        with open(f"{output_prefix}_sensitivity.json", 'w') as f:
            # Convert numpy and torch values to native Python types
            cleaned_map = {}
            for key, value in sensitivity_map.items():
                if isinstance(value, dict):
                    cleaned_map[key] = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            cleaned_map[key][k] = {
                                sk: float(sv) if isinstance(sv, (np.number, torch.Tensor)) else sv
                                for sk, sv in v.items()
                            }
                        else:
                            cleaned_map[key][k] = float(v) if isinstance(v, (np.number, torch.Tensor)) else v
                else:
                    cleaned_map[key] = float(value) if isinstance(value, (np.number, torch.Tensor)) else value
            
            json.dump(cleaned_map, f, indent=2)
        print(f"Sensitivity map saved to {output_prefix}_sensitivity.json")
    else:
        print("Skipping sensitivity analysis as requested")
        # Try to load sensitivity map from file
        try:
            with open(f"{output_prefix}_sensitivity.json", 'r') as f:
                sensitivity_map = json.load(f)
            print(f"Loaded sensitivity map from {output_prefix}_sensitivity.json")
        except:
            print("No sensitivity map found, will proceed without it")
    
    # Step 2: Run enhanced optimization
    print("\n=== Running Enhanced Optimization ===")
    optimizer = EnhancedOptimizer(
        toggle_framework=toggle,
        sensitivity_map=sensitivity_map,
        random_state=random_seed
    )
    
    start_time = time.time()
    best_config, best_results = optimizer.optimize(
        max_iterations=max_iterations,
        exploration_weight=exploration_weight
    )
    optimization_time = time.time() - start_time
    
    # Step 3: Save results and visualize
    results_path = f"{output_prefix}_results.json"
    config_path = f"{output_prefix}_config.json"
    
    # Create summary of optimization process
    optimization_summary = {
        'model_name': model_name,
        'num_iterations': max_iterations,
        'exploration_weight': exploration_weight,
        'optimization_time': optimization_time,
        'best_stl_score': float(best_results['stl_score']),
        'best_model_size': float(best_results['model_size']),
        'best_stl_satisfied': bool(best_results['stl_satisfied']),
        'cache_hits': len(optimizer.eval_cache) - len(optimizer.X),
        'total_evaluations': len(optimizer.X),
        'stl_thresholds': toggle.stl_thresholds
    }
    
    # Save best configuration
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
        print(f"Best configuration saved to {config_path}")
    
    # Save optimization results
    with open(results_path, 'w') as f:
        # Clean up non-serializable values
        cleaned_results = {}
        for k, v in best_results.items():
            if isinstance(v, dict):
                cleaned_results[k] = {
                    sk: float(sv) if isinstance(sv, (np.number, torch.Tensor)) else sv
                    for sk, sv in v.items()
                }
            else:
                cleaned_results[k] = float(v) if isinstance(v, (np.number, torch.Tensor)) else v
        
        result_dict = {
            'optimization_summary': optimization_summary,
            'best_results': cleaned_results
        }
        json.dump(result_dict, f, indent=2)
        print(f"Optimization results saved to {results_path}")
    
    # Create visualizations
    optimizer.visualize_optimization_progress(output_prefix)
    
    # Visualize best configuration
    toggle.visualize_bit_width_config(best_config)
    
    print("\n=== Optimization Complete ===")
    print(f"Best configuration: STL score={best_results['stl_score']:.4f}, "
          f"Model size={best_results['model_size']:.2f} MB, "
          f"Satisfied={best_results['stl_satisfied']}")
    print(f"Optimization took {optimization_time:.2f} seconds")
    print(f"Cache efficiency: {len(optimizer.eval_cache) - len(optimizer.X)} cache hits "
          f"out of {len(optimizer.eval_cache)} evaluations "
          f"({(len(optimizer.eval_cache) - len(optimizer.X)) / len(optimizer.eval_cache) * 100:.1f}%)")
    
    return best_config, best_results

def main():
    parser = argparse.ArgumentParser(description="Run Enhanced TOGGLE Optimization")
    parser.add_argument('--model', type=str, default='gpt2', help='Model name/path')
    parser.add_argument('--iterations', type=int, default=20, help='Maximum optimization iterations')
    parser.add_argument('--exploration', type=float, default=0.2, help='Exploration weight')
    parser.add_argument('--output', type=str, default='enhanced_opt', help='Output prefix')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-sensitivity', action='store_true', help='Skip sensitivity analysis')
    parser.add_argument('--relax-thresholds', type=float, default=0.0, 
                      help='Amount to relax STL thresholds (0.0-0.2)')
    
    args = parser.parse_args()
    
    run_optimization(
        model_name=args.model,
        max_iterations=args.iterations,
        exploration_weight=args.exploration,
        output_prefix=args.output,
        random_seed=args.seed,
        skip_sensitivity=args.skip_sensitivity,
        threshold_relax=args.relax_thresholds
    )

if __name__ == "__main__":
    main()