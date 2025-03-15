"""
Integration script for running the progressive compression approach
with GPU optimization for TOGGLE framework.
"""

import argparse
import json
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from toggle_dynamic_poc import DynamicPrecisionTransformer
from toggle_sensitivity_analysis import LayerSensitivityAnalyzer
from toggle_progressive_approach import ProgressiveCompression
from toggle_gpu_optimization import optimize_stl_evaluation

def run_progressive_optimization(model_name, max_iterations=100, 
                                output_prefix="progressive_opt", 
                                skip_sensitivity=False, threshold_relax=0.0):
    """
    Run progressive compression optimization with GPU acceleration
    
    Args:
        model_name: Name of the pre-trained model to use
        max_iterations: Maximum number of progressive iterations
        output_prefix: Prefix for output files
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
    
    # Apply GPU optimization
    print("\n=== Applying GPU Optimization ===")
    gpu_evaluator = optimize_stl_evaluation(toggle)
    
    # Measure GPU utilization
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    # Do a sample evaluation
    _ = toggle.evaluate_stl_properties(toggle.base_model)
    end_event.record()
    
    torch.cuda.synchronize()
    eval_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
    print(f"Optimized evaluation time: {eval_time:.4f} seconds")
    
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
    
    # Step 2: Run progressive compression
    print("\n=== Running Progressive Compression ===")
    optimizer = ProgressiveCompression(
        toggle_framework=toggle,
        sensitivity_map=sensitivity_map
    )
    
    start_time = time.time()
    best_config, best_results = optimizer.run_progressive_compression(
        max_iterations=max_iterations
    )
    optimization_time = time.time() - start_time
    
    # Step 3: Save results and visualize
    results_path = f"{output_prefix}_results.json"
    config_path = f"{output_prefix}_config.json"
    
    # Visualize compression progress
    optimizer.visualize_results(output_prefix)
    
    # Create summary of optimization process
    optimization_summary = {
        'model_name': model_name,
        'num_iterations': max_iterations,
        'optimization_time': optimization_time,
        'best_stl_score': float(best_results['stl_score']),
        'best_model_size': float(best_results['model_size']),
        'best_stl_satisfied': bool(best_results['stl_satisfied']),
        'total_evaluations': optimizer.eval_count,
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
    
    # Visualize best configuration
    toggle.visualize_bit_width_config(best_config)
    
    print("\n=== Progressive Optimization Complete ===")
    print(f"Best configuration: STL score={best_results['stl_score']:.4f}, "
          f"Model size={best_results['model_size']:.2f} MB, "
          f"Satisfied={best_results['stl_satisfied']}")
    print(f"Optimization took {optimization_time:.2f} seconds")
    print(f"Total evaluations: {optimizer.eval_count}")
    
    return best_config, best_results

def main():
    parser = argparse.ArgumentParser(description="Run Progressive TOGGLE Optimization")
    parser.add_argument('--model', type=str, default='gpt2', help='Model name/path')
    parser.add_argument('--iterations', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--output', type=str, default='progressive_opt', help='Output prefix')
    parser.add_argument('--skip-sensitivity', action='store_true', help='Skip sensitivity analysis')
    parser.add_argument('--relax-thresholds', type=float, default=0.0, 
                      help='Amount to relax STL thresholds (0.0-0.2)')
    
    args = parser.parse_args()
    
    run_progressive_optimization(
        model_name=args.model,
        max_iterations=args.iterations,
        output_prefix=args.output,
        skip_sensitivity=args.skip_sensitivity,
        threshold_relax=args.relax_thresholds
    )

if __name__ == "__main__":
    main()