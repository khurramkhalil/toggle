"""
Integration script for running the progressive compression approach
with GPU optimization and organized results storage.
"""

import argparse
import json
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from toggle_dynamic_poc import DynamicPrecisionTransformer
from toggle_sensitivity_analysis import LayerSensitivityAnalyzer
from toggle_gpu_optimization import optimize_stl_evaluation
from toggle_results_organization import ResultsManager

def run_progressive_optimization(model_name, max_iterations=100, 
                                output_prefix="progressive_opt", 
                                skip_sensitivity=False, threshold_relax=0.0,
                                results_dir="results",
                                use_real_datasets=False,
                                dataset_subset_size=50):
    """
    Run progressive compression optimization with GPU acceleration
    and organized results storage.
    
    Args:
        model_name: Name of the pre-trained model to use
        max_iterations: Maximum number of progressive iterations
        output_prefix: Prefix for output files
        skip_sensitivity: Skip sensitivity analysis if True
        threshold_relax: Amount to relax STL thresholds (0.0-0.2)
        results_dir: Directory for storing results
    """
    print(f"Initializing TOGGLE framework with model: {model_name}")
    
    # Initialize results manager
    results_manager = ResultsManager(base_dir=results_dir)
    
    # Get model-specific directory
    model_dir = results_manager.get_model_dir(model_name)
    
    # Update output prefix to include model directory
    output_prefix = os.path.join(model_dir, output_prefix)
    
    # Initialize TOGGLE framework
    toggle = DynamicPrecisionTransformer(model_name=model_name)
    
    # Store original thresholds
    original_thresholds = toggle.stl_thresholds.copy()
    
    # Optionally relax STL thresholds
    if threshold_relax > 0:
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
    # Apply GPU optimization
    print("\n=== Applying GPU Optimization ===")
    gpu_evaluator = optimize_stl_evaluation(
        toggle, 
        use_real_datasets=use_real_datasets,
        subset_size=dataset_subset_size
    )
    
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
    sensitivity_path = os.path.join(model_dir, f"{Path(model_name).name}_sensitivity.json")
    
    if not skip_sensitivity:
        print("\n=== Performing Layer Sensitivity Analysis ===")
        analyzer = LayerSensitivityAnalyzer(toggle)
        sensitivity_map = analyzer.analyze_layer_sensitivity()
        
        # Save sensitivity map
        with open(sensitivity_path, 'w') as f:
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
        print(f"Sensitivity map saved to {sensitivity_path}")
    else:
        print("Skipping sensitivity analysis as requested")
        # Try to load sensitivity map from file
        try:
            with open(sensitivity_path, 'r') as f:
                sensitivity_map = json.load(f)
            print(f"Loaded sensitivity map from {sensitivity_path}")
        except:
            print("No sensitivity map found, will proceed without it")
    
    # Step 2: Run progressive compression
    print("\n=== Running Progressive Compression ===")
    # Import here to avoid circular import
    from toggle_progressive_approach_fix import ProgressiveCompression
    optimizer = ProgressiveCompression(
        toggle_framework=toggle,
        sensitivity_map=sensitivity_map
    )
    
    start_time = time.time()
    best_config, best_results = optimizer.run_progressive_compression(
        max_iterations=max_iterations
    )
    optimization_time = time.time() - start_time
    
    # Generate file paths for visualizations
    progression_path = f"{output_prefix}_progression.png"
    compression_path = f"{output_prefix}_compression.png"
    bit_width_path = "bit_width_config.png"
    pruning_path = "pruning_config.png"
    
    # Step 3: Visualize results
    optimizer.visualize_results(output_prefix)
    toggle.visualize_bit_width_config(best_config)
    
    # Create optimization summary
    optimization_summary = {
        'model_name': model_name,
        'num_iterations': max_iterations,
        'optimization_time': optimization_time,
        'best_stl_score': float(best_results['stl_score']),
        'best_model_size': float(best_results['model_size']),
        'best_stl_satisfied': bool(best_results['stl_satisfied']),
        'total_evaluations': optimizer.eval_count,
        'original_stl_thresholds': original_thresholds,
        'relaxed_stl_thresholds': toggle.stl_thresholds
    }
    
    # Step 4: Save results using results manager
    results_path, config_path = results_manager.save_optimization_results(
        model_name=model_name,
        best_config=best_config,
        best_results=best_results,
        optimization_summary=optimization_summary,
        threshold_relaxation=threshold_relax
    )
    
    # Save plots
    plot_files = {
        progression_path: "progression",
        compression_path: "compression",
        bit_width_path: "bit_width_config",
        pruning_path: "pruning_config"
    }
    saved_plots = results_manager.save_plots(model_name, plot_files)
    
    print("\n=== Progressive Optimization Complete ===")
    print(f"Best configuration: STL score={best_results['stl_score']:.4f}, "
          f"Model size={best_results['model_size']:.2f} MB, "
          f"Satisfied={best_results['stl_satisfied']}")
    print(f"Optimization took {optimization_time:.2f} seconds")
    print(f"Total evaluations: {optimizer.eval_count}")
    print(f"\nResults organization complete:")
    print(f"  - Results saved to: {results_path}")
    print(f"  - Config saved to: {config_path}")
    print(f"  - Plots saved to: {results_manager.plots_dir}")
    print(f"  - Pareto analysis updated at: {results_manager.pareto_dir}")
    
    return best_config, best_results

def main():
    parser = argparse.ArgumentParser(description="Run Progressive TOGGLE Optimization with Organized Results")
    parser.add_argument('--model', type=str, default='gpt2', help='Model name/path')
    parser.add_argument('--iterations', type=int, default=100, help='Maximum iterations')
    parser.add_argument('--output', type=str, default='progressive_opt', help='Output prefix')
    parser.add_argument('--skip-sensitivity', action='store_true', help='Skip sensitivity analysis')
    parser.add_argument('--relax-thresholds', type=float, default=0.0, 
                      help='Amount to relax STL thresholds (0.0-0.2)')
    parser.add_argument('--results-dir', type=str, default='results',
                      help='Directory for storing results')
    parser.add_argument('--use-real-datasets', action='store_true',
                      help='Use real datasets instead of toy dataset')
    parser.add_argument('--dataset-subset-size', type=int, default=50,
                      help='Number of examples to use from each dataset')
    
    args = parser.parse_args()
    
    run_progressive_optimization(
        model_name=args.model,
        max_iterations=args.iterations,
        output_prefix=args.output,
        skip_sensitivity=args.skip_sensitivity,
        threshold_relax=args.relax_thresholds,
        results_dir=args.results_dir,
        use_real_datasets=args.use_real_datasets,
        dataset_subset_size=args.dataset_subset_size
    )

if __name__ == "__main__":
    main()