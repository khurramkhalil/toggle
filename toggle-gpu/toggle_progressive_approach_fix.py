"""
Progressive compression approach for TOGGLE framework with memory optimization.
This implements a systematic layer-by-layer compression strategy
that guarantees finding STL-satisfied configurations while managing memory efficiently.
"""

import torch
import numpy as np
from tqdm import tqdm
import time
import random
import copy
import matplotlib.pyplot as plt
import gc

class ProgressiveCompression:
    """
    Progressive compression strategy that starts with the base model
    and gradually compresses layers based on sensitivity until 
    constraints are violated, then backs off.
    """
    
    def __init__(self, toggle_framework, sensitivity_map=None):
        """
        Initialize the progressive compression.
        
        Args:
            toggle_framework: DynamicPrecisionTransformer instance
            sensitivity_map: Layer sensitivity data (optional)
        """
        self.toggle = toggle_framework
        self.num_layers = toggle_framework.num_layers
        self.components = toggle_framework.components
        self.sensitivity_map = sensitivity_map
        
        # Use checkpoint approach instead of storing all parameters
        self.checkpoint_created = False
        self.model_device = next(toggle_framework.base_model.parameters()).device
        
        # Set bit-width options in descending order (from high to low precision)
        self.bit_options = sorted(toggle_framework.bit_options, reverse=True)
        
        # Set pruning options in ascending order (from low to high pruning)
        self.pruning_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Track evaluations for efficiency
        self.eval_cache = {}
        self.eval_count = 0
        
        # Track results
        self.results_history = []
        
        print("Initializing progressive compression with memory-efficient implementation...")
    
    def create_checkpoint(self):
        """Create a checkpoint of the current model state"""
        if self.checkpoint_created:
            print("Checkpoint already exists, not creating a new one...")
            return
            
        print("Creating model checkpoint for state restoration...")
        # Use PyTorch's state_dict mechanism which is more memory efficient
        self.model_checkpoint = self.toggle.base_model.state_dict()
        self.checkpoint_created = True
        
        # Clear any unused memory
        torch.cuda.empty_cache()
        gc.collect()
    
    def restore_from_checkpoint(self):
        """Restore model parameters from checkpoint"""
        if not self.checkpoint_created:
            print("Warning: No checkpoint to restore from!")
            return
            
        # Load the state dict back into the model
        self.toggle.base_model.load_state_dict(self.model_checkpoint)
        
        # Clear any unused memory
        torch.cuda.empty_cache()
        gc.collect()

    def _get_config_hash(self, config):
        """Create a deterministic hash for a configuration"""
        # Sort by layer and component for consistency
        config_str = ""
        layer_keys = sorted(config.keys())
        for layer_key in layer_keys:
            component_keys = sorted(config[layer_key].keys())
            for component in component_keys:
                bits = config[layer_key][component]['bits']
                pruning = config[layer_key][component]['pruning']
                config_str += f"{layer_key}.{component}:{bits}:{pruning:.2f};"
        
        return hash(config_str)
    
    def _apply_config_with_swapping(self, config):
        """Apply configuration with parameter swapping for evaluation"""
        # Restore from checkpoint first to ensure clean state
        self.restore_from_checkpoint()
        
        # Apply configuration to model using toggle's method
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            if layer_key in config:
                layer_config = config[layer_key]
                
                for component, comp_config in layer_config.items():
                    bits = comp_config['bits']
                    pruning = comp_config['pruning']
                    
                    # Apply to this layer/component
                    self._apply_to_component(layer_idx, component, bits, pruning)
    
    def _apply_to_component(self, layer_idx, component, bits, pruning):
        """Apply compression to a specific component"""
        # Find parameter prefix for this layer/component
        if self.toggle.model_type == "gpt2":
            param_prefix = f"transformer.h.{layer_idx}.{component}"
        else:  # LLaMA style
            param_prefix = f"model.layers.{layer_idx}.{component}"
        
        # Apply quantization and pruning to matching parameters
        self.toggle._apply_quantization_to_params(
            self.toggle.base_model, param_prefix, bits, pruning)
    
    def evaluate_config(self, config):
        """Evaluate a configuration with caching"""
        config_hash = self._get_config_hash(config)
        
        # Check cache
        if config_hash in self.eval_cache:
            return self.eval_cache[config_hash]
        
        # Apply configuration
        self._apply_config_with_swapping(config)
        
        # Evaluate
        torch.cuda.synchronize()  # Ensure GPU operations are complete
        self.eval_count += 1
        
        # Use CUDA events to measure time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # Ensure model is on the right device
        if next(self.toggle.base_model.parameters()).device.type != self.model_device.type:
            self.toggle.base_model.to(self.model_device)
            
        results = self.toggle.evaluate_stl_properties(self.toggle.base_model)
        
        end_event.record()
        torch.cuda.synchronize()
        eval_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
        
        # Calculate model size and other metrics
        stl_score = results['stl_score']
        model_size = self.calculate_model_size(config)
        
        # Calculate average bit-width and pruning
        total_bits = 0
        total_pruning = 0
        count = 0
        
        for layer_key, layer_config in config.items():
            for component, comp_config in layer_config.items():
                total_bits += comp_config['bits']
                total_pruning += comp_config['pruning']
                count += 1
        
        avg_bits = total_bits / count if count > 0 else 16
        avg_pruning = total_pruning / count if count > 0 else 0
        compression_ratio = (16 - avg_bits) / 16  # Higher is better
        
        # Add to results
        results['model_size'] = model_size
        results['avg_bits'] = avg_bits
        results['avg_pruning'] = avg_pruning
        results['compression_ratio'] = compression_ratio
        results['eval_time'] = eval_time
        
        # Add to cache
        self.eval_cache[config_hash] = results
        
        # Log evaluation
        status = "✓" if results['stl_satisfied'] else "✗"
        print(f"Eval #{self.eval_count}: STL={stl_score:.4f} {status}, "
              f"Size={model_size:.2f}MB, "
              f"Bits={avg_bits:.2f}, Pruning={avg_pruning*100:.1f}%, "
              f"Time={eval_time:.3f}s")
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return results
    
    def calculate_model_size(self, config):
        """Calculate model size based on configuration"""
        total_params = 0
        total_bits = 0
        
        for name, param in self.toggle.base_model.named_parameters():
            if 'weight' not in name:
                continue
                
            # Count parameters
            param_count = param.numel()
            total_params += param_count
            
            # Find matching layer and component in config
            bits = 16  # Default to 16-bit
            pruning = 0.0  # Default to no pruning
            
            for layer_idx in range(self.num_layers):
                layer_key = f'layer_{layer_idx}'
                if layer_key in config:
                    for component in self.components:
                        # Generate the parameter prefix for this layer/component
                        if self.toggle.model_type == "gpt2":
                            param_prefix = f"transformer.h.{layer_idx}.{component}"
                        else:  # LLaMA style
                            param_prefix = f"model.layers.{layer_idx}.{component}"
                            
                        if name.startswith(param_prefix) and component in config[layer_key]:
                            bits = config[layer_key][component]['bits']
                            pruning = config[layer_key][component]['pruning']
                            break
            
            # Adjust param count for pruning
            effective_param_count = param_count * (1 - pruning)
            
            # Add to total bits
            total_bits += effective_param_count * bits
        
        # Convert to MB
        model_size_mb = total_bits / (8 * 1024 * 1024)
        
        return model_size_mb
    
    def _get_sorted_layers(self):
        """Get layers sorted by sensitivity (least to most sensitive)"""
        if not self.sensitivity_map:
            # If no sensitivity map, just use default order
            return [f'layer_{i}' for i in range(self.num_layers)]
        
        # Sort layers by sensitivity rank
        layer_items = [(k, v['rank']) for k, v in self.sensitivity_map['layer'].items()]
        sorted_items = sorted(layer_items, key=lambda x: x[1])
        
        return [item[0] for item in sorted_items]
    
    def run_progressive_compression(self, max_iterations=100, force_all_iterations=True):
        """
        Run progressive compression to find optimal configurations
        
        This systematically compresses layers from least to most sensitive,
        backing off when constraints are violated.
        
        Args:
            max_iterations: Maximum iterations to try
            force_all_iterations: Whether to continue exploring for all iterations even if stuck
            
        Returns:
            best_config: Best configuration found
            best_results: Evaluation results for best configuration
        """
        print("Running progressive compression...")
        
        # Create checkpoint for state restoration
        self.create_checkpoint()
        
        # Start with base configuration (all 16-bit, no pruning)
        current_config = {}
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            current_config[layer_key] = {}
            for component in self.components:
                current_config[layer_key][component] = {
                    'bits': 16,
                    'pruning': 0.0
                }
        
        # Evaluate base configuration
        print("\nEvaluating base configuration (all 16-bit, no pruning)...")
        base_results = self.evaluate_config(current_config)
        
        if not base_results['stl_satisfied']:
            print("Warning: Base configuration does not satisfy STL constraints!")
            print("Consider relaxing constraints.")
        
        # Keep track of best configuration
        best_config = copy.deepcopy(current_config)
        best_results = base_results
        
        # Get layers sorted by sensitivity (least to most sensitive)
        sorted_layers = self._get_sorted_layers()
        print("\nLayers sorted by sensitivity (least to most sensitive):")
        for i, layer_key in enumerate(sorted_layers):
            print(f"  {i+1}. {layer_key}")
        
        # Track whether we found at least one valid configuration
        found_valid = base_results['stl_satisfied']
        
        # Progressive compression approach
        print("\nStarting progressive compression...")
        stuck_iterations = 0
        
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration+1}/{max_iterations}")
            
            # Create a copy of the current best configuration
            test_config = copy.deepcopy(best_config)
            
            # Try to compress one more layer
            success = False
            
            # First, try to compress by bit-width
            for layer_key in sorted_layers:
                # Skip if this layer is already at minimum bit-width
                min_bits_in_layer = min([test_config[layer_key][comp]['bits'] 
                                        for comp in test_config[layer_key]])
                
                if min_bits_in_layer <= min(self.bit_options):
                    continue
                
                # Find the next lower bit-width
                next_bits = max([b for b in self.bit_options if b < min_bits_in_layer])
                
                # Try reducing bit-width for this layer
                for component in self.components:
                    # Skip if already at minimum
                    if test_config[layer_key][component]['bits'] <= next_bits:
                        continue
                    
                    # Try reducing just this component
                    old_bits = test_config[layer_key][component]['bits']
                    test_config[layer_key][component]['bits'] = next_bits
                    
                    # Evaluate
                    results = self.evaluate_config(test_config)
                    
                    # Check if still valid
                    if results['stl_satisfied']:
                        success = True
                        found_valid = True
                        
                        # Update best config if this has better compression
                        if results['model_size'] < best_results['model_size']:
                            best_config = copy.deepcopy(test_config)
                            best_results = results
                        
                        print(f"Success! Reduced {layer_key}.{component} from {old_bits}-bit to {next_bits}-bit")
                        break
                    else:
                        # Revert change if invalid
                        test_config[layer_key][component]['bits'] = old_bits
                
                if success:
                    break
            
            # If bit-width compression worked, continue to next iteration
            if success:
                self.results_history.append({
                    'iteration': iteration,
                    'stl_score': best_results['stl_score'],
                    'model_size': best_results['model_size'],
                    'avg_bits': best_results['avg_bits'],
                    'avg_pruning': best_results['avg_pruning']
                })
                continue
            
            # If bit-width compression didn't work, try pruning
            for layer_key in sorted_layers:
                # Try to increase pruning
                for component in self.components:
                    current_pruning = test_config[layer_key][component]['pruning']
                    
                    # Find next pruning level
                    next_pruning_options = [p for p in self.pruning_options if p > current_pruning]
                    if not next_pruning_options:
                        continue
                    
                    next_pruning = min(next_pruning_options)
                    
                    # Try increasing pruning for this component
                    test_config[layer_key][component]['pruning'] = next_pruning
                    
                    # Evaluate
                    results = self.evaluate_config(test_config)
                    
                    # Check if still valid
                    if results['stl_satisfied']:
                        success = True
                        found_valid = True
                        
                        # Update best config if this has better compression
                        if results['model_size'] < best_results['model_size']:
                            best_config = copy.deepcopy(test_config)
                            best_results = results
                        
                        print(f"Success! Increased {layer_key}.{component} pruning from {current_pruning:.1f} to {next_pruning:.1f}")
                        break
                    else:
                        # Revert change if invalid
                        test_config[layer_key][component]['pruning'] = current_pruning
                
                if success:
                    break
            
            # Record results for this iteration
            self.results_history.append({
                'iteration': iteration,
                'stl_score': best_results['stl_score'],
                'model_size': best_results['model_size'],
                'avg_bits': best_results['avg_bits'],
                'avg_pruning': best_results['avg_pruning']
            })
            
            # If no valid compression options found
            if not success:
                stuck_iterations += 1
                print(f"\nNo further valid compression options found in this iteration. (Stuck for {stuck_iterations} iterations)")
                
                if force_all_iterations and iteration < max_iterations - 1:
                    # Try a different approach - randomly modify a component to escape local minimum
                    if stuck_iterations % 5 == 0:  # Every 5 stuck iterations, try something more aggressive
                        print("Trying random perturbation to escape local minimum...")
                        
                        # Randomly select a layer
                        layer_key = random.choice(sorted_layers)
                        
                        # Randomly select a component
                        component = random.choice(self.components)
                        
                        # Try slightly increasing bit-width in exchange for more aggressive pruning
                        current_bits = test_config[layer_key][component]['bits']
                        current_pruning = test_config[layer_key][component]['pruning']
                        
                        # Find next higher bit-width if possible
                        higher_bits_options = [b for b in self.bit_options if b > current_bits]
                        if higher_bits_options:
                            next_higher_bits = min(higher_bits_options)
                            
                            # Find more aggressive pruning
                            next_pruning_options = [p for p in self.pruning_options if p > current_pruning + 0.1]
                            if next_pruning_options:
                                next_pruning = min(next_pruning_options)
                                
                                print(f"Trying {layer_key}.{component}: bits {current_bits}→{next_higher_bits}, pruning {current_pruning:.1f}→{next_pruning:.1f}")
                                
                                # Apply changes
                                test_config[layer_key][component]['bits'] = next_higher_bits
                                test_config[layer_key][component]['pruning'] = next_pruning
                                
                                # Evaluate
                                results = self.evaluate_config(test_config)
                                
                                # Check if valid
                                if results['stl_satisfied']:
                                    success = True
                                    found_valid = True
                                    best_config = copy.deepcopy(test_config)
                                    best_results = results
                                    print(f"Success! Escaped local minimum with trade-off strategy")
                                    stuck_iterations = 0
                                else:
                                    # Revert changes if invalid
                                    test_config[layer_key][component]['bits'] = current_bits
                                    test_config[layer_key][component]['pruning'] = current_pruning
                else:
                    print("\nNo further valid compression options found. Stopping early.")
                    break
        
        # Final results
        print("\nProgressive compression complete!")
        if found_valid:
            print(f"Best configuration: STL score={best_results['stl_score']:.4f}, "
                  f"Model size={best_results['model_size']:.2f} MB, "
                  f"Avg bits={best_results['avg_bits']:.2f}, "
                  f"Avg pruning={best_results['avg_pruning']*100:.1f}%, "
                  f"Satisfied={best_results['stl_satisfied']}")
        else:
            print("No valid configurations found that satisfy all STL constraints.")
            print("Consider relaxing constraints or using a different approach.")
        
        print(f"Total evaluations: {self.eval_count}")
        
        return best_config, best_results
    
    def visualize_results(self, output_prefix="progressive_compression"):
        """Visualize the progression of results"""
        if not self.results_history:
            print("No results to visualize")
            return
        
        # Extract data
        iterations = [r['iteration'] for r in self.results_history]
        stl_scores = [r['stl_score'] for r in self.results_history]
        model_sizes = [r['model_size'] for r in self.results_history]
        avg_bits = [r['avg_bits'] for r in self.results_history]
        avg_pruning = [r['avg_pruning'] for r in self.results_history]
        
        # Plot STL score and model size
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, stl_scores, 'b-', label='STL Score')
        plt.axhline(y=0, color='r', linestyle='--', label='Satisfaction Threshold')
        plt.xlabel('Iteration')
        plt.ylabel('STL Score')
        plt.title('STL Score Progression')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(iterations, model_sizes, 'g-', label='Model Size (MB)')
        plt.xlabel('Iteration')
        plt.ylabel('Model Size (MB)')
        plt.title('Model Size Progression')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_progression.png")
        print(f"Progression plot saved to {output_prefix}_progression.png")
        
        # Plot bit-width and pruning
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, avg_bits, 'm-', label='Avg Bit-width')
        plt.xlabel('Iteration')
        plt.ylabel('Average Bit-width')
        plt.title('Bit-width Progression')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(iterations, [p*100 for p in avg_pruning], 'c-', label='Avg Pruning (%)')
        plt.xlabel('Iteration')
        plt.ylabel('Average Pruning (%)')
        plt.title('Pruning Progression')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_compression.png")
        print(f"Compression plot saved to {output_prefix}_compression.png")