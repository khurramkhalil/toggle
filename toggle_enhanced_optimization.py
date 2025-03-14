"""
Enhanced optimization for TOGGLE framework using caching, parameter swapping, 
and guided search based on layer sensitivity analysis.
"""

import numpy as np
import torch
from scipy.stats import norm
import time
from tqdm import tqdm
import hashlib
import json
import matplotlib.pyplot as plt

class EnhancedOptimizer:
    """
    Enhanced Bayesian Optimization for TOGGLE compression configurations.
    Uses caching, parameter swapping, and layer sensitivity to improve efficiency.
    """
    
    def __init__(self, toggle_framework, sensitivity_map=None, random_state=None):
        """
        Initialize the Enhanced Optimizer.
        
        Args:
            toggle_framework: DynamicPrecisionTransformer instance
            sensitivity_map: Layer sensitivity data (optional)
            random_state: Random state for reproducibility
        """
        self.toggle = toggle_framework
        self.num_layers = toggle_framework.num_layers
        self.components = toggle_framework.components
        self.random_state = random_state
        self.sensitivity_map = sensitivity_map
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Set up search space
        self.bit_options = toggle_framework.bit_options
        
        # Initialize evaluation cache
        self.eval_cache = {}
        
        # Initialize history for GP
        self.X = []  # Configurations
        self.y = []  # Evaluation results
        
        # Store original parameters for swapping
        self.original_params = {}
        self.store_original_parameters()
        
        # Set up GP hyperparameters
        self.noise = 0.1
        self.length_scale = 1.0
        self.signal_variance = 1.0
    
    def store_original_parameters(self):
        """Store original model parameters for later restoration"""
        print("Storing original model parameters...")
        self.original_params = {}
        with torch.no_grad():
            for name, param in self.toggle.base_model.named_parameters():
                if 'weight' in name:
                    self.original_params[name] = param.data.clone()
    
    def restore_original_parameters(self):
        """Restore original model parameters"""
        with torch.no_grad():
            for name, param in self.toggle.base_model.named_parameters():
                if name in self.original_params:
                    param.data.copy_(self.original_params[name])
    
    def _get_config_hash(self, config):
        """Create a hash of the configuration for caching"""
        # Convert config to a standardized string format
        config_str = ""
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            if layer_key in config:
                for component in sorted(self.components):
                    if component in config[layer_key]:
                        bits = config[layer_key][component]['bits']
                        pruning = round(config[layer_key][component]['pruning'], 2)
                        config_str += f"{layer_key}.{component}:{bits}:{pruning};"
        
        # Generate hash
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _config_to_vector(self, config):
        """Convert a configuration dictionary to a flat vector for GP"""
        vector = []
        
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            if layer_key in config:
                layer_config = config[layer_key]
                
                for component in self.components:
                    if component in layer_config:
                        # Normalize bit-width to [0, 1]
                        bits = layer_config[component]['bits']
                        bits_idx = self.bit_options.index(bits) if bits in self.bit_options else -1
                        if bits_idx == -1:
                            # Handle case where bit value isn't in bit_options
                            closest_bit = min(self.bit_options, key=lambda x: abs(x - bits))
                            bits_idx = self.bit_options.index(closest_bit)
                        bits_norm = bits_idx / (len(self.bit_options) - 1)
                        
                        # Normalize pruning to [0, 1]
                        pruning = layer_config[component]['pruning']
                        pruning_norm = pruning / 0.5  # Max pruning is 0.5
                        
                        vector.extend([bits_norm, pruning_norm])
                    else:
                        # Default values if component not found
                        vector.extend([1.0, 0.0])  # 16-bit, no pruning
            else:
                # Default values if layer not found
                for _ in range(len(self.components)):
                    vector.extend([1.0, 0.0])  # 16-bit, no pruning
        
        return np.array(vector)
    
    def _vector_to_config(self, vector):
        """Convert a flat vector back to a configuration dictionary"""
        config = {}
        
        idx = 0
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            layer_config = {}
            
            for component in self.components:
                # Get bit-width from vector
                bits_norm = vector[idx]
                bits_idx = int(round(bits_norm * (len(self.bit_options) - 1)))
                bits_idx = max(0, min(bits_idx, len(self.bit_options) - 1))
                bits = self.bit_options[bits_idx]
                
                # Get pruning from vector
                pruning_norm = vector[idx + 1]
                pruning = pruning_norm * 0.5
                pruning = max(0.0, min(pruning, 0.5))
                
                layer_config[component] = {
                    'bits': bits,
                    'pruning': pruning
                }
                
                idx += 2
            
            config[layer_key] = layer_config
        
        return config
    
    def _random_config_vector(self):
        """Generate a random configuration vector biased by sensitivity"""
        vector = np.zeros(self.num_layers * len(self.components) * 2)
        
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            
            # Get layer sensitivity if available
            if self.sensitivity_map and layer_key in self.sensitivity_map['layer']:
                # Normalized rank (0-1, where 0 = least sensitive)
                norm_rank = self.sensitivity_map['layer'][layer_key]['rank'] / (self.num_layers - 1)
                
                # Less sensitive layers get more compression (lower bits, more pruning)
                # Use beta distribution with parameters controlled by sensitivity
                alpha_bits = 1 + 3 * norm_rank  # Higher for more sensitive layers (less compression)
                beta_bits = 1 + 3 * (1 - norm_rank)  # Higher for less sensitive layers (more compression)
                
                alpha_pruning = 1 + 3 * (1 - norm_rank)  # Higher for less sensitive layers (more pruning)
                beta_pruning = 1 + 3 * norm_rank  # Higher for more sensitive layers (less pruning)
            else:
                # Default values for random exploration
                alpha_bits = 2
                beta_bits = 5
                alpha_pruning = 1.5
                beta_pruning = 5
            
            # Generate biased random values for each component
            for comp_idx in range(len(self.components)):
                # Index for this layer/component in the flat vector
                idx = (layer_idx * len(self.components) + comp_idx) * 2
                
                # Bits (0-1, where 0 = lowest bits, 1 = highest bits)
                bits_value = np.random.beta(alpha_bits, beta_bits)
                
                # Pruning (0-1, where 0 = no pruning, 1 = max pruning)
                pruning_value = np.random.beta(alpha_pruning, beta_pruning)
                
                # Store in vector
                vector[idx] = bits_value
                vector[idx + 1] = pruning_value
        
        return vector
    
    def create_sensitivity_guided_configs(self, aggression_levels=None):
        """
        Create configurations guided by layer sensitivity at different compression levels
        
        Args:
            aggression_levels: List of aggression levels (0-1) to try
            
        Returns:
            List of configurations
        """
        if not self.sensitivity_map:
            return [self.toggle.default_config]
        
        aggression_levels = aggression_levels or [0.3, 0.5, 0.7]
        configs = []
        
        from toggle_sensitivity_analysis import LayerSensitivityAnalyzer
        analyzer = LayerSensitivityAnalyzer(self.toggle)
        
        for level in aggression_levels:
            config = analyzer.create_sensitivity_guided_config(
                self.sensitivity_map, aggression_level=level)
            configs.append(config)
        
        return configs
    
    def _kernel(self, x1, x2):
        """Compute RBF kernel between two vectors"""
        diff = x1 - x2
        return self.signal_variance * np.exp(-0.5 * np.sum(diff**2) / self.length_scale**2)
    
    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute kernel matrix between two sets of vectors"""
        if X2 is None:
            X2 = X1
            
        K = np.zeros((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i, j] = self._kernel(x1, x2)
                
        return K
    
    def _gp_predict(self, X_test):
        """
        Make predictions using Gaussian Process regression
        
        Args:
            X_test: Test points
            
        Returns:
            mean, variance
        """
        if len(self.X) == 0:
            # No data yet, return prior
            return np.zeros(len(X_test)), np.ones(len(X_test)) * self.signal_variance
        
        # Use at most 50 points for prediction to avoid cubic scaling
        if len(self.X) > 50:
            # Use most recent points plus some random historical points
            recent_indices = list(range(max(0, len(self.X) - 30), len(self.X)))
            remaining_indices = list(range(0, max(0, len(self.X) - 30)))
            if remaining_indices:
                random_indices = np.random.choice(
                    remaining_indices, 
                    min(20, len(remaining_indices)), 
                    replace=False
                ).tolist()
            else:
                random_indices = []
            indices = sorted(recent_indices + random_indices)
            
            X_train = np.array([self.X[i] for i in indices])
            y_train = np.array([self.y[i] for i in indices])
        else:
            X_train = np.array(self.X)
            y_train = np.array(self.y)
        
        # Compute kernel matrices
        K = self._compute_kernel_matrix(X_train)
        K_s = self._compute_kernel_matrix(X_train, X_test)
        K_ss = self._compute_kernel_matrix(X_test)
        
        # Add noise to diagonal
        K_noise = K + np.eye(len(K)) * self.noise
        
        # Compute posterior mean and variance
        try:
            # Use more stable computation
            L = np.linalg.cholesky(K_noise)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
            mu = K_s.T @ alpha
            
            v = np.linalg.solve(L, K_s)
            sigma2 = np.diag(K_ss - v.T @ v)
            
            # Ensure variances are positive
            sigma2 = np.maximum(sigma2, 1e-6)
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails
            try:
                K_inv = np.linalg.inv(K_noise)
                mu = K_s.T @ K_inv @ y_train
                sigma2 = np.diag(K_ss - K_s.T @ K_inv @ K_s)
                sigma2 = np.maximum(sigma2, 1e-6)
            except:
                print("Warning: Matrix inversion failed, using prior")
                return np.zeros(len(X_test)), np.ones(len(X_test)) * self.signal_variance
        
        return mu, sigma2
    
    def _expected_improvement(self, X, y_best):
        """
        Compute Expected Improvement acquisition function
        
        Args:
            X: Candidate points
            y_best: Current best objective value
            
        Returns:
            EI values
        """
        mu, sigma2 = self._gp_predict(X)
        sigma = np.sqrt(sigma2)
        
        # Handle case where sigma is zero
        mask = sigma > 1e-6
        
        ei = np.zeros_like(mu)
        
        if np.any(mask):
            z = (mu[mask] - y_best) / sigma[mask]
            ei[mask] = (mu[mask] - y_best) * norm.cdf(z) + sigma[mask] * norm.pdf(z)
        
        return ei
    
    def _obj_function(self, config):
        """
        Evaluate a configuration with continuous rewards for STL satisfaction
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Objective value (higher is better), results
        """
        # Check cache first
        config_hash = self._get_config_hash(config)
        if config_hash in self.eval_cache:
            return self.eval_cache[config_hash]
        
        # Apply configuration to base model using parameter swapping
        self._apply_config_with_swapping(config)
        
        # Evaluate with the current model state
        results = self.toggle.evaluate_stl_properties(self.toggle.base_model)
        
        # Calculate metrics and objective value
        stl_score = results['stl_score']
        model_size = results['model_size']
        robustness = results['robustness']
        
        # Calculate average bit-width
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
        
        # Create a continuous objective that rewards partial STL satisfaction
        # and lower bit-widths even when not fully satisfied
        if stl_score >= 0:
            # Satisfied: Maximize compression while maintaining satisfaction
            # Scale from 0 to 100 based on compression ratio
            obj_value = 100 + 50 * compression_ratio + 10 * stl_score
        else:
            # Not satisfied, but give partial credit for being close
            # Scale from -100 to 0 based on how close we are
            normalization = min(1.0, max(0.0, (stl_score + 1) / 2))
            obj_value = -100 + 100 * normalization + 10 * compression_ratio
        
        # Add evaluation metrics to results
        results['avg_bits'] = avg_bits
        results['avg_pruning'] = avg_pruning
        results['compression_ratio'] = compression_ratio
        results['obj_value'] = obj_value
        
        # Add to cache
        self.eval_cache[config_hash] = (obj_value, results)
        
        # Log evaluation
        status = "✓" if results['stl_satisfied'] else "✗"
        print(f"Eval: STL={stl_score:.4f} {status}, "
              f"Size={model_size:.2f}MB, "
              f"Bits={avg_bits:.2f}, Pruning={avg_pruning*100:.1f}%, "
              f"Obj={obj_value:.2f}")
        
        # Restore original parameters
        self.restore_original_parameters()
        
        return obj_value, results
    
    def _apply_config_with_swapping(self, config):
        """Apply configuration using parameter swapping"""
        # Make sure original parameters are stored
        if not self.original_params:
            self.store_original_parameters()
        
        # Apply configuration to base model
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
        for name, param in self.toggle.base_model.named_parameters():
            if name.startswith(param_prefix) and 'weight' in name:
                # Apply configuration using toggle's method
                self.toggle._apply_quantization_to_params(
                    self.toggle.base_model, param_prefix, bits, pruning)
    
    def optimize(self, max_iterations=50, exploration_weight=0.1):
        """
        Run enhanced optimization to find the best configuration
        
        Args:
            max_iterations: Maximum number of iterations
            exploration_weight: Weight for exploration in acquisition function
            
        Returns:
            best_config, best_results
        """
        print("Starting Enhanced Optimization...")
        
        # Create and evaluate sensitivity-guided configurations
        if self.sensitivity_map:
            guided_configs = self.create_sensitivity_guided_configs()
            print(f"Evaluating {len(guided_configs)} sensitivity-guided configurations...")
            
            for config in guided_configs:
                config_vector = self._config_to_vector(config)
                obj_value, results = self._obj_function(config)
                
                self.X.append(config_vector)
                self.y.append(obj_value)
        else:
            # Evaluate default configuration if no sensitivity map
            print("Evaluating default configuration...")
            default_config = self.toggle.default_config
            default_vector = self._config_to_vector(default_config)
            obj_value, results = self._obj_function(default_config)
            
            self.X.append(default_vector)
            self.y.append(obj_value)
            
            # Also evaluate a random configuration
            print("Evaluating a random configuration...")
            random_vector = self._random_config_vector()
            random_config = self._vector_to_config(random_vector)
            obj_value, results = self._obj_function(random_config)
            
            self.X.append(random_vector)
            self.y.append(obj_value)
        
        # Main optimization loop
        best_obj = max(self.y)
        best_idx = np.argmax(self.y)
        best_config = self._vector_to_config(self.X[best_idx])
        best_results = None
        
        print(f"Starting optimization loop for {max_iterations} iterations...")
        for i in tqdm(range(max_iterations)):
            # Generate random candidate points biased by sensitivity
            n_candidates = 1000
            X_candidates = [self._random_config_vector() for _ in range(n_candidates)]
            
            # Compute acquisition function values
            ei_values = self._expected_improvement(X_candidates, best_obj)
            
            # Add exploration bonus
            if exploration_weight > 0:
                _, var = self._gp_predict(X_candidates)
                ei_values = ei_values + exploration_weight * np.sqrt(var)
            
            # Select best candidate
            best_candidate_idx = np.argmax(ei_values)
            next_vector = X_candidates[best_candidate_idx]
            next_config = self._vector_to_config(next_vector)
            
            # Evaluate candidate
            obj_value, results = self._obj_function(next_config)
            
            # Update data
            self.X.append(next_vector)
            self.y.append(obj_value)
            
            # Update best result
            if obj_value > best_obj:
                best_obj = obj_value
                best_idx = len(self.y) - 1
                best_config = next_config
                best_results = results
                
                print(f"Iteration {i+1}: New best config found!")
                print(f"  STL score={results['stl_score']:.4f}, "
                      f"Model size={results['model_size']:.2f} MB, "
                      f"Satisfied={results['stl_satisfied']}")
            
            # Periodically update hyperparameters
            if (i + 1) % 10 == 0:
                self._update_hyperparameters()
        
        # Evaluate best configuration to get full results if not already done
        if best_results is None:
            _, best_results = self._obj_function(best_config)
        
        print("\nOptimization complete!")
        print(f"Best configuration: STL score={best_results['stl_score']:.4f}, "
              f"Model size={best_results['model_size']:.2f} MB, "
              f"Satisfied={best_results['stl_satisfied']}")
        
        return best_config, best_results
    
    def _update_hyperparameters(self):
        """Update GP hyperparameters using simple heuristics"""
        if len(self.X) < 5:
            return
        
        # Compute mean distance between points
        X_arr = np.array(self.X)
        dists = []
        for i in range(len(X_arr)):
            for j in range(i+1, len(X_arr)):
                dists.append(np.sum((X_arr[i] - X_arr[j])**2))
        
        mean_dist = np.mean(dists)
        
        # Update length scale based on mean distance
        self.length_scale = mean_dist / 2.0
        
        # Update signal variance based on y values
        y_var = np.var(self.y)
        if y_var > 0:
            self.signal_variance = y_var
    
    def visualize_optimization_progress(self, output_prefix="enhanced_opt"):
        """
        Create visualizations of the optimization process
        
        Args:
            output_prefix: Prefix for output files
        """
        # Plot objective value over iterations
        plt.figure(figsize=(10, 6))
        y_values = np.array(self.y)
        iterations = np.arange(len(y_values))
        
        # Mark if obj value > 0 (STL satisfied)
        is_satisfied = y_values > 100  # Values above 100 indicate satisfaction
        
        plt.scatter(iterations[is_satisfied], y_values[is_satisfied], 
                    label='STL Satisfied', color='green', marker='o')
        plt.scatter(iterations[~is_satisfied], y_values[~is_satisfied], 
                    label='STL Violated', color='red', marker='x')
        
        plt.plot(iterations, y_values, 'k--', alpha=0.3)
        
        plt.axhline(y=100, color='gray', linestyle='-', alpha=0.5,
                   label='Satisfaction Threshold')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Enhanced Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{output_prefix}_progress.png")
        print(f"Optimization progress saved to {output_prefix}_progress.png")
        
        # Plot best value found so far
        plt.figure(figsize=(10, 6))
        best_so_far = np.maximum.accumulate(y_values)
        plt.plot(iterations, best_so_far, 'b-', linewidth=2)
        plt.axhline(y=100, color='green', linestyle='--', 
                   label='Satisfaction Threshold')
        plt.xlabel('Iteration')
        plt.ylabel('Best Objective Value')
        plt.title('Best Value Found')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{output_prefix}_best_value.png")
        print(f"Best value plot saved to {output_prefix}_best_value.png")