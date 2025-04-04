"""
Improved Bayesian Optimization for TOGGLE compression configurations
This updated version uses a more effective objective function, better exploration,
and more flexible handling of configurations.
"""

import numpy as np
from scipy.stats import norm
import time
from tqdm import tqdm

class BayesianOptimization:
    """
    Bayesian Optimization for TOGGLE compression configurations.
    Uses a Gaussian Process (GP) as a surrogate model and Expected Improvement (EI)
    as the acquisition function to efficiently explore the configuration space.
    """
    
    def __init__(self, toggle_framework, num_layers, components, init_configs=None, random_state=None):
        """
        Initialize the Bayesian Optimization.
        
        Args:
            toggle_framework: DynamicPrecisionTransformer instance
            num_layers: Number of layers in the model
            components: List of component names per layer
            init_configs: List of initial configurations to evaluate (optional)
            random_state: Random state for reproducibility
        """
        self.toggle = toggle_framework
        self.num_layers = num_layers
        self.components = components
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Set up search space
        self.bit_options = toggle_framework.bit_options
        self.pruning_options = np.linspace(0.0, 0.5, 6)  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
        
        # Initialize history for Gaussian Process
        self.X = []  # Configurations
        self.y = []  # Evaluation results (STL score, model size)
        
        # Initial configs
        self.init_configs = init_configs or self.create_initial_configs()
        
        # Set up GP hyperparameters
        self.noise = 0.1
        self.length_scale = 1.0
        self.signal_variance = 1.0
    
    def create_initial_configs(self):
        """Create a diverse set of initial configurations to explore"""
        configs = []
        
        # Add a uniform 8-bit config
        uniform_8bit = {}
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            layer_config = {}
            for component in self.components:
                layer_config[component] = {
                    'bits': 8,
                    'pruning': 0.0
                }
            uniform_8bit[layer_key] = layer_config
        configs.append(uniform_8bit)
        
        # Add layer-wise decreasing precision (more bits for early layers)
        decreasing_config = {}
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            layer_config = {}
            
            # Lower layers have higher precision
            relative_depth = layer_idx / max(1, self.num_layers - 1)  # 0 to 1
            if relative_depth < 0.33:
                bits = 12
            elif relative_depth < 0.66:
                bits = 8
            else:
                bits = 6
                
            for component in self.components:
                layer_config[component] = {
                    'bits': bits,
                    'pruning': 0.0
                }
            decreasing_config[layer_key] = layer_config
        configs.append(decreasing_config)
        
        return configs
    
    def _config_to_vector(self, config):
        """Convert a configuration dictionary to a flat vector for GP"""
        vector = []
        
        # Check if config is a list or dictionary
        if isinstance(config, list):
            # Convert from list format to dictionary format
            dict_config = {}
            for layer_idx in range(self.num_layers):
                dict_config[f'layer_{layer_idx}'] = {}
                for comp_idx, component in enumerate(self.components):
                    dict_config[f'layer_{layer_idx}'][component] = config[layer_idx][comp_idx]
            config = dict_config
        
        # Process the dictionary format
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
    
    def _random_config(self):
        """Generate a random configuration vector biased toward lower bit-widths"""
        # Create biased random values for bit-widths (more likely to pick lower values)
        bit_values = np.random.beta(2, 5, self.num_layers * len(self.components))
        # Create random values for pruning
        pruning_values = np.random.beta(1.5, 5, self.num_layers * len(self.components))
        
        # Combine into a single vector
        vector = np.zeros(self.num_layers * len(self.components) * 2)
        vector[0::2] = bit_values    # Even indices for bit-widths
        vector[1::2] = pruning_values  # Odd indices for pruning
        
        return vector
    
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
            K_inv = np.linalg.inv(K_noise)
            mu = K_s.T @ K_inv @ y_train
            sigma2 = np.diag(K_ss - K_s.T @ K_inv @ K_s)
            
            # Ensure variances are positive
            sigma2 = np.maximum(sigma2, 1e-6)
        except np.linalg.LinAlgError:
            # Fallback if matrix inversion fails
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
        # Evaluate configuration
        results = self.toggle.evaluate_config(config)
        
        # Extract key metrics
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
        
        # Add the evaluation result to the log
        self.log_evaluation(config, results, avg_bits, avg_pruning, obj_value)
        
        return obj_value, results
    
    def log_evaluation(self, config, results, avg_bits, avg_pruning, obj_value):
        """Log evaluation results for analysis"""
        log_entry = {
            'obj_value': float(obj_value),
            'stl_score': float(results['stl_score']),
            'model_size': float(results['model_size']),
            'stl_satisfied': bool(results['stl_satisfied']),
            'avg_bits': float(avg_bits),
            'avg_pruning': float(avg_pruning),
            'robustness': {k: float(v) for k, v in results['robustness'].items()}
        }
        
        # Print summary
        status = "✓" if results['stl_satisfied'] else "✗"
        print(f"Eval: STL={results['stl_score']:.4f} {status}, "
              f"Size={results['model_size']:.2f}MB, "
              f"Bits={avg_bits:.2f}, Pruning={avg_pruning*100:.1f}%, "
              f"Obj={obj_value:.2f}")
        
        # Save to log file (optional)
        # try:
        #     with open('optimization_log.jsonl', 'a') as f:
        #         f.write(json.dumps(log_entry) + '\n')
        # except:
        #     pass
    
    def optimize(self, max_iterations=50, exploration_weight=0.1):
        """
        Run Bayesian Optimization to find the best configuration
        
        Args:
            max_iterations: Maximum number of iterations
            exploration_weight: Weight for exploration in acquisition function
            
        Returns:
            best_config, best_results
        """
        print("Starting Bayesian Optimization...")
        
        # Evaluate initial configurations
        if self.init_configs:
            print(f"Evaluating {len(self.init_configs)} initial configurations...")
            for config in tqdm(self.init_configs):
                config_vector = self._config_to_vector(config)
                obj_value, results = self._obj_function(config)
                
                self.X.append(config_vector)
                self.y.append(obj_value)
        
        # If no initial configs, evaluate default and a random one
        if not self.X:
            print("Evaluating default configuration...")
            default_config = self.toggle.default_config
            default_vector = self._config_to_vector(default_config)
            obj_value, results = self._obj_function(default_config)
            
            self.X.append(default_vector)
            self.y.append(obj_value)
            
            print("Evaluating a random configuration...")
            random_vector = self._random_config()
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
            # Generate random candidate points
            n_candidates = 1000
            X_candidates = [self._random_config() for _ in range(n_candidates)]
            
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