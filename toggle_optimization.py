"""
Bayesian Optimization for TOGGLE compression configurations
This module provides gradient-free optimization to find optimal bit-width
and pruning configurations that satisfy STL constraints while minimizing model size.
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
        self.init_configs = init_configs or []
        
        # Set up GP hyperparameters
        self.noise = 0.1
        self.length_scale = 1.0
        self.signal_variance = 1.0
    
    def _config_to_vector(self, config):
        """Convert a configuration dictionary to a flat vector for GP"""
        vector = []
        
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            layer_config = config[layer_key]
            
            for component in self.components:
                if component in layer_config:
                    # Normalize bit-width to [0, 1]
                    bits = layer_config[component]['bits']
                    bits_idx = self.bit_options.index(bits)
                    bits_norm = bits_idx / (len(self.bit_options) - 1)
                    
                    # Normalize pruning to [0, 1]
                    pruning = layer_config[component]['pruning']
                    pruning_norm = pruning / 0.5  # Max pruning is 0.5
                    
                    vector.extend([bits_norm, pruning_norm])
                else:
                    # Default values if component not found
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
        """Generate a random configuration vector"""
        vector = np.random.rand(self.num_layers * len(self.components) * 2)
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
        K_inv = np.linalg.inv(K_noise)
        mu = K_s.T @ K_inv @ y_train
        sigma2 = np.diag(K_ss - K_s.T @ K_inv @ K_s)
        
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
        Evaluate a configuration with respect to STL satisfaction and model size
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Objective value (higher is better)
        """
        # Evaluate configuration
        results = self.toggle.evaluate_config(config)
        
        # Extract key metrics
        stl_score = results['stl_score']
        model_size = results['model_size']
        stl_satisfied = results['stl_satisfied']
        
        # If STL constraints are satisfied, optimize for model size
        # Otherwise, optimize for STL score
        if stl_satisfied:
            # Penalize larger models, but keep obj value positive for satisfied configs
            obj_value = 100.0 - 0.1 * model_size
        else:
            # Make obj value negative for unsatisfied configs
            obj_value = stl_score - 100.0
        
        return obj_value, results
    
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
                
                print(f"Initial config: STL score={results['stl_score']:.4f}, "
                      f"Model size={results['model_size']:.2f} MB, "
                      f"Satisfied={results['stl_satisfied']}")
        
        # If no initial configs, evaluate default and a random one
        if not self.X:
            print("Evaluating default configuration...")
            default_config = self.toggle.default_config
            default_vector = self._config_to_vector(default_config)
            obj_value, results = self._obj_function(default_config)
            
            self.X.append(default_vector)
            self.y.append(obj_value)
            
            print(f"Default config: STL score={results['stl_score']:.4f}, "
                  f"Model size={results['model_size']:.2f} MB, "
                  f"Satisfied={results['stl_satisfied']}")
            
            print("Evaluating a random configuration...")
            random_vector = self._random_config()
            random_config = self._vector_to_config(random_vector)
            obj_value, results = self._obj_function(random_config)
            
            self.X.append(random_vector)
            self.y.append(obj_value)
            
            print(f"Random config: STL score={results['stl_score']:.4f}, "
                  f"Model size={results['model_size']:.2f} MB, "
                  f"Satisfied={results['stl_satisfied']}")
        
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