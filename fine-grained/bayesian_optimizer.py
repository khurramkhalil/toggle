import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

# Check if scikit-optimize is available, otherwise usse a simpler approach
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from model_compressor import ModelCompressor
from config import CompressionConfig, QuantizationConfig, PruningConfig, VerificationConfig


class BayesianCompressionOptimizer:
    """
    Uses Bayesian Optimization to find the optimal compression parameters
    that maximize compression while satisfying formal properties.
    """
    
    def __init__(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sample_inputs: List[str],
        property_configs: List[Dict[str, Any]],
        n_calls: int = 20,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize the optimizer
        
        Args:
            model: Model to optimize compression for
            tokenizer: Tokenizer for the model
            sample_inputs: List of input texts for evaluation
            property_configs: List of property configurations
            n_calls: Number of optimization iterations
            random_state: Random seed
            verbose: Whether to print progress
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sample_inputs = sample_inputs
        self.property_configs = property_configs
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        
        # Create search space
        self.dimensions = self._create_search_space()
        
        # Store results
        self.results = []
        self.best_config = None
        self.best_model = None
        self.best_score = float('-inf')
            
    def optimize(self) -> Tuple[Dict[str, Any], PreTrainedModel, float]:
            """
            Run Bayesian optimization to find optimal compression parameters
            
            Returns:
                Tuple of (best_config, best_model, best_score)
            """
            # Initialize with default config as fallback
            self.best_config = self._create_config_from_params([8, 'l1', 0.2, False])
            self.best_model = None
            self.best_score = float('-inf')
            
            if SKOPT_AVAILABLE:
                self._optimize_with_skopt()
            else:
                self._optimize_with_random_search()
            
            # If no valid configurations were found, try to compress with default config
            if self.best_model is None:
                print("Warning: No valid configurations found. Using default config.")
                try:
                    compressor = ModelCompressor(
                        self.model, 
                        self.tokenizer, 
                        config=self.best_config
                    )
                    self.best_model = compressor.compress()
                    self.best_score = 0.0  # Default neutral score
                except Exception as e:
                    print(f"Error with default config: {str(e)}")
                    # As last resort, just return the original model
                    self.best_model = self.model
                    self.best_score = 0.0
                
            return self.best_config, self.best_model, self.best_score
        
    def _optimize_with_skopt(self):
            """Use scikit-optimize for Bayesian optimization"""
            # First run a few random evaluations to initialize
            initial_random_points = min(5, self.n_calls // 2)
            random_y = []
            
            for _ in range(initial_random_points):
                # Sample a random point
                x = []
                for dim in self.dimensions:
                    if isinstance(dim, Integer):
                        x.append(np.random.randint(dim.low, dim.high + 1))
                    elif isinstance(dim, Real):
                        x.append(np.random.uniform(dim.low, dim.high))
                    elif isinstance(dim, Categorical) or isinstance(dim, list):
                        choices = dim.categories if isinstance(dim, Categorical) else dim
                        x.append(np.random.choice(choices))
                    else:
                        raise ValueError(f"Unknown dimension type: {type(dim)}")
                
                # Evaluate it
                y = self._objective_function(x)
                random_y.append(y)
            
            # Check if we have any valid points
            if not any(np.isfinite(y) for y in random_y):
                print("Warning: No valid configurations found in initial random search. " 
                    "Using random search instead of Bayesian optimization.")
                return self._optimize_with_random_search()
            
            # Now run GP minimize
            result = gp_minimize(
                self._objective_function,
                self.dimensions,
                n_calls=self.n_calls - initial_random_points,
                random_state=self.random_state,
                verbose=self.verbose
            )
            
            # Create best configuration
            if not self.best_config:  # If no valid configuration was found
                print("Warning: No valid configurations found during optimization.")
                return
                
            # Re-run with best parameters to get the model
            compressor = ModelCompressor(
                self.model, 
                self.tokenizer, 
                config=self.best_config
            )
            self.best_model = compressor.compress()
            self.best_score = -result.fun  # Negate because we minimized the negative score
        
    def _optimize_with_random_search(self):
        """Use random search when scikit-optimize is not available"""
        best_score = float('-inf')
        best_params = None
        
        for i in range(self.n_calls):
            # Sample random parameters
            params = []
            for dim in self.dimensions:
                if isinstance(dim, tuple) and len(dim) == 3:  # Real
                    low, high, _ = dim
                    params.append(np.random.uniform(low, high))
                elif isinstance(dim, tuple) and len(dim) == 2:  # Integer
                    low, high = dim
                    params.append(np.random.randint(low, high+1))
                else:  # Categorical
                    params.append(np.random.choice(dim))
            
            # Evaluate
            score = -self._objective_function(params)  # Negate because we're maximizing
            
            if self.verbose:
                print(f"Iteration {i+1}/{self.n_calls}: Score = {score}")
                
            if score > best_score:
                best_score = score
                best_params = params
                
        # Create best configuration
        self.best_config = self._create_config_from_params(best_params)
        
        # Re-run with best parameters to get the model
        compressor = ModelCompressor(
            self.model, 
            self.tokenizer, 
            config=self.best_config
        )
        self.best_model = compressor.compress()
        self.best_score = best_score
        
    def _objective_function(self, params: List[Any]) -> float:
            """
            Objective function to minimize (negative score)
            
            Args:
                params: List of parameter values
                
            Returns:
                Negative score (to minimize)
            """
            # Create configuration from parameters
            config = self._create_config_from_params(params)
            
            # Create model compressor
            compressor = ModelCompressor(
                self.model, 
                self.tokenizer, 
                config=config
            )
            
            try:
                # Compress model
                compressed_model = compressor.compress()
                
                # Verify formal properties
                verification_results = compressor.verify(self.sample_inputs)
                
                # Calculate score
                score = self._calculate_score(
                    compressor, 
                    compressed_model,
                    verification_results
                )
                
                # Store result
                self.results.append({
                    'params': params,
                    'config': config,
                    'score': score,
                    'verification': verification_results
                })
                
                # Update best if better
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = config
                    self.best_model = compressed_model
                    
                    if self.verbose:
                        print(f"New best score: {score}")
                        print(f"Configuration: {self._config_summary(config)}")
                
                # Return negative score (for minimization)
                # Make sure we don't return -infinity which causes problems with GP
                return max(-score, -1e8)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error during evaluation: {str(e)}")
                # Return a large but finite value instead of -infinity
                return 1e8  # A very large penalty, but not infinity
            
    def _calculate_score(
            self, 
            compressor: ModelCompressor, 
            compressed_model: PreTrainedModel,
            verification_results: Dict[str, Any]
        ) -> float:
            """
            Calculate optimization score
            
            Higher is better. Combines compression ratio with property satisfaction.
            
            Args:
                compressor: ModelCompressor instance
                compressed_model: Compressed model
                verification_results: Verification results
                
            Returns:
                Score value (higher is better)
            """
            # Get size reduction
            evaluation = compressor.evaluate(self.sample_inputs)
            size_reduction = evaluation["size_reduction"]
            reduction_ratio = size_reduction["reduction_ratio"]
            
            # Check if all properties are satisfied
            all_satisfied = verification_results["summary"]["verification_passed"]
            
            # Calculate minimum robustness (how well properties are satisfied)
            min_robustness = verification_results["summary"].get("min_robustness", 0)
            
            if all_satisfied:
                # If all properties are satisfied, score is reduction ratio
                score = reduction_ratio
            else:
                # If any property is violated, penalize the score
                # The more negative the robustness, the worse the score
                # But ensure the score stays bounded
                robustness_penalty = max(-1.0, min_robustness)
                score = max(0, reduction_ratio + robustness_penalty)
                
            return score
        
    def _create_search_space(self) -> List[Any]:
            """
            Create search space for optimization
            
            Returns:
                List of dimensions for the search space
            """
            if SKOPT_AVAILABLE:
                # Create scikit-optimize dimensions
                dimensions = [
                    # Quantization bit width
                    Integer(2, 16, name='w_bits'),
                    
                    # Pruning method
                    Categorical(['l1', 'l2', 'random'], name='pruning_method'),
                    
                    # Pruning amount - using categorical for discrete steps of 0.1
                    Categorical([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], name='pruning_amount'),
                    
                    # Weight layerwise quantization
                    Categorical([True, False], name='weight_layerwise')
                ]
            else:
                # Create simple dimensions for random search
                dimensions = [
                    (2, 16),  # w_bits (Integer)
                    ['l1', 'l2', 'random'],  # pruning_method (Categorical)
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # pruning_amount (Categorical)
                    [True, False]  # weight_layerwise (Categorical)
                ]
                
            return dimensions
        
    def _create_config_from_params(self, params: List[Any]) -> CompressionConfig:
        """
        Create CompressionConfig from optimization parameters
        
        Args:
            params: List of parameter values
            
        Returns:
            CompressionConfig instance
        """
        w_bits, pruning_method, pruning_amount, weight_layerwise = params
        
        # Convert float parameters to appropriate types
        w_bits = int(w_bits)
        pruning_amount = float(pruning_amount)
        
        return CompressionConfig(
            quantization=QuantizationConfig(
                enabled=True,
                target_layers=None,  # Quantize all layers
                w_bits=w_bits,
                weight_layerwise=weight_layerwise
            ),
            pruning=PruningConfig(
                enabled=True,
                method=pruning_method,
                amount=pruning_amount,
                make_permanent=True
            ),
            verification=VerificationConfig(
                properties=self.property_configs,
                max_violations=0  # No violations allowed
            )
        )
        
    def _config_summary(self, config: Optional[CompressionConfig]) -> Dict[str, Any]:
            """Create a summary of the configuration for display"""
            if config is None:
                return {
                    'w_bits': 'N/A',
                    'pruning_method': 'N/A',
                    'pruning_amount': 'N/A',
                    'weight_layerwise': 'N/A'
                }
                
            return {
                'w_bits': config.quantization.w_bits,
                'pruning_method': config.pruning.method,
                'pruning_amount': config.pruning.amount,
                'weight_layerwise': config.quantization.weight_layerwise
            }