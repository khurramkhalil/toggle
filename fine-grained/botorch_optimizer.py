import torch
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
    from botorch.acquisition.analytic import ConstrainedExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.transforms import normalize, standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.utils.sampling import draw_sobol_samples
    import gpytorch
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    print("BoTorch not available. Please install botorch with: pip install botorch gpytorch")

from model_compressor import ModelCompressor
from config import CompressionConfig, QuantizationConfig, PruningConfig, VerificationConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BoTorchOptimizer")


class BoTorchOptimizer:
    """
    Uses BoTorch for Bayesian Optimization to find optimal compression parameters
    that maximize compression while satisfying formal properties.
    """
    
    def __init__(
        self, 
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sample_inputs: List[str],
        property_configs: List[Dict[str, Any]],
        n_iterations: int = 20,
        initial_samples: int = 5,
        random_state: int = 42,
        verbose: bool = True,
        log_file: Optional[str] = "botorch_optimization.log"
    ):
        """
        Initialize the BoTorch optimizer
        
        Args:
            model: Model to optimize compression for
            tokenizer: Tokenizer for the model
            sample_inputs: List of input texts for evaluation
            property_configs: List of property configurations
            n_iterations: Number of optimization iterations
            initial_samples: Number of initial random samples
            random_state: Random seed
            verbose: Whether to print progress
            log_file: Path to log file (None for no file logging)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sample_inputs = sample_inputs
        self.property_configs = property_configs
        self.n_iterations = n_iterations
        self.initial_samples = initial_samples
        self.random_state = random_state
        self.verbose = verbose
        
        # Set up file logging if requested
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
        
        # Check if BoTorch is available
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required but not available. Please install botorch.")
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Define parameter space
        self._setup_parameter_space()
        
        # Store results
        self.train_X = []  # Parameter configurations
        self.train_Y = []  # Objective values
        self.train_constraints = []  # Constraint values
        
        self.best_config = None
        self.best_model = None
        self.best_score = float('-inf')
        self.best_idx = -1
        
        # Logging for hypothesis validation
        self.discrete_param_success_rate = {
            'bit_width': {},  # Map from bit_width to success rate
            'pruning_method': {}  # Map from method to success rate
        }
        self.constraint_satisfaction_history = []  # Track constraint satisfaction over iterations
        
        # Log initialization
        logger.info(f"Initialized BoTorch optimizer with {n_iterations} iterations and {initial_samples} initial samples")
        logger.info(f"Parameter bounds: {self.bounds}")
    
    def optimize(self) -> Tuple[Dict[str, Any], PreTrainedModel, float]:
        """
        Run Bayesian optimization to find optimal compression parameters
        
        Returns:
            Tuple of (best_config, best_model, best_score)
        """
        # Initialize with default config as fallback
        self.best_config = self._create_config_from_params([8, 0, 0.2, 0])  # 8-bit, l1, 0.2, not layerwise
        
        start_time = time.time()
        logger.info("Starting BoTorch optimization")
        
        # Generate initial samples using Sobol sequence
        logger.info(f"Generating {self.initial_samples} initial points using Sobol sequence")
        
        # Create bounds tensor with proper shape for botorch
        bounds_tensor = torch.tensor(self.bounds, dtype=torch.float64).T
        
        # Log the shape of bounds for debugging
        logger.info(f"Bounds tensor shape: {bounds_tensor.shape}")
        
        # Make sure bounds are properly shaped
        if bounds_tensor.shape[0] != 2 or bounds_tensor.shape[1] != 4:
            logger.warning(f"Unexpected bounds shape: {bounds_tensor.shape}, reshaping to (2, 4)")
            # Create proper bounds manually if needed
            bounds_tensor = torch.tensor([
                [2.0, 0.0, 0.0, 0.0],  # Lower bounds
                [16.0, 2.0, 1.0, 1.0]  # Upper bounds
            ], dtype=torch.float64)
        
        try:
            sobol = draw_sobol_samples(
                bounds=bounds_tensor,
                n=self.initial_samples,
                q=1,
                seed=self.random_state
            ).squeeze(1)
            
            logger.info(f"Generated Sobol points shape: {sobol.shape}")
            
            # Validate sobol points shape
            if len(sobol.shape) != 2 or sobol.shape[1] != 4:
                logger.warning(f"Unexpected sobol shape: {sobol.shape}, generating points manually")
                # Generate random points manually
                sobol = torch.zeros((self.initial_samples, 4), dtype=torch.float64)
                for i in range(self.initial_samples):
                    sobol[i, 0] = torch.rand(1).item() * (16 - 2) + 2  # bit_width
                    sobol[i, 1] = torch.rand(1).item() * 2  # pruning_method
                    sobol[i, 2] = torch.rand(1).item()  # pruning_amount
                    sobol[i, 3] = torch.rand(1).item()  # layerwise
        except Exception as e:
            logger.error(f"Error generating Sobol points: {str(e)}")
            # Generate random points as fallback
            sobol = torch.zeros((self.initial_samples, 4), dtype=torch.float64)
            for i in range(self.initial_samples):
                sobol[i, 0] = torch.rand(1).item() * (16 - 2) + 2  # bit_width
                sobol[i, 1] = torch.rand(1).item() * 2  # pruning_method
                sobol[i, 2] = torch.rand(1).item()  # pruning_amount
                sobol[i, 3] = torch.rand(1).item()  # layerwise
        
        # Convert Sobol points to parameter values
        initial_points = self._process_botorch_points(sobol)
        
        # Evaluate initial points
        logger.info("Evaluating initial points")
        for i, point in enumerate(initial_points):
            logger.info(f"Initial point {i+1}/{len(initial_points)}: {self._format_params(point)}")
            value, constraint = self._evaluate_point(point)
            self.train_X.append(point)
            self.train_Y.append(value)
            self.train_constraints.append(constraint)
            
            # Update best if better
            if value > self.best_score and constraint >= 0:
                self.best_score = value
                self.best_config = self._create_config_from_params(point)
                self.best_idx = len(self.train_Y) - 1
                logger.info(f"New best point from initialization: score={value:.4f}, constraint={constraint:.4f}")
        
        # Convert to tensors
        X = torch.tensor(self.train_X, dtype=torch.float64)
        Y = torch.tensor(self.train_Y, dtype=torch.float64).unsqueeze(-1)
        C = torch.tensor(self.train_constraints, dtype=torch.float64).unsqueeze(-1)
        
        # Normalize inputs
        bounds = torch.tensor(self.bounds, dtype=torch.float64)
        
        # Run optimization iterations
        for iteration in range(self.n_iterations):
            logger.info(f"Starting iteration {iteration+1}/{self.n_iterations}")
            
            # Fit GP model to objective
            gp = SingleTaskGP(X, Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            
            # Fit GP model to constraint
            constraint_gp = SingleTaskGP(X, C)
            constraint_mll = ExactMarginalLogLikelihood(constraint_gp.likelihood, constraint_gp)
            fit_gpytorch_mll(constraint_mll)
            
            # LOG: Check if discrete params affect model quality
            self._log_discrete_parameter_effects(X, Y, C)
            
            # Create acquisition function (Constrained Expected Improvement)
            acq_func = ConstrainedExpectedImprovement(
                model=gp,
                best_f=Y[self.best_idx].item() if self.best_idx >= 0 else 0.0,
                objective_index=0,
                constraints=[lambda Z: constraint_gp.posterior(Z).mean],
                constraint_thresholds=[0.0]  # Constraint should be >= 0
            )
            
            # Optimize acquisition function
            new_x, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
                options={"batch_limit": 5, "maxiter": 200},
            )
            
            # Process the point
            new_point = self._process_botorch_points(new_x)[0]
            
            # Evaluate the new point
            logger.info(f"Evaluating new point: {self._format_params(new_point)}")
            value, constraint = self._evaluate_point(new_point)
            
            # Update training data
            self.train_X.append(new_point)
            self.train_Y.append(value)
            self.train_constraints.append(constraint)
            
            # Update tensors
            X = torch.tensor(self.train_X, dtype=torch.float64)
            Y = torch.tensor(self.train_Y, dtype=torch.float64).unsqueeze(-1)
            C = torch.tensor(self.train_constraints, dtype=torch.float64).unsqueeze(-1)
            
            # LOG: Track constraint satisfaction rate
            satisfaction_rate = (C >= 0).float().mean().item()
            self.constraint_satisfaction_history.append(satisfaction_rate)
            logger.info(f"Constraint satisfaction rate: {satisfaction_rate:.2f}")
            
            # Update best if better
            if value > self.best_score and constraint >= 0:
                self.best_score = value
                self.best_config = self._create_config_from_params(new_point)
                self.best_idx = len(self.train_Y) - 1
                logger.info(f"New best point: score={value:.4f}, constraint={constraint:.4f}")
                
                # Try to get the best model
                try:
                    compressor = ModelCompressor(
                        self.model, 
                        self.tokenizer, 
                        config=self.best_config
                    )
                    self.best_model = compressor.compress()
                except Exception as e:
                    logger.error(f"Error saving best model: {str(e)}")
            
            # Log iteration results
            elapsed = time.time() - start_time
            logger.info(f"Iteration {iteration+1} complete in {elapsed:.2f}s. " 
                       f"Best score so far: {self.best_score:.4f}")
        
        # If we haven't saved the best model yet, do it now
        if self.best_model is None and self.best_config is not None:
            try:
                compressor = ModelCompressor(
                    self.model, 
                    self.tokenizer, 
                    config=self.best_config
                )
                self.best_model = compressor.compress()
            except Exception as e:
                logger.error(f"Error creating best model at end: {str(e)}")
        
        # Log validation of our hypotheses
        self._log_hypothesis_validation()
            
        return self.best_config, self.best_model, self.best_score
    
    def _setup_parameter_space(self):
        """Define the parameter space for optimization"""
        # Define bounds for continuous parameters
        # For discrete parameters, we use continuous relaxation and then round/map to discrete values
        self.bounds = [
            [2, 16],  # w_bits (Integer: 2-16)
            [0, 2],   # pruning_method (Categorical: l1=0, l2=1, random=2)
            [0, 1],   # pruning_amount (Real: 0.0-1.0)
            [0, 1]    # weight_layerwise (Boolean: False=0, True=1)
        ]
        
        # Define mapping for categorical/discrete parameters
        self.pruning_methods = ['l1', 'l2', 'random']
        
        logger.info("Parameter space defined")
        logger.info(f"Bit width range: [2, 16]")
        logger.info(f"Pruning methods: {self.pruning_methods}")
        logger.info(f"Pruning amount range: [0, 1]")
        logger.info(f"Weight layerwise: [False, True]")
    
    def _process_botorch_points(self, points: torch.Tensor) -> List[List[Any]]:
        """
        Convert BoTorch continuous points to actual parameter values
        
        Args:
            points: Tensor of points from BoTorch
            
        Returns:
            List of processed parameter lists
        """
        processed_points = []
        
        # Convert to numpy for easier processing
        points_np = points.cpu().numpy()
        
        for point in points_np:
            # Process each parameter according to its type
            bit_width = int(round(point[0]))  # Round to nearest integer
            bit_width = max(2, min(16, bit_width))  # Clamp to [2, 16]
            
            pruning_idx = int(round(point[1]))  # Round to nearest integer
            pruning_idx = max(0, min(len(self.pruning_methods)-1, pruning_idx))  # Clamp to valid indices
            
            pruning_amount = point[2]  # Keep as continuous value
            pruning_amount = max(0, min(1, pruning_amount))  # Clamp to [0, 1]
            
            # Round to nearest 0.1 for interpretability
            pruning_amount = round(pruning_amount * 10) / 10
            
            layerwise = round(point[3]) >= 0.5  # Convert to boolean
            layerwise_int = 1 if layerwise else 0  # Convert to int for storage
            
            processed_points.append([bit_width, pruning_idx, pruning_amount, layerwise_int])
        
        return processed_points
    
    def _evaluate_point(self, params: List[Any]) -> Tuple[float, float]:
        """
        Evaluate a parameter configuration
        
        Args:
            params: List of parameter values [bit_width, pruning_idx, pruning_amount, layerwise]
            
        Returns:
            Tuple of (objective_value, constraint_value)
            - objective_value: Higher is better (compression ratio)
            - constraint_value: Must be >= 0 to be feasible (min_robustness)
        """
        logger.info(f"Evaluating: bit_width={params[0]}, "
                   f"pruning_method={self.pruning_methods[int(params[1])]}, "
                   f"pruning_amount={params[2]}, "
                   f"layerwise={bool(params[3])}")
        
        # Create configuration
        config = self._create_config_from_params(params)
        
        try:
            # Create model compressor
            compressor = ModelCompressor(
                self.model, 
                self.tokenizer, 
                config=config
            )
            
            # Compress model
            compressor.compress()
            
            # Verify formal properties
            verification_results = compressor.verify(self.sample_inputs)
            
            # Evaluate the compression
            evaluation = compressor.evaluate(self.sample_inputs)
            size_reduction = evaluation["size_reduction"]
            
            # Extract reduction ratio and robustness
            reduction_ratio = size_reduction["reduction_ratio"]
            min_robustness = verification_results["summary"].get("min_robustness", -1.0)
            
            logger.info(f"Evaluation results: reduction_ratio={reduction_ratio:.4f}, min_robustness={min_robustness:.4f}")
            
            # Update parameter success tracking
            bit_width = params[0]
            pruning_method = self.pruning_methods[int(params[1])]
            
            # Track success of discrete parameters
            if bit_width not in self.discrete_param_success_rate['bit_width']:
                self.discrete_param_success_rate['bit_width'][bit_width] = []
            self.discrete_param_success_rate['bit_width'][bit_width].append(min_robustness >= 0)
            
            if pruning_method not in self.discrete_param_success_rate['pruning_method']:
                self.discrete_param_success_rate['pruning_method'][pruning_method] = []
            self.discrete_param_success_rate['pruning_method'][pruning_method].append(min_robustness >= 0)
            
            return reduction_ratio, min_robustness
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return 0.0, -1.0  # Unfeasible point with zero value
    
    def _create_config_from_params(self, params: List[Any]) -> CompressionConfig:
        """
        Create CompressionConfig from optimization parameters
        
        Args:
            params: Parameter values [bit_width, pruning_idx, pruning_amount, layerwise]
            
        Returns:
            CompressionConfig instance
        """
        bit_width, pruning_idx, pruning_amount, layerwise_int = params
        
        # Convert to proper types
        bit_width = int(bit_width)
        pruning_method = self.pruning_methods[int(pruning_idx)]
        pruning_amount = float(pruning_amount)
        weight_layerwise = bool(layerwise_int)
        
        return CompressionConfig(
            quantization=QuantizationConfig(
                enabled=True,
                target_layers=None,  # Quantize all layers
                w_bits=bit_width,
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
    
    def _format_params(self, params: List[Any]) -> str:
        """Format parameters for display"""
        bit_width, pruning_idx, pruning_amount, layerwise_int = params
        return (f"bit_width={int(bit_width)}, "
                f"pruning_method={self.pruning_methods[int(pruning_idx)]}, "
                f"pruning_amount={float(pruning_amount):.1f}, "
                f"layerwise={bool(layerwise_int)}")
    
    def _log_discrete_parameter_effects(self, X: torch.Tensor, Y: torch.Tensor, C: torch.Tensor):
        """Log effects of discrete parameters on outcomes for hypothesis validation"""
        # Extract bit widths and pruning methods
        bit_widths = X[:, 0].round().int().cpu().numpy()
        pruning_methods = X[:, 1].round().int().cpu().numpy()
        
        # Check constraint satisfaction for each bit width
        for bit_width in np.unique(bit_widths):
            mask = bit_widths == bit_width
            if mask.sum() > 0:
                satisfaction_rate = (C[mask] >= 0).float().mean().item()
                avg_obj = Y[mask].mean().item()
                logger.info(f"Bit width {bit_width}: "
                           f"satisfaction_rate={satisfaction_rate:.2f}, "
                           f"avg_objective={avg_obj:.4f}, "
                           f"count={mask.sum()}")
        
        # Check constraint satisfaction for each pruning method
        for method_idx in np.unique(pruning_methods):
            mask = pruning_methods == method_idx
            if mask.sum() > 0:
                method_name = self.pruning_methods[int(method_idx)]
                satisfaction_rate = (C[mask] >= 0).float().mean().item()
                avg_obj = Y[mask].mean().item()
                logger.info(f"Pruning method {method_name}: "
                           f"satisfaction_rate={satisfaction_rate:.2f}, "
                           f"avg_objective={avg_obj:.4f}, "
                           f"count={mask.sum()}")
    
    def _log_hypothesis_validation(self):
        """Log validation of our hypotheses"""
        logger.info("\n==== HYPOTHESIS VALIDATION ====")
        
        # 1. Analyze discrete parameter success rates
        logger.info("1. Discrete Parameter Effects:")
        
        # Bit width success rates
        logger.info("Bit Width Constraint Satisfaction Rates:")
        for bit_width, successes in sorted(self.discrete_param_success_rate['bit_width'].items()):
            if len(successes) > 0:
                rate = sum(successes) / len(successes)
                logger.info(f"  Bit width {bit_width}: {rate:.2f} ({sum(successes)}/{len(successes)})")
        
        # Pruning method success rates
        logger.info("Pruning Method Constraint Satisfaction Rates:")
        for method, successes in sorted(self.discrete_param_success_rate['pruning_method'].items()):
            if len(successes) > 0:
                rate = sum(successes) / len(successes)
                logger.info(f"  Method {method}: {rate:.2f} ({sum(successes)}/{len(successes)})")
        
        # 2. Analyze constraint handling effectiveness
        if len(self.constraint_satisfaction_history) > 0:
            logger.info("\n2. Constraint Handling Effectiveness:")
            
            # Check if constraint satisfaction improved over time
            initial_rate = self.constraint_satisfaction_history[:self.initial_samples]
            later_rate = self.constraint_satisfaction_history[self.initial_samples:]
            
            if len(initial_rate) > 0 and len(later_rate) > 0:
                initial_mean = sum(initial_rate) / len(initial_rate)
                later_mean = sum(later_rate) / len(later_rate) if len(later_rate) > 0 else 0
                
                logger.info(f"  Initial constraint satisfaction rate: {initial_mean:.2f}")
                logger.info(f"  Later constraint satisfaction rate: {later_mean:.2f}")
                logger.info(f"  Improvement: {later_mean - initial_mean:.2f}")
                
                if later_mean > initial_mean:
                    logger.info("  ✓ Hypothesis CONFIRMED: Constraint handling improved exploration of feasible region")
                else:
                    logger.info("  ✗ Hypothesis REJECTED: Constraint handling did not improve over random sampling")
        
        # Overall conclusion
        logger.info("\nFinal Analysis:")
        if self.best_score > 0 and self.best_config is not None:
            bit_width = self.best_config.quantization.w_bits
            pruning_method = self.best_config.pruning.method
            pruning_amount = self.best_config.pruning.amount
            
            logger.info(f"Best configuration: bit_width={bit_width}, pruning_method={pruning_method}, amount={pruning_amount}")
            logger.info(f"Best score: {self.best_score:.4f}")
            
            # Identify which hypothesis was more important
            bit_success_rates = [sum(s)/len(s) if len(s) > 0 else 0 
                                for _, s in self.discrete_param_success_rate['bit_width'].items()]
            method_success_rates = [sum(s)/len(s) if len(s) > 0 else 0 
                                for _, s in self.discrete_param_success_rate['pruning_method'].items()]
            
            bit_variance = np.var(bit_success_rates) if len(bit_success_rates) > 0 else 0
            method_variance = np.var(method_success_rates) if len(method_success_rates) > 0 else 0
            
            logger.info(f"Bit width success rate variance: {bit_variance:.4f}")
            logger.info(f"Pruning method success rate variance: {method_variance:.4f}")
            
            if bit_variance > method_variance:
                logger.info("✓ Hypothesis CONFIRMED: Handling discrete bit-width parameter was more important")
            else:
                logger.info("✓ Hypothesis CONFIRMED: Pruning method choice was more important than bit-width")