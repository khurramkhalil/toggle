import torch
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional


@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    
    enabled: bool = True
    
    # Which layers to quantize (if None, quantize all Linear layers)
    target_layers: Optional[List[str]] = None
    
    # Bit width for weight quantization
    w_bits: int = 8
    
    # Whether to use layerwise quantization
    weight_layerwise: bool = False
    
    # Whether to use symmetric quantization
    symmetric: bool = True


@dataclass
class PruningConfig:
    """Configuration for model pruning"""
    
    enabled: bool = True
    
    # Pruning method: 'l1', 'l2', 'random', 'structured_0', 'structured_1'
    method: str = "l1"
    
    # Which parameters to prune
    target_params: List[str] = field(default_factory=lambda: ["weight"])
    
    # Pruning amount (0.0 to 1.0) or dict mapping module names to amounts
    amount: Union[float, Dict[str, float]] = 0.2
    
    # Whether to make pruning permanent
    make_permanent: bool = True
    
    # Target modules for pruning (if None, prune all applicable modules)
    target_modules: Optional[List[str]] = None


@dataclass
class VerificationConfig:
    """Configuration for formal verification"""
    
    # Which properties to verify
    properties: List[Dict[str, Union[str, float]]] = field(default_factory=lambda: [
        {"name": "cosine_similarity", "threshold": 0.85},
        {"name": "topk_overlap", "k": 5, "min_overlap": 0.6}
    ])
    
    # Minimum robustness threshold for verification success
    min_robustness: float = 0.0
    
    # Maximum number of violations allowed
    max_violations: int = 0


@dataclass
class CompressionConfig:
    """Main configuration for model compression pipeline"""
    
    # Quantization configuration
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    
    # Pruning configuration
    pruning: PruningConfig = field(default_factory=PruningConfig)
    
    # Verification configuration
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Device to use for computation
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model precision (fp32, fp16, bf16)
    precision: str = "fp16"
    
    # Whether to print detailed statistics during compression
    verbose: bool = True