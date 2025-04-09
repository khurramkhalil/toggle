import torch
import random
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Union, Any

from transformers import PreTrainedModel, PreTrainedTokenizer

from config import CompressionConfig
from quantization.quant_layers import replace_linear_with_quantized
from pruning.pruners import apply_pruning, get_model_sparsity
from formal_verification.stl_specs import get_property
from formal_verification.property_verifier import PropertyVerifier
# from utils.size_calculator import calculate_size, calculate_size_reduction

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from quantization.quant_layers import QuantizeLinear


def calculate_size(model):
    total_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, QuantizeLinear):
            # For quantized linear layers
            if module.w_bits < 16:
                # Quantized weights use fewer bits
                total_bytes += module.weight.numel() * module.w_bits // 8
            else:
                # Regular weights
                total_bytes += module.weight.numel() * module.weight.element_size()
                
            # Add bias if present
            if module.bias is not None:
                total_bytes += module.bias.numel() * module.bias.element_size()
        elif isinstance(module, torch.nn.Linear):
            # Regular linear layers
            total_bytes += module.weight.numel() * module.weight.element_size()
            if module.bias is not None:
                total_bytes += module.bias.numel() * module.bias.element_size()
                
    # Convert to MB
    return total_bytes / (1024 ** 2)


def calculate_size_reduction(original_model: nn.Module, compressed_model: nn.Module) -> Dict[str, Any]:
    """
    Calculate the size reduction achieved by compression
    
    Args:
        original_model: Original uncompressed model
        compressed_model: Compressed model
        
    Returns:
        Dictionary with size reduction information
    """
    original_size = calculate_size(original_model)
    compressed_size = calculate_size(compressed_model)
    
    reduction_bytes = original_size - compressed_size
    reduction_ratio = reduction_bytes / original_size
    
    return {
        'original_mb': original_size,
        'compressed_mb': compressed_size,
        'reduction_mb': reduction_bytes / (1024 ** 2),
        'reduction_ratio': reduction_ratio,
        'reduction_percent': reduction_ratio * 100
    }

class ModelCompressor:
    """
    Model compression framework with formal verification
    
    Combines quantization and pruning techniques with formal verification
    to ensure behavioral properties are maintained during compression.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: CompressionConfig = None
    ):
        """
        Initialize the model compressor
        
        Args:
            model: Model to compress
            tokenizer: Tokenizer for the model
            config: Compression configuration
        """
        self.original_model = model
        self.compressed_model = None
        self.tokenizer = tokenizer
        self.config = config or CompressionConfig()
        
        # Set random seeds for reproducibility
        self._set_seed(self.config.seed)
        
    def compress(self) -> PreTrainedModel:
        """
        Apply compression techniques based on configuration
        
        Returns:
            Compressed model
        """
        # Create a deep copy of the model to avoid modifying the original
        self.compressed_model = self._clone_model(self.original_model)
        
        # Apply quantization if enabled
        if self.config.quantization.enabled:
            self._apply_quantization()
            
        # Apply pruning if enabled
        if self.config.pruning.enabled:
            self._apply_pruning()
            
        return self.compressed_model
        
    def verify(self, inputs: Union[List[str], Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Verify that the compressed model maintains specified properties
        
        Args:
            inputs: Text inputs or tokenized inputs for verification
            
        Returns:
            Dictionary with verification results
        """
        if self.compressed_model is None:
            raise ValueError("Model must be compressed before verification")
            
        # Initialize properties from config
        properties = []
        for prop_config in self.config.verification.properties:
            # Create a copy of the property config to avoid modifying the original
            prop_config_copy = prop_config.copy()
            name = prop_config_copy.pop("name")
            properties.append(get_property(name, **prop_config_copy))
            
        # Create property verifier
        verifier = PropertyVerifier(properties)
        
        # Run verification
        results = verifier.verify_all(
            self.original_model,
            self.compressed_model,
            self.tokenizer,
            inputs,
            device=self.config.device
        )
        
        # Add summary statistics
        satisfied_count = sum(1 for prop, res in results.items() if res["satisfied"])
        results["summary"] = {
            "total_properties": len(properties),
            "satisfied_properties": satisfied_count,
            "violated_properties": len(properties) - satisfied_count,
            "verification_passed": (len(properties) - satisfied_count) <= self.config.verification.max_violations
        }
        
        return results
        
    def evaluate(self, inputs: Union[List[str], Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Evaluate compression results including size reduction and verification
        
        Args:
            inputs: Text inputs or tokenized inputs for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        if self.compressed_model is None:
            raise ValueError("Model must be compressed before evaluation")
            
        # Calculate size reduction
        size_reduction = calculate_size_reduction(
            self.original_model,
            self.compressed_model
        )
        
        # Run verification
        verification_results = self.verify(inputs)
        
        # Measure sparsity
        sparsity = get_model_sparsity(self.compressed_model)
        avg_sparsity = sum(sparsity.values()) / len(sparsity) if sparsity else 0
        
        # Generate sample outputs
        if isinstance(inputs, list) and isinstance(inputs[0], str):
            sample_text = inputs[0]
        else:
            sample_text = "The model generates this text as a sample for comparison."
            
        # Generate from both models
        original_output = self._generate_text(self.original_model, sample_text)
        compressed_output = self._generate_text(self.compressed_model, sample_text)
        
        return {
            "size_reduction": size_reduction,
            "verification": verification_results,
            "sparsity": {
                "average": avg_sparsity,
                "by_layer": sparsity
            },
            "sample_generation": {
                "input": sample_text,
                "original_output": original_output,
                "compressed_output": compressed_output
            }
        }
        
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def _clone_model(self, model: PreTrainedModel) -> PreTrainedModel:
        """Create a deep copy of the model"""
        return copy.deepcopy(model)
        
    def _apply_quantization(self):
        """Apply quantization based on configuration"""
        if self.config.verbose:
            print(f"Quantizing model with {self.config.quantization.w_bits} bits...")
            
        # Apply quantization
        replace_linear_with_quantized(
            self.compressed_model,
            target_layers=self.config.quantization.target_layers,
            w_bits=self.config.quantization.w_bits,
            weight_layerwise=self.config.quantization.weight_layerwise
        )
        
        if self.config.verbose:
            print("Quantization complete")
            
    def _apply_pruning(self):
        """Apply pruning based on configuration"""
        if self.config.verbose:
            print(f"Pruning model using {self.config.pruning.method} method...")
            
        # Apply pruning
        apply_pruning(
            self.compressed_model,
            target_modules=self.config.pruning.target_modules,
            target_params=self.config.pruning.target_params,
            method=self.config.pruning.method,
            amount=self.config.pruning.amount,
            make_permanent=self.config.pruning.make_permanent
        )
        
        if self.config.verbose:
            print("Pruning complete")
            
    def _generate_text(self, model: PreTrainedModel, text: str, max_new_tokens: int = 50) -> str:
        """Generate text using the model"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.config.device)
        model = model.to(self.config.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False  # Deterministic generation
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)