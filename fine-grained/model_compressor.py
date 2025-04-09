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


def calculate_size(model: nn.Module, detailed: bool = False) -> Dict[str, Any]:
    """
    Calculate the memory size of a model, accounting for quantization and pruning
    
    Args:
        model: PyTorch model
        detailed: Whether to return detailed breakdown by layer
        
    Returns:
        Dictionary with size information in MB
    """
    total_bytes = 0
    details = {}
    
    # Find all modules
    for name, module in model.named_modules():
        # Skip containers/modules without parameters
        if not any(True for _ in module.parameters(recurse=False)):
            continue
            
        # Get parameters in this module
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{name}.{param_name}" if name else param_name
            
            # Calculate parameter size in bytes
            if isinstance(module, QuantizeLinear) and param_name == "weight":
                # Quantized weights use fewer bits
                w_bits = module.w_bits
                param_bytes = param.numel() * (w_bits // 8 if w_bits >= 8 else 1)
            else:
                param_bytes = param.numel() * param.element_size()
            
            # Calculate sparsity (percentage of zeros)
            non_zero = (param != 0).sum().item()
            sparsity = 1.0 - (non_zero / param.numel())
            
            # Account for pruning
            if sparsity > 0:
                param_bytes = int(param_bytes * (1 - sparsity))
            
            # Add to total
            total_bytes += param_bytes
            
            # Store details if requested
            if detailed:
                details[full_name] = {
                    'size_bytes': param_bytes,
                    'size_mb': param_bytes / (1024 ** 2),
                    'shape': list(param.shape),
                    'numel': param.numel(),
                    'dtype': str(param.dtype),
                    'sparsity': sparsity,
                    'quantized': isinstance(module, QuantizeLinear) and param_name == "weight",
                    'bits': module.w_bits if isinstance(module, QuantizeLinear) and param_name == "weight" else (param.element_size() * 8)
                }
    
    # Convert to MB
    total_mb = total_bytes / (1024 ** 2)
    
    result = {
        'total_bytes': total_bytes,
        'total_mb': total_mb
    }
    
    if detailed:
        result['details'] = details
    
    if detailed:
        # Print summary by layer type
        quantized_mb = sum(d['size_mb'] for d in details.values() if d['quantized'])
        pruned_mb = sum(d['size_mb'] * d['sparsity'] for d in details.values())
        full_precision_mb = sum(d['size_mb'] for d in details.values() if not d['quantized'])
        
        result['summary'] = {
            'quantized_mb': quantized_mb,
            'pruned_mb_savings': pruned_mb,
            'full_precision_mb': full_precision_mb
        }
        
    return result


def calculate_size_reduction(original_model: nn.Module, compressed_model: nn.Module) -> Dict[str, Any]:
    """
    Calculate the size reduction achieved by compression
    
    Args:
        original_model: Original uncompressed model
        compressed_model: Compressed model
        
    Returns:
        Dictionary with size reduction information
    """
    try:
        original_size = calculate_size(original_model)
        compressed_size = calculate_size(compressed_model)
        
        reduction_bytes = original_size['total_bytes'] - compressed_size['total_bytes']
        reduction_ratio = reduction_bytes / original_size['total_bytes'] if original_size['total_bytes'] > 0 else 0.0
        
        return {
            'original_mb': original_size['total_mb'],
            'compressed_mb': compressed_size['total_mb'],
            'reduction_mb': reduction_bytes / (1024 ** 2),
            'reduction_ratio': reduction_ratio,
            'reduction_percent': reduction_ratio * 100
        }
    except Exception as e:
        print(f"Error in calculate_size_reduction: {str(e)}")
        # Return default values
        return {
            'original_mb': 0.0,
            'compressed_mb': 0.0,
            'reduction_mb': 0.0,
            'reduction_ratio': 0.0,
            'reduction_percent': 0.0
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
            try:
                size_reduction = calculate_size_reduction(
                    self.original_model,
                    self.compressed_model
                )
            except Exception as e:
                print(f"Error calculating size reduction: {str(e)}")
                # Provide a default size reduction if calculation fails
                size_reduction = {
                    "original_mb": 0.0,
                    "compressed_mb": 0.0,
                    "reduction_ratio": 0.0,
                    "reduction_percent": 0.0
                }
            
            # Run verification
            try:
                verification_results = self.verify(inputs)
            except Exception as e:
                print(f"Error during verification: {str(e)}")
                # Provide default verification results if verification fails
                verification_results = {
                    "summary": {
                        "total_properties": 0,
                        "satisfied_properties": 0,
                        "violated_properties": 0,
                        "verification_passed": False
                    }
                }
            
            # Measure sparsity
            try:
                sparsity = get_model_sparsity(self.compressed_model)
                avg_sparsity = sum(sparsity.values()) / len(sparsity) if sparsity else 0
            except Exception as e:
                print(f"Error calculating sparsity: {str(e)}")
                sparsity = {}
                avg_sparsity = 0.0
            
            # Generate sample outputs
            try:
                if isinstance(inputs, list) and isinstance(inputs[0], str):
                    sample_text = inputs[0]
                else:
                    sample_text = "The model generates this text as a sample for comparison."
                    
                # Generate from both models
                original_output = self._generate_text(self.original_model, sample_text)
                compressed_output = self._generate_text(self.compressed_model, sample_text)
            except Exception as e:
                print(f"Error generating sample text: {str(e)}")
                sample_text = "Sample generation failed"
                original_output = ""
                compressed_output = ""
            
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