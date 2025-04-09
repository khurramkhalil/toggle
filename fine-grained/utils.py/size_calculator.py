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
    
    for name, param in model.named_parameters():
        # Skip parameters that don't require gradients (e.g., frozen layers)
        if not param.requires_grad:
            continue
            
        # Get the parent module
        module_name = '.'.join(name.split('.')[:-1])
        parent = model
        for component in module_name.split('.'):
            if component:
                parent = getattr(parent, component)
                
        # Calculate bytes based on parameter type and module
        param_bytes = 0
        
        # Check if the parameter is quantized
        if isinstance(parent, QuantizeLinear) and name.endswith('weight'):
            w_bits = parent.w_bits
            if w_bits < 16:
                # Quantized weights use fewer bits
                param_bytes = param.numel() * w_bits // 8
            else:
                # Regular float16 or float32
                param_bytes = param.numel() * param.element_size()
        else:
            # Regular parameter
            param_bytes = param.numel() * param.element_size()
            
        # Calculate non-zero elements for sparsity
        non_zero = (param != 0).sum().item()
        sparsity = 1.0 - (non_zero / param.numel())
        
        # Add to total
        total_bytes += param_bytes
        
        # Store details if requested
        if detailed:
            details[name] = {
                'size_bytes': param_bytes,
                'size_mb': param_bytes / (1024 ** 2),
                'shape': list(param.shape),
                'numel': param.numel(),
                'dtype': str(param.dtype),
                'sparsity': sparsity,
                'quantized': isinstance(parent, QuantizeLinear) if name.endswith('weight') else False,
                'bits': parent.w_bits if isinstance(parent, QuantizeLinear) and name.endswith('weight') else None
            }
    
    # Convert to MB
    total_mb = total_bytes / (1024 ** 2)
    
    result = {
        'total_bytes': total_bytes,
        'total_mb': total_mb
    }
    
    if detailed:
        result['details'] = details
        
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
    original_size = calculate_size(original_model)
    compressed_size = calculate_size(compressed_model)
    
    reduction_bytes = original_size['total_bytes'] - compressed_size['total_bytes']
    reduction_ratio = reduction_bytes / original_size['total_bytes']
    
    return {
        'original_mb': original_size['total_mb'],
        'compressed_mb': compressed_size['total_mb'],
        'reduction_mb': reduction_bytes / (1024 ** 2),
        'reduction_ratio': reduction_ratio,
        'reduction_percent': reduction_ratio * 100
    }