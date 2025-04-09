import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Dict, Union, Callable, Optional


class PruningMethod:
    """Base class for different pruning methods"""
    
    def __init__(self, name: str):
        self.name = name
    
    def apply(self, module: nn.Module, name: str, amount: float):
        """Apply pruning to the specified module and parameter"""
        raise NotImplementedError


class L1Pruning(PruningMethod):
    """L1 unstructured pruning (element-wise)"""
    
    def __init__(self):
        super().__init__("l1_unstructured")
    
    def apply(self, module: nn.Module, name: str, amount: float):
        prune.l1_unstructured(module, name, amount=amount)

class L2Pruning(PruningMethod):
    """L2 unstructured pruning (element-wise)"""
    
    def __init__(self):
        super().__init__("l2_unstructured")
    
    def apply(self, module: nn.Module, name: str, amount: float):
        # Since torch doesn't have l2_unstructured directly, we implement it
        if name not in module._parameters:
            raise ValueError(f"Parameter '{name}' does not exist in module")
            
        param = module._parameters[name]
        if param is None:
            return
            
        # Calculate L2 norm (squared) of each element
        l2_norm = param.pow(2)
        
        # Create a mask based on the L2 norms
        threshold = torch.quantile(l2_norm.flatten(), amount)
        mask = l2_norm > threshold
        
        # Apply the mask using PyTorch's custom pruning
        prune.CustomFromMask.apply(module, name, mask)


class RandomPruning(PruningMethod):
    """Random unstructured pruning (element-wise)"""
    
    def __init__(self):
        super().__init__("random_unstructured")
    
    def apply(self, module: nn.Module, name: str, amount: float):
        prune.random_unstructured(module, name, amount=amount)


class StructuredPruning(PruningMethod):
    """Structured pruning (removes entire rows/columns)"""
    
    def __init__(self, dim: int):
        super().__init__(f"ln_structured_{dim}")
        self.dim = dim
    
    def apply(self, module: nn.Module, name: str, amount: float):
        prune.ln_structured(module, name, amount=amount, n=2, dim=self.dim)


# Registry of available pruning methods
PRUNING_METHODS = {
    "l1": L1Pruning(),
    "l2": L2Pruning(),
    "random": RandomPruning(),
    "structured_0": StructuredPruning(0),  # Prune output channels/neurons
    "structured_1": StructuredPruning(1),  # Prune input channels/features
}


def apply_pruning(
    model: nn.Module,
    target_modules: Optional[List[nn.Module]] = None,
    target_params: List[str] = ["weight"],
    method: str = "l1",
    amount: Union[float, Dict[str, float]] = 0.2,
    make_permanent: bool = True
) -> nn.Module:
    """
    Apply pruning to specified modules in a model
    
    Args:
        model: PyTorch model to prune
        target_modules: List of modules to apply pruning to (if None, prune all applicable modules)
        target_params: List of parameter names to prune (e.g., 'weight', 'bias')
        method: Pruning method name from PRUNING_METHODS
        amount: Pruning ratio (0.0 to 1.0) or dict mapping module paths to pruning ratios
        make_permanent: Whether to make pruning permanent (remove pruning reparameterization)
        
    Returns:
        Pruned model
    """
    if method not in PRUNING_METHODS:
        raise ValueError(f"Unknown pruning method: {method}. Available methods: {list(PRUNING_METHODS.keys())}")
    
    pruning_method = PRUNING_METHODS[method]
    
    # If no specific modules provided, find all applicable modules
    if target_modules is None:
        target_modules = []
        for module_name, module in model.named_modules():
            if hasattr(module, target_params[0]):
                target_modules.append((module_name, module))
    
    # Apply pruning
    for module_name, module in (target_modules if isinstance(target_modules[0], tuple) else [(None, m) for m in target_modules]):
        for param_name in target_params:
            if not hasattr(module, param_name):
                continue
                
            # Determine pruning amount for this specific module
            module_amount = amount[module_name] if isinstance(amount, dict) and module_name in amount else amount
            
            # Apply pruning
            pruning_method.apply(module, param_name, amount=module_amount)
            
            # Make pruning permanent if requested
            if make_permanent:
                prune.remove(module, param_name)
                
    return model


def get_model_sparsity(model: nn.Module) -> Dict[str, float]:
    """
    Calculate sparsity for each parameter in the model
    
    Returns:
        Dictionary mapping parameter names to sparsity ratios
    """
    sparsity = {}
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Only consider matrices, not vectors
            zeros = (param == 0).float().sum()
            total = param.numel()
            sparsity[name] = (zeros / total).item()
            
    return sparsity