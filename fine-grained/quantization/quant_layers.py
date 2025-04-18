import torch
import torch.nn as nn
from .quantizers import LsqBinaryTernaryExtension, StretchedElasticQuant


class QuantizeLinear(nn.Linear):
    """
    Quantized linear layer that replaces standard torch.nn.Linear
    
    Supports different bit widths and quantization methods based on bit width:
    - 16+ bits: No quantization (original weights)
    - 3-4 bits: LsqBinaryTernaryExtension
    - 0-2 bits: StretchedElasticQuant
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        w_bits=16,
        a_bits=32,  # For future activation quantization
        symmetric=True,
        weight_layerwise=False,
    ):
        super(QuantizeLinear, self).__init__(in_features, out_features, bias=bias)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.symmetric = symmetric
        self.weight_layerwise = weight_layerwise
        
        # Initialize quantization parameters
        if self.w_bits < 16:
            self.weight_clip_val = nn.Parameter(torch.Tensor(self.weight.shape[0], 1))
            # Initialize clip value based on weight statistics
            with torch.no_grad():
                self.weight_clip_val.data.fill_(self.weight.abs().max().item())

    def forward(self, input_):
            # Ensure weights are 2D
            assert len(self.weight.size()) == 2
            real_weights = self.weight

            # Apply appropriate quantization based on bit width
            if self.w_bits >= 16:
                weight = self.weight
            elif self.w_bits <= 4:
                weight = LsqBinaryTernaryExtension.apply(
                    real_weights,
                    self.weight_clip_val,
                    self.w_bits,
                    self.weight_layerwise,
                ).to(input_.dtype)
            else:  # 5 <= w_bits <= 15
                # Use StretchedElasticQuant for bit widths 5-15
                weight = StretchedElasticQuant.apply(
                    real_weights,
                    self.weight_clip_val,
                    self.w_bits,
                    self.weight_layerwise,
                ).to(input_.dtype)

            # Perform the linear operation
            out = nn.functional.linear(input_, weight)
            if self.bias is not None:
                out += self.bias.view(1, -1).expand_as(out)

            return out


# def replace_linear_with_quantized(model, target_layers=None, w_bits=16, weight_layerwise=False):
#     """
#     Replace standard Linear layers with QuantizeLinear layers
    
#     Args:
#         model: PyTorch model
#         target_layers: List of layer names to quantize (if None, quantize all)
#         w_bits: Bit width for quantization
#         weight_layerwise: Whether to use layerwise quantization
#     """
#     for name, module in model.named_children():
#         if target_layers is not None and name not in target_layers:
#             continue
            
#         if isinstance(module, nn.Linear):
#             # Create quantized replacement
#             quant_layer = QuantizeLinear(
#                 module.in_features,
#                 module.out_features,
#                 bias=module.bias is not None,
#                 w_bits=w_bits,
#                 weight_layerwise=weight_layerwise
#             )
#             # Copy weights and biases
#             quant_layer.weight.data.copy_(module.weight.data)
#             if module.bias is not None:
#                 quant_layer.bias.data.copy_(module.bias.data)
                
#             # Replace the module
#             setattr(model, name, quant_layer)
#         else:
#             # Recursively quantize submodules
#             replace_linear_with_quantized(
#                 module, 
#                 target_layers, 
#                 w_bits, 
#                 weight_layerwise
#             )
    
#     return model

# def replace_linear_with_quantized(model, target_layers=None, w_bits=16, weight_layerwise=False):
#     """
#     Replace standard Linear layers with QuantizeLinear layers
    
#     Args:
#         model: PyTorch model
#         target_layers: List of layer names to quantize (if None, quantize all)
#         w_bits: Bit width for quantization
#         weight_layerwise: Whether to use layerwise quantization
#     """
#     replacements = {}
    
#     # First pass: identify all Linear layers
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             if target_layers is None or any(name.endswith(target) or target in name for target in target_layers):
#                 replacements[name] = (module, module._parameters.keys())
    
#     # Second pass: replace identified layers
#     for name, (module, param_names) in replacements.items():
#         # Create quantized replacement
#         quant_layer = QuantizeLinear(
#             module.in_features,
#             module.out_features,
#             bias=module.bias is not None,
#             w_bits=w_bits,
#             weight_layerwise=weight_layerwise
#         )
        
#         # Copy weights and biases
#         quant_layer.weight.data.copy_(module.weight.data)
#         if module.bias is not None:
#             quant_layer.bias.data.copy_(module.bias.data)
        
#         # Replace the module by finding its parent
#         parent_name = '.'.join(name.split('.')[:-1])
#         child_name = name.split('.')[-1]
        
#         if parent_name:
#             parent = model
#             for part in parent_name.split('.'):
#                 parent = getattr(parent, part)
#             setattr(parent, child_name, quant_layer)
#         else:
#             setattr(model, child_name, quant_layer)
    
#     print(f"Replaced {len(replacements)} Linear layers with QuantizeLinear layers")
#     return model

def replace_linear_with_quantized(model, target_layers=None, w_bits=16, weight_layerwise=False):
    """
    Replace standard Linear layers with QuantizeLinear layers
    
    Args:
        model: PyTorch model
        target_layers: List of layer names to quantize (if None, quantize all)
        w_bits: Bit width for quantization
        weight_layerwise: Whether to use layerwise quantization
    """
    replacements = {}
    
    # First pass: identify all Linear layers using named_modules for complete traversal
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_layers is None or any(target in name for target in target_layers):
                replacements[name] = module
    
    print(f"Found {len(replacements)} Linear layers in the model")
    
    # Second pass: replace identified layers
    replaced_count = 0
    for name, linear_module in replacements.items():
        # Create quantized replacement
        quant_layer = QuantizeLinear(
            linear_module.in_features,
            linear_module.out_features,
            bias=linear_module.bias is not None,
            w_bits=w_bits,
            weight_layerwise=weight_layerwise
        )
        
        # Copy weights and biases
        quant_layer.weight.data.copy_(linear_module.weight.data)
        if linear_module.bias is not None:
            quant_layer.bias.data.copy_(linear_module.bias.data)
        
        # Replace the module by finding its parent
        try:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, quant_layer)
                replaced_count += 1
            else:
                setattr(model, child_name, quant_layer)
                replaced_count += 1
        except AttributeError as e:
            print(f"Error replacing layer {name}: {str(e)}")
    
    print(f"Replaced {replaced_count} Linear layers with QuantizeLinear layers")
    return model