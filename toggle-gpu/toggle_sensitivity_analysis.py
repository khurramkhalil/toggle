"""
Layer Sensitivity Analysis for TOGGLE framework
This module analyzes layer-wise sensitivity to compression for more targeted optimization.
"""

import torch
import numpy as np
from tqdm import tqdm
import copy
import time

class LayerSensitivityAnalyzer:
    """
    Analyzes the sensitivity of each layer to different compression techniques.
    This helps identify which layers can be compressed more aggressively.
    """
    
    def __init__(self, toggle_framework, bit_width_to_test=None, pruning_to_test=None):
        """
        Initialize the sensitivity analyzer.
        
        Args:
            toggle_framework: DynamicPrecisionTransformer instance
            bit_width_to_test: List of bit-widths to test (optional)
            pruning_to_test: List of pruning ratios to test (optional)
        """
        self.toggle = toggle_framework
        self.num_layers = toggle_framework.num_layers
        self.components = toggle_framework.components
        
        # Define bit-widths and pruning ratios to test
        self.bit_width_to_test = bit_width_to_test or [4, 8]
        self.pruning_to_test = pruning_to_test or [0.0, 0.3]
        
        # Store original parameters for restoration
        self.original_params = {}
        
    def store_original_parameters(self):
        """Store original model parameters for later restoration"""
        print("Storing original model parameters...")
        self.original_params = {}
        with torch.no_grad():
            for name, param in self.toggle.base_model.named_parameters():
                if 'weight' in name:
                    self.original_params[name] = param.data.clone()
    
    def restore_original_parameters(self):
        """Restore original model parameters"""
        print("Restoring original model parameters...")
        with torch.no_grad():
            for name, param in self.toggle.base_model.named_parameters():
                if name in self.original_params:
                    param.data.copy_(self.original_params[name])
    
    def _modify_single_layer(self, layer_idx, component, bits, pruning):
        """
        Apply compression to a single layer component
        
        Args:
            layer_idx: Layer index
            component: Component name
            bits: Bit-width to use
            pruning: Pruning ratio to use
        """
        # Find parameter prefix for this layer/component
        if self.toggle.model_type == "gpt2":
            param_prefix = f"transformer.h.{layer_idx}.{component}"
        else:  # LLaMA style
            param_prefix = f"model.layers.{layer_idx}.{component}"
        
        # Apply quantization and pruning to matching parameters
        for name, param in self.toggle.base_model.named_parameters():
            if name.startswith(param_prefix) and 'weight' in name:
                # Make a copy to avoid modifying the original
                modified_data = param.data.clone()
                
                # Apply pruning if specified
                if pruning > 0:
                    # Create pruning mask based on magnitude
                    mask = torch.ones_like(modified_data)
                    threshold = torch.quantile(torch.abs(modified_data.flatten()), pruning)
                    mask[torch.abs(modified_data) < threshold] = 0
                    # Apply mask
                    modified_data.mul_(mask)
                
                # Apply quantization if bits < 16
                if bits < 16:
                    # For actual quantization, we'll use Meta's implementation
                    # First, determine a reasonable scaling factor (alpha)
                    if modified_data.dim() <= 1:
                        # For 1D tensors (bias, etc.), use simple scaling
                        alpha = torch.max(torch.abs(modified_data)).unsqueeze(0)
                    else:
                        # For weight matrices, use per-row scaling
                        alpha = torch.max(torch.abs(modified_data), dim=1, keepdim=True)[0]
                    
                    from toggle_dynamic_poc import StretchedElasticQuant, LsqBinaryTernaryExtension
                    
                    # Choose appropriate quantization method
                    if bits <= 2:
                        # Use StretchedElasticQuant for very low bit-width
                        modified_data = StretchedElasticQuant.apply(
                            modified_data, alpha, bits, True
                        )
                    else:
                        # Use LSQ for higher bit-widths
                        modified_data = LsqBinaryTernaryExtension.apply(
                            modified_data, alpha, bits, True
                        )
                
                # Replace original data with modified version
                param.data.copy_(modified_data)
    
    def analyze_layer_sensitivity(self):
        """
        Analyze the sensitivity of each layer to different compression techniques.
        
        Returns:
            sensitivity_map: Dictionary mapping layers to sensitivity scores
        """
        print("Analyzing layer sensitivity...")
        
        # Store original parameters
        self.store_original_parameters()
        
        # Create base config (16-bit, no pruning for all layers)
        base_config = self.toggle.default_config
        
        # Evaluate base model performance
        with torch.no_grad():
            base_results = self.toggle.evaluate_stl_properties(self.toggle.base_model)
            base_stl_score = base_results['stl_score']
        
        print(f"Base model STL score: {base_stl_score:.4f}")
        
        # Initialize sensitivity map
        sensitivity_map = {
            'layer': {},
            'component': {}
        }
        
        # Test bit-width sensitivity for each layer/component
        for layer_idx in range(self.num_layers):
            layer_key = f'layer_{layer_idx}'
            layer_sensitivity = {}
            
            for component in self.components:
                comp_sensitivity = {}
                
                for bits in self.bit_width_to_test:
                    # Restore original parameters
                    self.restore_original_parameters()
                    
                    # Apply compression to this layer/component only
                    self._modify_single_layer(layer_idx, component, bits, 0.0)
                    
                    # Evaluate impact
                    with torch.no_grad():
                        results = self.toggle.evaluate_stl_properties(self.toggle.base_model)
                        stl_score = results['stl_score']
                    
                    # Calculate sensitivity (how much the score changed)
                    # Normalized by base score to get percentage
                    if base_stl_score != 0:
                        sensitivity = (base_stl_score - stl_score) / abs(base_stl_score)
                    else:
                        sensitivity = base_stl_score - stl_score
                    
                    comp_sensitivity[f'bits={bits}'] = {
                        'stl_score': float(stl_score),
                        'sensitivity': float(sensitivity)
                    }
                    
                    print(f"Layer {layer_idx}, {component}, {bits} bits: "
                          f"STL={stl_score:.4f}, Sensitivity={sensitivity:.4f}")
                
                # Test pruning sensitivity
                for pruning in self.pruning_to_test:
                    if pruning > 0:
                        # Restore original parameters
                        self.restore_original_parameters()
                        
                        # Apply pruning to this layer/component only
                        self._modify_single_layer(layer_idx, component, 16, pruning)
                        
                        # Evaluate impact
                        with torch.no_grad():
                            results = self.toggle.evaluate_stl_properties(self.toggle.base_model)
                            stl_score = results['stl_score']
                        
                        # Calculate sensitivity
                        if base_stl_score != 0:
                            sensitivity = (base_stl_score - stl_score) / abs(base_stl_score)
                        else:
                            sensitivity = base_stl_score - stl_score
                        
                        comp_sensitivity[f'pruning={pruning:.1f}'] = {
                            'stl_score': float(stl_score),
                            'sensitivity': float(sensitivity)
                        }
                        
                        print(f"Layer {layer_idx}, {component}, pruning={pruning:.1f}: "
                              f"STL={stl_score:.4f}, Sensitivity={sensitivity:.4f}")
                
                # Store component sensitivity
                sensitivity_map['component'][f"{layer_key}.{component}"] = comp_sensitivity
            
            # Calculate overall layer sensitivity (average across components and operations)
            layer_sensitivities = []
            for component in self.components:
                comp_data = sensitivity_map['component'][f"{layer_key}.{component}"]
                for _, data in comp_data.items():
                    layer_sensitivities.append(data['sensitivity'])
            
            # Average sensitivity for this layer
            if layer_sensitivities:
                avg_sensitivity = sum(layer_sensitivities) / len(layer_sensitivities)
            else:
                avg_sensitivity = 0.0
            
            sensitivity_map['layer'][layer_key] = {
                'avg_sensitivity': float(avg_sensitivity),
                'rank': None  # Will be filled in after sorting
            }
        
        # Restore original parameters when done
        self.restore_original_parameters()
        
        # Rank layers by sensitivity (lower is less sensitive = can be compressed more)
        layer_items = list(sensitivity_map['layer'].items())
        sorted_layers = sorted(layer_items, key=lambda x: x[1]['avg_sensitivity'])
        
        # Assign ranks (0 = least sensitive)
        for i, (layer_key, _) in enumerate(sorted_layers):
            sensitivity_map['layer'][layer_key]['rank'] = i
        
        print("\nLayer Sensitivity Ranking (lower = less sensitive):")
        for i, (layer_key, data) in enumerate(sorted_layers):
            print(f"Rank {i}: {layer_key}, Sensitivity: {data['avg_sensitivity']:.4f}")
        
        return sensitivity_map
    
    def create_sensitivity_guided_config(self, sensitivity_map, aggression_level=0.7):
        """
        Create a configuration based on sensitivity analysis
        
        Args:
            sensitivity_map: Sensitivity data from analyze_layer_sensitivity
            aggression_level: How aggressive to be (0-1, higher = more compression)
            
        Returns:
            config: Layer-specific configuration
        """
        config = {}
        
        # Get normalized ranks for each layer (0-1 scale)
        max_rank = self.num_layers - 1
        normalized_ranks = {}
        
        for layer_key, data in sensitivity_map['layer'].items():
            normalized_ranks[layer_key] = data['rank'] / max_rank if max_rank > 0 else 0
        
        # Define compression levels based on sensitivity
        # Less sensitive layers get more compression
        bit_options = [2, 3, 4, 6, 8, 10, 12, 16]
        
        for layer_key, norm_rank in normalized_ranks.items():
            layer_config = {}
            
            # Compression increases with:
            # - Lower normalized rank (less sensitive)
            # - Higher aggression level
            # Combine these factors to determine compression level
            compression_level = (1 - norm_rank) * aggression_level
            
            # Scale the compression level to select bit-width
            # 0 = no compression (16-bit), 1 = max compression (2-bit)
            scaled_idx = int((len(bit_options) - 1) * compression_level)
            selected_bits = bit_options[scaled_idx]
            
            # Scale pruning similarly
            # More aggressive on less sensitive layers
            pruning_ratio = min(0.5, compression_level * 0.5)
            
            # Create component config
            for component in self.components:
                layer_config[component] = {
                    'bits': selected_bits,
                    'pruning': pruning_ratio
                }
            
            config[layer_key] = layer_config
        
        return config