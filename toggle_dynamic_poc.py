import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
import time
import json
from scipy.spatial.distance import jensenshannon
import argparse
import copy

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a small toy dataset for testing
TOY_DATASET = [
    "Large language models have revolutionized natural language processing.",
    "Compression techniques like quantization and pruning can reduce model size.",
    "Signal temporal logic provides formal verification of model properties.",
    "Long-range dependencies in text are crucial for maintaining coherence.",
    "The challenge is preserving model capabilities while reducing computational demands."
]

# Meta's Quantization Implementation
class LsqBinaryTernaryExtension(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle * (-q_w + q_w.round())
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class StretchedElasticQuant(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        if num_bits == 1 or num_bits == 0:
            Qn = -1
            Qp = 1
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        eps = torch.tensor(0.00001, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (
                torch.round(
                    torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift
                )
                + shift
            ) / n_levels
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )
        if ctx.num_bits == 1:
            if layerwise:
                grad_alpha = (
                    ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
                )
            else:
                grad_alpha = (input_.sign()) * grad_output * grad_scale
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            if layerwise:
                grad_alpha = (
                    (
                        (
                            indicate_small * Qn
                            + indicate_big * Qp
                            + indicate_middle
                            * (
                                -q_w
                                + (
                                    torch.round(
                                        torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                        - shift
                                    )
                                    + shift
                                )
                                / n_levels
                            )
                        )
                        * grad_output
                        * grad_scale
                    )
                    .sum()
                    .unsqueeze(dim=0)
                )
            else:
                grad_alpha = (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle
                        * (
                            -q_w
                            + (
                                torch.round(
                                    torch.clamp(q_w, -clip_val, clip_val) * n_levels
                                    - shift
                                )
                                + shift
                            )
                            / n_levels
                        )
                    )
                    * grad_output
                    * grad_scale
                )
                grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class DynamicPrecisionTransformer:
    def __init__(self, model_name="gpt2-small", dataset=None):
        """
        Initialize the TOGGLE dynamic precision framework.
        
        Args:
            model_name: Name of the pre-trained model to use
            dataset: List of text samples for evaluation
        """
        self.model_name = model_name
        self.dataset = dataset or TOY_DATASET
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print(f"Loading base model: {model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Identify model architecture and get number of layers
        if hasattr(self.base_model, 'transformer'):  # GPT-2 style
            self.model_type = "gpt2"
            self.num_layers = len(self.base_model.transformer.h)
            self.layer_prefix = 'transformer.h'
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):  # LLaMA style
            self.model_type = "llama"
            self.num_layers = len(self.base_model.model.layers)
            self.layer_prefix = 'model.layers'
        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")
        
        print(f"Model has {self.num_layers} layers")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(self.device)
        
        # Create a mapping of layer components for this architecture
        if self.model_type == "gpt2":
            self.components = [
                "attn.c_attn",    # Combined Q,K,V projection
                "attn.c_proj",    # Output projection
                "mlp.c_fc",       # MLP first layer
                "mlp.c_proj"      # MLP second layer
            ]
        elif self.model_type == "llama":
            self.components = [
                "self_attn.q_proj",  # Query projection
                "self_attn.k_proj",  # Key projection
                "self_attn.v_proj",  # Value projection
                "self_attn.o_proj",  # Output projection
                "mlp.gate_proj",     # MLP gate projection
                "mlp.up_proj",       # MLP up projection
                "mlp.down_proj"      # MLP down projection
            ]
        
        # Bit-width options
        self.bit_options = [2, 3, 4, 6, 8, 10, 12, 16]
        
        # STL thresholds
        self.stl_thresholds = {
            'coherence': 0.1,    # Max JSD between token distributions
            'attention': 0.8,    # Min cosine similarity between attention maps
            'context': 0.85,     # Min cosine similarity between embeddings
            'factual': 0.9       # Min probability ratio for factual correctness
        }
        
        # Tokenize dataset for easy access
        self.encoded_dataset = [self.tokenizer.encode(text, return_tensors="pt").to(self.device) 
                               for text in self.dataset]
        
        # Store base model outputs for comparison
        self.base_outputs = self.get_base_model_outputs()
        
        # Create default config (all 16-bit, no pruning)
        self.default_config = self.create_default_config()
    
    def create_default_config(self):
        """Create a default configuration with full precision and no pruning"""
        config = {}
        
        for layer_idx in range(self.num_layers):
            layer_config = {}
            for component in self.components:
                layer_config[component] = {
                    'bits': 16,
                    'pruning': 0.0
                }
            config[f'layer_{layer_idx}'] = layer_config
        
        return config
    
    def create_random_config(self):
        """Create a random configuration for experimentation"""
        config = {}
        
        for layer_idx in range(self.num_layers):
            layer_config = {}
            for component in self.components:
                # Randomly select bit width
                bits = np.random.choice(self.bit_options)
                # Randomly select pruning ratio (0-50%)
                pruning = np.random.uniform(0, 0.5)
                
                layer_config[component] = {
                    'bits': bits,
                    'pruning': pruning
                }
            config[f'layer_{layer_idx}'] = layer_config
        
        return config
    
    def get_base_model_outputs(self):
        """Compute and store base model outputs for later comparison"""
        print("Computing base model outputs...")
        base_outputs = []
        
        with torch.no_grad():
            for encoded_text in tqdm(self.encoded_dataset):
                # Get token probabilities
                outputs = self.base_model(encoded_text, output_attentions=True, output_hidden_states=True)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get attention maps and hidden states
                attentions = outputs.attentions
                hidden_states = outputs.hidden_states
                
                base_outputs.append({
                    'probs': probs.detach().cpu(),
                    'attentions': [att.detach().cpu() for att in attentions],
                    'hidden_states': [hs.detach().cpu() for hs in hidden_states]
                })
                
        return base_outputs
    
    def apply_dynamic_precision(self, config):
        """
        Apply dynamic precision and pruning according to the configuration
        
        Args:
            config: Dictionary with layer/component-specific configuration
                   Format: {layer_0: {component1: {bits: X, pruning: Y}, ...}, ...}
        
        Returns:
            Quantized model
        """
        print("Applying dynamic precision configuration...")
        
        # Create a fresh copy of the model
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Apply configuration to each layer and component
        for layer_name, layer_config in config.items():
            layer_idx = int(layer_name.split('_')[1])
            
            for component, comp_config in layer_config.items():
                bits = comp_config['bits']
                pruning_ratio = comp_config['pruning']
                
                # Find all weights that match this layer and component
                param_prefix = f"{self.layer_prefix}.{layer_idx}.{component}"
                
                # Apply simulated quantization
                self._apply_quantization_to_params(model, param_prefix, bits, pruning_ratio)
        
        model.to(self.device)
        return model
    
    def _apply_quantization_to_params(self, model, param_prefix, bits, pruning_ratio):
        """Apply real quantization and pruning to matching parameters"""
        for name, param in model.named_parameters():
            if name.startswith(param_prefix) and 'weight' in name:
                # Step 1: Apply pruning if specified
                if pruning_ratio > 0:
                    # Create pruning mask based on magnitude
                    mask = torch.ones_like(param.data)
                    threshold = torch.quantile(torch.abs(param.data.flatten()), pruning_ratio)
                    mask[torch.abs(param.data) < threshold] = 0
                    # Apply mask
                    param.data.mul_(mask)
                
                # Step 2: Apply quantization based on bit-width
                if bits < 16:
                    # For actual quantization, we'll use the Meta implementations
                    # First, determine a reasonable scaling factor (alpha)
                    if param.dim() <= 1:
                        # For 1D tensors (bias, etc.), use simple scaling
                        alpha = torch.max(torch.abs(param.data)).unsqueeze(0)
                    else:
                        # For weight matrices, use per-row scaling
                        alpha = torch.max(torch.abs(param.data), dim=1, keepdim=True)[0]
                    
                    # Choose the appropriate quantization method based on bit-width
                    if bits <= 2:
                        # Use StretchedElasticQuant for very low bit-width
                        param.data = StretchedElasticQuant.apply(
                            param.data, alpha, bits, True
                        )
                    else:
                        # Use LSQ for higher bit-widths
                        param.data = LsqBinaryTernaryExtension.apply(
                            param.data, alpha, bits, True
                        )
    
    # Update the evaluate_stl_properties method in DynamicPrecisionTransformer class
    def evaluate_stl_properties(self, model):
        """Evaluate all STL properties for the given model using STL specifications"""
        print("Evaluating STL properties...")
        
        # Import the STL monitor (assuming the file is in the same directory)
        from toggle_rtamt import SimpleSTLMonitor
        
        # Initialize STL monitor with our thresholds
        stl_monitor = SimpleSTLMonitor(stl_thresholds=self.stl_thresholds)
        
        stl_results = {
            'coherence': [],
            'attention': [],
            'context': [],
            'factual': []
        }
        
        with torch.no_grad():
            for i, encoded_text in enumerate(tqdm(self.encoded_dataset)):
                # Get outputs from the model
                outputs = model(encoded_text, output_attentions=True, output_hidden_states=True)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Get attention maps and hidden states
                attentions = outputs.attentions
                hidden_states = outputs.hidden_states
                
                # Base model outputs for comparison
                base_probs = self.base_outputs[i]['probs']
                base_attentions = self.base_outputs[i]['attentions']
                base_hidden_states = self.base_outputs[i]['hidden_states']
                
                # Prepare inputs for STL monitor
                base_outputs = {
                    'probs': base_probs,
                    'attentions': base_attentions,
                    'hidden_states': base_hidden_states
                }
                
                quant_outputs = {
                    'probs': probs.cpu(),
                    'attentions': [att.cpu() for att in attentions],
                    'hidden_states': [hs.cpu() for hs in hidden_states]
                }
                
                # Evaluate all properties using STL specifications
                robustness_values = stl_monitor.evaluate_all_properties(base_outputs, quant_outputs)
                
                # Store results
                for property_name, rob_value in robustness_values.items():
                    stl_results[property_name].append(rob_value)
        
        # Aggregate results - for STL robustness, the minimum value across all samples is what matters
        # because a single violation means the property is not satisfied
        min_results = {k: float(np.min(v)) for k, v in stl_results.items()}
        avg_results = {k: float(np.mean(v)) for k, v in stl_results.items()}
        
        # In STL with robustness semantics, positive values mean the property is satisfied
        # and negative values mean it's violated, with the magnitude indicating "how much"
        robustness = min_results
        
        # For compatibility with the rest of the code, we'll also compute average metrics
        metrics = avg_results
        
        # Calculate overall STL satisfaction score
        stl_score = min([
            robustness['coherence'],
            robustness['attention'],
            robustness['context'],
            robustness['factual']
        ])
        
        return {
            'metrics': metrics,
            'robustness': robustness,
            'stl_score': stl_score,
            'stl_satisfied': stl_score >= 0
        }
    
    def evaluate_config(self, config):
        """Evaluate a specific configuration and return results"""
        # Apply the configuration
        model = self.apply_dynamic_precision(config)
        
        # Calculate model size (estimate)
        model_size = self.calculate_model_size(config)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for encoded_text in self.encoded_dataset:
                _ = model(encoded_text)
        inference_time = (time.time() - start_time) / len(self.encoded_dataset)
        
        # Evaluate STL properties
        stl_results = self.evaluate_stl_properties(model)
        
        # Add model size and inference time to results
        stl_results['model_size'] = model_size
        stl_results['inference_time'] = inference_time
        stl_results['config'] = config
        
        return stl_results
    
    def calculate_model_size(self, config):
        """Calculate approximate model size with the given configuration"""
        # This is an estimation - real implementation would be more precise
        total_params = 0
        total_bits = 0
        
        for name, param in self.base_model.named_parameters():
            # Count parameters
            param_count = param.numel()
            total_params += param_count
            
            # Find matching layer and component in config
            bits = 16  # Default to 16-bit
            for layer_name, layer_config in config.items():
                layer_idx = int(layer_name.split('_')[1])
                for component, comp_config in layer_config.items():
                    if f"{self.layer_prefix}.{layer_idx}.{component}" in name and 'weight' in name:
                        bits = comp_config['bits']
                        pruning = comp_config['pruning']
                        # Adjust param count for pruning
                        param_count = param_count * (1 - pruning)
                        break
                        
            # Add to total bits
            total_bits += param_count * bits
        
        # Convert to MB
        model_size_mb = total_bits / (8 * 1024 * 1024)
        
        return model_size_mb
    
    def visualize_results(self, results):
        """Visualize the evaluation results"""
        # Extract metrics
        metrics = results['metrics']
        robustness = results['robustness']
        stl_score = results['stl_score']
        model_size = results['model_size']
        inference_time = results['inference_time']
        
        print("\n===== Evaluation Results =====")
        print(f"STL Score: {stl_score:.4f} ({'Satisfied' if stl_score >= 0 else 'Violated'})")
        print(f"Model Size: {model_size:.2f} MB")
        print(f"Inference Time: {inference_time:.4f} seconds per sample")
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nRobustness Scores:")
        for metric, value in robustness.items():
            status = "✓" if value >= 0 else "✗"
            print(f"  {metric}: {value:.4f} {status}")
        
        # Create a radar chart of the metrics
        categories = list(metrics.keys())
        values = [metrics[cat] for cat in categories]
        
        # Normalize values for the chart
        norm_values = []
        for i, cat in enumerate(categories):
            if cat == 'coherence':
                # For coherence (JSD), lower is better, so invert
                norm_values.append(1 - (values[i] / 0.2))  # Assuming 0.2 is max JSD
            else:
                # For similarities, higher is better
                norm_values.append(values[i])
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        # Close the loop
        norm_values.append(norm_values[0])
        angles.append(angles[0])
        categories.append(categories[0])
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, norm_values, 'o-', linewidth=2, label='Model Performance')
        ax.fill(angles, norm_values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.set_title('STL Property Evaluation', size=15)
        plt.tight_layout()
        plt.savefig('stl_radar_chart.png')
        print("Radar chart saved as 'stl_radar_chart.png'")
        
        # Create bit-width visualization
        self.visualize_bit_width_config(results['config'])
    
    def visualize_bit_width_config(self, config):
        """Visualize the bit-width configuration across layers and components"""
        # Extract bit-widths from config
        bit_widths = np.zeros((self.num_layers, len(self.components)))
        
        for layer_name, layer_config in config.items():
            layer_idx = int(layer_name.split('_')[1])
            for i, component in enumerate(self.components):
                if component in layer_config:
                    bit_widths[layer_idx, i] = layer_config[component]['bits']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(bit_widths, cmap='viridis')
        
        # Add labels
        ax.set_xticks(np.arange(len(self.components)))
        ax.set_yticks(np.arange(self.num_layers))
        ax.set_xticklabels(self.components)
        ax.set_yticklabels([f'Layer {i}' for i in range(self.num_layers)])
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Bit Width')
        
        # Add title
        ax.set_title("Layer-Component Bit-Width Configuration")
        
        # Loop over data dimensions and create text annotations
        for i in range(self.num_layers):
            for j in range(len(self.components)):
                text = ax.text(j, i, f"{bit_widths[i, j]:.0f}",
                              ha="center", va="center", color="w")
        
        fig.tight_layout()
        plt.savefig('bit_width_config.png')
        print("Bit-width configuration saved as 'bit_width_config.png'")
        
        # Create pruning visualization similarly
        pruning = np.zeros((self.num_layers, len(self.components)))
        
        for layer_name, layer_config in config.items():
            layer_idx = int(layer_name.split('_')[1])
            for i, component in enumerate(self.components):
                if component in layer_config:
                    pruning[layer_idx, i] = layer_config[component]['pruning'] * 100  # Convert to percentage
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pruning, cmap='Reds')
        
        # Add labels
        ax.set_xticks(np.arange(len(self.components)))
        ax.set_yticks(np.arange(self.num_layers))
        ax.set_xticklabels(self.components)
        ax.set_yticklabels([f'Layer {i}' for i in range(self.num_layers)])
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label('Pruning Ratio (%)')
        
        # Add title
        ax.set_title("Layer-Component Pruning Ratio Configuration")
        
        # Loop over data dimensions and create text annotations
        for i in range(self.num_layers):
            for j in range(len(self.components)):
                text = ax.text(j, i, f"{pruning[i, j]:.1f}%",
                              ha="center", va="center", color="w" if pruning[i, j] > 30 else "black")
        
        fig.tight_layout()
        plt.savefig('pruning_config.png')
        print("Pruning configuration saved as 'pruning_config.png'")
    
    def save_config(self, config, filename):
        """Save configuration to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {filename}")
    
    def load_config(self, filename):
        """Load configuration from a JSON file"""
        with open(filename, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {filename}")
        return config
    
    def update_component_config(self, layer_idx, component, bits=None, pruning=None):
        """
        Update configuration for a specific layer/component
        
        Args:
            layer_idx: Layer index
            component: Component name
            bits: New bit-width (if None, keep current)
            pruning: New pruning ratio (if None, keep current)
        
        Returns:
            Updated configuration
        """
        # Start with default config if none exists
        if not hasattr(self, 'current_config'):
            self.current_config = self.create_default_config()
        
        # Update the specified component
        layer_key = f'layer_{layer_idx}'
        if layer_key in self.current_config and component in self.current_config[layer_key]:
            if bits is not None:
                self.current_config[layer_key][component]['bits'] = bits
            if pruning is not None:
                self.current_config[layer_key][component]['pruning'] = pruning
        
        return self.current_config


def main():
    parser = argparse.ArgumentParser(description="TOGGLE Dynamic Precision PoC")
    parser.add_argument('--model', type=str, default='gpt2', help='Model name/path')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--save-config', type=str, help='Save configuration to file')
    parser.add_argument('--random', action='store_true', help='Use random configuration')
    args = parser.parse_args()
    
    # Initialize the framework
    toggle = DynamicPrecisionTransformer(model_name=args.model)
    
    # Determine which configuration to use
    if args.config:
        # Load from file
        config = toggle.load_config(args.config)
    elif args.random:
        # Use random configuration
        config = toggle.create_random_config()
    else:
        # Use default configuration
        config = toggle.default_config
    
    # Evaluate the configuration
    results = toggle.evaluate_config(config)
    
    # Visualize results
    toggle.visualize_results(results)
    
    # Save configuration if requested
    if args.save_config:
        toggle.save_config(config, args.save_config)


if __name__ == "__main__":
    main()