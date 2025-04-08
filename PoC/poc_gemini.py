# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# --- MODIFIED FOR TOGGLE FULL BO PoC ---

import math
import copy
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from datasets import load_dataset
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import rtamt
from skopt import gp_minimize # BO Library
from skopt.space import Integer, Real # Define search space
from skopt.utils import use_named_args

# --- Quantization Code (Provided by User - Assumed Correct & Included Here) ---
# [Include LsqBinaryTernaryExtension, StretchedElasticQuant, QuantizeLinear classes here]
class LsqBinaryTernaryExtension(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        ctx.num_bits = num_bits
        if num_bits >= 16: return input
        if num_bits == 1 or num_bits == 0: Qn, Qp = -1, 1
        else: Qn, Qp = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
        eps = torch.tensor(1e-5, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)
        grad_scale = 1.0 / math.sqrt(input.numel() * Qp) if Qp!=0 else 1.0 / math.sqrt(input.numel())
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1: q_w = input.sign()
        else: q_w = (input / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        return w_q
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16: return grad_output, None, None, None
        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.num_bits == 1:
            grad_alpha_term = (input_.sign()) * grad_output * grad_scale
        else:
            grad_alpha_term = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + q_w.round())) * grad_output * grad_scale)
        if layerwise: grad_alpha = grad_alpha_term.sum().unsqueeze(dim=0)
        else: grad_alpha = torch.sum(grad_alpha_term, dim=-1, keepdim=True)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class StretchedElasticQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        ctx.num_bits = num_bits
        if num_bits >= 16: return input
        eps = torch.tensor(1e-5, device=alpha.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)
        if num_bits == 1 or num_bits == 0: Qn, Qp = -1, 1
        else: Qn, Qp = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
        grad_scale = 1.0 / math.sqrt(input.numel() * Qp) if Qp!=0 else 1.0 / math.sqrt(input.numel())
        ctx.save_for_backward(input, alpha)
        clip_val = 1 - 1e-2
        if num_bits == 0: n_levels, shift = 1.5, 0
        else: n_levels, shift = 2 ** (num_bits - 1), 0.5
        Qp_stretch, Qn_stretch = (n_levels - shift) / n_levels, -((n_levels - shift) / n_levels)
        ctx.other = grad_scale, Qn_stretch, Qp_stretch, layerwise
        if num_bits == 1: q_w = input.sign()
        else: q_w = (torch.round(torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels
        w_q = q_w * alpha
        return w_q
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16: return grad_output, None, None, None
        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0: n_levels, shift = 1.5, 0
        else: n_levels, shift = 2 ** (ctx.num_bits - 1), 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.num_bits == 1:
            grad_alpha_term = (input_.sign()) * grad_output * grad_scale
        else:
            grad_alpha_term = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (-q_w + (torch.round(torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift) + shift) / n_levels)) * grad_output * grad_scale)
        if layerwise: grad_alpha = grad_alpha_term.sum().unsqueeze(dim=0)
        else: grad_alpha = torch.sum(grad_alpha_term, dim=-1, keepdim=True)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None

class QuantizeLinear(nn.Linear):
    def __init__(self, *kargs, bias=False, w_bits=16, weight_layerwise=False, **kwargs):
        in_features, out_features = kargs[0], kargs[1]
        super(QuantizeLinear, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self.w_bits = w_bits
        self.weight_layerwise = weight_layerwise
        if self.w_bits < 16:
            if self.w_bits == 1 or self.w_bits == 0: Qp_init = 1
            else: Qp_init = 2 ** (self.w_bits - 1) - 1
            init_val = 2*torch.mean(torch.abs(self.weight.data))/math.sqrt(Qp_init) if Qp_init > 0 else torch.mean(torch.abs(self.weight.data))
            # Ensure init_val is a scalar before filling
            init_val_scalar = init_val.item() if torch.is_tensor(init_val) else init_val
            if self.weight_layerwise: self.weight_clip_val = nn.Parameter(torch.Tensor([init_val_scalar]))
            else: self.weight_clip_val = nn.Parameter(torch.full((self.weight.shape[0], 1), init_val_scalar))
        else: self.weight_clip_val = None

    def forward(self, input_):
        real_weights = self.weight
        if self.w_bits >= 16 or self.weight_clip_val is None: weight = self.weight
        elif self.w_bits <= 2:
             if self.w_bits == 1: weight = LsqBinaryTernaryExtension.apply(real_weights, self.weight_clip_val, self.w_bits, self.weight_layerwise)
             else: weight = StretchedElasticQuant.apply(real_weights, self.weight_clip_val, self.w_bits, self.weight_layerwise)
        elif self.w_bits <= 4: weight = LsqBinaryTernaryExtension.apply(real_weights, self.weight_clip_val, self.w_bits, self.weight_layerwise)
        else:
            Qn, Qp = -(2 ** (self.w_bits - 1)), 2 ** (self.w_bits - 1) - 1
            weight_alpha = self.weight_clip_val
            q_w = (real_weights / weight_alpha).round().clamp(Qn, Qp)
            weight = q_w * weight_alpha
        weight = weight.to(input_.dtype)
        out = nn.functional.linear(input_, weight, self.bias)
        return out
# --- End Quantization Code ---


# --- Configuration & Global Variables ---
MODEL_NAME = "gpt2"
DATASET_NAME = "lambada"
DATASET_SPLIT = "validation"
NUM_SAMPLES_STL = 5 # Keep STL evaluation very small for speed
NUM_SAMPLES_CALIB = 0 # No calibration needed for this simplified PoC
MAX_LENGTH_T_PRIME = 8 # Keep STL trace very short
SEED = 42
N_BO_CALLS = 15 # Number of BO iterations (keep low for demo)

# STL Parameters
EPSILON_PHI1 = 0.15 # Threshold for JSD
RHO_TH_PHI1 = 0.0 # Target minimum robustness for phi_1

# Global variables for model, tokenizer, dataset (load once)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL = None
TOKENIZER = None
EVAL_DATASET = None
MODEL_PARAMS_INFO = None # To store estimated param info

# Penalty for constraint violation
CONSTRAINT_PENALTY = 1e10

# --- Helper Functions (Quant/Prune/Cost/STL Eval - Adapted) ---

# [Include estimate_gpt2_params, calculate_detailed_cost functions here - Assume they take kappa]
# [Include replace_linear_with_quantized, apply_pruning_to_layer, apply_compression functions here]
# [Include get_next_token_probs, calculate_jsd functions here]
# NOTE: Ensure these functions are defined before being called in objective_func

# --- estimate_gpt2_params, calculate_detailed_cost ---
def estimate_gpt2_params():
    hidden_size = 768
    params_attn_qkv = 3 * hidden_size**2
    params_attn_out = hidden_size**2
    params_ffn_up = hidden_size * 4 * hidden_size
    params_ffn_down = 4 * hidden_size * hidden_size
    num_layers = 12
    config = GPT2Config.from_pretrained(MODEL_NAME)
    params_lm_head = hidden_size * config.vocab_size
    components_per_layer = {
        'c_attn.weight': params_attn_qkv,
        'c_proj.weight': params_attn_out,
        'mlp.c_fc.weight': params_ffn_up,
        'mlp.c_proj.weight': params_ffn_down
    }
    components_global = {'lm_head.weight': params_lm_head}
    total_params_approx = num_layers * sum(components_per_layer.values()) + sum(components_global.values())
    # print(f"Estimated total core parameters: ~{total_params_approx/1e6:.1f}M")
    return total_params_approx, components_per_layer, components_global, num_layers

def calculate_detailed_cost(config_kappa, components_per_layer, components_global, num_layers, S=512, b_ref=16, C=2):
    estimated_flops = 0.0
    for l in range(num_layers):
        b_lc, p_lc = config_kappa.get(l, (16, 0.0))
        for W_lc_approx in components_per_layer.values():
            term = (1.0 - p_lc) * W_lc_approx * S * (b_lc / b_ref)
            estimated_flops += term
    b_lc, p_lc = config_kappa.get('lm_head', (16, 0.0))
    W_lc_approx = components_global.get('lm_head.weight', 0)
    term = (1.0 - p_lc) * W_lc_approx * S * (b_lc / b_ref)
    estimated_flops += term
    return C * estimated_flops

# --- replace_linear_with_quantized, apply_pruning_to_layer ---
def replace_linear_with_quantized(module, layer_config):
    target_bits = layer_config[0]
    if not isinstance(module, nn.Linear) or isinstance(module, QuantizeLinear) or target_bits >= 16:
        return module
    quantized_layer = QuantizeLinear(
        module.in_features, module.out_features, bias=module.bias is not None,
        w_bits=target_bits, weight_layerwise=False
    )
    quantized_layer.weight.data.copy_(module.weight.data)
    if module.bias is not None: quantized_layer.bias.data.copy_(module.bias.data)
    return quantized_layer

def apply_pruning_to_layer(layer_module, layer_config):
    target_pruning = layer_config[1]
    if target_pruning <= 0: return

    modules_to_prune = []
    for name, module in layer_module.named_modules():
         if isinstance(module, (nn.Linear, QuantizeLinear)):
              # Check for main weight matrices by common GPT-2 naming conventions
              is_main_weight = False
              for keyword in ['c_attn', 'c_proj', 'c_fc', 'lm_head']:
                   if keyword in name:
                       # Heuristic: Avoid pruning biases if layer name ends with bias?
                       # This basic check might need refinement for complex nested structures.
                       # Let's assume for PoC we only target layers with 'weight' explicitly available.
                       # A more robust way is needed for production.
                       if hasattr(module, 'weight'): # Only prune layers with weights
                            is_main_weight = True
                            break
              if is_main_weight:
                    modules_to_prune.append((module, 'weight'))


    # Need to handle potential duplicates if named_modules yields nested results
    unique_modules_to_prune = list({id(mod): (mod, name) for mod, name in modules_to_prune}.values())

    for module, param_name in unique_modules_to_prune:
        # Check if already pruned (might happen with shared layers/references if not careful)
        if not prune.is_pruned(module):
             try:
                 prune.l1_unstructured(module, name=param_name, amount=target_pruning)
                 prune.remove(module, param_name) # Make permanent
             except Exception as e:
                 print(f"Warning: Pruning failed for {name}: {e}")


# --- apply_compression ---
def apply_compression(base_model, config_kappa):
    compressed_model = copy.deepcopy(base_model)
    # Apply Pruning first
    # print("Applying Pruning...")
    for i, layer in enumerate(compressed_model.transformer.h):
        if i in config_kappa: apply_pruning_to_layer(layer, config_kappa[i])
    if 'lm_head' in config_kappa: apply_pruning_to_layer(compressed_model.lm_head, config_kappa['lm_head'])
    # print("Pruning applied.")

    # Apply Quantization
    # print("Applying Quantization...")
    for i in range(compressed_model.config.n_layer):
         if i in config_kappa and config_kappa[i][0] < 16:
              layer = compressed_model.transformer.h[i]
              # Replace specific submodules - MUST match actual GPT-2 structure
              if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_attn'): layer.attn.c_attn = replace_linear_with_quantized(layer.attn.c_attn, config_kappa[i])
              if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_proj'): layer.attn.c_proj = replace_linear_with_quantized(layer.attn.c_proj, config_kappa[i])
              if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'c_fc'): layer.mlp.c_fc = replace_linear_with_quantized(layer.mlp.c_fc, config_kappa[i])
              if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'c_proj'): layer.mlp.c_proj = replace_linear_with_quantized(layer.mlp.c_proj, config_kappa[i])
    if 'lm_head' in config_kappa and config_kappa['lm_head'][0] < 16:
         compressed_model.lm_head = replace_linear_with_quantized(compressed_model.lm_head, config_kappa['lm_head'])
    # print("Quantization applied.")
    compressed_model.eval()
    return compressed_model

# --- get_next_token_probs, calculate_jsd ---
@torch.no_grad()
def get_next_token_probs(model, input_ids, device):
    model.to(device); model.eval()
    try:
        outputs = model(input_ids.to(device))
        logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error during forward pass: {e}")
        # Return uniform distribution on error? Or handle upstream?
        # Returning None for now to indicate failure
        return None


def calculate_jsd(p, q, base=2, epsilon_div=1e-10):
    if p is None or q is None or len(p) != len(q): return float('inf') # Indicate error/mismatch
    p = np.asarray(p) + epsilon_div
    q = np.asarray(q) + epsilon_div
    p /= np.sum(p)
    q /= np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m + epsilon_div)) # Add epsilon inside log
    kl_qm = np.sum(q * np.log(q / m + epsilon_div))
    jsd_val = 0.5 * (kl_pm + kl_qm)
    if jsd_val < 0: jsd_val = 0.0
    # JSD definition varies; this matches common information theory def.
    # Ensure it aligns with robustness interpretation (lower JSD is better)
    return jsd_val

# --- evaluate_stl_phi1_robustness ---
@torch.no_grad()
def evaluate_stl_phi1_robustness(base_model, compressed_model, tokenizer, dataset_subset, device, epsilon):
    """Evaluates STL spec phi_1 using RTAMT."""
    spec = rtamt.STLSpecification(semantics=rtamt.Semantics.STANDARD)
    spec.name = 'Sequential Coherence Phi_1'
    spec.declare_var('jsd_signal', 'float')
    spec.spec = f'always[0:{MAX_LENGTH_T_PRIME-1}] (jsd_signal <= {epsilon})'
    try:
        spec.parse()
    except rtamt.RTAMTException as err:
        print(f'RTAMT Parsing Exception: {err}'); return -float('inf')

    min_robustness_overall = float('inf')
    num_evaluated_samples = 0

    for sample in dataset_subset: # Iterate only over the provided subset
        text = sample['text']
        last_space_idx = text.rfind(' ')
        if last_space_idx == -1: continue
        context = text[:last_space_idx].strip()

        input_ids = tokenizer.encode(context, return_tensors='pt')
        # Ensure context is not too long initially
        if input_ids.shape[1] >= base_model.config.n_positions: continue

        base_input_ids = input_ids.clone()
        comp_input_ids = input_ids.clone()

        trace = {'time': [], 'jsd_signal': []}
        valid_steps = 0

        for t in range(MAX_LENGTH_T_PRIME):
            if base_input_ids.shape[1] >= base_model.config.n_positions: break

            probs_base = get_next_token_probs(base_model, base_input_ids, device)
            probs_comp = get_next_token_probs(compressed_model, comp_input_ids, device)

            jsd = calculate_jsd(probs_base, probs_comp)
            if math.isinf(jsd): # Skip step if JSD calculation failed
                print(f"Warning: Infinite JSD at step {t}. Skipping step.")
                continue

            trace['time'].append(float(t))
            trace['jsd_signal'].append(jsd)
            valid_steps += 1

            if probs_base is None: break # Stop if base model failed
            next_token_id = np.argmax(probs_base)
            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            base_input_ids = torch.cat([base_input_ids, next_token_tensor], dim=1)
            comp_input_ids = torch.cat([comp_input_ids, next_token_tensor], dim=1)

            if next_token_id == tokenizer.eos_token_id: break

        if valid_steps > 0:
            try:
                robustness_trace = spec.evaluate(trace)
                sample_robustness = robustness_trace[0][1] if robustness_trace else -float('inf')
                min_robustness_overall = min(min_robustness_overall, sample_robustness)
                num_evaluated_samples += 1
            except rtamt.RTAMTException as err:
                print(f'RTAMT Eval Exception: {err}'); min_robustness_overall = min(min_robustness_overall, -float('inf'))
            # print(f"Sample Rob: {sample_robustness:.4f}") # Debug

    # print(f"Evaluated {num_evaluated_samples} samples for robustness.")
    if num_evaluated_samples == 0: return -float('inf')
    return min_robustness_overall


# --- BO Objective Function ---
# Decorator to handle named arguments from skopt
@use_named_args(dimensions=None) # Dimensions will be set dynamically
def objective_func(**params):
    """
    Objective function for Bayesian Optimization.
    Takes keyword arguments corresponding to dimensions in search_space.
    Returns the cost E(kappa), penalized if constraints are violated.
    """
    global BASE_MODEL, TOKENIZER, EVAL_DATASET, DEVICE, MODEL_PARAMS_INFO, RHO_TH_PHI1, EPSILON_PHI1, CONSTRAINT_PENALTY

    start_time = time.time()

    # 1. Construct kappa from BO parameters
    # Simplified: Apply same (bits, pruning) to all components in a layer
    kappa_config = {}
    num_layers = MODEL_PARAMS_INFO[3]
    for i in range(num_layers):
        bits = params[f'layer_{i}_bits']
        prune_ratio = params[f'layer_{i}_prune']
        kappa_config[i] = (bits, prune_ratio)
    kappa_config['lm_head'] = (params['lm_head_bits'], params['lm_head_prune'])

    # print(f"Evaluating Kappa: {kappa_config}") # Debug

    # 2. Apply Compression
    # Critical: Ensure base_model exists and is on CPU to avoid GPU memory clash if BO runs sequentially
    if BASE_MODEL is None: raise ValueError("Base model not loaded")
    BASE_MODEL.to('cpu') # Ensure base is on CPU before deepcopy
    torch.cuda.empty_cache() # Clear cache before creating new model
    compressed_model = apply_compression(BASE_MODEL, kappa_config)
    compressed_model.to(DEVICE) # Move compressed model to GPU for eval

    # 3. Calculate Cost (Fast)
    cost = calculate_detailed_cost(kappa_config, *MODEL_PARAMS_INFO[1:])

    # 4. Evaluate Robustness (Slow) - ONLY phi_1 for this PoC
    # Move base model to GPU for evaluation if not already there
    BASE_MODEL.to(DEVICE)
    phi1_robustness = evaluate_stl_phi1_robustness(
        BASE_MODEL, compressed_model, TOKENIZER, EVAL_DATASET, DEVICE, EPSILON_PHI1
    )
    # Placeholder for other robustness values
    # phi2_robustness = ...
    # phi3_robustness = ...
    # phi4_robustness = ...

    # Clean up GPU memory after evaluation
    del compressed_model
    BASE_MODEL.to('cpu') # Move base back to CPU
    torch.cuda.empty_cache()

    # 5. Check Constraints & Apply Penalty
    constraint_satisfied = (phi1_robustness >= RHO_TH_PHI1)
    # In full version: check all phi_j >= rho_th_j

    final_objective_value = cost
    if not constraint_satisfied:
        # Apply penalty: higher penalty for larger violation
        violation_magnitude = abs(phi1_robustness - RHO_TH_PHI1)
        final_objective_value += CONSTRAINT_PENALTY * (1 + violation_magnitude) # Penalize more for larger violation
        print(f"Constraint violated! Rho1={phi1_robustness:.4f} < {RHO_TH_PHI1}. Penalized Cost={final_objective_value:.4e}")
    else:
        print(f"Constraint met. Rho1={phi1_robustness:.4f}. Cost={final_objective_value:.4e}")


    elapsed = time.time() - start_time
    print(f"Iteration completed in {elapsed:.2f}s")

    # gp_minimize minimizes the returned value
    return final_objective_value


# --- Main Execution ---
if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    warnings.filterwarnings("ignore", category=UserWarning) # Suppress some harmless warnings

    print(f"Using device: {DEVICE}")

    # Load shared resources ONCE
    print(f"Loading base model: {MODEL_NAME}")
    BASE_MODEL = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    TOKENIZER = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    TOKENIZER.pad_token = TOKENIZER.eos_token
    BASE_MODEL.eval()
    print(f"Loading dataset: {DATASET_NAME}")
    full_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    EVAL_DATASET = full_dataset.shuffle(seed=SEED).select(range(NUM_SAMPLES_STL))
    print(f"STL evaluation subset size: {len(EVAL_DATASET)}")
    MODEL_PARAMS_INFO = estimate_gpt2_params() # Get param info (total, comps, layers)

    # Define Search Space for BO (Simplified: Layer-wise bits/pruning + head)
    search_space = []
    num_layers = MODEL_PARAMS_INFO[3]
    bit_space = (2, 16) # Bit range (inclusive)
    prune_space = (0.0, 0.5) # Pruning range (inclusive)

    for i in range(num_layers):
        search_space.append(Integer(low=bit_space[0], high=bit_space[1], name=f'layer_{i}_bits'))
        search_space.append(Real(low=prune_space[0], high=prune_space[1], name=f'layer_{i}_prune'))
    search_space.append(Integer(low=bit_space[0], high=bit_space[1], name='lm_head_bits'))
    search_space.append(Real(low=prune_space[0], high=prune_space[1], name='lm_head_prune'))

    # Dynamically set dimensions for the objective function decorator
    objective_func.dimensions = search_space

    # Run Bayesian Optimization
    print(f"\n--- Starting Bayesian Optimization ({N_BO_CALLS} calls) ---")
    start_bo = time.time()

    # gp_minimize tries to find parameters that minimize objective_func
    result = gp_minimize(
        func=objective_func,
        dimensions=search_space,
        n_calls=N_BO_CALLS,       # Total number of evaluations
        n_initial_points=5, # How many random points to sample initially
        acq_func='EI',      # Expected Improvement acquisition function
        random_state=SEED,
        verbose=True        # Print progress
    )

    end_bo = time.time()
    print(f"\n--- Bayesian Optimization Finished ({end_bo - start_bo:.2f}s) ---")

    # Extract Best Result found by BO (lowest objective value)
    best_params_list = result.x
    best_objective_value = result.fun

    # Reconstruct best kappa from list
    best_kappa = {}
    idx = 0
    for i in range(num_layers):
        best_kappa[i] = (best_params_list[idx], best_params_list[idx+1])
        idx += 2
    best_kappa['lm_head'] = (best_params_list[idx], best_params_list[idx+1])

    print("\nBest configuration found by BO:")
    print(f"Kappa: {best_kappa}")
    print(f"Achieved Objective Value (Penalized Cost): {best_objective_value:.4e}")

    # Re-evaluate the best configuration to get its true cost and robustness
    print("\nRe-evaluating best configuration found...")
    final_cost = calculate_detailed_cost(best_kappa, *MODEL_PARAMS_INFO[1:])
    final_comp_model = apply_compression(BASE_MODEL, best_kappa)
    final_comp_model.to(DEVICE)
    BASE_MODEL.to(DEVICE)
    final_robustness = evaluate_stl_phi1_robustness(
         BASE_MODEL, final_comp_model, TOKENIZER, EVAL_DATASET, DEVICE, EPSILON_PHI1
    )

    print("\n--- Final Performance of Best Found Kappa ---")
    print(f"Estimated FLOPs Cost (E(kappa)): {final_cost:.4e}")
    print(f"Minimum Robustness (rho_min,1): {final_robustness:.6f}")
    if final_robustness >= RHO_TH_PHI1:
        print("Constraint Met.")
    else:
        print("Constraint Violated (BO might have settled here due to penalties/search space).")

    # Post-processing (finding feasible points, selecting modes) would follow here
    # Requires storing results from all BO iterations (result.x_iters, result.func_vals)
    # and re-evaluating robustness for points where cost wasn't penalized.
    print("\nFull implementation would involve post-processing all evaluated points to find feasible set and select modes.")
