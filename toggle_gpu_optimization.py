"""
GPU-optimized STL evaluation for TOGGLE framework.
This module replaces the original STL evaluation with GPU-optimized
implementations that fully utilize available hardware.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class GPUOptimizedSTLEvaluator:
    """
    Optimized STL evaluator that leverages GPU acceleration.
    """
    
    def __init__(self, toggle_framework):
        """
        Initialize the GPU-optimized STL evaluator.
        
        Args:
            toggle_framework: DynamicPrecisionTransformer instance
        """
        self.toggle = toggle_framework
        self.device = next(toggle_framework.base_model.parameters()).device
        self.stl_thresholds = toggle_framework.stl_thresholds
        
        # Ensure model is on GPU
        if self.device.type != 'cuda':
            print("Warning: Model not on CUDA device. Moving to GPU...")
            self.toggle.base_model.to('cuda')
            self.device = torch.device('cuda')
    
    @torch.no_grad()
    def evaluate_stl_properties(self, model, batch_size=None):
        """
        GPU-optimized evaluation of STL properties.
        
        Args:
            model: Model to evaluate
            batch_size: Batch size for evaluation (optional)
            
        Returns:
            Dictionary with evaluation results
        """
        # Process all encoded dataset samples
        if batch_size and len(self.toggle.encoded_dataset) > batch_size:
            # Batch processing for large datasets
            return self._evaluate_in_batches(model, batch_size)
        else:
            # Process all at once for small datasets
            return self._evaluate_all_samples(model)
    
    @torch.no_grad()
    def _evaluate_in_batches(self, model, batch_size):
        """Process evaluation in batches"""
        # Initialize aggregated results
        all_robustness = {
            'coherence': [],
            'attention': [],
            'context': [],
            'factual': []
        }
        
        # Process in batches
        for i in range(0, len(self.toggle.encoded_dataset), batch_size):
            batch_encoded = self.toggle.encoded_dataset[i:i+batch_size]
            batch_base_outputs = self.toggle.base_outputs[i:i+batch_size]
            
            # Create single batch by padding sequences
            max_length = max(enc.shape[1] for enc in batch_encoded)
            batch_input_ids = torch.zeros(len(batch_encoded), max_length, dtype=batch_encoded[0].dtype, device=self.device)
            attention_mask = torch.zeros(len(batch_encoded), max_length, dtype=torch.bool, device=self.device)
            
            for j, enc in enumerate(batch_encoded):
                length = enc.shape[1]
                batch_input_ids[j, :length] = enc[0]
                attention_mask[j, :length] = True
            
            # Forward pass with model
            outputs = model(batch_input_ids, attention_mask=attention_mask, 
                           output_attentions=True, output_hidden_states=True)
            
            # Extract and keep all tensors on GPU
            logits = outputs.logits
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states
            
            # Evaluate properties for this batch
            batch_results = self._evaluate_properties(
                batch_input_ids, logits, attentions, hidden_states, batch_base_outputs)
            
            # Aggregate results
            for k, v in batch_results.items():
                all_robustness[k].extend(v)
        
        # Calculate minimum robustness for each property
        min_robustness = {}
        for property_name, values in all_robustness.items():
            tensor_values = torch.tensor(values, device=self.device)
            min_robustness[property_name] = tensor_values.min().item()
        
        # Calculate average metrics
        avg_metrics = {}
        for property_name, values in all_robustness.items():
            tensor_values = torch.tensor(values, device=self.device)
            avg_metrics[property_name] = tensor_values.mean().item()
        
        # Calculate overall STL score
        stl_score = min(min_robustness.values())
        
        return {
            'metrics': avg_metrics,
            'robustness': min_robustness,
            'stl_score': stl_score,
            'stl_satisfied': stl_score >= 0
        }
    
    @torch.no_grad()
    def _evaluate_all_samples(self, model):
        """Evaluate all samples at once"""
        all_robustness = {
            'coherence': [],
            'attention': [],
            'context': [],
            'factual': []
        }
        
        # Process each sample
        for i, encoded_text in enumerate(self.toggle.encoded_dataset):
            # Forward pass with model - keep all operations on GPU
            outputs = model(encoded_text, output_attentions=True, output_hidden_states=True)
            
            # Extract outputs - keep on GPU
            logits = outputs.logits
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states
            
            # Get base model outputs
            base_outputs = self.toggle.base_outputs[i]
            
            # Evaluate STL properties
            sample_robustness = self._evaluate_sample(encoded_text, logits, attentions, hidden_states, base_outputs)
            
            # Aggregate results
            for k, v in sample_robustness.items():
                all_robustness[k].append(v)
        
        # Calculate minimum robustness for each property
        min_robustness = {}
        for property_name, values in all_robustness.items():
            tensor_values = torch.tensor(values, device=self.device)
            min_robustness[property_name] = tensor_values.min().item()
        
        # Calculate average metrics
        avg_metrics = {}
        for property_name, values in all_robustness.items():
            tensor_values = torch.tensor(values, device=self.device)
            avg_metrics[property_name] = tensor_values.mean().item()
        
        # Calculate overall STL score
        stl_score = min(min_robustness.values())
        
        return {
            'metrics': avg_metrics,
            'robustness': min_robustness,
            'stl_score': stl_score,
            'stl_satisfied': stl_score >= 0
        }
    
    @torch.no_grad()
    def _evaluate_properties(self, input_ids, logits, attentions, hidden_states, base_outputs_list):
        """
        Evaluate STL properties for a batch of samples.
        Keep all operations on GPU for maximum performance.
        """
        batch_size = input_ids.shape[0]
        
        # Initialize results
        robustness = {
            'coherence': [],
            'attention': [],
            'context': [],
            'factual': []
        }
        
        # Process each sample
        for i in range(batch_size):
            # Extract slices for this sample
            sample_logits = logits[i:i+1]
            sample_attentions = [att[i:i+1] for att in attentions]
            sample_hidden_states = [hs[i:i+1] for hs in hidden_states]
            
            # Get base model outputs for this sample
            base_outputs = base_outputs_list[i]
            
            # Evaluate STL properties
            sample_robustness = self._evaluate_sample(
                input_ids[i:i+1], sample_logits, sample_attentions, sample_hidden_states, base_outputs)
            
            # Store results
            for k, v in sample_robustness.items():
                robustness[k].append(v)
        
        return robustness
    
    @torch.no_grad()
    def _evaluate_sample(self, input_ids, logits, attentions, hidden_states, base_outputs):
        """
        Evaluate STL properties for a single sample.
        All operations are performed on GPU for maximum performance.
        """
        # Move base outputs to GPU if needed
        base_probs = self._to_gpu(base_outputs['probs'])
        base_attentions = [self._to_gpu(att) for att in base_outputs['attentions']]
        base_hidden_states = [self._to_gpu(hs) for hs in base_outputs['hidden_states']]
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # 1. Evaluate coherence using JSD
        coherence_rob = self._evaluate_coherence(base_probs, probs)
        
        # 2. Evaluate attention similarity
        attention_rob = self._evaluate_attention(base_attentions, attentions)
        
        # 3. Evaluate context similarity
        context_rob = self._evaluate_context(base_hidden_states[-1], hidden_states[-1])
        
        # 4. Evaluate factual accuracy
        factual_rob = self._evaluate_factual(base_probs, probs)
        
        return {
            'coherence': coherence_rob,
            'attention': attention_rob,
            'context': context_rob,
            'factual': factual_rob
        }
    
    def _to_gpu(self, tensor):
        """Move a tensor to GPU if it's not already there"""
        if isinstance(tensor, torch.Tensor) and tensor.device.type != 'cuda':
            return tensor.to(self.device)
        return tensor
    
    @torch.no_grad()
    def _evaluate_coherence(self, base_probs, quant_probs):
        """
        GPU-optimized evaluation of coherence using JSD.
        
        Args:
            base_probs: Token probabilities from base model
            quant_probs: Token probabilities from quantized model
            
        Returns:
            Coherence robustness value
        """
        # Extract last token probabilities
        base_last = base_probs[0, -1]
        quant_last = quant_probs[0, -1]
        
        # Get top tokens (using GPU operations)
        _, base_top_indices = torch.topk(base_last, k=min(100, base_last.size(0)))
        _, quant_top_indices = torch.topk(quant_last, k=min(100, quant_last.size(0)))
        
        # Combine indices
        top_indices = torch.cat([base_top_indices, quant_top_indices]).unique()
        
        # Extract probabilities for these indices
        base_top_probs = base_last[top_indices]
        quant_top_probs = quant_last[top_indices]
        
        # Normalize
        base_top_probs = base_top_probs / base_top_probs.sum()
        quant_top_probs = quant_top_probs / quant_top_probs.sum()
        
        # Calculate JSD (GPU optimized)
        m = 0.5 * (base_top_probs + quant_top_probs)
        base_kl = F.kl_div(m.log(), base_top_probs, reduction='sum')
        quant_kl = F.kl_div(m.log(), quant_top_probs, reduction='sum')
        jsd = 0.5 * (base_kl + quant_kl)
        
        # Calculate robustness
        coherence_rob = self.stl_thresholds['coherence'] - jsd.item()
        
        return coherence_rob
    
    @torch.no_grad()
    def _evaluate_attention(self, base_attentions, quant_attentions):
        """
        GPU-optimized evaluation of attention similarity.
        
        Args:
            base_attentions: Attention maps from base model
            quant_attentions: Attention maps from quantized model
            
        Returns:
            Attention robustness value
        """
        similarities = []
        
        for base_att, quant_att in zip(base_attentions, quant_attentions):
            # Flatten attention maps
            base_flat = base_att.view(-1)
            quant_flat = quant_att.view(-1)
            
            # Calculate cosine similarity on GPU
            similarity = F.cosine_similarity(base_flat.unsqueeze(0), quant_flat.unsqueeze(0))
            similarities.append(similarity)
        
        # Average similarity across layers
        avg_similarity = torch.stack(similarities).mean()
        
        # Calculate robustness
        attention_rob = avg_similarity.item() - self.stl_thresholds['attention']
        
        return attention_rob
    
    @torch.no_grad()
    def _evaluate_context(self, base_embs, quant_embs):
        """
        GPU-optimized evaluation of contextual consistency.
        
        Args:
            base_embs: Context embeddings from base model
            quant_embs: Context embeddings from quantized model
            
        Returns:
            Context robustness value
        """
        # Extract embeddings for the last token
        base_emb = base_embs[0, -1]
        quant_emb = quant_embs[0, -1]
        
        # Calculate cosine similarity on GPU
        similarity = F.cosine_similarity(base_emb.unsqueeze(0), quant_emb.unsqueeze(0))
        
        # Calculate robustness
        context_rob = similarity.item() - self.stl_thresholds['context']
        
        return context_rob
    
    @torch.no_grad()
    def _evaluate_factual(self, base_probs, quant_probs):
        """
        GPU-optimized evaluation of factual accuracy.
        
        Args:
            base_probs: Token probabilities from base model
            quant_probs: Token probabilities from quantized model
            
        Returns:
            Factual robustness value
        """
        # Get last token probabilities
        base_last = base_probs[0, -1]
        quant_last = quant_probs[0, -1]
        
        # Get most likely token from base model
        most_likely_token = base_last.argmax()
        
        # Get probabilities for this token
        base_prob = base_last[most_likely_token]
        quant_prob = quant_last[most_likely_token]
        
        # Calculate probability ratio
        ratio = (quant_prob / (base_prob + 1e-10))
        
        # Calculate robustness
        factual_rob = ratio.item() - self.stl_thresholds['factual']
        
        return factual_rob


# Integration function to replace STL evaluation in toggle_dynamic_poc.py
def optimize_stl_evaluation(toggle_framework):
    """
    Replace the STL evaluation function in the toggle framework
    with the GPU-optimized version.
    
    Args:
        toggle_framework: DynamicPrecisionTransformer instance
        
    Returns:
        GPU-optimized STL evaluator
    """
    # Create GPU-optimized evaluator
    gpu_evaluator = GPUOptimizedSTLEvaluator(toggle_framework)
    
    # Replace the evaluate_stl_properties method
    toggle_framework.evaluate_stl_properties = gpu_evaluator.evaluate_stl_properties
    
    # Verify GPU utilization
    with torch.no_grad():
        # Create a dummy input to warm up GPU
        dummy_input = toggle_framework.encoded_dataset[0]
        _ = toggle_framework.base_model(dummy_input)
        
        # Check GPU utilization metrics
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        print(f"GPU memory usage: {gpu_memory:.2f} GB")
        
        # Force synchronization to accurate measure GPU usage
        torch.cuda.synchronize()
    
    print("STL evaluation GPU-optimized successfully!")
    return gpu_evaluator