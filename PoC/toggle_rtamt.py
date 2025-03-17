"""
Fixed STL Monitor for TOGGLE Framework
"""

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

class SimpleSTLMonitor:
    """
    A simplified STL monitor that doesn't rely on external libraries
    but still evaluates the core STL properties for LLM compression.
    """
    
    def __init__(self, stl_thresholds=None):
        """
        Initialize the STL monitor with specified thresholds.
        
        Args:
            stl_thresholds: Dictionary with thresholds for STL properties
        """
        self.stl_thresholds = stl_thresholds or {
            'coherence': 0.1,    # Max JSD between token distributions
            'attention': 0.8,    # Min cosine similarity between attention maps
            'context': 0.85,     # Min cosine similarity between embeddings
            'factual': 0.9       # Min probability ratio for factual correctness
        }
    
    def evaluate_coherence(self, base_probs, quant_probs):
        """
        Evaluate sequential coherence using JSD between token distributions
        
        Args:
            base_probs: Token probabilities from base model
            quant_probs: Token probabilities from quantized model
            
        Returns:
            Robustness value for coherence property
        """
        # Convert to numpy arrays if needed
        if isinstance(base_probs, torch.Tensor):
            base_probs = base_probs.cpu().numpy()
        if isinstance(quant_probs, torch.Tensor):
            quant_probs = quant_probs.cpu().numpy()
        
        # For simplicity, we'll compute JSD for the last token
        last_token_base_probs = base_probs[0, -1]
        last_token_quant_probs = quant_probs[0, -1]
        
        # Get top-p tokens (p=0.9)
        base_top_indices = np.argsort(last_token_base_probs)[::-1]
        quant_top_indices = np.argsort(last_token_quant_probs)[::-1]
        
        # Union of top indices that cover 90% of probability mass
        base_cumsum = np.cumsum(last_token_base_probs[base_top_indices])
        quant_cumsum = np.cumsum(last_token_quant_probs[quant_top_indices])
        
        base_top_p_indices = base_top_indices[base_cumsum <= 0.9]
        quant_top_p_indices = quant_top_indices[quant_cumsum <= 0.9]
        
        top_indices = np.union1d(base_top_p_indices, quant_top_p_indices)
        
        # Compute JSD only on these top tokens
        jsd = jensenshannon(
            last_token_base_probs[top_indices] / np.sum(last_token_base_probs[top_indices]),
            last_token_quant_probs[top_indices] / np.sum(last_token_quant_probs[top_indices])
        )
        
        # Return robustness value (threshold - jsd), positive means satisfied
        return self.stl_thresholds['coherence'] - jsd
    
    def evaluate_attention(self, base_attentions, quant_attentions):
        """
        Evaluate long-range dependencies using attention map similarity
        
        Args:
            base_attentions: Attention maps from base model
            quant_attentions: Attention maps from quantized model
            
        Returns:
            Robustness value for attention property
        """
        # Convert to numpy arrays if needed
        if isinstance(base_attentions[0], torch.Tensor):
            base_attentions = [att.cpu().numpy() for att in base_attentions]
        if isinstance(quant_attentions[0], torch.Tensor):
            quant_attentions = [att.cpu().numpy() for att in quant_attentions]
        
        # Compute cosine similarity between attention maps
        layer_sims = []
        for l in range(len(base_attentions)):
            base_att = base_attentions[l][0].flatten()
            quant_att = quant_attentions[l][0].flatten()
            
            # Compute cosine similarity
            sim = np.dot(base_att, quant_att) / (np.linalg.norm(base_att) * np.linalg.norm(quant_att))
            layer_sims.append(sim)
        
        # Average similarity across layers
        similarity = np.mean(layer_sims)
        
        # Return robustness value (sim - threshold), positive means satisfied
        return similarity - self.stl_thresholds['attention']
    
    def evaluate_context(self, base_embs, quant_embs):
        """
        Evaluate contextual consistency using embedding similarity
        
        Args:
            base_embs: Context embeddings from base model
            quant_embs: Context embeddings from quantized model
            
        Returns:
            Robustness value for context property
        """
        # Convert to numpy arrays if needed
        if isinstance(base_embs, torch.Tensor):
            base_embs = base_embs.cpu().numpy()
        if isinstance(quant_embs, torch.Tensor):
            quant_embs = quant_embs.cpu().numpy()
        
        # Extract embeddings for the last token
        if base_embs.ndim > 2:  # Handle batch dimension
            base_emb = base_embs[0, -1]
            quant_emb = quant_embs[0, -1]
        else:
            base_emb = base_embs[-1]
            quant_emb = quant_embs[-1]
        
        # Compute cosine similarity
        sim = np.dot(base_emb, quant_emb) / (np.linalg.norm(base_emb) * np.linalg.norm(quant_emb))
        
        # Return robustness value (sim - threshold), positive means satisfied
        return sim - self.stl_thresholds['context']
    
    def evaluate_factual(self, base_probs, quant_probs):
        """
        Evaluate factual accuracy using probability ratio
        
        Args:
            base_probs: Token probabilities from base model
            quant_probs: Token probabilities from quantized model
            
        Returns:
            Robustness value for factual property
        """
        # Convert to numpy arrays if needed
        if isinstance(base_probs, torch.Tensor):
            base_probs = base_probs.cpu().numpy()
        if isinstance(quant_probs, torch.Tensor):
            quant_probs = quant_probs.cpu().numpy()
        
        # Get most likely token from base model
        most_likely_token = base_probs[0, -1].argmax().item()
        base_prob = base_probs[0, -1, most_likely_token].item()
        quant_prob = quant_probs[0, -1, most_likely_token].item()
        
        # Compute ratio
        ratio = quant_prob / (base_prob + 1e-10)  # Avoid division by zero
        
        # Return robustness value (ratio - threshold), positive means satisfied
        return ratio - self.stl_thresholds['factual']
    
    def evaluate_all_properties(self, base_outputs, quant_outputs):
        """
        Evaluate all STL properties and return robustness values
        
        Args:
            base_outputs: Dictionary with base model outputs
            quant_outputs: Dictionary with quantized model outputs
            
        Returns:
            Dictionary with robustness values for each property
        """
        # Extract outputs
        base_probs = base_outputs['probs']
        quant_probs = quant_outputs['probs']
        base_attentions = base_outputs['attentions']
        quant_attentions = quant_outputs['attentions']
        base_hidden_states = base_outputs['hidden_states']
        quant_hidden_states = quant_outputs['hidden_states']
        
        # Evaluate each property
        coherence_rob = self.evaluate_coherence(base_probs, quant_probs)
        attention_rob = self.evaluate_attention(base_attentions, quant_attentions)
        context_rob = self.evaluate_context(base_hidden_states[-1], quant_hidden_states[-1])
        factual_rob = self.evaluate_factual(base_probs, quant_probs)
        
        # Return robustness values
        return {
            'coherence': coherence_rob,
            'attention': attention_rob,
            'context': context_rob,
            'factual': factual_rob
        }


def test_stl_monitor():
    """Test the STL monitor with synthetic data"""
    # Create STL monitor
    monitor = SimpleSTLMonitor()
    
    # Create synthetic data
    vocab_size = 1000
    seq_len = 10
    num_layers = 4
    hidden_dim = 768
    
    # Base model probabilities (random)
    base_probs = torch.softmax(torch.randn(1, seq_len, vocab_size), dim=-1)
    
    # Quantized model probabilities (slightly perturbed)
    quant_probs = torch.softmax(base_probs + 0.1 * torch.randn_like(base_probs), dim=-1)
    
    # Base model attention maps
    base_attentions = [torch.softmax(torch.randn(1, 1, seq_len, seq_len), dim=-1) for _ in range(num_layers)]
    
    # Quantized model attention maps (slightly perturbed)
    quant_attentions = [torch.softmax(att + 0.1 * torch.randn_like(att), dim=-1) for att in base_attentions]
    
    # Base model hidden states
    base_hidden = [torch.randn(1, seq_len, hidden_dim) for _ in range(num_layers+1)]
    
    # Quantized model hidden states (slightly perturbed)
    quant_hidden = [h + 0.1 * torch.randn_like(h) for h in base_hidden]
    
    # Test each property
    coherence_rob = monitor.evaluate_coherence(base_probs, quant_probs)
    attention_rob = monitor.evaluate_attention(base_attentions, quant_attentions)
    context_rob = monitor.evaluate_context(base_hidden[-1], quant_hidden[-1])
    factual_rob = monitor.evaluate_factual(base_probs, quant_probs)
    
    print("Coherence robustness:", coherence_rob)
    print("Attention robustness:", attention_rob)
    print("Context robustness:", context_rob)
    print("Factual robustness:", factual_rob)
    
    # Test evaluate_all_properties
    base_outputs = {
        'probs': base_probs,
        'attentions': base_attentions,
        'hidden_states': base_hidden
    }
    quant_outputs = {
        'probs': quant_probs,
        'attentions': quant_attentions,
        'hidden_states': quant_hidden
    }
    
    all_rob = monitor.evaluate_all_properties(base_outputs, quant_outputs)
    print("All robustness values:", all_rob)
    
    return all_rob

if __name__ == "__main__":
    test_stl_monitor()