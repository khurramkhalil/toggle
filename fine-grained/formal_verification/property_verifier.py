import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

from .stl_specs import STLProperty, get_property


class PropertyVerifier:
    """
    Verifies formal properties between original and compressed models
    """
    
    def __init__(self, properties: List[STLProperty]):
        """
        Initialize with a list of properties to verify
        
        Args:
            properties: List of STLProperty instances
        """
        self.properties = properties
        
    def verify_all(
        self,
        original_model: PreTrainedModel,
        compressed_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        inputs: Union[List[str], Dict[str, torch.Tensor]],
        device: str = "cpu"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Verify all properties for the given inputs
        
        Args:
            original_model: Original (uncompressed) model
            compressed_model: Compressed model
            tokenizer: Tokenizer for text inputs
            inputs: List of text inputs or tokenized inputs
            device: Device to run models on
            
        Returns:
            Dictionary mapping property names to verification results
        """
        results = {}
        
        # Move models to device
        original_model = original_model.to(device)
        compressed_model = compressed_model.to(device)
        
        # Set models to evaluation mode
        original_model.eval()
        compressed_model.eval()
        
        # Tokenize inputs if they are strings
        if isinstance(inputs, list) and isinstance(inputs[0], str):
            encoded_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        else:
            encoded_inputs = inputs
            if isinstance(encoded_inputs, dict):
                encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
        
        # Verify each property
        for prop in self.properties:
            results[prop.name] = self._verify_property(
                prop, original_model, compressed_model, encoded_inputs
            )
            
        return results
    
    def _verify_property(
        self, 
        property: STLProperty,
        original_model: PreTrainedModel,
        compressed_model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Verify a single property
        
        Args:
            property: STLProperty to verify
            original_model: Original model
            compressed_model: Compressed model
            inputs: Tokenized inputs
            
        Returns:
            Dictionary with verification results
        """
        # Create appropriate signals for this property
        if property.name == "logit_cosine_similarity":
            signals = self._create_cosine_similarity_signals(original_model, compressed_model, inputs)
        elif property.name == "topk_overlap":
            signals = self._create_topk_overlap_signals(original_model, compressed_model, inputs)
        elif property.name == "probability_deviation":
            signals = self._create_probability_deviation_signals(original_model, compressed_model, inputs)
        elif property.name == "response_time":
            signals = self._create_response_time_signals(original_model, compressed_model, inputs)
        else:
            raise ValueError(f"Unknown property: {property.name}")
        
        # Evaluate property
        min_robustness, robustness_trace = property.evaluate(signals)
        
        # Create result dictionary
        result = {
            "satisfied": min_robustness >= 0,
            "min_robustness": min_robustness,
            "robustness_trace": robustness_trace,
            "signals": signals
        }
        
        return result
    
    def _create_cosine_similarity_signals(
        self,
        original_model: PreTrainedModel,
        compressed_model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, List]:
        """Create signals for cosine similarity property"""
        with torch.no_grad():
            # Get logits from both models
            orig_outputs = original_model(**inputs).logits
            comp_outputs = compressed_model(**inputs).logits
        
        # Create signals dictionary
        signals = {
            'time': [],
            'cos_sim': []
        }
        
        # Calculate cosine similarity for each position in the sequence
        for t in range(orig_outputs.shape[1]):
            for b in range(orig_outputs.shape[0]):  # Loop through batch dimension
                orig_vec = orig_outputs[b, t].float()
                comp_vec = comp_outputs[b, t].float()
                
                # Calculate cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    orig_vec.unsqueeze(0), 
                    comp_vec.unsqueeze(0)
                ).item()
                
                # Add to signals
                signals['time'].append(len(signals['time']))
                signals['cos_sim'].append(cos_sim)
        
        return signals
    
    def _create_topk_overlap_signals(
        self,
        original_model: PreTrainedModel,
        compressed_model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        k: int = 5
    ) -> Dict[str, List]:
        """Create signals for top-k overlap property"""
        with torch.no_grad():
            # Get logits from both models
            orig_outputs = original_model(**inputs).logits
            comp_outputs = compressed_model(**inputs).logits
        
        # Create signals dictionary
        signals = {
            'time': [],
            'overlap_ratio': []
        }
        
        # Calculate top-k overlap for each position in the sequence
        for t in range(orig_outputs.shape[1]):
            for b in range(orig_outputs.shape[0]):  # Loop through batch dimension
                # Get top-k indices for both models
                orig_topk = torch.topk(orig_outputs[b, t], k).indices.cpu().numpy()
                comp_topk = torch.topk(comp_outputs[b, t], k).indices.cpu().numpy()
                
                # Calculate overlap
                overlap = len(set(orig_topk) & set(comp_topk))
                overlap_ratio = overlap / k
                
                # Add to signals
                signals['time'].append(len(signals['time']))
                signals['overlap_ratio'].append(overlap_ratio)
        
        return signals
    
    def _create_probability_deviation_signals(
        self,
        original_model: PreTrainedModel,
        compressed_model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, List]:
        """Create signals for probability deviation property"""
        with torch.no_grad():
            # Get logits from both models
            orig_outputs = original_model(**inputs).logits
            comp_outputs = compressed_model(**inputs).logits
            
            # Convert to probabilities
            orig_probs = torch.nn.functional.softmax(orig_outputs, dim=-1)
            comp_probs = torch.nn.functional.softmax(comp_outputs, dim=-1)
        
        # Create signals dictionary
        signals = {
            'time': [],
            'prob_diff': []
        }
        
        # Calculate max probability difference for each position
        for t in range(orig_outputs.shape[1]):
            for b in range(orig_outputs.shape[0]):  # Loop through batch dimension
                # Calculate max absolute difference
                max_diff = torch.max(torch.abs(orig_probs[b, t] - comp_probs[b, t])).item()
                
                # Add to signals
                signals['time'].append(len(signals['time']))
                signals['prob_diff'].append(max_diff)
        
        return signals
    
    def _create_response_time_signals(
        self,
        original_model: PreTrainedModel,
        compressed_model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, List]:
        """Create signals for response time property"""
        signals = {
            'time': [],
            'time_ratio': []
        }
        
        # Measure inference time for both models
        for _ in range(5):  # Multiple runs for stability
            # Time original model
            start_time = time.time()
            with torch.no_grad():
                _ = original_model(**inputs)
            orig_time = time.time() - start_time
            
            # Time compressed model
            start_time = time.time()
            with torch.no_grad():
                _ = compressed_model(**inputs)
            comp_time = time.time() - start_time
            
            # Calculate time ratio
            time_ratio = comp_time / orig_time if orig_time > 0 else 1.0
            
            # Add to signals
            signals['time'].append(len(signals['time']))
            signals['time_ratio'].append(time_ratio)
        
        return signals
        
    def summarize_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize verification results across all properties
        
        Args:
            results: Dictionary mapping property names to verification results
            
        Returns:
            Summary dictionary
        """
        properties_count = len(results)
        satisfied_count = sum(1 for prop, res in results.items() if res["satisfied"])
        min_robustness = min(res["min_robustness"] for res in results.values())
        
        return {
            "total_properties": properties_count,
            "satisfied_properties": satisfied_count,
            "violated_properties": properties_count - satisfied_count,
            "all_satisfied": satisfied_count == properties_count,
            "min_robustness": min_robustness
        }
    
    def get_robustness_bounds(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Get the robustness bounds for each property
        
        Args:
            results: Dictionary mapping property names to verification results
            
        Returns:
            Dictionary mapping property names to robustness bounds
        """
        bounds = {}
        
        for prop_name, result in results.items():
            bounds[prop_name] = result["min_robustness"]
            
        return bounds
    
    def plot_robustness(self, results: Dict[str, Dict[str, Any]], property_name: str):
        """
        Plot the robustness trace for a specific property
        
        Args:
            results: Dictionary mapping property names to verification results
            property_name: Name of the property to plot
            
        Returns:
            Matplotlib figure (if matplotlib is available)
        """
        try:
            import matplotlib.pyplot as plt
            
            if property_name not in results:
                raise ValueError(f"Property {property_name} not found in results")
                
            result = results[property_name]
            trace = result["robustness_trace"]
            
            times = [t for t, _ in trace]
            robustness = [r for _, r in trace]
            
            plt.figure(figsize=(10, 5))
            plt.plot(times, robustness)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.xlabel('Time Step')
            plt.ylabel('Robustness')
            plt.title(f'Robustness Trace for {property_name}')
            plt.grid(True, alpha=0.3)
            
            return plt.gcf()
        except ImportError:
            print("Matplotlib is not available for plotting")
            return None