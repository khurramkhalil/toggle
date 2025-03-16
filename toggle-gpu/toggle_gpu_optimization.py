"""
GPU-optimized STL evaluation for TOGGLE framework.
This module replaces the original STL evaluation with GPU-optimized
implementations that fully utilize available hardware.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import random

def load_evaluation_datasets(dataset_names=None, subset_size=50):
    """
    Load datasets for STL evaluation.
    
    Args:
        dataset_names: List of dataset names to load (default: predefined set)
        subset_size: Number of examples to use from each dataset
        
    Returns:
        Dictionary of datasets by STL property
    """
    # Default dataset mapping
    default_datasets = {
        'coherence': ['lambada'],
        'attention': ['hotpot_qa'],
        'context': ['coqa'],
        'factual': ['truthful_qa']
    }
    
    datasets = {}
    
    # Load specified datasets or defaults
    dataset_map = dataset_names or default_datasets
    
    for property_name, dataset_list in dataset_map.items():
        datasets[property_name] = []
        
        for dataset_name in dataset_list:
            try:
                # Load dataset (using HuggingFace datasets)
                if dataset_name == 'lambada':
                    dataset = load_dataset("lambada", split='validation')
                elif dataset_name == 'hotpot_qa':
                    dataset = load_dataset("hotpot_qa", split='validation')
                elif dataset_name == 'coqa':
                    dataset = load_dataset("coqa", split='validation')
                elif dataset_name == 'truthful_qa':
                    dataset = load_dataset("truthful_qa", 'multiple_choice', split='validation')
                else:
                    dataset = load_dataset(dataset_name)
                    # Get appropriate split
                    split = 'validation' if 'validation' in dataset else 'test' if 'test' in dataset else 'train'
                    dataset = dataset[split]
                
                # Take subset
                if len(dataset) > subset_size:
                    indices = random.sample(range(len(dataset)), subset_size)
                    subset = dataset.select(indices)
                else:
                    subset = dataset
                
                # Add to property-specific datasets
                datasets[property_name].append({
                    'name': dataset_name,
                    'data': subset
                })
                
                print(f"Loaded {len(subset)} examples from {dataset_name} for {property_name}")
                
            except Exception as e:
                print(f"Error loading {dataset_name}: {str(e)}")
                print("Using toy dataset instead")
                # Create a fallback toy dataset for this property
                datasets[property_name].append({
                    'name': 'toy_dataset',
                    'data': None  # Will use the toggle's toy dataset
                })
    
    return datasets
class GPUOptimizedSTLEvaluator:
    """
    Optimized STL evaluator that leverages GPU acceleration.
    """
    
    def __init__(self, toggle_framework, use_real_datasets=False, subset_size=50):
        """
        Initialize the GPU-optimized STL evaluator.
        
        Args:
            toggle_framework: DynamicPrecisionTransformer instance
            use_real_datasets: Whether to use real datasets instead of toy dataset
            subset_size: Number of examples to use from each dataset
        """
        self.toggle = toggle_framework
        self.device = next(toggle_framework.base_model.parameters()).device
        self.stl_thresholds = toggle_framework.stl_thresholds
        
        # Load real datasets if requested
        self.use_real_datasets = use_real_datasets
        if use_real_datasets:
            print("Loading evaluation datasets...")
            self.datasets = load_evaluation_datasets(subset_size=subset_size)
            # Process datasets
            self.processed_datasets = self._process_datasets(self.datasets)
            print("Datasets loaded and processed")
        else:
            self.datasets = None
            self.processed_datasets = None
            print("Using toy dataset for evaluation")
        
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
        # If using real datasets
        if self.use_real_datasets and self.processed_datasets:
            return self._evaluate_with_real_datasets(model)
        
        # Otherwise use the toy dataset
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
    
    def _process_datasets(self, datasets):
        """Process datasets for efficient evaluation"""
        processed = {}
        
        # Process coherence datasets
        if 'coherence' in datasets:
            processed['coherence'] = []
            for dataset_info in datasets['coherence']:
                if dataset_info['name'] == 'lambada':
                    # Process LAMBADA dataset
                    examples = []
                    for item in dataset_info['data']:
                        text = item['text']
                        # Get all but the last token for context
                        context = ' '.join(text.split()[:-1])
                        target = text.split()[-1]
                        
                        # Tokenize
                        inputs = self.toggle.tokenizer(context, return_tensors="pt").to(self.device)
                        target_id = self.toggle.tokenizer.encode(' ' + target)[1]  # Get ID of the target token
                        
                        examples.append({
                            'inputs': inputs,
                            'target_id': target_id,
                            'full_text': text
                        })
                    processed['coherence'].append({
                        'name': dataset_info['name'],
                        'examples': examples
                    })
            
        # Process attention datasets
        if 'attention' in datasets:
            processed['attention'] = []
            for dataset_info in datasets['attention']:
                if dataset_info['name'] == 'hotpot_qa':
                    # Process HotpotQA dataset
                    examples = []
                    for item in dataset_info['data']:
                        context = ' '.join(item['context']['sentences'])
                        question = item['question']
                        answer = item['answer']
                        
                        # Tokenize
                        inputs = self.toggle.tokenizer(context + ' ' + question, return_tensors="pt").to(self.device)
                        
                        examples.append({
                            'inputs': inputs,
                            'answer': answer,
                            'question': question,
                            'context': context
                        })
                    processed['attention'].append({
                        'name': dataset_info['name'],
                        'examples': examples
                    })
        
        # Process context datasets
        if 'context' in datasets:
            processed['context'] = []
            for dataset_info in datasets['context']:
                if dataset_info['name'] == 'coqa':
                    # Process CoQA dataset
                    examples = []
                    for item in dataset_info['data']:
                        story = item['story']
                        questions = item['questions']
                        answers = item['answers']
                        
                        # Use first few Q&A pairs
                        for i in range(min(3, len(questions))):
                            question = questions[i]
                            answer = answers[i]
                            
                            # Tokenize
                            inputs = self.toggle.tokenizer(story + ' ' + question, return_tensors="pt").to(self.device)
                            
                            examples.append({
                                'inputs': inputs,
                                'answer': answer,
                                'question': question,
                                'story': story
                            })
                    processed['context'].append({
                        'name': dataset_info['name'],
                        'examples': examples
                    })
        
        # Process factual datasets
        if 'factual' in datasets:
            processed['factual'] = []
            for dataset_info in datasets['factual']:
                if dataset_info['name'] == 'truthful_qa':
                    # Process TruthfulQA dataset
                    examples = []
                    for item in dataset_info['data']:
                        question = item['question']
                        choices = item['mc1_targets']['choices']
                        labels = item['mc1_targets']['labels']
                        
                        # Find correct answer
                        correct_idx = labels.index(1) if 1 in labels else 0
                        correct_answer = choices[correct_idx]
                        
                        # Tokenize
                        inputs = self.toggle.tokenizer(question, return_tensors="pt").to(self.device)
                        
                        examples.append({
                            'inputs': inputs,
                            'choices': choices,
                            'correct_answer': correct_answer,
                            'question': question
                        })
                    processed['factual'].append({
                        'name': dataset_info['name'],
                        'examples': examples
                    })
        
        return processed

    @torch.no_grad()
    def _evaluate_with_real_datasets(self, model):
        """Evaluate STL properties using real datasets"""
        all_robustness = {
            'coherence': [],
            'attention': [],
            'context': [],
            'factual': []
        }
        
        # Evaluate coherence using LAMBADA
        if 'coherence' in self.processed_datasets:
            for dataset in self.processed_datasets['coherence']:
                examples = dataset['examples']
                
                for example in examples:
                    inputs = example['inputs']
                    target_id = example['target_id']
                    
                    # Get model prediction
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits[:, -1], dim=-1)
                    
                    # Check probability of correct token
                    prob_correct = probs[0, target_id].item()
                    
                    # Get base model prediction
                    base_outputs = self.toggle.base_model(**inputs)
                    base_logits = base_outputs.logits
                    base_probs = F.softmax(base_logits[:, -1], dim=-1)
                    
                    # Check probability of correct token in base model
                    base_prob_correct = base_probs[0, target_id].item()
                    
                    # Calculate coherence score (ratio of probabilities)
                    coherence_score = prob_correct / (base_prob_correct + 1e-10)
                    
                    # Calculate robustness
                    coherence_rob = coherence_score - self.stl_thresholds['coherence']
                    all_robustness['coherence'].append(coherence_rob)
        
        # Evaluate attention using HotpotQA (simplified)
        if 'attention' in self.processed_datasets:
            for dataset in self.processed_datasets['attention']:
                examples = dataset['examples']
                
                for example in examples:
                    inputs = example['inputs']
                    
                    # Get model outputs with attention
                    outputs = model(**inputs, output_attentions=True)
                    attentions = outputs.attentions
                    
                    # Get base model outputs with attention
                    base_outputs = self.toggle.base_model(**inputs, output_attentions=True)
                    base_attentions = base_outputs.attentions
                    
                    # Compare attention patterns
                    attention_similarities = []
                    for l in range(len(attentions)):
                        # Flatten attention maps
                        att_flat = attentions[l].view(-1)
                        base_att_flat = base_attentions[l].view(-1)
                        
                        # Calculate cosine similarity
                        similarity = F.cosine_similarity(att_flat.unsqueeze(0), base_att_flat.unsqueeze(0))
                        attention_similarities.append(similarity.item())
                    
                    # Average similarity across layers
                    avg_similarity = sum(attention_similarities) / len(attention_similarities)
                    
                    # Calculate robustness
                    attention_rob = avg_similarity - self.stl_thresholds['attention']
                    all_robustness['attention'].append(attention_rob)
        
        # Evaluate context using CoQA (simplified)
        if 'context' in self.processed_datasets:
            for dataset in self.processed_datasets['context']:
                examples = dataset['examples']
                
                for example in examples:
                    inputs = example['inputs']
                    
                    # Get model hidden states
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1][:, -1]  # Last token of last layer
                    
                    # Get base model hidden states
                    base_outputs = self.toggle.base_model(**inputs, output_hidden_states=True)
                    base_hidden_states = base_outputs.hidden_states[-1][:, -1]
                    
                    # Calculate cosine similarity
                    similarity = F.cosine_similarity(hidden_states, base_hidden_states)
                    
                    # Calculate robustness
                    context_rob = similarity.item() - self.stl_thresholds['context']
                    all_robustness['context'].append(context_rob)
        
        # Evaluate factual using TruthfulQA (simplified)
        if 'factual' in self.processed_datasets:
            for dataset in self.processed_datasets['factual']:
                examples = dataset['examples']
                
                for example in examples:
                    inputs = example['inputs']
                    choices = example['choices']
                    
                    # Get logits for question
                    outputs = model(**inputs)
                    logits = outputs.logits[:, -1]
                    
                    # Get base model logits
                    base_outputs = self.toggle.base_model(**inputs)
                    base_logits = base_outputs.logits[:, -1]
                    
                    # Get choice tokens
                    choice_ids = [self.toggle.tokenizer.encode(' ' + choice)[1] for choice in choices]
                    
                    # Get probabilities for choices
                    probs = F.softmax(logits, dim=-1)[0, choice_ids].cpu().numpy()
                    base_probs = F.softmax(base_logits, dim=-1)[0, choice_ids].cpu().numpy()
                    
                    # Calculate probability ratio
                    max_idx = np.argmax(base_probs)
                    prob_ratio = probs[max_idx] / (base_probs[max_idx] + 1e-10)
                    
                    # Calculate robustness
                    factual_rob = prob_ratio - self.stl_thresholds['factual']
                    all_robustness['factual'].append(factual_rob)
        
        # Calculate minimum robustness for each property
        min_robustness = {}
        for property_name, values in all_robustness.items():
            if values:
                min_robustness[property_name] = min(values)
            else:
                # Fallback to toy dataset evaluation if no examples
                if property_name == 'coherence':
                    min_robustness[property_name] = self._evaluate_coherence_toy(model)
                elif property_name == 'attention':
                    min_robustness[property_name] = self._evaluate_attention_toy(model)
                elif property_name == 'context':
                    min_robustness[property_name] = self._evaluate_context_toy(model)
                elif property_name == 'factual':
                    min_robustness[property_name] = self._evaluate_factual_toy(model)
        
        # Calculate average metrics
        avg_metrics = {}
        for property_name, values in all_robustness.items():
            if values:
                avg_metrics[property_name] = sum(values) / len(values)
            else:
                avg_metrics[property_name] = min_robustness[property_name]
        
        # Calculate overall STL score
        stl_score = min(min_robustness.values())
        
        return {
            'metrics': avg_metrics,
            'robustness': min_robustness,
            'stl_score': stl_score,
            'stl_satisfied': stl_score >= 0
        }

    @torch.no_grad()
    def _evaluate_coherence_toy(self, model):
        """Fallback to toy dataset for coherence evaluation"""
        return self._evaluate_sample(self.toggle.encoded_dataset[0], model)['coherence']

    @torch.no_grad()
    def _evaluate_attention_toy(self, model):
        """Fallback to toy dataset for attention evaluation"""
        return self._evaluate_sample(self.toggle.encoded_dataset[0], model)['attention']

    @torch.no_grad()
    def _evaluate_context_toy(self, model):
        """Fallback to toy dataset for context evaluation"""
        return self._evaluate_sample(self.toggle.encoded_dataset[0], model)['context']

    @torch.no_grad()
    def _evaluate_factual_toy(self, model):
        """Fallback to toy dataset for factual evaluation"""
        return self._evaluate_sample(self.toggle.encoded_dataset[0], model)['factual']

# Integration function to replace STL evaluation in toggle_dynamic_poc.py
def optimize_stl_evaluation(toggle_framework, use_real_datasets=False, subset_size=50):
    """
    Replace the STL evaluation function in the toggle framework
    with the GPU-optimized version.
    
    Args:
        toggle_framework: DynamicPrecisionTransformer instance
        use_real_datasets: Whether to use real datasets
        subset_size: Number of examples from each dataset to use
        
    Returns:
        GPU-optimized STL evaluator
    """
    # Create GPU-optimized evaluator
    gpu_evaluator = GPUOptimizedSTLEvaluator(
        toggle_framework, 
        use_real_datasets=use_real_datasets,
        subset_size=subset_size
    )
    
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