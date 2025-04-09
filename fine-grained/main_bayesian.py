import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from datetime import datetime

from bayesian_optimizer import BayesianCompressionOptimizer
from model_compressor import ModelCompressor

def main():
    # Define model and tokenizer
    model_name = "gpt2"  # Can be any HuggingFace model: "facebook/opt-125m", "meta-llama/Llama-2-7b-hf", etc.
    print(f"Loading model: {model_name}")
    
    # Load model with appropriate dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Make sure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Sample inputs for evaluation
    sample_inputs = [
        "The Eiffel Tower is located in the city of",
        "Quantum computing uses qubits instead of",
        "The theory of relativity was developed by",
        "Machine learning algorithms require",
        "The capital of France is"
    ]
    
    # Define property configurations
    property_configs = [
        {"name": "cosine_similarity", "threshold": 0.85},
        # {"name": "topk_overlap", "k": 5, "min_overlap": 0.6},
    ]
    
    # Create Bayesian optimizer
    print("Initializing Bayesian optimization...")
    optimizer = BayesianCompressionOptimizer(
        model=model,
        tokenizer=tokenizer,
        sample_inputs=sample_inputs,
        property_configs=property_configs,
        n_calls=20,  # Number of optimization iterations
        verbose=True
    )
    
    # Run optimization
    print("Running Bayesian optimization...")
    best_config, best_model, best_score = optimizer.optimize()
    
    # Print optimization results
    print("\n========== Optimization Results ==========")
    print(f"Best score: {best_score:.4f}")
    print("Best configuration:")
    for key, value in optimizer._config_summary(best_config).items():
        print(f"  {key}: {value}")
    
    # Evaluate best model
    print("\n========== Best Model Evaluation ==========")
    compressor = ModelCompressor(model, tokenizer, best_config)
    evaluation = compressor.evaluate(sample_inputs)
    
    # Print size reduction
    size_reduction = evaluation["size_reduction"]
    print("\nSize Reduction:")
    print(f"Original size: {size_reduction['original_mb']:.2f} MB")
    print(f"Compressed size: {size_reduction['compressed_mb']:.2f} MB")
    print(f"Reduction: {size_reduction['reduction_percent']:.1f}%")
    
    # Print verification results
    verification = evaluation["verification"]
    print("\nVerification Results:")
    print(f"Total properties: {verification['summary']['total_properties']}")
    print(f"Satisfied: {verification['summary']['satisfied_properties']}")
    print(f"Violated: {verification['summary']['violated_properties']}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/optimization_{timestamp}.json", "w") as f:
        json.dump({
            "model": model_name,
            "best_score": best_score,
            "best_config": optimizer._config_summary(best_config),
            "size_reduction": {
                "original_mb": float(size_reduction['original_mb']),
                "compressed_mb": float(size_reduction['compressed_mb']),
                "reduction_percent": float(size_reduction['reduction_percent'])
            },
            "verification": {
                "total_properties": verification['summary']['total_properties'],
                "satisfied_properties": verification['summary']['satisfied_properties'],
                "violated_properties": verification['summary']['violated_properties']
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {results_dir}/optimization_{timestamp}.json")
    
    # Save best model
    model_dir = f"{results_dir}/best_model_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    best_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Best model saved to {model_dir}")


if __name__ == "__main__":
    main()
