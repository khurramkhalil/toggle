import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import logging
from datetime import datetime

from botorch_optimizer import BoTorchOptimizer
from model_compressor import ModelCompressor

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("BotorchExample")
    
    # Define model and tokenizer
    model_name = "gpt2"  # Can be replaced with any HuggingFace model
    logger.info(f"Loading model: {model_name}")
    
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
        {"name": "topk_overlap", "k": 5, "min_overlap": 0.6},
    ]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "botorch_results"
    os.makedirs(results_dir, exist_ok=True)
    log_file = f"{results_dir}/botorch_optimization_{timestamp}.log"
    
    # Create BoTorch optimizer
    logger.info("Initializing BoTorch optimization...")
    optimizer = BoTorchOptimizer(
        model=model,
        tokenizer=tokenizer,
        sample_inputs=sample_inputs,
        property_configs=property_configs,
        n_iterations=15,
        initial_samples=5,
        verbose=True,
        log_file=log_file
    )
    
    # Run optimization
    logger.info("Running BoTorch optimization...")
    best_config, best_model, best_score = optimizer.optimize()
    
    # Print optimization results
    logger.info("\n========== Optimization Results ==========")
    logger.info(f"Best score: {best_score:.4f}")
    
    if best_config is None:
        logger.error("No valid configuration found!")
        return
    
    logger.info("Best configuration:")
    logger.info(f"  w_bits: {best_config.quantization.w_bits}")
    logger.info(f"  pruning_method: {best_config.pruning.method}")
    logger.info(f"  pruning_amount: {best_config.pruning.amount}")
    logger.info(f"  weight_layerwise: {best_config.quantization.weight_layerwise}")
    
    # Create a new model compressor with the best configuration
    logger.info("\nInitializing model compressor with best configuration...")
    compressor = ModelCompressor(model, tokenizer, best_config)
    
    # Explicitly compress the model
    logger.info("Compressing model with best configuration...")
    try:
        compressed_model = compressor.compress()
        logger.info("Compression successful!")
        
        # Print property verification results for best model
        logger.info("\nProperty Verification Results:")
        verification = compressor.verify(sample_inputs)
        for prop_name, result in verification.items():
            if prop_name != "summary":
                satisfied = "✓" if result["satisfied"] else "✗"
                robustness = result.get("min_robustness", 0)
                logger.info(f"  {prop_name}: {satisfied} (robustness={robustness:.4f})")
        
        logger.info(f"\nOverall: {verification['summary']['satisfied_properties']}/{verification['summary']['total_properties']} properties satisfied")
        logger.info(f"Minimum robustness across all properties: {verification['summary'].get('min_robustness', 0):.4f}")
        
        # Evaluate the compressed model
        logger.info("\n========== Best Model Evaluation ==========")
        evaluation = compressor.evaluate(sample_inputs)
        
        # Print size reduction
        size_reduction = evaluation["size_reduction"]
        logger.info("\nSize Reduction:")
        logger.info(f"Original size: {size_reduction['original_mb']:.2f} MB")
        logger.info(f"Compressed size: {size_reduction['compressed_mb']:.2f} MB")
        logger.info(f"Reduction: {size_reduction['reduction_percent']:.1f}%")
        
        # Save the compressed model
        model_dir = f"{results_dir}/best_model_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        compressed_model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        logger.info(f"Best model saved to {model_dir}")
        
        # Save optimization results to JSON
        results_file = f"{results_dir}/results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "model": model_name,
                "best_score": float(best_score),
                "best_config": {
                    "w_bits": best_config.quantization.w_bits,
                    "pruning_method": best_config.pruning.method,
                    "pruning_amount": float(best_config.pruning.amount),
                    "weight_layerwise": best_config.quantization.weight_layerwise
                },
                "size_reduction": {
                    "original_mb": float(size_reduction['original_mb']),
                    "compressed_mb": float(size_reduction['compressed_mb']),
                    "reduction_percent": float(size_reduction['reduction_percent'])
                },
                "verification": {
                    "total_properties": verification['summary']['total_properties'],
                    "satisfied_properties": verification['summary']['satisfied_properties'],
                    "violated_properties": verification['summary']['violated_properties'],
                    "min_robustness": float(verification['summary'].get('min_robustness', 0.0))
                },
                "timestamp": timestamp
            }, f, indent=2)
        logger.info(f"Results saved to {results_file}")
        
        # Analyze bit width and pruning method combinations
        logger.info("\nAnalysis of Parameter Combinations:")
        bit_width_success = optimizer.discrete_param_success_rate['bit_width']
        pruning_method_success = optimizer.discrete_param_success_rate['pruning_method']
        
        # Print bit width analysis
        logger.info("Bit Width Success Rates:")
        for bit_width, successes in sorted(bit_width_success.items()):
            if len(successes) > 0:
                success_rate = sum(successes) / len(successes)
                logger.info(f"  {int(bit_width)}-bit: {success_rate:.2f} ({sum(successes)}/{len(successes)} trials)")
        
        # Print pruning method analysis
        logger.info("\nPruning Method Success Rates:")
        for method, successes in sorted(pruning_method_success.items()):
            if len(successes) > 0:
                success_rate = sum(successes) / len(successes)
                logger.info(f"  {method}: {success_rate:.2f} ({sum(successes)}/{len(successes)} trials)")
        
    except Exception as e:
        logger.error(f"Error during compression or evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()