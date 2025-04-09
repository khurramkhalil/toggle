import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_compressor import ModelCompressor
from config import CompressionConfig, QuantizationConfig, PruningConfig, VerificationConfig

# Example usage of the formal LLM compression framework

def main():
    # Load model and tokenizer
    model_name = "gpt2"  # Can be replaced with any HuggingFace model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Make sure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure compression
    config = CompressionConfig(
        quantization=QuantizationConfig(
            enabled=True,
            # Only quantize specific layers (first two layers in this example)
            # target_layers=["transformer.h.0", "transformer.h.1"],
            target_layers=["h.0", "h.1"],
            w_bits=4,  # 4-bit quantization
            weight_layerwise=False
        ),
        pruning=PruningConfig(
            enabled=True,
            method="l1",
            amount=0.3,  # Prune 30% of weights
            make_permanent=True
        ),
        verification=VerificationConfig(
            properties=[
                {"name": "cosine_similarity", "threshold": 0.85},
                # {"name": "topk_overlap", "k": 5, "min_overlap": 0.6},
                # {"name": "probability_deviation", "max_deviation": 0.1}
            ],
            max_violations=0  # No violations allowed
        ),
        verbose=True
    )
    
    # Create model compressor
    compressor = ModelCompressor(model, tokenizer, config)
    
    # Compress model
    compressed_model = compressor.compress()
    
    # Sample inputs for verification
    sample_inputs = [
        "The Eiffel Tower is located in the city of",
        "Quantum computing uses qubits instead of",
        "The theory of relativity was developed by"
    ]
    
    # Verify compression results
    verification_results = compressor.verify(sample_inputs)
    
    # Print verification summary
    print("\nVerification Results:")
    print(f"Total properties: {verification_results['summary']['total_properties']}")
    print(f"Satisfied: {verification_results['summary']['satisfied_properties']}")
    print(f"Violated: {verification_results['summary']['violated_properties']}")
    print(f"Verification passed: {verification_results['summary']['verification_passed']}")
    
    # Run full evaluation
    evaluation_results = compressor.evaluate(sample_inputs)
    
    # Print size reduction
    size_reduction = evaluation_results["size_reduction"]
    print("\nSize Reduction:")
    print(f"Original size: {size_reduction['original_mb']:.2f} MB")
    print(f"Compressed size: {size_reduction['compressed_mb']:.2f} MB")
    print(f"Reduction: {size_reduction['reduction_percent']:.1f}%")
    
    # Print sample generation comparison
    print("\nSample Text Generation:")
    sample = evaluation_results["sample_generation"]
    print(f"Input: {sample['input']}")
    print(f"Original: {sample['original_output']}")
    print(f"Compressed: {sample['compressed_output']}")


if __name__ == "__main__":
    main()