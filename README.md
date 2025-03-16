# TOGGLE Framework

**TOGGLE** (Transformer Optimization with Guaranteed Logic Evaluation) is a dynamic precision framework for optimizing transformer-based language models through quantization and pruning while preserving model quality guarantees through Signal Temporal Logic (STL) evaluation.

## Overview

Large Language Models (LLMs) have shown remarkable capabilities but often come with significant computational requirements. The TOGGLE framework addresses this challenge by:

1. Applying variable precision and pruning across different components of transformer models
2. Using Signal Temporal Logic (STL) to formalize and guarantee model quality requirements
3. Finding optimal compression configurations through systematic exploration
4. Providing GPU-optimized evaluation for efficient processing

## Key Components

The framework consists of several integrated modules:

- **Core Framework** (`toggle_dynamic_poc.py`): Implements quantization functions, model configuration, and STL property evaluation
- **GPU Optimization** (`toggle_gpu_optimization.py`): Accelerates STL evaluation with GPU-optimized implementations
- **Sensitivity Analysis** (`toggle_sensitivity_analysis.py`): Identifies which layers can be compressed more aggressively
- **Progressive Compression** (`toggle_progressive_approach_fix.py`): Implements layer-by-layer compression strategy with guarantees
- **Results Organization** (`toggle_results_organization.py`): Manages storage and visualization of results
- **Pareto Analysis** (`toggle_pareto_analyzer.py`): Analyzes and visualizes tradeoffs between model size and quality

## STL Properties

TOGGLE evaluates four key properties of a compressed model compared to the base model:

1. **Coherence**: Measures distribution similarity of token predictions
2. **Attention**: Evaluates consistency of attention patterns
3. **Context**: Measures similarity of context representations
4. **Factual**: Ensures factual correctness is preserved

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/toggle-framework.git
cd toggle-framework

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- CUDA-capable GPU (recommended: A100, V100, or similar)
- 16GB+ GPU memory for small models, 80GB+ for larger models

## Usage

### Basic Optimization

```bash
python toggle-gpu/toggle_progressive_integration_organized.py \
  --model gpt2 \
  --iterations 100 \
  --output my_optimization
```

### With Advanced Options

```bash
python toggle-gpu/toggle_progressive_integration_organized.py \
  --model deepseek-ai/deepseek-llm-7b-base \
  --iterations 150 \
  --relax-thresholds 0.1 \
  --use-real-datasets \
  --dataset-subset-size 20 \
  --output deepseek_optimization
```

### Command Line Arguments

- `--model`: Model name/path from Hugging Face (default: gpt2)
- `--iterations`: Maximum optimization iterations (default: 100)
- `--output`: Output prefix for results (default: progressive_opt)
- `--skip-sensitivity`: Skip sensitivity analysis if set
- `--relax-thresholds`: Amount to relax STL thresholds (0.0-0.2, default: 0.0)
- `--results-dir`: Directory for storing results (default: results)
- `--use-real-datasets`: Use real datasets instead of toy dataset
- `--dataset-subset-size`: Number of examples from each dataset (default: 50)

## Results Analysis

After optimization, review the analysis in the results directory:

```bash
# Generate Pareto front analysis
python toggle-gpu/toggle_pareto_analyzer.py --full

# View specific model results
python toggle-gpu/toggle_pareto_analyzer.py --models gpt2 deepseek-ai/deepseek-llm-7b-base
```

## Memory Optimization

For larger models (>7B parameters), the framework includes memory optimization techniques:

1. Sampling-based quantile calculation for pruning
2. Batch processing for evaluation
3. Efficient tensor operations with GPU pinning

## Example Workflow

1. **Analyze layer sensitivity** to determine which layers can be compressed more
2. **Run progressive compression** to find optimal configuration
3. **Evaluate STL properties** to ensure quality guarantees
4. **Analyze Pareto front** to understand model size vs. quality tradeoffs
5. **Apply optimal configuration** for deployment

## Customization

You can customize STL thresholds in `toggle_dynamic_poc.py`:

```python
self.stl_thresholds = {
    'coherence': 0.15,   # Max JSD between token distributions
    'attention': 0.75,   # Min cosine similarity between attention maps
    'context': 0.80,     # Min cosine similarity between embeddings
    'factual': 0.85      # Min probability ratio for factual correctness
}
```

## Citation

If you use TOGGLE in your research, please cite:

```
@article{toggle2025,
  title={TOGGLE: Transformer Optimization with Guaranteed Logic Evaluation},
  author={Khurram Khalil, Khaza Hoque},
  journal={arXiv preprint},
  year={2025}
}
```

## License

[MIT License](LICENSE)

## Acknowledgements

This framework builds upon several open-source projects:
- Hugging Face Transformers
- PyTorch
- Meta's Quantization Implementation

## Contributing

Contributions are welcome! Please check the issues page for current tasks or create a new issue to discuss potential improvements.