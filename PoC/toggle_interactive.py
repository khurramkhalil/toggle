#!/usr/bin/env python3
"""
Interactive command-line tool for experimenting with TOGGLE dynamic precision configurations.
"""
import argparse
import json
import os
from toggle_dynamic_poc import DynamicPrecisionTransformer

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_config_summary(config):
    """Print a summary of the configuration"""
    # Count layers and components
    num_layers = len(config)
    total_components = sum(len(layer_config) for layer_config in config.values())
    
    # Calculate average bit-width and pruning
    total_bits = 0
    total_pruning = 0
    for layer_config in config.values():
        for comp_config in layer_config.values():
            total_bits += comp_config['bits']
            total_pruning += comp_config['pruning']
    
    avg_bits = total_bits / total_components
    avg_pruning = total_pruning / total_components * 100  # Convert to percentage
    
    print(f"Configuration summary:")
    print(f"  Number of layers: {num_layers}")
    print(f"  Total components: {total_components}")
    print(f"  Average bit-width: {avg_bits:.2f} bits")
    print(f"  Average pruning ratio: {avg_pruning:.2f}%")
    
    # Print bit-width distribution
    bit_counts = {}
    for layer_config in config.values():
        for comp_config in layer_config.values():
            bits = comp_config['bits']
            bit_counts[bits] = bit_counts.get(bits, 0) + 1
    
    print("\nBit-width distribution:")
    for bits, count in sorted(bit_counts.items()):
        percentage = count / total_components * 100
        print(f"  {bits}-bit: {count} components ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="TOGGLE Interactive Configuration Tool")
    parser.add_argument('--model', type=str, default='gpt2', help='Model name/path')
    parser.add_argument('--config', type=str, help='Path to initial configuration file')
    args = parser.parse_args()
    
    # Initialize the framework
    toggle = DynamicPrecisionTransformer(model_name=args.model)
    
    # Load initial configuration
    if args.config and os.path.exists(args.config):
        current_config = toggle.load_config(args.config)
    else:
        current_config = toggle.default_config
    
    toggle.current_config = current_config
    
    # Print welcome message
    print_header("TOGGLE Dynamic Precision Configuration Tool")
    print(f"Model: {args.model} ({toggle.num_layers} layers)")
    print(f"Components per layer: {', '.join(toggle.components)}")
    print("\nType 'help' for available commands")
    
    # Interactive loop
    last_results = None
    
    while True:
        try:
            cmd = input("\nTOGGLE> ").strip().lower()
            
            if cmd == 'exit' or cmd == 'quit':
                break
                
            elif cmd == 'help':
                print("\nAvailable commands:")
                print("  help              - Show this help message")
                print("  show              - Show current configuration")
                print("  summary           - Show configuration summary stats")
                print("  eval              - Evaluate current configuration")
                print("  random            - Generate a random configuration")
                print("  default           - Reset to default configuration")
                print("  set L C bits P    - Set layer L, component C to bits-width B with pruning P")
                print("                       Example: set 0 attn.c_attn 4 0.3")
                print("  uniform B         - Set all components to B bits")
                print("  save FILENAME     - Save configuration to file")
                print("  load FILENAME     - Load configuration from file")
                print("  exit/quit         - Exit the program")
                
            elif cmd == 'show':
                # Pretty print current configuration
                print("\nCurrent configuration:")
                print(json.dumps(current_config, indent=2))
                
            elif cmd == 'summary':
                print_config_summary(current_config)
                
            elif cmd == 'eval':
                print("\nEvaluating current configuration...")
                results = toggle.evaluate_config(current_config)
                last_results = results
                
                # Print key results
                print(f"\nSTL Score: {results['stl_score']:.4f} ({'Satisfied' if results['stl_score'] >= 0 else 'Violated'})")
                print(f"Model Size: {results['model_size']:.2f} MB")
                print(f"Inference Time: {results['inference_time']:.4f} seconds per sample")
                
                # Print metrics
                print("\nMetrics:")
                for metric, value in results['metrics'].items():
                    print(f"  {metric}: {value:.4f}")
                
                # Print robustness
                print("\nRobustness:")
                for metric, value in results['robustness'].items():
                    status = "✓" if value >= 0 else "✗"
                    print(f"  {metric}: {value:.4f} {status}")
                    
                # Ask if user wants to visualize
                viz = input("\nVisualize results? (y/n): ").strip().lower()
                if viz == 'y':
                    toggle.visualize_results(results)
                    
            elif cmd == 'random':
                current_config = toggle.create_random_config()
                toggle.current_config = current_config
                print("Generated random configuration")
                print_config_summary(current_config)
                
            elif cmd == 'default':
                current_config = toggle.create_default_config()
                toggle.current_config = current_config
                print("Reset to default configuration (all 16-bit, no pruning)")
                
            elif cmd.startswith('set '):
                # Parse command: set layer component bits pruning
                parts = cmd.split()
                if len(parts) != 5:
                    print("Error: 'set' command requires 4 arguments: layer, component, bits, pruning")
                    print("Example: set 0 attn.c_attn 4 0.3")
                    continue
                
                try:
                    layer_idx = int(parts[1])
                    component = parts[2]
                    bits = int(parts[3])
                    pruning = float(parts[4])
                    
                    # Validate inputs
                    if layer_idx < 0 or layer_idx >= toggle.num_layers:
                        print(f"Error: Layer index must be between 0 and {toggle.num_layers-1}")
                        continue
                        
                    if component not in toggle.components:
                        print(f"Error: Component must be one of: {', '.join(toggle.components)}")
                        continue
                        
                    if bits not in toggle.bit_options:
                        print(f"Error: Bit-width must be one of: {', '.join(map(str, toggle.bit_options))}")
                        continue
                        
                    if pruning < 0 or pruning > 0.9:
                        print("Error: Pruning ratio must be between 0 and 0.9")
                        continue
                    
                    # Update configuration
                    current_config = toggle.update_component_config(layer_idx, component, bits, pruning)
                    print(f"Updated layer {layer_idx}, component {component} to {bits} bits with {pruning*100:.1f}% pruning")
                    
                except ValueError:
                    print("Error: Invalid argument format")
                    
            elif cmd.startswith('uniform '):
                # Set uniform bit-width for all components
                try:
                    bits = int(cmd.split()[1])
                    
                    if bits not in toggle.bit_options:
                        print(f"Error: Bit-width must be one of: {', '.join(map(str, toggle.bit_options))}")
                        continue
                    
                    # Update all components
                    for layer_idx in range(toggle.num_layers):
                        for component in toggle.components:
                            current_config = toggle.update_component_config(layer_idx, component, bits, None)
                    
                    print(f"Set all components to {bits} bits (pruning unchanged)")
                    
                except (ValueError, IndexError):
                    print("Error: 'uniform' command requires a bit-width argument")
                    
            elif cmd.startswith('save '):
                # Save configuration to file
                try:
                    filename = cmd.split(maxsplit=1)[1]
                    toggle.save_config(current_config, filename)
                except IndexError:
                    print("Error: 'save' command requires a filename")
                    
            elif cmd.startswith('load '):
                # Load configuration from file
                try:
                    filename = cmd.split(maxsplit=1)[1]
                    if os.path.exists(filename):
                        current_config = toggle.load_config(filename)
                        toggle.current_config = current_config
                    else:
                        print(f"Error: File '{filename}' not found")
                except IndexError:
                    print("Error: 'load' command requires a filename")
                    
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nOperation interrupted")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nExiting TOGGLE configuration tool")


if __name__ == "__main__":
    main()