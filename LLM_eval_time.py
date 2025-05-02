import torch
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import gc
from transformers import AutoTokenizer, OPTForCausalLM, AutoConfig
from torch.cuda.amp import autocast
from typing import Dict, List, Tuple
import os

class ProfilingHook:
    def __init__(self, layer_name: str, layer_idx: int, hook_type: str, device_type: str = "cuda"):
        self.layer_name = layer_name
        self.layer_idx = layer_idx
        self.hook_type = hook_type
        self.execution_times = []
        self.is_active = False
        self.device_type = device_type
        
        if device_type == "cuda":
            # CUDA events for precise GPU timing
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            # For CPU, we'll use time.time()
            self.start_time = 0
        
    def pre_hook(self, module, input):
        if self.is_active:
            if self.device_type == "cuda":
                # Record the start event
                torch.cuda.synchronize()  # Ensure all previous CUDA operations are completed
                self.start_event.record()
            else:
                # CPU timing
                self.start_time = time.perf_counter()  # More precise than time.time()
        
    def post_hook(self, module, input, output):
        if self.is_active:
            if self.device_type == "cuda":
                # Record the end event
                self.end_event.record()
                torch.cuda.synchronize()  # Wait for the end event to complete
                
                # Calculate elapsed time in milliseconds
                exec_time_ms = self.start_event.elapsed_time(self.end_event)
            else:
                # CPU timing
                end_time = time.perf_counter()  # More precise than time.time()
                exec_time_ms = (end_time - self.start_time) * 1000  # Convert to milliseconds
                
            self.execution_times.append(exec_time_ms)

def register_hooks_subset(model: OPTForCausalLM, device_type: str = "cuda", 
                          start_layer: int = 0, num_layers: int = None) -> Dict[str, ProfilingHook]:
    """Register hooks for a subset of attention and FFN layers in the model."""
    hooks = {}
    hook_handles = []
    
    # Get total number of layers
    total_layers = len(model.model.decoder.layers)
    
    # If num_layers is not specified, use all layers from start_layer
    if num_layers is None:
        num_layers = total_layers - start_layer
    
    # Ensure we don't exceed the total number of layers
    end_layer = min(start_layer + num_layers, total_layers)
    
    print(f"Profiling layers {start_layer} to {end_layer-1} out of {total_layers} total layers")
    
    # Iterate through the selected decoder layers
    for i in range(start_layer, end_layer):
        layer = model.model.decoder.layers[i]
        
        # Hook for attention layer
        attn_hook = ProfilingHook(f"layer_{i}_attention", i, "attention", device_type)
        attn_pre_handle = layer.self_attn.register_forward_pre_hook(attn_hook.pre_hook)
        attn_post_handle = layer.self_attn.register_forward_hook(attn_hook.post_hook)
        hook_handles.extend([attn_pre_handle, attn_post_handle])
        hooks[f"layer_{i}_attention"] = attn_hook
        
        # Hook for FFN (Feed Forward Network) - we'll measure both fc1 and fc2 together
        ffn_hook = ProfilingHook(f"layer_{i}_ffn", i, "ffn", device_type)
        ffn_pre_handle = layer.fc1.register_forward_pre_hook(ffn_hook.pre_hook)
        ffn_post_handle = layer.fc2.register_forward_hook(ffn_hook.post_hook)
        hook_handles.extend([ffn_pre_handle, ffn_post_handle])
        hooks[f"layer_{i}_ffn"] = ffn_hook
    
    return hooks, hook_handles

def run_profiling_with_low_memory(model_id: str = "facebook/opt-125m", 
                                 sequence_length: int = 128, 
                                 num_iterations: int = 3,
                                 batch_size: int = 1,
                                 use_fp16: bool = False,
                                 warmup_iterations: int = 2,
                                 use_cpu: bool = False,
                                 start_layer: int = 0,
                                 num_layers: int = 4,
                                 offload_to_disk: bool = False) -> pd.DataFrame:
    """
    Profile the OPT model's layer-wise execution times with optimizations for low memory devices.
    
    Args:
        model_id: The HuggingFace model ID for OPT (use small model like opt-125m)
        sequence_length: Input sequence length (reduced from original)
        num_iterations: Number of inference passes to profile (reduced)
        batch_size: Batch size for inference (keep at 1 for low memory)
        use_fp16: Whether to use FP16 precision
        warmup_iterations: Number of warmup iterations (reduced)
        use_cpu: Whether to use CPU instead of GPU
        start_layer: First layer to profile (for partial profiling)
        num_layers: Number of layers to profile (for partial profiling)
        offload_to_disk: Whether to offload model to disk between iterations
        
    Returns:
        DataFrame with profiling results
    """
    device = torch.device("cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Running profiling on CPU.")
        # Disable FP16 for CPU
        use_fp16 = False
    elif device.type != "cuda" and not use_cpu:
        print("Warning: CUDA not available. Profiling on CPU.")
        # Disable FP16 for CPU
        use_fp16 = False
    
    # Load the model configuration first to check size
    config = AutoConfig.from_pretrained(model_id)
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    
    print(f"Model info: {model_id}, Layers: {num_hidden_layers}, Hidden size: {hidden_size}")
    
    # Estimate approximate memory requirements (very rough estimate)
    approx_memory_gb = (num_hidden_layers * hidden_size * hidden_size * 4 * 4) / (1024**3)
    print(f"Approximate model memory: {approx_memory_gb:.2f} GB")
    
    # Memory optimization - Clean up before loading model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Memory optimization - Low precision
    torch_dtype = torch.float16 if use_fp16 and device.type == "cuda" else torch.float32
    
    # Load model with memory optimizations
    print(f"Loading model {model_id} with dtype {torch_dtype}...")
    
    model = OPTForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
    )
    
    model.to(device)
    model.eval()
    
    # Load tokenizer separately to manage memory
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Register hooks only for a subset of layers
    print("Registering hooks...")
    hooks, hook_handles = register_hooks_subset(
        model, 
        device.type, 
        start_layer=start_layer, 
        num_layers=num_layers
    )
    
    # Create small dummy input
    input_ids = torch.randint(100, 30000, (batch_size, sequence_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Minimal Warmup
    print(f"Warming up for {warmup_iterations} iterations...")
    with torch.no_grad():
        for _ in range(warmup_iterations):
            if device.type == "cuda" and use_fp16:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Free memory after each iteration
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    # Activate hooks for profiling
    for hook in hooks.values():
        hook.is_active = True
    
    # Results collector
    all_results = []
    
    # Run profiling iterations
    print(f"Running {num_iterations} profiling iterations...")
    with torch.no_grad():
        for i in range(num_iterations):
            print(f"Iteration {i+1}/{num_iterations}")
            
            if device.type == "cuda":
                # Clear cache between iterations to ensure consistent timing
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Perturb input slightly to avoid any caching effects
            if i > 0:
                # Small perturbation that won't significantly affect execution time
                input_ids = torch.randint(100, 30000, (batch_size, sequence_length), device=device)
                
            if use_fp16 and device.type == "cuda":
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            # Collect intermediate results after each iteration
            interim_results = []
            for name, hook in hooks.items():
                layer_idx = hook.layer_idx
                layer_type = hook.hook_type
                
                # Only process the latest result
                if hook.execution_times:
                    latest_exec_time = hook.execution_times[-1]
                    interim_results.append({
                        "layer_idx": layer_idx,
                        "layer_type": layer_type,
                        "iteration": i,
                        "execution_time_ms": latest_exec_time
                    })
                    
                    # Clear the execution times after collecting to save memory
                    hook.execution_times = [latest_exec_time]
            
            all_results.extend(interim_results)
            
            # Optional extreme memory saving - offload model between iterations
            if offload_to_disk and i < num_iterations - 1:
                print("Offloading model to disk to save memory...")
                torch.save(model.state_dict(), "temp_model.pt")
                del model
                torch.cuda.empty_cache() if device.type == "cuda" else None
                gc.collect()
                
                # Reload model for next iteration
                print("Reloading model...")
                model = OPTForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
                model.load_state_dict(torch.load("temp_model.pt"))
                model.to(device)
                model.eval()
                
                # Reapply hooks
                hooks, hook_handles = register_hooks_subset(
                    model, 
                    device.type, 
                    start_layer=start_layer, 
                    num_layers=num_layers
                )
                # Reactivate hooks
                for hook in hooks.values():
                    hook.is_active = True
                
                # Clean up temp file
                if os.path.exists("temp_model.pt"):
                    os.remove("temp_model.pt")
    
    # Clean up hooks
    for handle in hook_handles:
        handle.remove()
    
    # Clean up to free memory
    del model
    del tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    return df

def visualize_results(df: pd.DataFrame, save_path: str = "opt_profiling_results_low_mem.png"):
    """Visualize the profiling results with memory-efficient plotting."""
    # Calculate average execution time per layer and type
    avg_times = df.groupby(["layer_idx", "layer_type"])["execution_time_ms"].mean().reset_index()
    std_times = df.groupby(["layer_idx", "layer_type"])["execution_time_ms"].std().reset_index()
    
    # Merge average and std
    merged_df = pd.merge(avg_times, std_times, on=["layer_idx", "layer_type"], suffixes=('_mean', '_std'))
    
    # Create simpler plot to save memory
    plt.figure(figsize=(10, 6))
    
    # Plot by layer type
    for layer_type in df['layer_type'].unique():
        layer_data = merged_df[merged_df['layer_type'] == layer_type]
        plt.bar(
            [str(idx) + "-" + layer_type for idx in layer_data['layer_idx']], 
            layer_data['execution_time_ms_mean'],
            yerr=layer_data['execution_time_ms_std'],
            alpha=0.7,
            label=layer_type
        )
    
    plt.xlabel("Layer Index - Type")
    plt.ylabel("Execution Time (ms)")
    plt.title("OPT Model Layer-wise Execution Time")
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    
    return None

def main():
    # Configuration for low memory
    model_id = "facebook/opt-125m"  # Use smallest OPT model
    sequence_length = 64  # Reduce sequence length
    num_iterations = 3  # Reduce number of iterations
    batch_size = 1  # Keep batch size at 1
    use_fp16 = True  # Use FP16 to save memory
    use_cpu = False  # Use CPU if no GPU or not enough GPU memory
    start_layer = 0  # Start layer (for partial profiling)
    num_layers = 4   # Number of layers to profile (None to profile all from start_layer)
    offload_to_disk = False  # Set to True for extreme memory saving but slower execution
    
    # Parse command line arguments if any
    import argparse
    parser = argparse.ArgumentParser(description='Profile OPT model layers with low memory footprint.')
    parser.add_argument('--model', type=str, default=model_id, help='Model ID from HuggingFace')
    parser.add_argument('--seq_len', type=int, default=sequence_length, help='Input sequence length')
    parser.add_argument('--iterations', type=int, default=num_iterations, help='Number of profiling iterations')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
    parser.add_argument('--fp16', action='store_true', default=use_fp16, help='Use FP16 precision')
    parser.add_argument('--cpu', action='store_true', default=use_cpu, help='Run on CPU instead of GPU')
    parser.add_argument('--start_layer', type=int, default=start_layer, help='First layer to profile')
    parser.add_argument('--num_layers', type=int, default=num_layers, help='Number of layers to profile (None for all)')
    parser.add_argument('--offload', action='store_true', default=offload_to_disk, help='Offload model to disk between iterations')
    args = parser.parse_args()
    
    # Check available memory before starting
    if not args.cpu and torch.cuda.is_available():
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_mem_free = gpu_mem_total - gpu_mem_reserved
        
        print(f"GPU Memory: Total {gpu_mem_total:.2f} GB, Reserved {gpu_mem_reserved:.2f} GB, "
              f"Allocated {gpu_mem_allocated:.2f} GB, Free ~{gpu_mem_free:.2f} GB")
        
        # Warn if memory might be too low
        if gpu_mem_free < 1.5:  # Less than 1.5GB free
            print("WARNING: Very low GPU memory available. Consider using --cpu option or a smaller model.")
    
    # Create timestamp for output files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    device_name = "cpu" if args.cpu else "gpu"
    # Remove facebook/ from model name for filename
    model_short_name = args.model.split('/')[-1] if '/' in args.model else args.model
    
    # Run profiling
    try:
        results_df = run_profiling_with_low_memory(
            model_id=args.model,
            sequence_length=args.seq_len,
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            use_fp16=args.fp16,
            use_cpu=args.cpu,
            start_layer=args.start_layer,
            num_layers=args.num_layers,
            offload_to_disk=args.offload
        )
        
        # Create output filenames
        base_filename = f"opt_{model_short_name}_layers{args.start_layer}-{args.start_layer+args.num_layers-1}_{device_name}_{timestamp}"
        
        # Save raw results (memory efficient - save immediately)
        raw_csv = f"{base_filename}_raw.csv"
        results_df.to_csv(raw_csv, index=False)
        print(f"Raw results saved to {raw_csv}")
        
        # Calculate basic statistics (memory efficient)
        stats = results_df.groupby(["layer_idx", "layer_type"])["execution_time_ms"].agg(
            ["mean", "std", "min", "max", "count"]
        ).reset_index()
        
        # Print summary
        print("\nProfiling Statistics Summary:")
        print(stats)
        
        # Save statistics
        stats_csv = f"{base_filename}_stats.csv"
        stats.to_csv(stats_csv, index=False)
        print(f"Statistics saved to {stats_csv}")
        
        # Calculate overall statistics by layer type
        print("\nSummary by layer type:")
        layer_type_stats = results_df.groupby("layer_type")["execution_time_ms"].agg(
            ["mean", "std", "min", "max", "sum"]
        )
        print(layer_type_stats)
        
        # Basic visualization (memory efficient)
        viz_path = f"{base_filename}_plot.png"
        visualize_results(results_df, save_path=viz_path)
        print(f"Visualization saved to {viz_path}")
        
        return results_df, stats, layer_type_stats, base_filename
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    # Clean up before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Display system info
    print("\n=== System Information ===")
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM: Total {memory.total / (1024**3):.2f} GB, Available {memory.available / (1024**3):.2f} GB")
    except ImportError:
        print("psutil not available for memory info")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA: Not available")
    
    print("\n=== Profiling Options ===")
    print("1. Profile smaller model (opt-125m) with 4 layers")
    print("2. Profile single layer of a larger model (opt-350m)")
    print("3. Run on CPU with larger model (opt-1.3b, single layer)")
    print("4. Custom configuration")
    
    choice = input("Choose an option (1-4, default=1): ").strip() or "1"
    
    if choice == "1":
        # Option 1: Default small model, few layers
        import sys
        sys.argv = [sys.argv[0], "--model", "facebook/opt-1.3b", "--seq_len", "64", 
                   "--start_layer", "0", "--num_layers", "4", "--iterations", "3"]
        main()
    elif choice == "2":
        # Option 2: Single layer of medium model
        import sys
        sys.argv = [sys.argv[0], "--model", "facebook/opt-350m", "--seq_len", "32", 
                   "--start_layer", "0", "--num_layers", "1", "--iterations", "3"]
        main()
    elif choice == "3":
        # Option 3: CPU with single layer of larger model
        import sys
        sys.argv = [sys.argv[0], "--model", "facebook/opt-1.3b", "--seq_len", "32", 
                   "--start_layer", "0", "--num_layers", "1", "--iterations", "2",
                   "--cpu"]
        main()
    elif choice == "4":
        # Option 4: Custom
        model = input("Model name (e.g., facebook/opt-125m): ").strip() or "facebook/opt-125m"
        seq_len = input("Sequence length (default=64): ").strip() or "64"
        start = input("Start layer (default=0): ").strip() or "0"
        num = input("Number of layers (default=2): ").strip() or "2"
        iters = input("Iterations (default=3): ").strip() or "3"
        cpu_option = input("Use CPU? (y/n, default=n): ").strip().lower() == 'y'
        
        import sys
        cmd = [sys.argv[0], "--model", model, "--seq_len", seq_len, 
               "--start_layer", start, "--num_layers", num, "--iterations", iters]
        if cpu_option:
            cmd.append("--cpu")
        sys.argv = cmd
        main()
    else:
        print("Invalid choice. Using default options.")
        main()