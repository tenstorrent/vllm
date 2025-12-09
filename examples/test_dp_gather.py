"""
Benchmark script comparing torch.distributed.all_gather_into_tensor vs
torch.distributed.gather on CPU with gloo and MPI backends.

Configuration:
- DP sizes: 2, 4, 8
- Gather only to rank 0
- Tensor sizes: (32, 2048), (32, 1024), (32, 128), (32, 4096)
- Backends: gloo (CPU) and MPI (CPU)

Usage:
- For gloo backend (single host): python test_dp_gather.py
- For gloo backend (multi-host via mpirun/ttrun): mpirun -n <DP_SIZE> python test_dp_gather.py
- For MPI backend (requires PyTorch compiled with MPI): mpirun -n <DP_SIZE> python test_dp_gather.py
  (will auto-detect and use MPI backend if available, otherwise falls back to gloo)

Note: When launched under mpirun/ttrun, the script can use gloo backend even if
PyTorch's MPI backend is not available. The processes are launched by mpirun,
but communication uses gloo backend.
"""

import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Tuple, Optional


def is_running_under_mpi() -> bool:
    """Check if we're running under mpirun/mpiexec."""
    # Check for common MPI environment variables
    mpi_vars = ["OMPI_COMM_WORLD_RANK", "PMI_RANK", "MPIRUN_RANK"]
    return any(var in os.environ for var in mpi_vars)


def get_mpi_rank_and_size() -> Optional[tuple[int, int]]:
    """Get rank and world_size from MPI environment if available."""
    # OpenMPI
    if "OMPI_COMM_WORLD_RANK" in os.environ and "OMPI_COMM_WORLD_SIZE" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        return rank, size
    # Intel MPI / PMI
    if "PMI_RANK" in os.environ and "PMI_SIZE" in os.environ:
        rank = int(os.environ["PMI_RANK"])
        size = int(os.environ["PMI_SIZE"])
        return rank, size
    # MPICH
    if "MPIRUN_RANK" in os.environ:
        # Try to get size from other env vars or assume it's set
        rank = int(os.environ["MPIRUN_RANK"])
        size = int(os.environ.get("MPIRUN_SIZE", "1"))
        return rank, size
    return None


def setup_distributed(rank: int, world_size: int, backend: str = "gloo"):
    """Initialize the distributed environment."""
    if backend == "gloo":
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
    elif backend == "mpi":
        # MPI backend requires running under mpirun/mpiexec
        # PyTorch can auto-detect rank/world_size from MPI when running under mpirun
        # First try without explicit rank/world_size (preferred method)
        try:
            dist.init_process_group(backend=backend)
            # Get the actual rank and world_size from the process group
            actual_rank = dist.get_rank()
            actual_world_size = dist.get_world_size()
            if actual_rank != rank or actual_world_size != world_size:
                print(f"Warning: MPI detected rank={actual_rank}, world_size={actual_world_size}, "
                      f"but expected rank={rank}, world_size={world_size}")
        except RuntimeError as e:
            # If auto-detection fails, try with explicit rank/world_size from environment
            mpi_info = get_mpi_rank_and_size()
            if mpi_info is None:
                error_msg = (
                    "MPI backend initialization failed. Possible causes:\n"
                    "1. PyTorch was not compiled with MPI support\n"
                    "2. MPI libraries are not available\n"
                    "3. Not running under mpirun/mpiexec\n\n"
                    f"Original error: {e}\n\n"
                    "To check if PyTorch has MPI support, run:\n"
                    "  python -c 'import torch.distributed as dist; print(dist.is_mpi_available())'"
                )
                raise RuntimeError(error_msg) from e
            
            mpi_rank, mpi_size = mpi_info
            try:
                dist.init_process_group(
                    backend=backend,
                    rank=mpi_rank,
                    world_size=mpi_size
                )
            except RuntimeError as e2:
                error_msg = (
                    "MPI backend initialization failed. Possible causes:\n"
                    "1. PyTorch was not compiled with MPI support\n"
                    "2. MPI libraries are not available\n\n"
                    f"Original error: {e2}\n\n"
                    "To check if PyTorch has MPI support, run:\n"
                    "  python -c 'import torch.distributed as dist; print(dist.is_mpi_available())'"
                )
                raise RuntimeError(error_msg) from e2
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def benchmark_gather(
    rank: int,
    world_size: int,
    tensor_size: Tuple[int, int] = (32, 2048),
    num_warmup: int = 10,
    num_iterations: int = 100,
):
    """Benchmark torch.distributed.gather (gathers to rank 0)."""
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Create input tensor on each rank
    input_tensor = torch.randn(*tensor_size, dtype=dtype, device=device)
    
    # Prepare gather list (only rank 0 needs output)
    gather_list = None
    if rank == 0:
        gather_list = [torch.empty_like(input_tensor) for _ in range(world_size)]
    
    # Warmup
    for _ in range(num_warmup):
        dist.gather(input_tensor, gather_list, dst=0)
    
    # Synchronize before timing
    dist.barrier()
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        dist.gather(input_tensor, gather_list, dst=0)
    dist.barrier()
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    avg_time_ms = (elapsed_time / num_iterations) * 1000
    
    return avg_time_ms


def benchmark_all_gather_into_tensor(
    rank: int,
    world_size: int,
    tensor_size: Tuple[int, int] = (32, 2048),
    num_warmup: int = 10,
    num_iterations: int = 100,
):
    """Benchmark torch.distributed.all_gather_into_tensor (all ranks get result)."""
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Create input tensor on each rank
    input_tensor = torch.randn(*tensor_size, dtype=dtype, device=device)
    
    # Create output tensor (all ranks get full result)
    output_tensor = torch.empty(
        (world_size * tensor_size[0], tensor_size[1]),
        dtype=dtype,
        device=device
    )
    
    # Warmup
    for _ in range(num_warmup):
        dist.all_gather_into_tensor(output_tensor, input_tensor)
    
    # Synchronize before timing
    dist.barrier()
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        dist.all_gather_into_tensor(output_tensor, input_tensor)
    dist.barrier()
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    avg_time_ms = (elapsed_time / num_iterations) * 1000
    
    return avg_time_ms


def run_benchmark(rank: int, world_size: int, backend: str):
    """Run benchmarks on a single rank."""
    setup_distributed(rank, world_size, backend=backend)
    
    # Tensor sizes to benchmark (sorted by second dimension)
    tensor_sizes = [
        (32, 128),
        (32, 1024),
        (32, 2048),
        (32, 4096),
    ]
    num_warmup = 10
    num_iterations = 100
    
    try:
        results = []
        
        for tensor_size in tensor_sizes:
            # Benchmark gather (only rank 0 receives)
            gather_time = benchmark_gather(
                rank, world_size, tensor_size, num_warmup, num_iterations
            )
            
            # Benchmark all_gather_into_tensor (all ranks receive, but we only care about rank 0)
            all_gather_time = benchmark_all_gather_into_tensor(
                rank, world_size, tensor_size, num_warmup, num_iterations
            )
            
            results.append((tensor_size, gather_time, all_gather_time))
        
        # Print results (only rank 0 prints to avoid clutter)
        if rank == 0:
            print("\n" + "=" * 80)
            print(f"Benchmark Results: gather vs all_gather_into_tensor ({backend.upper()} backend)")
            print("=" * 80)
            print(f"Configuration:")
            print(f"  - World size: {world_size}")
            print(f"  - Backend: {backend} (CPU)")
            print(f"  - Warmup iterations: {num_warmup}")
            print(f"  - Benchmark iterations: {num_iterations}")
            print("\nResults (average time per operation):")
            print("-" * 80)
            print(f"{'Tensor Size':<20} {'gather (ms)':<20} {'all_gather_into_tensor (ms)':<30} {'Speedup':<10}")
            print("-" * 80)
            
            for tensor_size, gather_time, all_gather_time in results:
                speedup = gather_time / all_gather_time if all_gather_time > 0 else float('inf')
                speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x"
                faster_method = "all_gather" if speedup > 1 else "gather"
                
                print(f"{str(tensor_size):<20} {gather_time:<20.4f} {all_gather_time:<30.4f} {speedup_str} ({faster_method})")
            
            print("=" * 80 + "\n")
    
    finally:
        cleanup_distributed()


def run_benchmark_gloo(rank: int, world_size: int):
    """Wrapper to run gloo benchmark."""
    run_benchmark(rank, world_size, backend="gloo")


def run_benchmark_mpi():
    """Run MPI benchmark - called directly when running under mpirun."""
    # Check if MPI backend is available in PyTorch
    if not dist.is_mpi_available():
        raise RuntimeError(
            "MPI backend is not available in this PyTorch installation.\n"
            "PyTorch must be compiled with MPI support to use the MPI backend.\n"
            "You can check availability with:\n"
            "  python -c 'import torch.distributed as dist; print(dist.is_mpi_available())'"
        )
    
    mpi_info = get_mpi_rank_and_size()
    if mpi_info is None:
        # Try to get rank/size from PyTorch after initialization
        # For now, use dummy values - setup_distributed will handle actual init
        rank = 0
        world_size = 4
    else:
        rank, world_size = mpi_info
    
    run_benchmark(rank, world_size, backend="mpi")


def run_benchmark_gloo_under_mpi():
    """Run gloo benchmark when launched under mpirun/ttrun."""
    # Get rank and world_size from MPI environment
    mpi_info = get_mpi_rank_and_size()
    if mpi_info is None:
        raise RuntimeError("Running under mpirun but could not detect MPI rank/size from environment")
    
    rank, world_size = mpi_info
    run_benchmark(rank, world_size, backend="gloo")


def main():
    """Main entry point."""
    # DP sizes to benchmark
    dp_sizes = [2, 4, 8]
    
    # Check if we're running under mpirun/ttrun
    if is_running_under_mpi():
        # Running under mpirun/ttrun - can use either MPI or gloo backend
        mpi_info = get_mpi_rank_and_size()
        if mpi_info is None:
            raise RuntimeError("Running under mpirun but could not detect MPI rank/size from environment")
        actual_rank, actual_world_size = mpi_info
        
        print(f"Detected MPI environment (rank={actual_rank}, world_size={actual_world_size})")
        print("=" * 80)
        
        # Check if MPI backend is available
        if dist.is_mpi_available():
            print("\nPyTorch MPI backend is available. Running MPI backend benchmark...")
            print("=" * 80)
            try:
                run_benchmark_mpi()
            except RuntimeError as e:
                print(f"\nError running MPI benchmark: {e}")
                print("\nFalling back to gloo backend...")
                print("=" * 80)
                run_benchmark_gloo_under_mpi()
        else:
            print("\nPyTorch MPI backend is not available.")
            print("Using gloo backend instead (processes launched by mpirun, communication via gloo).")
            print("=" * 80)
            run_benchmark_gloo_under_mpi()
    else:
        # Running normally - run gloo benchmark with mp.spawn for each DP size
        print("Starting benchmark for DP sizes: 2, 4, 8")
        print("This will compare gather vs all_gather_into_tensor on CPU with gloo backend.")
        print("\nNote: To run under mpirun/ttrun (e.g., for multi-host), use:")
        print("  mpirun -n <DP_SIZE> python test_dp_gather.py")
        print("  or")
        print("  ttrun --rank-binding <config.yaml> python test_dp_gather.py")
        print("=" * 80)
        
        # Benchmark gloo backend for each DP size
        for world_size in dp_sizes:
            print("\n" + "=" * 80)
            print(f"Running GLOO backend benchmark with {world_size} processes...")
            print("=" * 80)
            mp.spawn(
                run_benchmark_gloo,
                args=(world_size,),
                nprocs=world_size,
                join=True
            )


if __name__ == "__main__":
    main()
