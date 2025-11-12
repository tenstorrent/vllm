# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from typing import Optional
import subprocess
import tempfile
import yaml
import cloudpickle
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.abstract import UniProcExecutor

logger = init_logger(__name__)

def parse_tt_mpi_params(vllm_config: VllmConfig) -> tuple[str, set[int], set[int]]:
    """
    Parse override_tt_config for a rank binding file (required for launching 
    TT MPI processes), and compute device and local DP ranks.
    Returns tuple with:
      - rank_binding_file: str
      - non_device_dp_ranks: set[int]
    """
    dp_size = vllm_config.parallel_config.data_parallel_size
    override_tt_config = vllm_config.model_config.override_tt_config or {}
    rank_binding_file = override_tt_config.get("rank_binding")
    non_device_dp_ranks: set[int] = set()
    if rank_binding_file:
        if not isinstance(rank_binding_file, str):
            raise RuntimeError(
                "override_tt_config['rank_binding'] must be a non-empty string")
        try:
            with open(rank_binding_file, "r") as f:
                rb = yaml.safe_load(f)
            mpi_world = len(rb.get("rank_bindings", []))
        except Exception as e:
            raise RuntimeError(
                f"Failed to read rank binding '{rank_binding_file}': {e}") from e
        if mpi_world <= 0 or dp_size % mpi_world != 0:
            raise RuntimeError(
                f"data_parallel_size ({dp_size}) must be divisible by number of "
                f"device MPI ranks ({mpi_world})")
        # Assume DP world is evenly split into mpi_world groups and set
        # device DP ranks as the first rank in each group.
        dp_size_per_mpi_rank = dp_size // mpi_world
        device_dp_ranks = {i * dp_size_per_mpi_rank  for i in range(mpi_world)}
        non_device_dp_ranks = {i for i in range(dp_size) if i not in device_dp_ranks}

    return rank_binding_file, non_device_dp_ranks

def tt_run_launch(handshake_address: str, vllm_config: VllmConfig,
                  rank_binding_file: str) -> None:
    """
    Launch TT MPI processes via tt-run from rank 0.
    Uses args from override_tt_config:
      - rank_binding: str (required, already parsed)
      - mpi_args: str (optional, parsed here)
      - config_pkl_dir: str (required for multi-host systems, parsed here)
    """

    assert rank_binding_file and isinstance(rank_binding_file, str), (
        "rank_binding must be provided to tt_run_launch as a non-empty string")
    
    # Parse override_tt_config for optional fields.
    override_tt_config = vllm_config.model_config.override_tt_config or {}
    mpi_args = override_tt_config.get("mpi_args", "")
    config_pkl_dir = override_tt_config.get("config_pkl_dir")

    # Serialize vllm_config for remote engines to load.
    dir_for_cfg = None
    if config_pkl_dir:
        if not os.path.isdir(config_pkl_dir):
            raise RuntimeError(
                "override_tt_config['config_pkl_dir'] must be a directory")
        dir_for_cfg = config_pkl_dir
    cfg_dir = dir_for_cfg if dir_for_cfg is not None else tempfile.gettempdir()
    serialized_config_path = os.path.join(cfg_dir, "tmp_vllm_tt_cfg.pkl")
    with open(serialized_config_path, "wb") as tf:
        cloudpickle.dump(vllm_config, tf)

    # Create a temporary rank binding file that augments global_env
    # with the handshake address and config pickle path.
    with open(rank_binding_file, "r") as f:
        rb = yaml.safe_load(f)
    rb.setdefault("global_env", {})
    rb["global_env"]["VLLM_TT_HANDSHAKE_ADDR"] = str(handshake_address)
    rb["global_env"]["VLLM_TT_CONFIG_PICKLE"] = str(serialized_config_path)
    tmp_rb_path = os.path.join(tempfile.gettempdir(), "tmp_vllm_tt_rank_binding.yaml")
    with open(tmp_rb_path, "w") as tf:
        yaml.safe_dump(rb, tf)

    cmd = ["tt-run", "--rank-binding", tmp_rb_path]
    if mpi_args:
        # Pass raw string; tt-run will shlex.split it
        cmd.extend(["--mpi-args", mpi_args])
    # Program to run per MPI rank: engine entrypoint
    cmd.extend([sys.executable, "-m", "vllm.v1.entrypoints.tt_engine_core"])

    child_env = os.environ.copy()
    logger.info("Launching engines with tt-run: %s", " ".join(cmd))
    subprocess.Popen(cmd, env=child_env)

def _get_env(name: str, default: Optional[str] = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val

def main() -> None:
    # Read handshake address and config pickle path from env.
    handshake_address = _get_env("VLLM_TT_HANDSHAKE_ADDR")
    config_pickle_path = _get_env("VLLM_TT_CONFIG_PICKLE")

    # Load vllm config.
    with open(config_pickle_path, "rb") as f:
        vllm_config: VllmConfig = cloudpickle.load(f)

    # Ensure TT model classes are registered in this process (MPI rank).
    try:
        from examples.offline_inference_tt import register_tt_models
        register_tt_models()
    except Exception:
        # If examples module is unavailable, continue; custom registration may
        # be handled elsewhere.
        pass

    # Derive MPI ranks if present (device ranks).
    has_mpi = ("OMPI_COMM_WORLD_SIZE" in os.environ
               or "PMI_SIZE" in os.environ)
    mpi_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                                  os.environ.get("PMI_RANK", "0")))
    mpi_world = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                                   os.environ.get("PMI_SIZE", "1")))

    # Determine the global DP topology from the serialized config.
    pc = vllm_config.parallel_config
    dp_size = pc.data_parallel_size

    # Map MPI ranks (subset) deterministically into global DP ranks:
    # dp_rank = mpi_rank * (dp_size / mpi_world), assuming divisibility.
    # This picks the first local DP rank on each host.
    if not has_mpi:
        raise RuntimeError("tt_engine_core must be launched under MPI (tt-run)")
    assert dp_size % mpi_world == 0, (
        f"dp_size ({dp_size}) must be divisible by mpi_world ({mpi_world})"
    )
    segment = dp_size // mpi_world
    pc.data_parallel_rank = mpi_rank * segment
    pc.data_parallel_rank_local = 0  # device executor per host uses local dp rank 0

    # Ensure uniproc in engine process (worker inline).
    assert pc.distributed_executor_backend == "uni", (
        "TT MPI must be used with uniproc executor backend")

    # Run engine core busy loop.
    # Local client is False since only non-device ranks are spawned in-process.
    EngineCoreProc.run_engine_core(vllm_config=vllm_config,
                                   local_client=False,
                                   handshake_address=handshake_address,
                                   executor_class=UniProcExecutor,
                                   log_stats=False,
                                   dp_rank=pc.data_parallel_rank,
                                   local_dp_rank=pc.data_parallel_rank_local)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("tt_engine_core failed")
        sys.exit(1)


