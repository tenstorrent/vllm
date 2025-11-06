# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import pickle
import sys
from typing import Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine.core import EngineCoreProc
import cloudpickle
from vllm.v1.executor.abstract import UniProcExecutor

logger = init_logger(__name__)


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
    # This picks the first local DP rank on each host (e.g., 0 and dp_size/2).
    if has_mpi:
        assert dp_size % mpi_world == 0, (
            f"dp_size ({dp_size}) must be divisible by mpi_world ({mpi_world})"
        )
        segment = dp_size // mpi_world
        dp_rank = mpi_rank * segment
        dp_local = 0  # device executor per host uses local dp rank 0
    else:
        # Non-MPI ranks: keep values as provided by the front-end/launcher.
        dp_rank = pc.data_parallel_rank
        dp_local = pc.data_parallel_rank_local

    # Set DP ranks in config (do not change dp_size).
    pc.data_parallel_rank = dp_rank
    pc.data_parallel_rank_local = dp_local

    # Ensure uniproc in engine process (worker inline).
    pc.distributed_executor_backend = "uni"

    # Run engine core busy loop.
    EngineCoreProc.run_engine_core(vllm_config=vllm_config,
                                   local_client=False,
                                   handshake_address=handshake_address,
                                   executor_class=UniProcExecutor,
                                   log_stats=False,
                                   dp_rank=dp_rank,
                                   local_dp_rank=dp_local)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("tt_engine_core failed")
        sys.exit(1)


