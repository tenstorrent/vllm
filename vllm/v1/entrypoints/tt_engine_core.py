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

    # Derive DP ranks from MPI envs (OMPI or PMI).
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                              os.environ.get("PMI_RANK", "0")))
    world = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                               os.environ.get("PMI_SIZE", "1")))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK",
                                    os.environ.get("MPI_LOCALRANKID",
                                                   "0")))

    # Set DP ranks in config.
    pc = vllm_config.parallel_config
    pc.data_parallel_rank = rank
    pc.data_parallel_size = world
    pc.data_parallel_rank_local = local_rank

    # Ensure uniproc in engine process (worker inline).
    pc.distributed_executor_backend = "uni"

    # Run engine core busy loop.
    EngineCoreProc.run_engine_core(vllm_config=vllm_config,
                                   local_client=(rank == 0),
                                   handshake_address=handshake_address,
                                   executor_class=UniProcExecutor,
                                   log_stats=False,
                                   dp_rank=rank,
                                   local_dp_rank=local_rank)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("tt_engine_core failed")
        sys.exit(1)


