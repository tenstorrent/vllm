# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import fnmatch
import os
import shlex
import socket
import subprocess
import sys
import threading
import weakref

import cloudpickle
import yaml

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip
from vllm.utils.system_utils import get_mp_context, kill_process_tree
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.executor.abstract import UniProcExecutor

logger = init_logger(__name__)


def _validate_launch_from_rank0_host(mpi_args: str, host_ip: str) -> None:
    # Parse --map-by to locate rankfile and ensure rank 0 host matches host_ip.
    if not mpi_args:
        return
    try:
        argv = shlex.split(mpi_args)
    except Exception:
        argv = []
    mapby_path = None
    for i, tok in enumerate(argv):
        if tok.startswith("--map-by"):
            # Support either --map-by=<VALUE> or "--map-by <VALUE>"
            if "=" in tok:
                value = tok.split("=", 1)[1]
            elif i + 1 < len(argv):
                value = argv[i + 1]
            else:
                value = ""
            # Extract file=... from VALUE like "rankfile:file=/path"
            if "file=" in value:
                mapby_path = value.split("file=", 1)[1].split(",", 1)[0].strip()
            # If value itself is a path, accept it as fallback
            if not mapby_path and value and os.path.isfile(value):
                mapby_path = value
            break
    if not (mapby_path and os.path.isfile(mapby_path)):
        return

    rank0_host = None
    try:
        with open(mapby_path) as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                # Format: rank 0=HOST slot=...
                tokens = s.split()
                if len(tokens) >= 2 and tokens[0] == "rank" and "=" in tokens[1]:
                    # left: "0", right: "HOST"
                    left, right = tokens[1].split("=", 1)
                    try:
                        rnum = int(left)
                    except Exception:
                        rnum = -1
                    if rnum == 0:
                        rank0_host = right.strip()
                        break
    except Exception:
        rank0_host = None
    if not rank0_host:
        return

    resolved_ips: set[str] = set()
    # If rank0_host is already an IP address, add it directly
    if all(c.isdigit() or c == "." for c in rank0_host):
        resolved_ips.add(rank0_host)
    # Also try to resolve hostname to IP addresses
    try:
        info = socket.getaddrinfo(rank0_host, None, proto=socket.IPPROTO_TCP)
        for ai in info:
            resolved_ips.add(ai[4][0])
    except Exception as e:
        logger.warning(
            "Failed to resolve IP address for rank 0 host %s: %s", rank0_host, e
        )
        # If resolution failed and rank0_host is not an IP, we can't validate
        if not resolved_ips:
            return
    assert host_ip in resolved_ips, (
        f"MPI rank 0 host {rank0_host} from rankfile {mapby_path} "
        f"(resolves to {sorted(resolved_ips)}) does not match "
        f"launcher host IP {host_ip} (must launch from rank 0 host "
        f"{rank0_host})."
    )
    logger.info("Validated launching from MPI rank 0 host %s", rank0_host)


def parse_tt_mpi_params(vllm_config: VllmConfig) -> tuple[str | None, int, int]:
    """
    Parse override_tt_config for a rank binding file (required for launching
    TT MPI processes), and compute MPI topology information.
    Returns tuple with:
      - rank_binding_file: str
      - mpi_world: int (number of MPI ranks / hosts)
      - dp_size_per_mpi_rank: int (DP ranks per host segment)
    """

    parallel_config = vllm_config.parallel_config
    assert parallel_config.data_parallel_backend != "ray", (
        "TT does not support ray-based data parallel backend"
    )
    dp_size = parallel_config.data_parallel_size
    override_tt_config = vllm_config.model_config.override_tt_config or {}
    rank_binding_file = override_tt_config.get("rank_binding")
    mpi_world = 0
    dp_size_per_mpi_rank = 0
    if rank_binding_file:
        if not isinstance(rank_binding_file, str):
            raise RuntimeError(
                "override_tt_config['rank_binding'] must be a non-empty string"
            )
        try:
            with open(rank_binding_file) as f:
                rb = yaml.safe_load(f)
            mpi_world = len(rb.get("rank_bindings", []))
        except Exception as e:
            raise RuntimeError(
                f"Failed to read rank binding '{rank_binding_file}': {e}"
            ) from e
        if mpi_world <= 0 or dp_size % mpi_world != 0:
            raise RuntimeError(
                f"data_parallel_size ({dp_size}) must be divisible by number "
                f"of device MPI ranks ({mpi_world})"
            )
        # Assume DP world is evenly split into mpi_world groups and set
        # device DP ranks as the first rank in each group.
        dp_size_per_mpi_rank = dp_size // mpi_world

    return rank_binding_file, mpi_world, dp_size_per_mpi_rank


def tt_run_launch(
    handshake_address: str,
    vllm_config: VllmConfig,
    rank_binding_file: str,
    log_stats: bool,
    cleanup_target: object,
):
    """
    Launch TT MPI processes via tt-run from rank 0.
    Uses args from override_tt_config:
      - rank_binding: str (required, already parsed)
      - mpi_args: str (optional, parsed here)
      - extra_ttrun_args: str (optional, parsed here)
      - config_pkl_dir: str (required, parsed here)
    Args:
        handshake_address: ZMQ address for engine handshake communication.
        vllm_config: Configuration object containing model and parallel config.
        rank_binding_file: Path to YAML file specifying MPI rank bindings.
        log_stats: Whether to enable statistics logging in engine processes.
        cleanup_target: Object whose lifecycle determines when to clean up
            the MPI subprocess. A finalizer will be set up automatically.
    """

    assert rank_binding_file and isinstance(rank_binding_file, str), (
        "rank_binding must be provided to tt_run_launch as a non-empty string"
    )

    # Parse override_tt_config for optional fields.
    override_tt_config = vllm_config.model_config.override_tt_config or {}
    mpi_args = override_tt_config.get("mpi_args", "")
    extra_ttrun_args = override_tt_config.get("extra_ttrun_args")
    cfg_dir = override_tt_config.get("config_pkl_dir")

    if not cfg_dir:
        raise RuntimeError(
            "override_tt_config['config_pkl_dir'] is required for TT MPI launch"
        )
    if not os.path.isdir(cfg_dir):
        raise RuntimeError("override_tt_config['config_pkl_dir'] must be a directory")

    host_ip = get_ip()
    # Data parallel master IP must be the same as the launcher host's IP since
    # the launcher binds the rendezvous port and sends it to all engines.
    parallel_config = vllm_config.parallel_config
    assert parallel_config.data_parallel_master_ip == host_ip, (
        f"data_parallel_master_ip {parallel_config.data_parallel_master_ip} "
        f"must be the same as the launcher host's IP {host_ip}"
    )
    # Launch must be done on host with MPI rank 0 since we are setting that
    # process's DP rank to 0 and torch distributed uses DP rank 0 to bind TCP
    # rendezvous endpoint.
    _validate_launch_from_rank0_host(mpi_args, host_ip)

    # Serialize vllm_config for remote engines to load.
    serialized_config_path = os.path.join(cfg_dir, "tmp_vllm_tt_cfg.pkl")
    with open(serialized_config_path, "wb") as tf:
        cloudpickle.dump(vllm_config, tf)

    # Create a temporary rank binding file that augments global_env with
    # any env_passthrough variables.
    with open(rank_binding_file) as f:
        rb = yaml.safe_load(f)
    rb.setdefault("global_env", {})
    # Whitelist-based env passthrough (patterns) to avoid copying full env.
    default_env_patterns = ["VLLM_*", "MESH_DEVICE"]
    env_passthrough = override_tt_config.get("env_passthrough", default_env_patterns)
    if isinstance(env_passthrough, (list, tuple)):
        to_inject = {}
        for key, val in os.environ.items():
            for pattern in env_passthrough:
                if fnmatch.fnmatch(key, pattern):
                    to_inject[key] = val
                    break
        # Do not override existing keys in global_env.
        for k, v in to_inject.items():
            rb["global_env"].setdefault(k, v)

    tmp_rb_path = os.path.join(cfg_dir, "tmp_vllm_tt_rank_binding.yaml")
    with open(tmp_rb_path, "w") as tf:
        yaml.safe_dump(rb, tf)

    normalized_extra_ttrun_args: list[str] = []
    if extra_ttrun_args:
        if not isinstance(extra_ttrun_args, str):
            raise RuntimeError(
                "override_tt_config['extra_ttrun_args'] must be a string"
            )
        normalized_extra_ttrun_args = shlex.split(extra_ttrun_args)

        # Prevent ambiguous/duplicated flags that vLLM is responsible for.
        reserved_flags = {"--rank-binding", "--mpi-args"}
        if any(
            (tok in reserved_flags)
            or any(tok.startswith(f"{flag}=") for flag in reserved_flags)
            for tok in normalized_extra_ttrun_args
        ):
            raise RuntimeError(
                "override_tt_config['extra_ttrun_args'] must not include "
                "--rank-binding or --mpi-args (vLLM sets these)"
            )

    cmd = ["tt-run"]
    if normalized_extra_ttrun_args:
        cmd.extend(normalized_extra_ttrun_args)
    cmd.extend(["--rank-binding", tmp_rb_path])
    if mpi_args:
        # Pass raw string; tt-run will shlex.split it
        cmd.extend(["--mpi-args", mpi_args])
    # Program to run per MPI rank: engine entrypoint with explicit args
    cmd.extend(
        [
            sys.executable,
            "-m",
            "vllm.v1.engine.tt_core_launcher",
            "--handshake",
            str(handshake_address),
            "--config-pkl",
            str(serialized_config_path),
            "--log-stats",
            ("1" if log_stats else "0"),
        ]
    )

    child_env = os.environ.copy()
    pretty_cmd = shlex.join(cmd)
    logger.info("Launching engines with tt-run: %s", pretty_cmd)
    mpi_proc = subprocess.Popen(cmd, env=child_env)

    # Set up finalizer for MPI subprocess cleanup
    _setup_mpi_proc_finalizer(mpi_proc, cleanup_target)


def _setup_mpi_proc_finalizer(
    mpi_proc: subprocess.Popen, cleanup_target: object
) -> None:
    """
    Set up a weakref finalizer to clean up MPI subprocess when cleanup_target
    is garbage collected. This ensures graceful shutdown of TT-MPI processes.
    Args:
        mpi_proc: The subprocess.Popen object for the tt-run process
        cleanup_target: Obj whose lifecycle decides when to clean up mpi_proc
    """

    def _finalize_mpi(proc_ref):
        proc = proc_ref()
        if proc is None:
            return
        # Check if process is already dead
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning(
                "tt-run subprocess did not exit within timeout, sending SIGKILL"
            )
            if proc.pid is not None:
                kill_process_tree(proc.pid)

    weakref.finalize(cleanup_target, _finalize_mpi, weakref.ref(mpi_proc))


def _spawn_local_non_device_dp_ranks(
    *,
    handshake_address: str,
    vllm_config: VllmConfig,
    log_stats: bool,
    mpi_rank: int,
    mpi_world: int,
) -> list[object]:
    """Spawn additional (non-device) DP ranks on this host.

    tt-run launches exactly one MPI rank per host (the "device" rank). The rest
    of the DP ranks for that host are spawned locally as plain subprocesses to
    keep the MPI world size equal to the number of hosts.

    We intentionally do **not** spawn any extra ranks on MPI rank 0, because the
    vLLM launcher process (running on host 0) already spawns those ranks locally
    so it can track their sentinels and surface failures early.
    """
    pc = vllm_config.parallel_config
    dp_size = pc.data_parallel_size
    if mpi_world <= 0:
        raise RuntimeError(f"Invalid mpi_world={mpi_world}")
    if dp_size % mpi_world != 0:
        raise RuntimeError(
            f"dp_size ({dp_size}) must be divisible by mpi_world ({mpi_world})"
        )
    segment = dp_size // mpi_world
    if segment <= 1:
        return []
    if mpi_rank == 0:
        return []

    # This MPI rank corresponds to the first DP rank in its segment.
    segment_start = mpi_rank * segment
    dp_ranks_to_spawn = list(range(segment_start + 1, segment_start + segment))
    if not dp_ranks_to_spawn:
        return []

    # local_dp_rank is per-host, so it is computed relative to segment_start.
    # The device rank on this host uses local_dp_rank=0, so spawned ranks map to
    # [1, segment-1].
    local_rank_map = {r: (r - segment_start) for r in dp_ranks_to_spawn}
    logger.info(
        "TT-MPI balanced launch (mpi_rank=%d/%d): spawning non-device DP ranks %s "
        "with local_dp_rank mapping %s (device dp_rank=%d local_dp_rank=0)",
        mpi_rank,
        mpi_world,
        dp_ranks_to_spawn,
        local_rank_map,
        segment_start,
    )

    ctx = get_mp_context()
    procs = []
    common_kwargs = {
        "vllm_config": vllm_config,
        # "Remote" from the launcher/front-end process on host 0.
        "local_client": False,
        "handshake_address": handshake_address,
        "executor_class": UniProcExecutor,
        "log_stats": log_stats,
    }

    for dp_rank in dp_ranks_to_spawn:
        # Preserve mixed-launch convention: device rank uses local_dp_rank=0,
        # and spawned ranks use local_dp_rank in [1, segment-1] for this host.
        local_dp_rank = dp_rank - segment_start
        p = ctx.Process(
            target=EngineCoreProc.run_engine_core,
            name=f"EngineCore_DP{dp_rank}",
            kwargs=common_kwargs
            | {
                "dp_rank": dp_rank,
                "local_dp_rank": local_dp_rank,
            },
        )
        p.start()
        procs.append(p)

    # If any spawned process exits, terminate this MPI rank. This helps the
    # rank-0 launcher fail fast instead of waiting indefinitely for a missing
    # handshake from a remote engine.
    def _watch_children() -> None:
        try:
            import signal
            from multiprocessing import connection

            connection.wait([p.sentinel for p in procs])
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            # Best-effort: if something goes wrong here, let the main engine run.
            logger.exception("Failed while watching spawned non-device ranks")

    threading.Thread(target=_watch_children, daemon=True).start()
    return procs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TT engine core entrypoint")
    parser.add_argument("--handshake", required=True, help="Handshake address")
    parser.add_argument(
        "--config-pkl",
        required=True,
        dest="config_pkl",
        help="Path to serialized VllmConfig pickle",
    )
    parser.add_argument(
        "--log-stats",
        required=True,
        choices=["0", "1"],
        dest="log_stats",
        help="Enable stat logging (1) or disable (0)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    handshake_address = args.handshake
    config_pickle_path = args.config_pkl
    log_stats = args.log_stats == "1"

    # Derive MPI ranks if present (device ranks).
    has_mpi = "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ
    mpi_rank = int(
        os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0"))
    )
    mpi_world = int(
        os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "1"))
    )

    # Load vllm config.
    if not os.path.isfile(config_pickle_path):
        raise RuntimeError(
            f"Config file doesn't exist or isn't a file: {config_pickle_path}"
        )
    if os.path.islink(config_pickle_path):
        raise RuntimeError(
            f"Config file is a symlink (not allowed): {config_pickle_path}"
        )
    with open(config_pickle_path, "rb") as f:
        vllm_config: VllmConfig = cloudpickle.load(f)

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
    pc.data_parallel_rank_local = 0  # Device processes use local DP rank 0.

    # Ensure uniproc in engine process (worker inline).
    assert pc.distributed_executor_backend == "uni", (
        "TT MPI must be used with uniproc executor backend"
    )

    # Spawn the rest of the DP ranks for this host (if any) before entering the
    # device engine busy-loop (which blocks).
    _spawn_local_non_device_dp_ranks(
        handshake_address=handshake_address,
        vllm_config=vllm_config,
        log_stats=log_stats,
        mpi_rank=mpi_rank,
        mpi_world=mpi_world,
    )

    # Run engine core busy loop.
    # Local client is False since only non-device ranks are spawned in-process.
    EngineCoreProc.run_engine_core(
        vllm_config=vllm_config,
        local_client=False,
        handshake_address=handshake_address,
        executor_class=UniProcExecutor,
        log_stats=log_stats,
        dp_rank=pc.data_parallel_rank,
        local_dp_rank=pc.data_parallel_rank_local,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("tt_engine_core failed")
        sys.exit(1)
