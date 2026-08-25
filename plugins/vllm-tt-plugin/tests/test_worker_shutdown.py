from types import SimpleNamespace

import pytest

from vllm_tt_plugin import worker as worker_module
from vllm_tt_plugin.model_runner import TTModelRunner


class _Runner:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    def shutdown(self):
        self.events.append("runner")
        if self.error:
            raise self.error


def _worker(events, mesh=object(), runner_error=None):
    worker = object.__new__(worker_module.TTWorker)
    worker.model_runner = _Runner(events, runner_error)
    worker.mesh_device = mesh
    worker.device_config = SimpleNamespace(device=mesh)
    worker.vllm_config = object()
    return worker


def test_worker_shutdown_is_ordered_and_idempotent(monkeypatch):
    events = []
    mesh = object()
    worker = _worker(events, mesh)
    monkeypatch.setattr(worker_module, "get_tt_config", lambda _cfg: "config")
    monkeypatch.setattr(
        worker_module,
        "close_mesh_device",
        lambda got_mesh, got_cfg: events.append(("mesh", got_mesh, got_cfg)),
    )

    worker.shutdown()
    worker.shutdown()

    assert events == ["runner", ("mesh", mesh, "config")]
    assert worker.model_runner is None
    assert worker.mesh_device is None
    assert worker.device_config.device is None


def test_worker_shutdown_attempts_mesh_close_after_runner_failure(monkeypatch):
    events = []
    failure = RuntimeError("model close failed")
    worker = _worker(events, runner_error=failure)
    monkeypatch.setattr(worker_module, "get_tt_config", lambda _cfg: None)
    monkeypatch.setattr(
        worker_module,
        "close_mesh_device",
        lambda _mesh, _cfg: events.append("mesh"),
    )

    with pytest.raises(RuntimeError, match="model close failed"):
        worker.shutdown()

    assert events == ["runner", "mesh"]
    worker.shutdown()  # ownership was cleared before the failed cleanup
    assert events == ["runner", "mesh"]


def test_non_device_rank_releases_runner_without_closing_mesh(monkeypatch):
    events = []
    worker = _worker(events, mesh=None)
    monkeypatch.setattr(
        worker_module,
        "close_mesh_device",
        lambda *_args: pytest.fail("non-device rank tried to close a mesh"),
    )

    worker.shutdown()

    assert events == ["runner"]


def test_model_runner_shutdown_calls_optional_model_close_once():
    events = []
    model = SimpleNamespace(close=lambda: events.append("model"))
    runner = object.__new__(TTModelRunner)
    runner.model = model
    runner.mesh_device = object()

    runner.shutdown()
    runner.shutdown()

    assert events == ["model"]
    assert runner.model is None
    assert runner.mesh_device is None


def test_close_mesh_attempts_all_steps_and_reraises_first_error(monkeypatch):
    events = []
    profiler_error = RuntimeError("profiler failed")
    mesh = SimpleNamespace(
        get_num_devices=lambda: 4,
        get_submeshes=lambda: ["submesh-a", "submesh-b"],
    )

    def read_profiler(_mesh):
        events.append("profiler")
        raise profiler_error

    monkeypatch.setattr(worker_module.ttnn, "ReadDeviceProfiler", read_profiler)
    monkeypatch.setattr(
        worker_module.ttnn,
        "close_mesh_device",
        lambda target: events.append(("close", target)),
    )
    monkeypatch.setattr(
        worker_module,
        "reset_fabric",
        lambda cfg, count: events.append(("fabric", cfg, count)),
    )

    with pytest.raises(RuntimeError, match="profiler failed"):
        worker_module.close_mesh_device(mesh, "config")

    assert events == [
        "profiler",
        ("close", "submesh-a"),
        ("close", "submesh-b"),
        ("close", mesh),
        ("fabric", "config", 4),
    ]
