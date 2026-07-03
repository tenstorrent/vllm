# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for TT DP mode selection and TT MPI launch semantics."""

import pathlib
from types import SimpleNamespace

import pytest
import vllm_tt_plugin.platform as tt_platform
from vllm_tt_plugin.launcher import parse_tt_mpi_params
from vllm_tt_plugin.platform import (
    TTPlatform,
    _maybe_reorder_standard_dp_visible_device_groups,
)
from vllm_tt_plugin.worker import TTWorker, _rank_owns_mesh, _resolve_mesh_grid

import vllm.v1.engine.utils as engine_utils


class TestDPModes:
    """Tests how successfully we set DP mode."""

    @pytest.fixture
    def vllm_config(self) -> SimpleNamespace:
        """Fixture for a dummy vLLM config."""
        return SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                worker_cls="auto",
            ),
            model_config=SimpleNamespace(
                model="dummy",
                hf_config=SimpleNamespace(architectures=["DummyModel"]),
                max_logprobs=10,
            ),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                async_scheduling=False,
                scheduler_cls=None,
                max_num_seqs=4,
            ),
            speculative_config=None,
            lora_config=None,
            cache_config=SimpleNamespace(enable_prefix_caching=False),
        )

    @pytest.fixture
    def dummy_model_class(self) -> type:
        """Fixture for a dummy model class."""
        return type(
            "DummyModel",
            (),
            {"__module__": "models.tt_transformers.tt.generator_vllm"},
        )

    @staticmethod
    def register_dummy_model(
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        visible_device_groups: list[str] | None = None,
    ) -> None:
        """Registers a dummy model class for testing."""
        with monkeypatch.context() as m:
            m.setattr("vllm_tt_plugin.platform.register_tt_models", lambda _: None)
            m.setattr(
                "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
                lambda _cfg: visible_device_groups,
            )
            m.setattr(
                "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
                lambda: ["TTDummyModel"],
            )
            m.setattr(
                "vllm.model_executor.model_loader.utils.get_model_architecture",
                lambda _model_config: (dummy_model_class, None),
            )

            TTPlatform.check_and_update_config(vllm_config)

    def test_upstream_dp_engine_core_is_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        """Default mode should route TT to upstream vLLM DP core proc."""
        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert (
            vllm_config.parallel_config.engine_core_cls
            == "vllm.v1.engine.core.EngineCore"
        ), "Expected `EngineCore` to be the default engine core class for TT models."

        assert (
            vllm_config.parallel_config.engine_core_proc_cls
            == "vllm.v1.engine.core.EngineCoreProc"
        ), (
            "Expected `EngineCoreProc` to be the default engine core proc class for "
            "TT models."
        )

        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        ), (
            "Expected `DPEngineCoreProc` to be the default DP engine core proc class "
            "for TT models."
        )

    def test_lane_mode_keeps_upstream_dp_engine_core(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        """Lane mode should not select the removed TT DP core proc."""
        vllm_config.additional_config = {"_tt_resolved_lane_count": 2}
        vllm_config.parallel_config.data_parallel_size = 1

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert (
            vllm_config.parallel_config.engine_core_cls
            == "vllm.v1.engine.core.EngineCore"
        ), "Expected `EngineCore` to be the default engine core class for TT models."

        assert (
            vllm_config.parallel_config.engine_core_proc_cls
            == "vllm.v1.engine.core.EngineCoreProc"
        ), (
            "Expected `EngineCoreProc` to be the default engine core proc class for "
            "TT models."
        )

        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        ), (
            "Expected lane mode to leave `DPEngineCoreProc` on the upstream "
            "default path."
        )

    def test_standard_dp_all_ranks_own_mesh(self) -> None:
        parallel_config = SimpleNamespace(
            data_parallel_size=4,
            data_parallel_rank_local=3,
        )

        assert _rank_owns_mesh(parallel_config)

    def test_collapsed_standard_dp_rank_still_owns_mesh(self) -> None:
        parallel_config = SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank_local=3,
            data_parallel_index=3,
        )

        assert _rank_owns_mesh(parallel_config)

    def test_single_process_only_rank_zero_owns_mesh(self) -> None:
        assert _rank_owns_mesh(
            SimpleNamespace(data_parallel_size=1, data_parallel_rank_local=0)
        )
        assert not _rank_owns_mesh(
            SimpleNamespace(data_parallel_size=1, data_parallel_rank_local=1)
        )

    def test_collapsed_standard_dp_rank_warms_up_model(self) -> None:
        worker = TTWorker.__new__(TTWorker)
        worker.enable_model_warmup = True
        worker.parallel_config = SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank_local=7,
            data_parallel_index=7,
        )
        warmup_calls: list[str] = []
        worker.model_runner = SimpleNamespace(
            warmup_model=lambda: warmup_calls.append("warmup")
        )

        timings = TTWorker.compile_or_warm_up_model(worker)

        assert warmup_calls == ["warmup"]
        assert timings.language_model >= 0.0

    def test_visible_devices_override_full_machine_mesh_preset(self) -> None:
        assert _resolve_mesh_grid("TG", 1, "0") == (1, 1)
        assert _resolve_mesh_grid("TG", 8, "0,1,2,3,4,5,6,7") == (1, 8)
        assert _resolve_mesh_grid("P150x8", 8, "3") == (1, 1)

    def test_single_host_standard_dp_uses_upstream_launcher(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.parallel_config.data_parallel_size = 4

        self.register_dummy_model(
            monkeypatch,
            vllm_config,
            dummy_model_class,
            visible_device_groups=["24,25", "26,27", "3,2", "1,0"],
        )

        assert (
            vllm_config.parallel_config.engine_core_launcher_cls
            == "vllm.v1.engine.utils.CoreEngineLauncher"
        )
        assert TTPlatform._standard_dp_visible_device_groups == [
            "24,25",
            "26,27",
            "3,2",
            "1,0",
        ]

    def test_wh_galaxy_dp4_groups_follow_known_good_mesh_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            tt_platform.ttnn.cluster,
            "get_cluster_type",
            lambda: tt_platform.ttnn.cluster.ClusterType.GALAXY,
        )

        groups = [
            "0,1,2,3,4,5,6,7",
            "8,9,10,11,12,13,14,15",
            "16,17,18,19,20,21,22,23",
            "24,25,26,27,28,29,30,31",
        ]

        assert _maybe_reorder_standard_dp_visible_device_groups(
            groups,
            (4, 8),
            4,
        ) == [
            "0,1,2,3,4,5,6,7",
            "16,17,18,19,20,21,22,23",
            "24,25,26,27,28,29,30,31",
            "8,9,10,11,12,13,14,15",
        ]

    def test_rank_binding_keeps_tt_launcher(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.parallel_config.data_parallel_size = 4
        vllm_config.additional_config = {
            "tt": {"rank_binding": "/tmp/rank_binding.yaml"}
        }

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert (
            vllm_config.parallel_config.engine_core_launcher_cls
            == "vllm_tt_plugin.launcher.TTCoreEngineLauncher"
        )
        assert TTPlatform._standard_dp_visible_device_groups is None

    def test_standard_dp_visible_device_groups_feed_upstream_env_assignment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(engine_utils, "current_platform", TTPlatform)
        monkeypatch.setattr(
            TTPlatform,
            "_standard_dp_visible_device_groups",
            ["24,25,26,27,3,2,1,0", "16,17,18,19,20,21,22,23"],
        )

        assert (
            engine_utils.get_device_indices(
                TTPlatform.device_control_env_var,
                local_dp_rank=1,
                world_size=1,
            )
            == "16,17,18,19,20,21,22,23"
        )

    def test_legacy_gathered_override_is_ignored_by_platform(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        """Legacy TT gathered-DP config should not affect platform routing."""
        vllm_config.additional_config = {"tt": {"tt_data_parallel_size": 4}}
        vllm_config.parallel_config.data_parallel_size = 4

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert vllm_config.parallel_config.data_parallel_size == 4
        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        )
        assert (
            vllm_config.scheduler_config.scheduler_cls
            == "vllm_tt_plugin.scheduler.TTScheduler"
        )

    def test_standard_dp_uses_all_device_ranks(
        self,
        tmp_path: pathlib.Path,
        vllm_config: SimpleNamespace,
    ) -> None:
        """Standard DP should use one MPI rank per DP rank."""
        rank_binding = tmp_path / "rank_binding.json"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    mesh_id: 0\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "0"\n'
            "\n"
            "  - rank: 1\n"
            "    mesh_id: 1\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "3"\n'
            "\n"
            "  - rank: 2\n"
            "    mesh_id: 2\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "1"\n'
            "\n"
            "  - rank: 3\n"
            "    mesh_id: 3\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "2"\n'
        )

        vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 4

        parsed_rank_binding, non_device_dp_ranks = parse_tt_mpi_params(vllm_config)

        assert parsed_rank_binding == str(rank_binding), (
            "Expected rank binding to be returned correctly."
        )
        assert non_device_dp_ranks == set(), (
            "Expected no non-device DP ranks in standard DP mode."
        )

    def test_standard_dp_rejects_mismatched_mpi_world(
        self,
        tmp_path: pathlib.Path,
        vllm_config: SimpleNamespace,
    ) -> None:
        """Standard DP must map one TT MPI rank to one DP rank."""
        rank_binding = tmp_path / "rank_binding.json"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    mesh_id: 0\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "0, 1"\n'
            "\n"
            "  - rank: 1\n"
            "    mesh_id: 1\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "2, 3"\n'
        )

        vllm_config.additional_config = {"tt": {"rank_binding": str(rank_binding)}}
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 4

        with pytest.raises(
            RuntimeError,
            match="Standard DP mode requires one TT MPI rank per DP rank",
        ):
            parse_tt_mpi_params(vllm_config)

    def test_legacy_gathered_override_is_ignored_by_launcher(
        self,
        tmp_path: pathlib.Path,
        vllm_config: SimpleNamespace,
    ) -> None:
        """Launcher should ignore the legacy TT gathered-DP override."""
        rank_binding = tmp_path / "rank_binding.json"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    mesh_id: 0\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "0"\n'
            "\n"
            "  - rank: 1\n"
            "    mesh_id: 1\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "1"\n'
            "\n"
            "  - rank: 2\n"
            "    mesh_id: 2\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "2"\n'
            "\n"
            "  - rank: 3\n"
            "    mesh_id: 3\n"
            "    env_overrides:\n"
            '      TT_VISIBLE_DEVICES: "3"\n'
        )

        vllm_config.additional_config = {
            "tt": {
                "rank_binding": str(rank_binding),
                "tt_data_parallel_size": 4,
            }
        }
        vllm_config.parallel_config.data_parallel_backend = "mp"
        vllm_config.parallel_config.data_parallel_size = 4

        parsed_rank_binding, non_device_dp_ranks = parse_tt_mpi_params(vllm_config)

        assert parsed_rank_binding == str(rank_binding)
        assert non_device_dp_ranks == set()
