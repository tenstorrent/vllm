# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for TT DP mode selection and TT MPI launch semantics."""

import pathlib
from types import SimpleNamespace

import pytest
from vllm_tt_plugin.launcher import parse_tt_mpi_params
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.worker import _rank_owns_mesh, _resolve_mesh_grid


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
    ) -> None:
        """Registers a dummy model class for testing."""
        with monkeypatch.context() as m:
            m.setattr("vllm_tt_plugin.platform.register_tt_models", lambda _: None)
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

    def test_single_process_only_rank_zero_owns_mesh(self) -> None:
        assert _rank_owns_mesh(
            SimpleNamespace(data_parallel_size=1, data_parallel_rank_local=0)
        )
        assert not _rank_owns_mesh(
            SimpleNamespace(data_parallel_size=1, data_parallel_rank_local=1)
        )

    def test_visible_devices_override_full_machine_mesh_preset(self) -> None:
        assert _resolve_mesh_grid("TG", 1, "0") == (1, 1)
        assert _resolve_mesh_grid("TG", 8, "0,1,2,3,4,5,6,7") == (1, 8)

    def test_removed_gathered_override_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        """The removed TT gathered-DP override should fail fast."""
        vllm_config.additional_config = {"tt": {"tt_data_parallel_size": 4}}
        vllm_config.parallel_config.data_parallel_size = 4

        with pytest.raises(ValueError, match="no longer supported"):
            self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

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

    def test_removed_gathered_override_is_rejected_by_launcher(
        self,
        tmp_path: pathlib.Path,
        vllm_config: SimpleNamespace,
    ) -> None:
        """Launcher should reject the removed TT gathered-DP override too."""
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

        with pytest.raises(ValueError, match="no longer supported"):
            parse_tt_mpi_params(vllm_config)
