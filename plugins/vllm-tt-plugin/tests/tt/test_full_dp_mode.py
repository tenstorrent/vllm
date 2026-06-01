# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm_tt_plugin.config import should_open_mesh_for_rank
from vllm_tt_plugin.launcher import (
    TTCoreEngineLauncher,
    TTLaunchPlan,
    parse_tt_mpi_params,
)
from vllm_tt_plugin.platform import TTPlatform


class TestFullDPMode:
    """Tests how successfully we set DP modes - from TT to vLLM."""

    def test_tt_platform_records_full_dp_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tests if ``full_dp_mode=True`` is recorded correctly."""
        vllm_config = SimpleNamespace(
            plugin_config={"tt": {"full_dp_mode": True}},
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                async_scheduling=False,
                scheduler_cls=None,
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            speculative_config=None,
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                worker_cls="auto",
            ),
            lora_config=None,
            model_config=SimpleNamespace(
                max_logprobs=10,
                model="dummy-model",
                hf_config=SimpleNamespace(architectures=["DummyModel"]),
                get_sliding_window=lambda: None,
            ),
        )

        dummy_model_class = type(
            "DummyModel",
            (),
            {"__module__": "models.tt_transformers.tt.generator_vllm"},
        )

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

        assert TTPlatform.full_dp_mode is True
        assert (
            vllm_config.parallel_config.engine_core_cls
            == "vllm.v1.engine.core.EngineCore"
        )
        assert (
            vllm_config.parallel_config.engine_core_proc_cls
            == "vllm.v1.engine.core.EngineCoreProc"
        )
        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        )


    def test_tt_platform_default_uses_tt_dp_engine_core(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tests if ``full_dp_mode=False`` (default) configured to TT DP."""
        vllm_config = SimpleNamespace(
            plugin_config={"tt": {}},
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                async_scheduling=False,
                scheduler_cls=None,
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            speculative_config=None,
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                worker_cls="auto",
            ),
            lora_config=None,
            model_config=SimpleNamespace(
                max_logprobs=10,
                model="dummy-model",
                hf_config=SimpleNamespace(architectures=["DummyModel"]),
                get_sliding_window=lambda: None,
            ),
        )

        dummy_model_class = type(
            "DummyModel",
            (),
            {"__module__": "models.tt_transformers.tt.generator_vllm"},
        )

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

        assert TTPlatform.full_dp_mode is False
        assert (
            vllm_config.parallel_config.engine_core_cls
            == "vllm_tt_plugin.engine.TTEngineCore"
        )
        assert (
            vllm_config.parallel_config.engine_core_proc_cls
            == "vllm_tt_plugin.engine.TTEngineCoreProc"
        )
        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm_tt_plugin.engine.TTDPEngineCoreProc"
        )


    def test_tt_launcher_propagates_full_dp_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tests if ``full_dp_mode=True`` is correctly propagated to the launcher."""
        vllm_config = SimpleNamespace(
            plugin_config={"tt": {"full_dp_mode": True}},
            parallel_config=SimpleNamespace(
                data_parallel_master_ip="",
                data_parallel_size_local=0,
            ),
        )

        with monkeypatch.context() as m:
            m.setattr(
                "vllm_tt_plugin.launcher.parse_tt_mpi_params",
                lambda _cfg: (None, set()),
            )
            plan = TTCoreEngineLauncher().prepare_launch(vllm_config)

        assert isinstance(plan, TTLaunchPlan)
        assert plan.full_dp_mode is True
        assert plan.remote_launched is False

    def test_parse_tt_mpi_params_full_dp_uses_all_device_ranks(
        self,
        tmp_path,
    ) -> None:
        """``full_dp_mode`` uses one MPI rank per DP rank (no non-device ranks)."""
        rank_binding = tmp_path / "rank_binding.yaml"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "  - rank: 1\n"
            "  - rank: 2\n"
            "  - rank: 3\n",
            encoding="utf-8",
        )

        vllm_config = SimpleNamespace(
            plugin_config={
                "tt": {
                    "full_dp_mode": True,
                    "rank_binding": str(rank_binding),
                }
            },
            parallel_config=SimpleNamespace(
                data_parallel_backend="mp",
                data_parallel_size=4,
            ),
        )

        parsed_rank_binding, non_device_dp_ranks = parse_tt_mpi_params(vllm_config)
        assert parsed_rank_binding == str(rank_binding)
        assert non_device_dp_ranks == set()

    def test_parse_tt_mpi_params_full_dp_rejects_mismatched_world(
        self,
        tmp_path,
    ) -> None:
        """``full_dp_mode`` rejects rank layouts that imply local non-device ranks."""
        rank_binding = tmp_path / "rank_binding.yaml"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "  - rank: 1\n",
            encoding="utf-8",
        )

        vllm_config = SimpleNamespace(
            plugin_config={
                "tt": {
                    "full_dp_mode": True,
                    "rank_binding": str(rank_binding),
                }
            },
            parallel_config=SimpleNamespace(
                data_parallel_backend="mp",
                data_parallel_size=4,
            ),
        )

        with pytest.raises(RuntimeError, match="full_dp_mode requires one TT MPI"):
            parse_tt_mpi_params(vllm_config)

    def test_worker_device_rank_selection_for_full_dp_mode(self) -> None:
        """Worker opens mesh on all ranks in full-DP, rank 0 only otherwise."""
        assert should_open_mesh_for_rank(0, full_dp_mode=False) is True
        assert should_open_mesh_for_rank(1, full_dp_mode=False) is False
        assert should_open_mesh_for_rank(1, full_dp_mode=True) is True
