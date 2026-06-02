# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm_tt_plugin import engine as tt_engine
from vllm_tt_plugin.config import should_open_mesh_for_rank
from vllm_tt_plugin.launcher import parse_tt_mpi_params
from vllm_tt_plugin.platform import TTPlatform


class TestFullDPMode:
    """Tests how successfully we set DP modes - from TT to vLLM."""

    def test_tt_platform_default_uses_upstream_dp_engine_core(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default mode should route TT to upstream vLLM DP cores."""
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

        assert TTPlatform.gathered_dp_mode is False
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

    def test_tt_platform_tt_data_parallel_size_uses_tt_dp_engine_core(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting ``tt_data_parallel_size`` enables TT gathered-DP mode."""
        vllm_config = SimpleNamespace(
            plugin_config={"tt": {"tt_data_parallel_size": 4}},
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

        assert TTPlatform.gathered_dp_mode is True
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

    def test_parse_tt_mpi_params_standard_dp_uses_all_device_ranks(
        self,
        tmp_path,
    ) -> None:
        """Standard DP uses one MPI rank per DP rank (no non-device ranks)."""
        rank_binding = tmp_path / "rank_binding.yaml"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    env_overrides:\n"
            "      TT_VISIBLE_DEVICES: '0'\n"
            "  - rank: 1\n"
            "    env_overrides:\n"
            "      TT_VISIBLE_DEVICES: '1'\n"
            "  - rank: 2\n"
            "    env_overrides:\n"
            "      TT_VISIBLE_DEVICES: '2'\n"
            "  - rank: 3\n"
            "    env_overrides:\n"
            "      TT_VISIBLE_DEVICES: '3'\n",
            encoding="utf-8",
        )

        vllm_config = SimpleNamespace(
            plugin_config={
                "tt": {
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

    def test_parse_tt_mpi_params_standard_dp_requires_visible_devices(
        self,
        tmp_path,
    ) -> None:
        """Standard DP requires TT_VISIBLE_DEVICES for each rank binding."""
        rank_binding = tmp_path / "rank_binding.yaml"
        rank_binding.write_text(
            "rank_bindings:\n"
            "  - rank: 0\n"
            "    env_overrides:\n"
            "      TT_VISIBLE_DEVICES: '0'\n"
            "  - rank: 1\n"
            "    env_overrides: {}\n",
            encoding="utf-8",
        )

        vllm_config = SimpleNamespace(
            plugin_config={
                "tt": {
                    "rank_binding": str(rank_binding),
                }
            },
            parallel_config=SimpleNamespace(
                data_parallel_backend="mp",
                data_parallel_size=2,
            ),
        )

        with pytest.raises(RuntimeError, match="TT_VISIBLE_DEVICES"):
            parse_tt_mpi_params(vllm_config)

    def test_parse_tt_mpi_params_standard_dp_rejects_mismatched_world(
        self,
        tmp_path,
    ) -> None:
        """Standard DP rejects rank layouts with local non-device ranks."""
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
                    "rank_binding": str(rank_binding),
                }
            },
            parallel_config=SimpleNamespace(
                data_parallel_backend="mp",
                data_parallel_size=4,
            ),
        )

        with pytest.raises(RuntimeError, match="Standard DP mode requires"):
            parse_tt_mpi_params(vllm_config)

    def test_parse_tt_mpi_params_gathered_dp_uses_non_device_ranks(
        self,
        tmp_path,
    ) -> None:
        """Gathered-DP uses only the first local rank in each MPI segment."""
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
                    "tt_data_parallel_size": 4,
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
        assert non_device_dp_ranks == {1, 3}

    def test_worker_device_rank_selection_for_gathered_dp_mode(self) -> None:
        """Worker opens mesh on all ranks in standard DP, rank 0 only in gathered."""
        assert should_open_mesh_for_rank(0, gathered_dp_mode=False) is True
        assert should_open_mesh_for_rank(1, gathered_dp_mode=False) is True
        assert should_open_mesh_for_rank(0, gathered_dp_mode=True) is True
        assert should_open_mesh_for_rank(1, gathered_dp_mode=True) is False

    def test_tt_gathered_dp_engine_rejects_standard_mode(self) -> None:
        """Gathered-DP core must not be constructible for standard DP mode."""
        vllm_config = SimpleNamespace(plugin_config={"tt": {}})
        with pytest.raises(ValueError, match="standard DP mode"):
            tt_engine.TTDPEngineCoreProc(vllm_config)

    def test_tt_gathered_dp_engine_accepts_gathered_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit ``tt_data_parallel_size`` keeps the TT gathered-DP core path."""
        called = {"base_init": False}

        def _fake_base_init(self, *args, **kwargs):
            called["base_init"] = True
            self.batch_queue = None

        monkeypatch.setattr(tt_engine.DPEngineCoreProc, "__init__", _fake_base_init)

        vllm_config = SimpleNamespace(
            plugin_config={"tt": {"tt_data_parallel_size": 4}}
        )
        proc = tt_engine.TTDPEngineCoreProc(vllm_config)
        assert called["base_init"] is True
        assert proc._dp_in_flight is None
