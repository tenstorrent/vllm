# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for the explicit TT decode reload contract.

These tests exercise only controller state and plain torch tensors. Importing
the plugin still requires the normal ttnn-enabled test environment.
"""

from types import SimpleNamespace

import numpy as np
import torch
from vllm_tt_plugin.async_decode import (
    CompletedDecodeStep,
    SubmittedStepContext,
    TTAsyncDecodeController,
)
from vllm_tt_plugin.model_input import TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner


def _controller(current_req_ids=("req-0",), *, trace_mode="decode_only"):
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={req_id: i for i, req_id in enumerate(current_req_ids)}
        ),
        model=SimpleNamespace(model_capabilities={"supports_async_decode": True}),
        trace_mode=trace_mode,
    )
    return TTAsyncDecodeController(runner)


def _decode_input(*, device_sampling=True, decode_layout_changed=False, page=0):
    return SimpleNamespace(
        perform_device_sampling=device_sampling,
        decode_layout_changed=decode_layout_changed,
        block_tables_per_group=[torch.tensor([[page, 0]], dtype=torch.int32)],
    )


def _submit(controller, model_input):
    plan = controller.plan_decode_reload(model_input)
    controller.commit_decode_submission(model_input, plan)
    return plan


def test_first_device_decode_fully_initializes_forward_and_sampling_state():
    plan = _submit(_controller(), _decode_input())

    assert plan.reload_inputs
    assert not plan.reload_page_table
    assert plan.reload_sampling_params
    assert plan.reset_sampling_state
    assert not plan.overlap_safe


def test_steady_device_decode_reuses_resident_inputs():
    controller = _controller()
    _submit(controller, _decode_input(page=1))
    submitted_page_tables = controller._submitted_page_tables

    plan = _submit(controller, _decode_input(page=1))

    assert not plan.reload_inputs
    assert not plan.reload_page_table
    assert not plan.reload_sampling_params
    assert not plan.reset_sampling_state
    assert plan.overlap_safe
    assert controller._submitted_page_tables is submitted_page_tables


def test_model_without_async_decode_support_reloads_inputs_every_step():
    controller = _controller()
    controller.runner.model.model_capabilities["supports_async_decode"] = False
    _submit(controller, _decode_input(page=1))

    plan = _submit(controller, _decode_input(page=1))

    assert plan.reload_inputs
    assert not plan.reload_page_table
    assert not plan.reload_sampling_params
    assert not plan.reset_sampling_state
    assert not plan.overlap_safe


def test_decode_without_trace_reloads_inputs_every_step():
    controller = _controller(trace_mode="none")
    _submit(controller, _decode_input(page=1))

    plan = _submit(controller, _decode_input(page=1))

    assert plan.reload_inputs
    assert not plan.reload_page_table
    assert not plan.reload_sampling_params
    assert not plan.reset_sampling_state
    assert not plan.overlap_safe


def test_page_table_only_refresh_is_overlap_safe():
    controller = _controller()
    _submit(controller, _decode_input(page=1))

    plan = _submit(controller, _decode_input(page=2))

    assert not plan.reload_inputs
    assert plan.reload_page_table
    assert not plan.reload_sampling_params
    assert not plan.reset_sampling_state
    assert plan.overlap_safe


def test_host_sampling_reloads_every_step_and_device_switch_reinitializes():
    controller = _controller()
    first_host = _submit(controller, _decode_input(device_sampling=False))
    second_host = _submit(controller, _decode_input(device_sampling=False))
    device = _submit(controller, _decode_input(device_sampling=True))

    assert first_host.reload_inputs
    assert second_host.reload_inputs
    assert not second_host.reload_sampling_params
    assert device.reload_inputs
    assert device.reload_sampling_params
    assert device.reset_sampling_state


def test_prefill_and_layout_change_break_the_decode_chain():
    controller = _controller()
    _submit(controller, _decode_input())
    controller.note_prefill_submitted()

    after_prefill = _submit(controller, _decode_input())
    layout_change = _submit(controller, _decode_input(decode_layout_changed=True))

    assert after_prefill.reload_inputs and after_prefill.reset_sampling_state
    assert layout_change.reload_inputs and layout_change.reset_sampling_state


def test_drained_decode_updates_next_host_token_and_position_before_transition():
    req_state = SimpleNamespace(output_token_ids=[])
    input_batch = SimpleNamespace(
        req_id_to_index={"req-0": 0},
        num_tokens=np.array([3], dtype=np.int32),
        token_ids_cpu=np.zeros((1, 8), dtype=np.int32),
    )
    runner = SimpleNamespace(
        requests={"req-0": req_state},
        input_batch=input_batch,
        model_config=SimpleNamespace(max_model_len=8),
        model=SimpleNamespace(model_capabilities={"supports_async_decode": True}),
        trace_mode="decode_only",
    )
    runner._apply_sampled_tokens_to_state = lambda **kwargs: (
        TTModelRunner._apply_sampled_tokens_to_state(runner, **kwargs)
    )
    controller = TTAsyncDecodeController(runner)
    controller.note_dp_decode_submitted(True)
    completed = CompletedDecodeStep(
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        logprobs=None,
        context=SubmittedStepContext(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            request_states=(req_state,),
            submit_time_ns=1,
        ),
        completion_time_ns=2,
    )

    controller.apply_completed_decode_step(completed)
    controller.note_prefill_submitted()
    plan = controller.plan_decode_reload(_decode_input())

    next_position = int(input_batch.num_tokens[0]) - 1
    assert input_batch.token_ids_cpu[0, next_position] == 7
    assert next_position == 3
    assert req_state.output_token_ids == [7]
    assert plan.reload_inputs
    assert plan.reset_sampling_state


def test_dp_submission_mirrors_resident_mode_and_prefill_invalidates_it():
    controller = _controller()

    controller.note_dp_decode_submitted(True)

    assert controller._decode_chain_valid
    assert controller._previous_device_sampling is True

    controller.note_prefill_submitted()

    assert not controller._decode_chain_valid


def test_dp_slot_remap_is_offset_into_global_slot_namespace():
    rank_local = torch.tensor([[3, 1, 2, 0], [2, 0, 3, 1]], dtype=torch.int32)

    global_remap = TTModelRunner._globalize_dp_slot_remap(rank_local)

    assert global_remap.tolist() == [3, 1, 2, 0, 6, 4, 7, 5]


def test_explicit_contract_keeps_layout_hint_inside_planner():
    captured = {}

    class FakeModel:
        decode_input_update_contract = 1
        model_capabilities = {"supports_async_decode": True}

        def decode_forward(self, **kwargs):
            captured.update(kwargs)
            return object()

    runner = SimpleNamespace(
        model=FakeModel(),
        trace_mode="decode_only",
        kv_caches=object(),
        request_specific_rope=False,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        input_batch=SimpleNamespace(commit_slot_remap=lambda: None),
    )
    controller = TTAsyncDecodeController(runner)
    model_input = SimpleNamespace(
        input_tokens=torch.zeros((1, 1), dtype=torch.int32),
        input_positions=torch.zeros((1,), dtype=torch.int32),
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        block_tables_per_group=[torch.zeros((1, 1), dtype=torch.int32)],
        block_tables_per_layer=None,
        unpadded_batch_size=1,
        tt_sampling_params=TTSamplingParams(
            temperature=torch.tensor([1.0]),
            top_k=torch.tensor([32]),
            top_p=torch.tensor([1.0]),
            presence_penalty=torch.tensor([0.0]),
            frequency_penalty=torch.tensor([0.0]),
            repetition_penalty=torch.tensor([1.0]),
            seed=torch.tensor([-1]),
            num_logprobs=torch.tensor([-2]),
            enable_log_probs=torch.tensor([False]),
        ),
        perform_device_sampling=True,
        prompt_tokens=None,
        output_tokens=None,
        decode_layout_changed=True,
        slot_remap=torch.tensor([0], dtype=torch.int32),
    )

    controller.submit_decode(model_input, read_from_device=False, async_read=False)

    assert "reset_batch" not in captured
    assert "decode_layout_changed" not in captured
    assert captured["reload_inputs"] is True
    assert captured["reload_sampling_params"] is True
    assert captured["reset_sampling_state"] is True


def test_legacy_contract_receives_reset_batch_without_explicit_commands(
    monkeypatch,
):
    captured = {}
    warnings = []
    monkeypatch.setattr(
        "vllm_tt_plugin.async_decode.logger.warning",
        lambda *args: warnings.append(args),
    )

    class FakeLegacyModel:
        model_capabilities = {"supports_async_decode": True}

        def decode_forward(self, **kwargs):
            captured.update(kwargs)
            return object()

    runner = SimpleNamespace(
        model=FakeLegacyModel(),
        trace_mode="decode_only",
        kv_caches=object(),
        request_specific_rope=False,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        input_batch=SimpleNamespace(commit_slot_remap=lambda: None),
    )
    controller = TTAsyncDecodeController(runner)
    model_input = SimpleNamespace(
        input_tokens=torch.zeros((1, 1), dtype=torch.int32),
        input_positions=torch.zeros((1,), dtype=torch.int32),
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        block_tables_per_group=[torch.zeros((1, 1), dtype=torch.int32)],
        block_tables_per_layer=None,
        unpadded_batch_size=1,
        tt_sampling_params=TTSamplingParams(
            temperature=torch.tensor([1.0]),
            top_k=torch.tensor([32]),
            top_p=torch.tensor([1.0]),
            presence_penalty=torch.tensor([0.0]),
            frequency_penalty=torch.tensor([0.0]),
            repetition_penalty=torch.tensor([1.0]),
            seed=torch.tensor([-1]),
            num_logprobs=torch.tensor([-2]),
            enable_log_probs=torch.tensor([False]),
        ),
        perform_device_sampling=True,
        prompt_tokens=None,
        output_tokens=None,
        decode_layout_changed=True,
        slot_remap=torch.tensor([0], dtype=torch.int32),
    )

    first_submission = controller.submit_decode(
        model_input, read_from_device=False, async_read=False
    )
    second_submission = controller.submit_decode(
        model_input, read_from_device=False, async_read=False
    )

    assert captured["reset_batch"] is True
    assert "decode_layout_changed" not in captured
    assert "reload_inputs" not in captured
    assert "reload_page_table" not in captured
    assert "reload_sampling_params" not in captured
    assert "reset_sampling_state" not in captured
    assert first_submission.reload_plan is None
    assert second_submission.reload_plan is None
    assert controller._decode_chain_valid
    assert controller._previous_device_sampling is True
    assert len(warnings) == 1


def test_contract_version_does_not_change_steady_decode_eligibility():
    runner = SimpleNamespace(
        model=SimpleNamespace(model_capabilities={"supports_async_decode": True}),
        non_dp_async_scheduling=True,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank_local=0,
        ),
        trace_mode="decode_only",
    )
    controller = TTAsyncDecodeController(runner)

    assert controller.steady_decode_base_enabled(dp_gather=False)

    runner.model.decode_input_update_contract = 1

    assert controller.steady_decode_base_enabled(dp_gather=False)


def test_scheduler_layout_prediction_detects_add_remove_and_preemption():
    controller = _controller(("req-0", "req-1"))

    same = SimpleNamespace(num_scheduled_tokens={"req-0": 1, "req-1": 1})
    removed = SimpleNamespace(num_scheduled_tokens={"req-0": 1})
    added = SimpleNamespace(num_scheduled_tokens={"req-0": 1, "req-1": 1, "req-2": 1})

    assert controller.scheduler_preserves_decode_layout(same)
    assert not controller.scheduler_preserves_decode_layout(removed)
    assert not controller.scheduler_preserves_decode_layout(added)


def test_cancelled_or_resumed_request_is_not_applied_to_runner_state():
    captured = []
    controller = _controller()
    controller.runner._apply_sampled_tokens_to_state = lambda **kwargs: captured.append(
        kwargs
    )
    completed = CompletedDecodeStep(
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        logprobs=None,
        context=SubmittedStepContext(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            request_states=(object(),),
            submit_time_ns=1,
        ),
        completion_time_ns=2,
    )

    controller.apply_completed_decode_step(completed, skip_req_ids={"req-0"})

    assert captured[0]["skip_req_ids"] == {"req-0"}


def test_reused_request_id_suppresses_cached_non_dp_scheduler_output():
    old_req_state = SimpleNamespace(output_token_ids=[])
    new_req_state = SimpleNamespace(output_token_ids=[])
    captured = []
    runner_output = SimpleNamespace(
        sampled_token_ids=[[7]],
        req_id_to_index={"req-0": 0},
    )
    controller = _controller()
    controller.runner.requests = {"req-0": new_req_state}
    controller.runner._apply_sampled_tokens_to_state = lambda **kwargs: captured.append(
        kwargs
    )
    completed = CompletedDecodeStep(
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        logprobs=None,
        context=SubmittedStepContext(
            req_ids=["req-0"],
            req_id_to_index={"req-0": 0},
            request_states=(old_req_state,),
            submit_time_ns=1,
        ),
        completion_time_ns=2,
        runner_output=runner_output,
    )

    controller.apply_completed_decode_step(completed)

    assert runner_output.sampled_token_ids == [[]]
    assert captured[0]["skip_req_ids"] == {"req-0"}


def test_unscheduled_live_request_keeps_completed_token_in_cached_state():
    req_state = SimpleNamespace(output_token_ids=[])
    runner = SimpleNamespace(
        requests={"req-0": req_state},
        input_batch=SimpleNamespace(req_id_to_index={}),
        model_config=SimpleNamespace(max_model_len=32),
    )

    TTModelRunner._apply_sampled_tokens_to_state(
        runner,
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        req_ids=["req-0"],
        request_states=(req_state,),
        skip_req_ids=set(),
    )

    assert req_state.output_token_ids == [7]


def test_reused_request_id_rejects_captured_dp_request_identity():
    old_req_state = SimpleNamespace(output_token_ids=[])
    new_req_state = SimpleNamespace(output_token_ids=[])
    runner = SimpleNamespace(
        requests={"req-0": new_req_state},
        input_batch=SimpleNamespace(req_id_to_index={"req-0": 0}),
        model_config=SimpleNamespace(max_model_len=32),
    )

    TTModelRunner._apply_sampled_tokens_to_state(
        runner,
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        req_ids=["req-0"],
        request_states=(old_req_state,),
    )

    assert old_req_state.output_token_ids == []
    assert new_req_state.output_token_ids == []


def test_reused_request_id_suppresses_old_dp_scheduler_output():
    old_req_state = SimpleNamespace(output_token_ids=[])
    new_req_state = SimpleNamespace(output_token_ids=[])
    runner = SimpleNamespace(
        requests={"req-0": new_req_state},
        input_batch=SimpleNamespace(num_reqs=1),
        _dp_request_state_snapshots={4: (old_req_state,)},
        apply_and_build_runner_output=lambda *args, **kwargs: SimpleNamespace(
            sampled_token_ids=[[7]],
            req_id_to_index={"req-0": 0},
        ),
    )

    output = TTModelRunner.apply_dp_execution_result(
        runner,
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        req_ids=["req-0"],
        req_id_to_index={"req-0": 0},
        request_state_snapshot_id=4,
    )

    assert output.sampled_token_ids == [[]]
    assert runner._dp_request_state_snapshots == {}
