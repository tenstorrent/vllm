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
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_input import TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.worker import TTWorker


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


def _submission_input(*, device_sampling: bool, slot_remap=(3, 1, 2, 3)):
    return SimpleNamespace(
        input_tokens=torch.zeros((4, 1), dtype=torch.int32),
        input_positions=torch.zeros((4,), dtype=torch.int32),
        block_tables=torch.zeros((4, 1), dtype=torch.int32),
        block_tables_per_group=[torch.zeros((4, 1), dtype=torch.int32)],
        block_tables_per_layer=None,
        unpadded_batch_size=4,
        tt_sampling_params=TTSamplingParams(
            temperature=torch.ones(4),
            top_k=torch.full((4,), 32),
            top_p=torch.ones(4),
            presence_penalty=torch.zeros(4),
            frequency_penalty=torch.zeros(4),
            repetition_penalty=torch.ones(4),
            seed=torch.full((4,), -1),
            num_logprobs=torch.full((4,), -2),
            enable_log_probs=torch.zeros(4, dtype=torch.bool),
        ),
        perform_device_sampling=device_sampling,
        prompt_tokens=None,
        output_tokens=None,
        decode_layout_changed=True,
        slot_remap=torch.tensor(slot_remap, dtype=torch.int32),
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


def test_empty_batch_does_not_consume_pending_slot_remap():
    batch = SimpleNamespace(
        num_reqs=0,
        _req_ids=[None],
        req_output_token_ids=[None],
        _slot_remap=torch.tensor([3, 1, 2, 3], dtype=torch.int32),
    )

    InputBatch.condense(batch, [0])

    assert batch._req_ids == []
    assert batch.req_output_token_ids == []
    assert batch._slot_remap.tolist() == [3, 1, 2, 3]


def test_dp_slot_remap_commit_respects_contract_and_sampling_mode():
    commits = []
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            input_batch=SimpleNamespace(
                commit_slot_remap=lambda: commits.append("v1-host")
            ),
        )
    )

    TTWorker.commit_dp_slot_updates(worker, device_sampling=False, contract_version=1)

    worker.model_runner.input_batch.commit_slot_remap = lambda: commits.append(
        "v0-device"
    )
    TTWorker.commit_dp_slot_updates(worker, device_sampling=False, contract_version=0)
    TTWorker.commit_dp_slot_updates(worker, device_sampling=True, contract_version=0)

    assert commits == ["v1-host", "v0-device"]


def test_contract_version_probe_tolerates_a_rank_without_a_model():
    """Non-device DP ranks never load a model, so they cannot read the version."""
    controller = TTAsyncDecodeController(
        SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=4))
    )
    worker = SimpleNamespace(model_runner=SimpleNamespace(async_decode=controller))

    assert TTWorker.decode_input_update_contract_version(worker) == -1

    controller.runner.model = SimpleNamespace(decode_input_update_contract=1)

    assert TTWorker.decode_input_update_contract_version(worker) == 1


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


def test_explicit_contract_delivers_and_commits_slot_remap_for_host_sampling():
    captured = {}
    commits = []

    class FakeModel:
        decode_input_update_contract = 1
        model_capabilities = {"supports_async_decode": False}

        def decode_forward(self, **kwargs):
            captured.update(kwargs)
            return object()

    runner = SimpleNamespace(
        model=FakeModel(),
        trace_mode="decode_only",
        kv_caches=object(),
        request_specific_rope=False,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        input_batch=SimpleNamespace(commit_slot_remap=lambda: commits.append(True)),
    )
    controller = TTAsyncDecodeController(runner)

    controller.submit_decode(
        _submission_input(device_sampling=False),
        read_from_device=False,
        async_read=False,
    )

    assert captured["slot_remap"].tolist() == [3, 1, 2, 3]
    assert commits == [True]


def test_layout_change_refreshes_request_rope_even_when_request_id_is_reused():
    captured = {}

    class FakeModel:
        decode_input_update_contract = 1
        model_capabilities = {"supports_async_decode": False}

        def decode_forward(self, **kwargs):
            captured.update(kwargs)
            return object()

    runner = SimpleNamespace(
        model=FakeModel(),
        trace_mode="decode_only",
        kv_caches=object(),
        request_specific_rope=True,
        previous_req_ids={"req-0"},
        requests={"req-0": SimpleNamespace(mrope_position_delta=17)},
        parallel_config=SimpleNamespace(data_parallel_size=1),
        input_batch=SimpleNamespace(
            req_ids=["req-0"],
            commit_slot_remap=lambda: None,
        ),
    )
    controller = TTAsyncDecodeController(runner)

    controller.submit_decode(
        _submission_input(device_sampling=False),
        read_from_device=False,
        async_read=False,
    )

    assert captured["rope_deltas_all_users"] == [17]


def test_slot_remap_is_not_committed_when_decode_submission_fails():
    commits = []

    class FakeModel:
        decode_input_update_contract = 1
        model_capabilities = {"supports_async_decode": False}

        def decode_forward(self, **kwargs):
            raise RuntimeError("submission rejected")

    runner = SimpleNamespace(
        model=FakeModel(),
        trace_mode="decode_only",
        kv_caches=object(),
        request_specific_rope=False,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        input_batch=SimpleNamespace(commit_slot_remap=lambda: commits.append(True)),
    )
    controller = TTAsyncDecodeController(runner)

    try:
        controller.submit_decode(
            _submission_input(device_sampling=False),
            read_from_device=False,
            async_read=False,
        )
    except RuntimeError as exc:
        assert str(exc) == "submission rejected"
    else:
        raise AssertionError("decode submission should have failed")

    assert commits == []


def test_legacy_host_sampling_keeps_slot_remap_pending_for_device_sampling(
    monkeypatch,
):
    calls = []
    commits = []
    monkeypatch.setattr(
        "vllm_tt_plugin.async_decode.logger.warning",
        lambda *args: None,
    )

    class FakeLegacyModel:
        model_capabilities = {"supports_async_decode": True}

        def decode_forward(self, **kwargs):
            calls.append(kwargs)
            return object()

    runner = SimpleNamespace(
        model=FakeLegacyModel(),
        trace_mode="decode_only",
        kv_caches=object(),
        request_specific_rope=False,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        input_batch=SimpleNamespace(commit_slot_remap=lambda: commits.append(True)),
    )
    controller = TTAsyncDecodeController(runner)
    model_input = _submission_input(device_sampling=False)

    controller.submit_decode(
        model_input,
        read_from_device=False,
        async_read=False,
    )
    model_input.perform_device_sampling = True
    controller.submit_decode(
        model_input,
        read_from_device=False,
        async_read=False,
    )

    assert "slot_remap" not in calls[0]
    assert calls[1]["slot_remap"].tolist() == [3, 1, 2, 3]
    assert commits == [True]


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


def test_contract_version_does_not_change_non_dp_steady_decode_eligibility():
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


def test_gathered_dp_overlap_requires_the_explicit_contract():
    def _controller_for(dp_size, model):
        return TTAsyncDecodeController(
            SimpleNamespace(
                model=model,
                parallel_config=SimpleNamespace(data_parallel_size=dp_size),
            )
        )

    legacy = SimpleNamespace()
    v0 = SimpleNamespace(decode_input_update_contract=0)
    v1 = SimpleNamespace(decode_input_update_contract=1)

    assert not _controller_for(4, legacy).gathered_dp_overlap_permitted()
    assert not _controller_for(4, v0).gathered_dp_overlap_permitted()
    assert _controller_for(4, v1).gathered_dp_overlap_permitted()

    # Lane mode is single-process; its overlap path is unchanged by the gate.
    assert _controller_for(1, legacy).gathered_dp_overlap_permitted()

    # A rank that holds no model abstains instead of vetoing the global vote.
    abstaining = TTAsyncDecodeController(
        SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=4))
    )
    assert abstaining.decode_input_update_contract_version() is None
    assert abstaining.gathered_dp_overlap_permitted()


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
            submit_time_ns=1,
        ),
        completion_time_ns=2,
    )

    controller.apply_completed_decode_step(completed, skip_req_ids={"req-0"})

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
        skip_req_ids=set(),
    )

    assert req_state.output_token_ids == [7]


def _steady_eligible_runner(current_req_ids=("req-0",)):
    """Runner mock whose every steady-decode invariant is satisfied."""
    sampling = SimpleNamespace(
        bad_words_token_ids={},
        has_active_logitsprocs=lambda: False,
    )
    return SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={req_id: i for i, req_id in enumerate(current_req_ids)},
            no_penalties=True,
            no_allowed_token_ids=True,
            max_num_logprobs=None,
            sampling=sampling,
        ),
        requests={
            req_id: SimpleNamespace(sampling_params=None) for req_id in current_req_ids
        },
        model=SimpleNamespace(model_capabilities={"supports_async_decode": True}),
        model_config=SimpleNamespace(logits_processors=None),
        trace_mode="decode_only",
        _decode_layout_changed_since_last_decode=False,
        check_perform_device_sampling=lambda **_kwargs: True,
    )


def _steady_scheduler_output(current_req_ids=("req-0",), **overrides):
    fields = {
        "num_scheduled_tokens": {req_id: 1 for req_id in current_req_ids},
        "scheduled_new_reqs": [],
        "scheduled_cached_reqs": SimpleNamespace(
            req_ids=list(current_req_ids), resumed_req_ids=set()
        ),
        "pending_structured_output_tokens": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_layout_change_causes_are_all_rejected_before_update_states():
    """Every ``_update_states`` layout-change cause must fail the pre-drain check.

    The drain decision is taken from the scheduler output before the persistent
    batch is mutated, so it has to reject each event that would later set
    ``_decode_layout_changed_since_last_decode`` and force a host reload.
    """
    controller = _controller(("req-0", "req-1"))
    controller.note_dp_decode_submitted(True)
    controller.runner = _steady_eligible_runner(("req-0", "req-1"))
    controller.runner.input_batch.req_id_to_index = {"req-0": 0, "req-1": 1}

    baseline = _steady_scheduler_output(("req-0", "req-1"))
    assert controller.steady_decode_scheduler_invariants_met(baseline, None)

    # Finished or unscheduled: present in the batch, absent from this step.
    removed = _steady_scheduler_output(("req-0",))
    removed.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["req-0"], resumed_req_ids=set()
    )
    assert not controller.steady_decode_scheduler_invariants_met(removed, None)

    # Added: scheduled this step, absent from the batch.
    added = _steady_scheduler_output(("req-0", "req-1", "req-2"))
    assert not controller.steady_decode_scheduler_invariants_met(added, None)

    # Resumed from preemption: membership is unchanged, so only the prefill
    # check rejects it. This leg is why the membership diff alone is not enough.
    resumed = _steady_scheduler_output(("req-0", "req-1"))
    resumed.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["req-0", "req-1"], resumed_req_ids={"req-1"}
    )
    assert controller.scheduler_preserves_decode_layout(resumed)
    assert not controller.steady_decode_scheduler_invariants_met(resumed, None)


def test_dp_block_table_width_follows_allocation_not_host_tokens():
    """A one-step-stale token count must not narrow the gathered page table.

    At a block boundary the scheduler has already allocated the block the
    device is about to write, while host ``num_tokens`` still lags by one.
    Trimming to the token-derived width would drop that block, and the DP
    concat zero-pads it back to block id 0.
    """
    block_size = 32
    allocated_blocks = 2
    stale_num_tokens = 32  # the applied token count lags the device by one

    group = SimpleNamespace(
        num_blocks_per_row=np.array([allocated_blocks, 0], dtype=np.int32)
    )
    batch = SimpleNamespace(block_table=SimpleNamespace(block_tables=[group]))

    assert InputBatch.allocated_blocks_for_rows(batch, [0]) == allocated_blocks
    assert stale_num_tokens // block_size < allocated_blocks


def test_gathered_dp_result_rejects_requests_invalidated_since_submit():
    """Under overlap the next step is processed before the previous is applied.

    A request that step finished or resumed must get neither a runner-state
    update nor a scheduler-visible token.
    """
    live = SimpleNamespace(output_token_ids=[])
    resumed = SimpleNamespace(output_token_ids=[])
    runner = SimpleNamespace(
        requests={"live": live, "resumed": resumed},
        input_batch=SimpleNamespace(req_id_to_index={}, num_reqs=2),
        model_config=SimpleNamespace(max_model_len=32),
        _invalidated_req_ids={"resumed"},
    )
    runner._consume_invalidated_req_ids = lambda: (
        TTModelRunner._consume_invalidated_req_ids(runner)
    )
    runner._apply_sampled_tokens_to_state = lambda **kwargs: (
        TTModelRunner._apply_sampled_tokens_to_state(runner, **kwargs)
    )
    runner._build_runner_output = lambda **kwargs: SimpleNamespace(
        req_id_to_index=dict(kwargs["req_id_to_index"]),
        sampled_token_ids=[[5], [6]],
    )
    runner.apply_and_build_runner_output = lambda *args, **kwargs: (
        TTModelRunner.apply_and_build_runner_output(runner, *args, **kwargs)
    )

    output = TTModelRunner.apply_dp_execution_result(
        runner,
        torch.tensor([[5], [6]], dtype=torch.int32),
        None,
        req_ids=["live", "resumed"],
        req_id_to_index={"live": 0, "resumed": 1},
    )

    assert live.output_token_ids == [5]
    assert resumed.output_token_ids == []
    assert output.sampled_token_ids == [[5], []]
    assert runner._invalidated_req_ids == set()


def test_slot_reuse_clears_a_stale_pending_remap_entry():
    """Prefill steps neither deliver nor commit a remap, so entries persist.

    Two condense-then-reuse rounds without an intervening decode would
    otherwise hand a brand-new request another slot's source index.
    """
    batch = InputBatch.__new__(InputBatch)
    batch.max_num_reqs = 4
    batch._slot_remap = torch.tensor([0, 2, 2, 3], dtype=torch.int32)

    InputBatch._reset_slot_remap_entry(batch, 1)

    assert batch._slot_remap.tolist() == [0, 1, 2, 3]
