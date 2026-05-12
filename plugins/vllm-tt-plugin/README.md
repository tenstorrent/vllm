# vLLM TT Plugin

Phase 1 Tenstorrent backend plugin package for the TT vLLM fork.

Install from the repository root with:

```bash
pip install -e plugins/vllm-tt-plugin
```

This package registers TT model architectures through `vllm.general_plugins`
and exposes `vllm_tt_plugin.platform.TTPlatform` through
`vllm.platform_plugins`. During Phase 1 it still relies on TT execution hooks
that remain in the forked vLLM core.
