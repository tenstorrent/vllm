# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger

logger = init_logger(__name__)


def register() -> None:
    """Register TT models and reasoning parsers in every vLLM process."""
    from vllm_tt_plugin.model_registry import register_tt_models_from_plugin

    register_tt_models_from_plugin()
    _register_tt_reasoning_parsers()


def _register_tt_reasoning_parsers() -> None:
    """Register reasoning parsers for TT-served models that aren't upstream.

    Kept in the plugin (rather than patched into ``vllm.reasoning``) so it
    carries over unchanged when switching to upstream vLLM. Registered lazily so
    the parser module is only imported when ``--reasoning-parser`` selects it.
    """
    from vllm.reasoning import ReasoningParserManager

    ReasoningParserManager.register_lazy_module(
        "gemma4",
        "vllm_tt_plugin.gemma4_reasoning_parser",
        "Gemma4ReasoningParser",
    )


def platform_plugin() -> str | None:
    """Return the TT platform class when TT runtime libraries are present."""
    try:
        import ttnn  # noqa: F401
    except Exception as exc:
        logger.debug("TT plugin platform is not available because: %s", exc)
        return None

    logger.debug("Confirmed TT plugin platform is available because ttnn is found.")
    return "vllm_tt_plugin.platform.TTPlatform"
