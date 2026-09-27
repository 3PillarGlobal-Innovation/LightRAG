"""Bedrock inference parameters must be opt-in, not defaulted.

``lightrag_server`` used to hardcode::

    kwargs["temperature"] = get_env_value("BEDROCK_LLM_TEMPERATURE", 1.0, float)

so every Converse call carried ``inferenceConfig.temperature`` even though
``BEDROCK_LLM_TEMPERATURE`` ships commented out in ``env.example``. Models that
reject the field rather than ignoring it -- the reasoning-tier inference
profiles among them -- failed every completion::

    ValidationException: This model doesn't support the temperature field.
    Remove temperature and try again.

which broke document ingestion outright. Bedrock now goes through
``BedrockLLMOptions`` like every other binding: options register with
``argparse.SUPPRESS``, so one the operator never set is absent from the
Namespace, absent from ``options_dict()``, and absent from the request.

These tests pin both halves -- that an unset option never materialises, and
that the driver only forwards what it was actually given.
"""

import sys
from argparse import ArgumentParser
from unittest.mock import MagicMock, patch

import pytest

import lightrag.llm.bedrock as bedrock_module
from lightrag.llm.bedrock import bedrock_complete_if_cache
from lightrag.llm.binding_options import BedrockLLMOptions

from tests.test_bedrock_token_tracker import _FakeBedrockClient, _FakeSession

_BEDROCK_ENV_VARS = (
    "BEDROCK_LLM_TEMPERATURE",
    "BEDROCK_LLM_MAX_TOKENS",
    "BEDROCK_LLM_TOP_P",
    "BEDROCK_LLM_STOP_SEQUENCES",
    "BEDROCK_LLM_EXTRA_FIELDS",
)


@pytest.fixture
def fake_client(monkeypatch):
    """Local copy of the token-tracker suite's fixture.

    It could be shared through ``tests/conftest.py``, but that file is imported
    for every test in the suite, and hoisting this would put ``aioboto3`` --
    which only these two modules need -- on the import path of all of them.
    """
    client = _FakeBedrockClient()
    monkeypatch.setattr(
        bedrock_module.aioboto3, "Session", lambda: _FakeSession(client)
    )
    return client


def _options_dict(monkeypatch, **env):
    """Parse an empty command line under ``env`` and return the options dict.

    The environment is read when the arguments are *registered*, so every
    variable has to be set before ``add_args`` -- not just before parsing.
    """
    for name in _BEDROCK_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    parser = ArgumentParser()
    BedrockLLMOptions.add_args(parser)
    return BedrockLLMOptions.options_dict(parser.parse_args([]))


@pytest.mark.offline
def test_unset_options_are_absent_not_defaulted(monkeypatch):
    """The regression: no configuration means no inference parameters at all.

    ``BedrockLLMOptions.temperature`` defaults to ``DEFAULT_TEMPERATURE`` so the
    value can be documented in ``--help`` and the generated sample .env. That
    default must never reach a request.
    """
    assert _options_dict(monkeypatch) == {}


@pytest.mark.offline
def test_configured_options_are_typed_and_passed_through(monkeypatch):
    options = _options_dict(
        monkeypatch,
        BEDROCK_LLM_TEMPERATURE="0.3",
        BEDROCK_LLM_MAX_TOKENS="512",
    )

    assert options == {"temperature": 0.3, "max_tokens": 512}


@pytest.mark.offline
def test_one_configured_option_does_not_drag_in_the_others(monkeypatch):
    """Setting one option must not materialise the defaults of the rest."""
    options = _options_dict(monkeypatch, BEDROCK_LLM_TOP_P="0.9")

    assert options == {"top_p": 0.9}


@pytest.mark.offline
async def test_no_inference_config_when_nothing_is_supplied(fake_client):
    """With no inference parameters the key is omitted, not sent empty."""
    assert await bedrock_complete_if_cache("us.openai.gpt-5.6-luna", "hi") == "hello"

    ((_, sent),) = fake_client.calls
    assert "inferenceConfig" not in sent


@pytest.mark.offline
async def test_none_valued_parameters_are_dropped(fake_client):
    """``max_tokens=None`` means "inherit the provider default", not "send null"."""
    await bedrock_complete_if_cache(
        "us.openai.gpt-5.6-luna", "hi", max_tokens=None, temperature=0.2
    )

    ((_, sent),) = fake_client.calls
    assert sent["inferenceConfig"] == {"temperature": 0.2}


@pytest.mark.offline
async def test_supplied_parameters_are_mapped_to_converse_names(fake_client):
    await bedrock_complete_if_cache(
        "us.anthropic.claude-sonnet-4-6",
        "hi",
        max_tokens=64,
        top_p=0.9,
        stop_sequences=["</s>"],
    )

    ((_, sent),) = fake_client.calls
    assert sent["inferenceConfig"] == {
        "maxTokens": 64,
        "topP": 0.9,
        "stopSequences": ["</s>"],
    }


@pytest.mark.offline
async def test_extra_fields_become_additional_model_request_fields(fake_client):
    """``extra_fields`` is not an inferenceConfig key; it has its own slot.

    Without the pass-through it would survive into ``converse(**kwargs)`` and
    botocore would reject the call.
    """
    reasoning = {"reasoningConfig": {"type": "enabled"}}
    await bedrock_complete_if_cache(
        "us.anthropic.claude-sonnet-4-6", "hi", extra_fields=reasoning
    )

    ((_, sent),) = fake_client.calls
    assert sent["additionalModelRequestFields"] == reasoning
    assert "extra_fields" not in sent


# The server-side half: the closure ``create_app`` actually builds. Patching
# ``LightRAG`` lets ``_build_rag`` run without any storage backend, and the
# function it was handed is the one the server would call per completion --
# the same technique ``tests/test_path_prefixes.py`` uses to exercise
# ``create_app`` offline.
_SERVER_ENV_TO_ISOLATE = (
    "LLM_BINDING",
    "LLM_MODEL",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_BINDING",
    "EMBEDDING_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
)


def _bedrock_model_complete(monkeypatch, **env):
    """Return the Bedrock completion function a Bedrock-configured server builds."""
    for name in _SERVER_ENV_TO_ISOLATE + _BEDROCK_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BINDING", "aws_bedrock")
    monkeypatch.setenv("LLM_MODEL", "us.openai.gpt-5.6-luna")
    monkeypatch.setenv("EMBEDDING_BINDING", "aws_bedrock")
    monkeypatch.setenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    from lightrag.api.config import parse_args
    from lightrag.api.lightrag_server import create_app

    args = parse_args()
    with patch("lightrag.api.lightrag_server.LightRAG") as rag:
        rag.return_value = MagicMock()
        create_app(args)

    return rag.call_args.kwargs["llm_model_func"]


@pytest.mark.offline
async def test_server_sends_no_inference_config_when_unconfigured(
    monkeypatch, fake_client
):
    """The regression, at the line that caused it.

    A Bedrock server with no inference options configured must issue a Converse
    request carrying no ``inferenceConfig`` at all. Re-adding a manufactured
    temperature to ``bedrock_model_complete`` fails here and nowhere else.
    """
    complete = _bedrock_model_complete(monkeypatch)

    assert await complete("hi") == "hello"

    ((_, sent),) = fake_client.calls
    assert "inferenceConfig" not in sent


@pytest.mark.offline
async def test_server_applies_configured_options(monkeypatch, fake_client):
    """Configuration still reaches the request -- the knob is not merely removed."""
    complete = _bedrock_model_complete(
        monkeypatch, BEDROCK_LLM_TEMPERATURE="0.4", BEDROCK_LLM_MAX_TOKENS="256"
    )

    await complete("hi")

    ((_, sent),) = fake_client.calls
    assert sent["inferenceConfig"] == {"temperature": 0.4, "maxTokens": 256}


@pytest.mark.offline
async def test_server_configuration_is_overridable_by_the_caller(
    monkeypatch, fake_client
):
    """``{**options, **kwargs}`` -- an explicit argument beats server config.

    Deliberately the opposite of the OpenAI path, which applies its options with
    ``kwargs.update()``. Switching the merge order fails here.
    """
    complete = _bedrock_model_complete(monkeypatch, BEDROCK_LLM_TEMPERATURE="0.4")

    await complete("hi", temperature=0.9)

    ((_, sent),) = fake_client.calls
    assert sent["inferenceConfig"] == {"temperature": 0.9}
