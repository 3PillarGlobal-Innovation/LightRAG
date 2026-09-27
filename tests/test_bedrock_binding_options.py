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

from argparse import ArgumentParser

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

    Fixtures are not shared between test modules, only through conftest, and
    the Converse fake is not worth promoting there for two callers.
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


@pytest.mark.offline
async def test_server_options_are_overridable_by_the_caller(monkeypatch):
    """The server merge is ``{**options, **kwargs}`` -- an explicit call wins."""
    client = _FakeBedrockClient()
    monkeypatch.setattr(
        bedrock_module.aioboto3, "Session", lambda: _FakeSession(client)
    )
    options = _options_dict(monkeypatch, BEDROCK_LLM_TEMPERATURE="0.3")

    await bedrock_complete_if_cache(
        "us.anthropic.claude-sonnet-4-6", "hi", **{**options, "temperature": 0.9}
    )

    ((_, sent),) = client.calls
    assert sent["inferenceConfig"] == {"temperature": 0.9}
