"""The Bedrock binding must absorb LightRAG's ``token_tracker`` kwarg.

``lightrag/utils.py`` injects ``kwargs["token_tracker"]`` into every LLM call
and ``EmbeddingFunc.__call__`` does the same for embeddings. A binding that
does not declare the parameter forwards it to the provider SDK, and botocore
rejects the request::

    ParamValidationError: Unknown parameter in input: "token_tracker"

which broke every document ingestion on ``LLM_BINDING=aws_bedrock``. These
tests pin that the kwarg is consumed on both paths and that the usage Bedrock
reports lands in the tracker, without a network call — the aioboto3 session is
replaced by a fake client that records what it was asked for.
"""

import json
from typing import ClassVar

import pytest

import lightrag.llm.bedrock as bedrock_module
from lightrag.llm.bedrock import bedrock_complete_if_cache, bedrock_embed
from lightrag.utils import TokenTracker


class _FakeBody:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _FakeBedrockClient:
    """Stands in for ``bedrock-runtime``; fails on any unknown kwarg like botocore."""

    _CONVERSE_PARAMS: ClassVar[set[str]] = {
        "modelId",
        "messages",
        "system",
        "inferenceConfig",
        "toolConfig",
        "guardrailConfig",
    }

    def __init__(self, stream_events=None):
        self.calls = []
        self._stream_events = stream_events or []

    def _validate(self, kwargs):
        unknown = set(kwargs) - self._CONVERSE_PARAMS
        if unknown:
            raise bedrock_module.ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": f"Unknown parameter in input: {sorted(unknown)}",
                    }
                },
                "Converse",
            )

    async def converse(self, **kwargs):
        self._validate(kwargs)
        self.calls.append(("converse", kwargs))
        return {
            "output": {"message": {"content": [{"text": "hello"}]}},
            "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
        }

    async def converse_stream(self, **kwargs):
        self._validate(kwargs)
        self.calls.append(("converse_stream", kwargs))

        async def _events():
            for event in self._stream_events:
                yield event

        return {"stream": _events()}

    async def invoke_model(self, **kwargs):
        self.calls.append(("invoke_model", kwargs))
        text = json.loads(kwargs["body"])["inputText"]
        return {
            "body": _FakeBody(
                {"embedding": [0.1] * 1024, "inputTextTokenCount": len(text.split())}
            )
        }

    # aioboto3's ``session.client(...)`` is used both as an async context
    # manager and via explicit ``__aenter__`` / ``__aexit__`` in the stream path.
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, client):
        self._client = client

    def client(self, service_name, region_name=None):
        assert service_name == "bedrock-runtime"
        return self._client


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeBedrockClient()
    monkeypatch.setattr(
        bedrock_module.aioboto3, "Session", lambda: _FakeSession(client)
    )
    return client


@pytest.mark.offline
async def test_non_streaming_absorbs_token_tracker_and_records_usage(fake_client):
    tracker = TokenTracker()

    result = await bedrock_complete_if_cache(
        "us.openai.gpt-5.6-luna",
        "hi",
        system_prompt="sys",
        token_tracker=tracker,
        hashing_kv=object(),
        max_tokens=64,
    )

    assert result == "hello"
    ((_, sent),) = fake_client.calls
    assert "token_tracker" not in sent
    assert sent["inferenceConfig"] == {"maxTokens": 64}
    assert (tracker.prompt_tokens, tracker.completion_tokens, tracker.total_tokens) == (
        11,
        7,
        18,
    )
    assert tracker.call_count == 1


@pytest.mark.offline
async def test_non_streaming_without_tracker_still_works(fake_client):
    assert await bedrock_complete_if_cache("us.openai.gpt-5.6-luna", "hi") == "hello"


@pytest.mark.offline
async def test_streaming_reads_usage_from_metadata_after_message_stop(monkeypatch):
    """Bedrock emits ``metadata`` after ``messageStop``; the old loop broke on
    ``messageStop`` and would never have seen the usage block."""
    client = _FakeBedrockClient(
        stream_events=[
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "hel"}}},
            {"contentBlockDelta": {"delta": {"text": "lo"}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7}
                }
            },
        ]
    )
    monkeypatch.setattr(
        bedrock_module.aioboto3, "Session", lambda: _FakeSession(client)
    )
    tracker = TokenTracker()

    stream = await bedrock_complete_if_cache(
        "us.anthropic.claude-sonnet-4-6", "hi", stream=True, token_tracker=tracker
    )
    chunks = [chunk async for chunk in stream]

    assert chunks == ["hel", "lo"]
    ((_, sent),) = client.calls
    assert "token_tracker" not in sent and "stream" not in sent
    assert (tracker.prompt_tokens, tracker.completion_tokens, tracker.total_tokens) == (
        5,
        2,
        7,
    )


@pytest.mark.offline
async def test_embed_absorbs_token_tracker_and_sums_titan_input_tokens(fake_client):
    tracker = TokenTracker()

    vectors = await bedrock_embed(
        ["one two three", "four five"],
        model="amazon.titan-embed-text-v2:0",
        token_tracker=tracker,
    )

    assert vectors.shape == (2, 1024)
    assert all("token_tracker" not in sent for _, sent in fake_client.calls)
    assert tracker.prompt_tokens == 5
    assert tracker.completion_tokens == 0
    assert tracker.total_tokens == 5
    assert tracker.call_count == 1


@pytest.mark.offline
async def test_embed_without_tracker_records_nothing(fake_client):
    vectors = await bedrock_embed(["one"], model="amazon.titan-embed-text-v2:0")
    assert vectors.shape == (1, 1024)
