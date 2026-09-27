"""The Bedrock answer is not reliably the first content block.

``bedrock_complete_if_cache`` read the Converse answer positionally::

    content = response["output"]["message"]["content"][0]["text"]

A reasoning-tier model returns a ``reasoningContent`` block *ahead* of the
answer once the prompt is involved enough to trigger reasoning, and an entity
extraction prompt is. The validation directly above that line only checked the
content list was non-empty, so the index still landed on the reasoning block
and raised ``KeyError: 'text'``, which ``_handle_bedrock_exception`` turned
into a ``BedrockError``. Every document ingestion failed on such a model.

The shape varies per *call*, not per model — the same model answers a trivial
prompt with a bare text block — so it cannot be predicted from the model id,
which is why the fix scans rather than branching on a family. These tests pin
that, and that a single-text-block response is unaffected.

The streaming path never had the defect: its loop reads ``delta.get("text")``
and skips a reasoning delta already.
"""

import pytest

import lightrag.llm.bedrock as bedrock_module
from lightrag.llm.bedrock import BedrockError, bedrock_complete_if_cache

from tests.test_bedrock_token_tracker import _FakeBedrockClient, _FakeSession

# Both Bedrock families this deployment switches between. Nothing in the fix
# branches on the model id, so they must behave identically.
_OPENAI_PROFILE = "us.openai.gpt-5.6-luna"
_CLAUDE_PROFILE = "us.anthropic.claude-sonnet-4-6"

_REASONING_BLOCK = {"reasoningContent": {"redactedContent": "...opaque..."}}


def _client(monkeypatch, content):
    client = _FakeBedrockClient(converse_content=content)
    monkeypatch.setattr(
        bedrock_module.aioboto3, "Session", lambda: _FakeSession(client)
    )
    return client


@pytest.mark.offline
@pytest.mark.parametrize("model", [_OPENAI_PROFILE, _CLAUDE_PROFILE])
async def test_reasoning_block_before_the_answer(monkeypatch, model):
    """The regression: this raised KeyError: 'text' -> BedrockError."""
    _client(monkeypatch, [_REASONING_BLOCK, {"text": "Entity: Foo"}])

    assert await bedrock_complete_if_cache(model, "extract") == "Entity: Foo"


@pytest.mark.offline
async def test_single_text_block_unchanged(monkeypatch):
    """The common case must be untouched by the scan."""
    _client(monkeypatch, [{"text": "hello"}])

    assert await bedrock_complete_if_cache(_OPENAI_PROFILE, "hi") == "hello"


@pytest.mark.offline
async def test_several_reasoning_blocks_before_the_answer(monkeypatch):
    _client(
        monkeypatch,
        [_REASONING_BLOCK, _REASONING_BLOCK, {"text": "answer"}],
    )

    assert await bedrock_complete_if_cache(_OPENAI_PROFILE, "hi") == "answer"


@pytest.mark.offline
async def test_first_text_block_wins(monkeypatch):
    _client(monkeypatch, [_REASONING_BLOCK, {"text": "first"}, {"text": "second"}])

    assert await bedrock_complete_if_cache(_OPENAI_PROFILE, "hi") == "first"


@pytest.mark.offline
async def test_empty_text_block_is_skipped(monkeypatch):
    """``block.get("text")`` is a truthiness test, so an empty block is not
    mistaken for the answer and the real one is still found."""
    _client(monkeypatch, [{"text": ""}, {"text": "real answer"}])

    assert await bedrock_complete_if_cache(_OPENAI_PROFILE, "hi") == "real answer"


@pytest.mark.offline
async def test_non_dict_block_does_not_crash_the_scan(monkeypatch):
    """The isinstance guard: a malformed block must not mask the answer."""
    _client(monkeypatch, ["unexpected", {"text": "answer"}])

    assert await bedrock_complete_if_cache(_OPENAI_PROFILE, "hi") == "answer"


@pytest.mark.offline
async def test_reasoning_only_response_still_raises_empty_content(monkeypatch):
    """No text anywhere is a real failure and must stay one.

    ``next(..., None)`` yields None, which the existing emptiness check turns
    into the same BedrockError it always raised — not a KeyError, and not a
    silent empty string handed to the extractor.
    """
    _client(monkeypatch, [_REASONING_BLOCK])

    with pytest.raises(BedrockError, match="empty content"):
        await bedrock_complete_if_cache(_OPENAI_PROFILE, "hi")


@pytest.mark.offline
async def test_whitespace_only_answer_still_raises(monkeypatch):
    _client(monkeypatch, [_REASONING_BLOCK, {"text": "   "}])

    with pytest.raises(BedrockError, match="empty content"):
        await bedrock_complete_if_cache(_OPENAI_PROFILE, "hi")


@pytest.mark.offline
async def test_empty_content_list_still_raises_invalid_structure(monkeypatch):
    """The pre-existing structural guard is unchanged."""
    _client(monkeypatch, [])

    with pytest.raises(BedrockError, match="Invalid response structure"):
        await bedrock_complete_if_cache(_OPENAI_PROFILE, "hi")
