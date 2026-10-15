"""Query-answer cache freshness (``cache_valid_after``) and cache-hit reporting.

A query answer is cached under its question and parameters only. Once the
documents behind it change, asking the same question again used to return the
answer built from the old documents. A caller now passes the time its
documents last changed; an older cached answer is regenerated, and the result
reports whether the answer came from the cache.
"""

import sys

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _note_query_cache, _usable_query_cache
from lightrag.utils import CacheData, handle_cache, save_to_cache

# Run by the offline CI job (`pytest -m offline`): no network, no real LLM or storage.
pytestmark = pytest.mark.offline


class FakeKV:
    """The two calls the cache helpers make, with the timestamp rules of the JSON store."""

    def __init__(self, now=1000):
        self.data = {}
        self.now = now
        self.global_config = {
            "enable_llm_cache": True,
            "enable_llm_cache_for_entity_extract": True,
        }

    async def get_by_id(self, key):
        entry = self.data.get(key)
        return dict(entry) if entry else None

    async def upsert(self, items):
        for key, value in items.items():
            value = dict(value)
            if key in self.data:
                # Like JsonKVStorage: an overwrite refreshes update_time and drops create_time.
                value.pop("create_time", None)
                value["update_time"] = self.now
            else:
                value["create_time"] = self.now
                value["update_time"] = self.now
            self.data[key] = value


def _query_entry(content="answer"):
    return CacheData(
        args_hash="h1", content=content, prompt="q", mode="mix", cache_type="query"
    )


class TestUsableQueryCache:
    def test_without_a_cutoff_any_cached_answer_is_used(self):
        assert _usable_query_cache(("a", 5), QueryParam()) == ("a", 5)

    def test_an_answer_older_than_the_cutoff_is_regenerated(self):
        assert _usable_query_cache(("a", 5), QueryParam(cache_valid_after=6)) is None

    def test_an_answer_written_after_the_cutoff_is_used(self):
        assert _usable_query_cache(("a", 7), QueryParam(cache_valid_after=6)) == (
            "a",
            7,
        )

    def test_a_miss_stays_a_miss(self):
        assert _usable_query_cache(None, QueryParam(cache_valid_after=6)) is None


class TestCacheTimestamps:
    @pytest.mark.asyncio
    async def test_handle_cache_reports_when_the_answer_was_last_written(self):
        kv = FakeKV(now=100)
        await save_to_cache(kv, _query_entry("old"))
        kv.now = 200
        await save_to_cache(kv, _query_entry("new"))

        content, written_at = await handle_cache(
            kv, "h1", "q", "mix", cache_type="query"
        )

        # The overwrite dropped create_time; update_time is the moment that counts.
        assert (content, written_at) == ("new", 200)

    @pytest.mark.asyncio
    async def test_regenerating_an_unchanged_answer_moves_its_timestamp_forward(self):
        """Otherwise the entry stays older than the cutoff and every later ask pays for a new answer."""
        kv = FakeKV(now=100)
        await save_to_cache(kv, _query_entry("same"))
        kv.now = 300
        await save_to_cache(kv, _query_entry("same"))

        _, written_at = await handle_cache(kv, "h1", "q", "mix", cache_type="query")
        assert written_at == 300
        assert (
            _usable_query_cache(("same", written_at), QueryParam(cache_valid_after=250))
            is not None
        )

    @pytest.mark.asyncio
    async def test_an_unchanged_extraction_result_is_not_rewritten(self):
        kv = FakeKV(now=100)
        extraction = CacheData(
            args_hash="e1",
            content="x",
            prompt="p",
            mode="default",
            cache_type="extract",
        )
        await save_to_cache(kv, extraction)
        kv.now = 300
        await save_to_cache(kv, extraction)

        assert kv.data["default:extract:e1"]["update_time"] == 100


class TestNoteQueryCache:
    def test_a_hit_records_when_the_answer_was_cached(self):
        raw = {}
        _note_query_cache(raw, 1234)
        assert raw["query_cache"] == {"hit": True, "cached_at": 1234}

    def test_a_miss_is_recorded_too(self):
        raw = {}
        _note_query_cache(raw, None)
        assert raw["query_cache"] == {"hit": False, "cached_at": None}

    def test_no_result_data_is_left_alone(self):
        _note_query_cache(None, 1234)


# ---------------------------------------------------------------------------
# Wiring: the query path and the route, not just the helpers
# ---------------------------------------------------------------------------


class FakeChunks:
    """A chunks vector store that always finds one chunk."""

    cosine_better_than_threshold = 0.2

    async def query(self, query, top_k, query_embedding=None):
        return [
            {
                "id": "chunk-1",
                "content": "Refunds are accepted within 45 days of delivery.",
                "file_path": "Refund Window.md",
                "full_doc_id": "doc-1",
                "distance": 0.9,
                "created_at": 1,
            }
        ]


class FakeTokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


def _naive_setup():
    calls = []

    async def llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        calls.append(prompt)
        return f"answer {len(calls)}"

    kv = FakeKV(now=1000)
    global_config = {
        "llm_model_func": llm,
        "tokenizer": FakeTokenizer(),
        "max_total_tokens": 30000,
        "rerank_model_func": None,
        "min_rerank_score": 0.0,
        "kg_chunk_pick_method": "VECTOR",
    }
    return calls, kv, global_config


@pytest.mark.asyncio
async def test_naive_query_reuses_its_cache_and_regenerates_after_the_cutoff():
    """A refactor of naive_query that dropped either helper would pass the helper
    tests above; this runs the query itself with a stubbed LLM."""
    from lightrag.operate import naive_query

    calls, kv, global_config = _naive_setup()

    def ask(cutoff=None):
        param = QueryParam(mode="naive", enable_rerank=False, cache_valid_after=cutoff)
        return naive_query(
            "refund window?", FakeChunks(), param, global_config, hashing_kv=kv
        )

    first = await ask()
    assert (first.content, first.raw_data["query_cache"]["hit"]) == ("answer 1", False)

    kv.now = 1500
    second = await ask()
    assert second.content == "answer 1"
    assert second.raw_data["query_cache"] == {"hit": True, "cached_at": 1000}
    assert len(calls) == 1

    kv.now = 3000
    third = await ask(cutoff=2000)  # the documents changed after the answer was cached
    assert (third.content, third.raw_data["query_cache"]["hit"]) == ("answer 2", False)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_the_query_route_passes_the_cutoff_and_reports_the_cache_hit(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # The API package parses the server's command line on import.
    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    from lightrag.api.routers.query_routes import create_query_routes

    seen = {}

    class FakeRag:
        async def aquery_llm(self, query, param):
            seen["cache_valid_after"] = param.cache_valid_after
            return {
                "llm_response": {"content": "45 days"},
                "data": {"references": []},
                "query_cache": {"hit": True, "cached_at": 1000},
            }

    async def get_rag(request):
        return FakeRag()

    app = FastAPI()
    app.include_router(create_query_routes(get_rag))
    response = TestClient(app).post(
        "/query",
        json={
            "query": "refund window?",
            "mode": "naive",
            "cache_valid_after": 1760526125.5,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["response"], body["cached"], body["cached_at"]) == (
        "45 days",
        True,
        1000,
    )
    assert seen["cache_valid_after"] == 1760526125.5
