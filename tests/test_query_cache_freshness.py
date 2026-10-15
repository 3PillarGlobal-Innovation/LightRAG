"""Query-answer cache freshness (``cache_valid_after``) and cache-hit reporting.

A query answer is cached under its question and parameters only. Once the
documents behind it change, asking the same question again used to return the
answer built from the old documents. A caller now passes the time its
documents last changed; an older cached answer is regenerated, and the result
reports whether the answer came from the cache.
"""

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _note_query_cache, _usable_query_cache
from lightrag.utils import CacheData, handle_cache, save_to_cache


class FakeKV:
    """The two calls the cache helpers make, with the timestamp rules of the JSON store."""

    def __init__(self, now=1000):
        self.data = {}
        self.now = now
        self.global_config = {"enable_llm_cache": True, "enable_llm_cache_for_entity_extract": True}

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
    return CacheData(args_hash="h1", content=content, prompt="q", mode="mix", cache_type="query")


class TestUsableQueryCache:
    def test_without_a_cutoff_any_cached_answer_is_used(self):
        assert _usable_query_cache(("a", 5), QueryParam()) == ("a", 5)

    def test_an_answer_older_than_the_cutoff_is_regenerated(self):
        assert _usable_query_cache(("a", 5), QueryParam(cache_valid_after=6)) is None

    def test_an_answer_written_after_the_cutoff_is_used(self):
        assert _usable_query_cache(("a", 7), QueryParam(cache_valid_after=6)) == ("a", 7)

    def test_a_miss_stays_a_miss(self):
        assert _usable_query_cache(None, QueryParam(cache_valid_after=6)) is None


class TestCacheTimestamps:
    @pytest.mark.asyncio
    async def test_handle_cache_reports_when_the_answer_was_last_written(self):
        kv = FakeKV(now=100)
        await save_to_cache(kv, _query_entry("old"))
        kv.now = 200
        await save_to_cache(kv, _query_entry("new"))

        content, written_at = await handle_cache(kv, "h1", "q", "mix", cache_type="query")

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
        assert _usable_query_cache(("same", written_at), QueryParam(cache_valid_after=250)) is not None

    @pytest.mark.asyncio
    async def test_an_unchanged_extraction_result_is_not_rewritten(self):
        kv = FakeKV(now=100)
        extraction = CacheData(args_hash="e1", content="x", prompt="p", mode="default", cache_type="extract")
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
