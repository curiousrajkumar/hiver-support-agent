"""Unit tests for BM25 retrieval index and search."""

import os
import pytest
from src.retrieval import SupportRetrievalEngine, BM25Index

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "sample_apple.csv")


def test_retrieval_initialization_and_search(tmp_path):
    temp_index_path = str(tmp_path / "test_retrieval_index.json")
    engine = SupportRetrievalEngine(index_path=temp_index_path)
    engine.initialize(csv_path=SAMPLE_CSV, force_rebuild=True)

    assert os.path.exists(temp_index_path)
    assert engine.index.corpus_size > 0

    results = engine.retrieve("my battery is draining and shuts down", top_k=3)
    assert len(results) > 0
    assert len(results) <= 3
    assert results[0].score > 0
    assert any("battery" in p.customer_text.lower() for p in results)


def test_reloading_cached_index(tmp_path):
    temp_index_path = str(tmp_path / "test_retrieval_index.json")
    engine1 = SupportRetrievalEngine(index_path=temp_index_path)
    engine1.initialize(csv_path=SAMPLE_CSV, force_rebuild=True)

    # Reload in new instance
    engine2 = SupportRetrievalEngine(index_path=temp_index_path)
    engine2.initialize(force_rebuild=False)
    assert engine2.index.corpus_size == engine1.index.corpus_size

    res = engine2.retrieve("how do i transfer photos to my mac", top_k=2)
    assert len(res) > 0
