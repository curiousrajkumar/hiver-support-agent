"""Unit tests for dataset loading and conversation thread reconstruction."""

import os
import pytest
from data.loader import DatasetLoader, ConversationThread

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "sample_apple.csv")


def test_loader_finds_sample_csv():
    loader = DatasetLoader(brand_name="AppleSupport")
    path = loader.get_dataset_path(SAMPLE_CSV)
    assert os.path.exists(path)


def test_thread_reconstruction():
    loader = DatasetLoader(brand_name="AppleSupport")
    threads = loader.load_and_reconstruct_threads(SAMPLE_CSV, max_threads=20)
    assert len(threads) > 0
    assert len(threads) <= 20

    first_thread = threads[0]
    assert isinstance(first_thread, ConversationThread)
    assert first_thread.customer_text != ""
    assert first_thread.brand_reply_text != ""
    assert first_thread.thread_id.startswith("thread_")


def test_jsonl_serialization(tmp_path):
    loader = DatasetLoader(brand_name="AppleSupport")
    threads = loader.load_and_reconstruct_threads(SAMPLE_CSV, max_threads=5)
    temp_jsonl = tmp_path / "test_threads.jsonl"

    loader.save_threads_jsonl(threads, str(temp_jsonl))
    assert os.path.exists(temp_jsonl)

    reloaded = DatasetLoader.load_threads_jsonl(str(temp_jsonl))
    assert len(reloaded) == len(threads)
    assert reloaded[0].thread_id == threads[0].thread_id
