"""Retrieval Index and Search Engine for Historical AppleSupport Conversation Pairs.

Implements BM25 Okapi ranking over reconstructed customer -> brand reply pairs.
Saves and reloads persistent index to ensure reproducible, sub-second queries (<15 min pipeline).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from data.loader import ConversationThread, DatasetLoader


@dataclass
class RetrievedPair:
    thread_id: str
    customer_text: str
    brand_reply: str
    score: float


class BM25Index:
    """Self-contained, highly optimized BM25 Okapi search index with disk persistence."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_pairs: List[dict] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_id, term_freq)]

    def _tokenize(self, text: str) -> List[str]:
        """Clean and tokenize text into lowercase word tokens."""
        text = text.lower()
        # Remove URLs and Twitter handles
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        tokens = re.findall(r"\b[a-z0-9_]{2,}\b", text)
        return tokens

    def fit(self, threads: List[ConversationThread]):
        """Index conversation threads based on customer text."""
        self.corpus_size = len(threads)
        self.doc_lengths = []
        self.doc_pairs = []
        self.doc_freqs = {}
        self.inverted_index = {}

        total_length = 0

        for doc_id, thread in enumerate(threads):
            tokens = self._tokenize(thread.customer_text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            self.doc_pairs.append({
                "thread_id": thread.thread_id,
                "customer_text": thread.customer_text,
                "brand_reply": thread.brand_reply_text,
            })

            # Calculate term frequencies for this document
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for term, count in tf.items():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_id, count))

        self.avg_doc_len = total_length / max(1, self.corpus_size)

        # Compute IDF with standard Lucene BM25 formula
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            # idf = ln(1 + (N - n + 0.5) / (n + 0.5))
            self.idf[term] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def search(self, query: str, top_k: int = 3) -> List[RetrievedPair]:
        """Rank historical pairs using BM25 scoring."""
        query_tokens = self._tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores: Dict[int, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf_val = self.idf.get(token, 0.0)
            postings = self.inverted_index[token]

            for doc_id, tf in postings:
                doc_len = self.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score_contribution = idf_val * (numerator / denominator)
                scores[doc_id] = scores.get(doc_id, 0.0) + score_contribution

        # Sort documents by BM25 score descending
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for doc_id, raw_score in sorted_docs:
            pair = self.doc_pairs[doc_id]
            # Normalize score to 0..1 scale approximately for confidence thresholds
            norm_score = round(raw_score / (raw_score + 10.0), 3)
            results.append(
                RetrievedPair(
                    thread_id=pair["thread_id"],
                    customer_text=pair["customer_text"],
                    brand_reply=pair["brand_reply"],
                    score=norm_score,
                )
            )

        return results

    def save(self, filepath: str):
        """Persist index state to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "corpus_size": self.corpus_size,
            "avg_doc_len": self.avg_doc_len,
            "doc_lengths": self.doc_lengths,
            "doc_pairs": self.doc_pairs,
            "doc_freqs": self.doc_freqs,
            "idf": self.idf,
            "inverted_index": self.inverted_index,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> BM25Index:
        """Load persistent index from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        index = cls(k1=data["k1"], b=data["b"])
        index.corpus_size = data["corpus_size"]
        index.avg_doc_len = data["avg_doc_len"]
        index.doc_lengths = data["doc_lengths"]
        index.doc_pairs = data["doc_pairs"]
        index.doc_freqs = data["doc_freqs"]
        index.idf = data["idf"]
        # Convert inverted_index lists back to tuples
        index.inverted_index = {k: [(doc_id, count) for doc_id, count in v] for k, v in data["inverted_index"].items()}
        return index


class SupportRetrievalEngine:
    """Manages retrieval index creation, caching, and querying."""

    DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "retrieval_index.json")

    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path or self.DEFAULT_INDEX_PATH
        self.index: Optional[BM25Index] = None

    def initialize(self, csv_path: Optional[str] = None, force_rebuild: bool = False):
        """Load cached index or construct from CSV."""
        if not force_rebuild and os.path.exists(self.index_path):
            try:
                self.index = BM25Index.load(self.index_path)
                return
            except Exception:
                pass  # Fall through to rebuild

        loader = DatasetLoader(brand_name="AppleSupport")
        threads = loader.load_and_reconstruct_threads(csv_path)
        self.index = BM25Index()
        self.index.fit(threads)
        self.index.save(self.index_path)

    def retrieve(self, customer_query: str, top_k: int = 3) -> List[RetrievedPair]:
        """Retrieve top-K historical customer->brand reply pairs."""
        if not self.index:
            self.initialize()
        return self.index.search(customer_query, top_k=top_k)
