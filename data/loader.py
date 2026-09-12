"""Dataset loader and thread reconstruction pipeline for Twitter Customer Support dataset.

Filters tweets involving @AppleSupport and reconstructs:
Customer (T1) -> Brand Reply (T2) -> Customer Follow-up (T3) threads.

Supports:
1. Full Kaggle CSV (`twcs.csv`):
   kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/ --unzip
2. Custom local CSV path
3. Bundled stratified sample (`data/sample_apple.csv`) for <15 min quickstart
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, Generator, Iterator, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConversationThread:
    thread_id: str
    customer_tweet_id: str
    customer_text: str
    brand_reply_id: str
    brand_reply_text: str
    customer_followup_id: Optional[str] = None
    customer_followup_text: Optional[str] = None
    created_at: Optional[str] = None


class DatasetLoader:
    def __init__(self, brand_name: str = "AppleSupport"):
        self.brand_name = brand_name.lower()

    def get_dataset_path(self, custom_path: Optional[str] = None) -> str:
        """Find dataset file prioritizing user path > twcs.csv > sample_apple.csv."""
        if custom_path and os.path.exists(custom_path):
            return custom_path
        
        default_full = os.path.join(os.path.dirname(__file__), "twcs.csv")
        if os.path.exists(default_full):
            return default_full

        sample_path = os.path.join(os.path.dirname(__file__), "sample_apple.csv")
        if os.path.exists(sample_path):
            return sample_path

        raise FileNotFoundError(
            f"No dataset found at {custom_path or default_full}. "
            "Please run:\n"
            "  kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/ --unzip\n"
            "or use the bundled sample."
        )

    def load_and_reconstruct_threads(
        self,
        csv_path: Optional[str] = None,
        max_threads: Optional[int] = None,
    ) -> List[ConversationThread]:
        """Reconstruct customer -> brand -> customer threads from CSV."""
        path = self.get_dataset_path(csv_path)
        logger.info(f"Loading and reconstructing threads from: {path}")

        # Pass 1: Index relevant tweets
        # We need:
        # - Tweets by brand
        # - Inbound tweets to brand
        # In Kaggle dataset: tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
        
        tweets_by_id: Dict[str, dict] = {}
        brand_replies: List[dict] = []

        with open(path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tweet_id = str(row.get("tweet_id", "")).strip()
                author_id = str(row.get("author_id", "")).strip().lower()
                inbound = str(row.get("inbound", "")).strip().lower() == "true"
                in_resp = str(row.get("in_response_to_tweet_id", "")).strip()
                # Clean in_resp if float string like 115712.0
                if in_resp.endswith(".0"):
                    in_resp = in_resp[:-2]

                text = row.get("text", "").strip()
                created_at = row.get("created_at", "").strip()

                record = {
                    "tweet_id": tweet_id,
                    "author_id": author_id,
                    "inbound": inbound,
                    "in_response_to_tweet_id": in_resp if in_resp and in_resp != "nan" else None,
                    "text": text,
                    "created_at": created_at,
                }

                tweets_by_id[tweet_id] = record

                if author_id == self.brand_name and record["in_response_to_tweet_id"]:
                    brand_replies.append(record)

        logger.info(f"Found {len(brand_replies)} replies by @{self.brand_name}.")

        # Pass 2: Reconstruct threads
        threads: List[ConversationThread] = []
        for brand_reply in brand_replies:
            parent_id = brand_reply["in_response_to_tweet_id"]
            if not parent_id or parent_id not in tweets_by_id:
                continue

            customer_tweet = tweets_by_id[parent_id]
            # Verify parent was an inbound tweet from customer
            if not customer_tweet["inbound"]:
                continue

            # Look for optional customer follow-up (where in_response_to is this brand_reply)
            # Find any tweet in tweets_by_id where in_response_to_tweet_id == brand_reply['tweet_id']
            # We can scan or match if available
            customer_followup = None
            for tid, t_data in tweets_by_id.items():
                if t_data.get("in_response_to_tweet_id") == brand_reply["tweet_id"] and t_data["inbound"]:
                    customer_followup = t_data
                    break

            thread = ConversationThread(
                thread_id=f"thread_{customer_tweet['tweet_id']}_{brand_reply['tweet_id']}",
                customer_tweet_id=customer_tweet["tweet_id"],
                customer_text=customer_tweet["text"],
                brand_reply_id=brand_reply["tweet_id"],
                brand_reply_text=brand_reply["text"],
                customer_followup_id=customer_followup["tweet_id"] if customer_followup else None,
                customer_followup_text=customer_followup["text"] if customer_followup else None,
                created_at=customer_tweet.get("created_at"),
            )
            threads.append(thread)

            if max_threads and len(threads) >= max_threads:
                break

        logger.info(f"Successfully reconstructed {len(threads)} conversation threads.")
        return threads

    def save_threads_jsonl(self, threads: List[ConversationThread], output_path: str):
        """Save reconstructed threads to a JSONL file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for thread in threads:
                f.write(json.dumps(asdict(thread), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(threads)} threads to {output_path}")

    @staticmethod
    def load_threads_jsonl(input_path: str) -> List[ConversationThread]:
        """Load reconstructed threads from a JSONL file."""
        threads = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    threads.append(ConversationThread(**data))
        return threads
