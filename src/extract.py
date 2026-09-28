import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from pydantic import BaseModel
from tqdm import tqdm

from config import BATCH_SIZE, CACHE_FILE, EXTRACTION_MODEL, MAX_WORKERS
from llm import ask
from prompts import EXTRACTION_SYSTEM, EXTRACTION_USER


class Insight(BaseModel):
    review_id: int
    insight: str
    quote: str
    sentiment: Literal["positive", "negative", "mixed"]


class Extraction(BaseModel):
    insights: list[Insight]


def normalize(text):
    return " ".join(text.lower().split())


def is_verbatim(quote, review):
    quote = normalize(quote)
    return bool(quote) and quote in normalize(review)


def batch_key(reviews, batch):
    content = "\n".join(f"{i}:{reviews[i]}" for i in batch)
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def load_cache():
    if not CACHE_FILE.exists():
        return {}
    cache = {}
    for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
        # a line can be cut short if the script was killed while writing
        try:
            entry = json.loads(line)
            cache[entry["key"]] = entry["insights"]
        except (ValueError, KeyError, TypeError):
            continue
    return cache


def extract_batch(client, reviews, batch):
    block = "\n\n".join(f'<review id="{i}">\n{reviews[i]}\n</review>' for i in batch)
    prompt = EXTRACTION_USER.format(reviews=block)
    result = ask(client, EXTRACTION_MODEL, EXTRACTION_SYSTEM, prompt, Extraction)
    return [insight.model_dump() for insight in result.insights]


def extract_insights(client, reviews):
    """Returns the insights whose quote was found in the review, and the number rejected."""
    ids = list(reviews)
    batches = [ids[i : i + BATCH_SIZE] for i in range(0, len(ids), BATCH_SIZE)]
    cache = load_cache()

    found = []
    todo = {}
    for batch in batches:
        key = batch_key(reviews, batch)
        if key in cache:
            found += cache[key]
        else:
            todo[key] = batch
    if len(todo) < len(batches):
        print(f"{len(batches) - len(todo)} of {len(batches)} batches already done, resuming.")

    CACHE_FILE.parent.mkdir(exist_ok=True)
    with ThreadPoolExecutor(MAX_WORKERS) as pool, CACHE_FILE.open("a", encoding="utf-8") as f:
        futures = {
            pool.submit(extract_batch, client, reviews, batch): key
            for key, batch in todo.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading reviews"):
            insights = future.result()
            f.write(json.dumps({"key": futures[future], "insights": insights}) + "\n")
            f.flush()
            found += insights

    kept = [
        i
        for i in found
        if i["review_id"] in reviews and is_verbatim(i["quote"], reviews[i["review_id"]])
    ]
    kept.sort(key=lambda i: i["review_id"])
    return kept, len(found) - len(kept)
