from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_FILE = ROOT / ".cache" / "extraction.jsonl"
ENV_FILE = ROOT / ".env"

REVIEW_COLUMN = "REVIEW TRANSLATED"

EXTRACTION_MODEL = "claude-haiku-4-5"
SYNTHESIS_MODEL = "claude-opus-5"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 20
MAX_WORKERS = 4

# Cosine distance above which two groups of insights are not merged.
# Kept low on purpose: the model can merge groups that are too narrow,
# but cannot split a group that mixes two topics.
DISTANCE_THRESHOLD = 0.25
MIN_CLUSTER_SIZE = 3
MAX_CLUSTERS = 60
MAX_THEMES = 15
QUOTES_PER_THEME = 5
