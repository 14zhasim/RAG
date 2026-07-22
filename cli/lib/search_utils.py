import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOVIE_PATH = PROJECT_ROOT/'data'/'movies.json'
STOP_WORD_PATH = PROJECT_ROOT/'data'/'stopwords.txt'
DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
CACHE_DIR = Path("cache")
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP_SIZE = 0
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
SCORE_PRECISION = 2


def load_movies():
    with open(MOVIE_PATH, "r") as f:
        data = json.load(f)
        return data["movies"]

def load_stop_words() -> list[str]:
    with open(STOP_WORD_PATH, "r") as f:
        data = f.read().splitlines()
        return data

def format_search_result(doc_id, title, document, score, metadata=None):
    return {
        "id": doc_id,
        "title": title,
        "document": document[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }

STOPWORDS = load_stop_words()
