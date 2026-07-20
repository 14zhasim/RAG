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

def load_movies():
    with open(MOVIE_PATH, "r") as f:
        data = json.load(f)
        return data["movies"]

def load_stop_words() -> list[str]:
    with open(STOP_WORD_PATH, "r") as f:
        data = f.read().splitlines()
        return data

STOPWORDS = load_stop_words()