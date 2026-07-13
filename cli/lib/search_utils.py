import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/'data'/'movies.json'
DEFAULT_SEARCH_LIMIT = 5

def load_movies():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        return data["movies"]