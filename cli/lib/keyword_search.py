from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies
import string


def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text_tokens = text.split()
    return [token for token in text_tokens if token]

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    preprocessed_query = preprocess_text(query)
    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        if any(
            query_token in title_token
            for query_token in preprocessed_query
            for title_token in preprocessed_title
        ):
            results.append(movie)
            if len(results) >= limit:
                break
    return results
