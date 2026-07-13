from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stop_words
import string


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    stop_words = load_stop_words()
    results = []
    preprocessed_stop_words = preprocess_list(stop_words)
    preprocessed_query = preprocess_text(query)
    for movie in movies:
        preprocessed_title = preprocess_text(movie["title"])
        if has_matching_token(preprocessed_query, preprocessed_title, preprocessed_stop_words):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text_tokens = text.split()
    return [token for token in text_tokens if token]

def preprocess_list(texts: list[str]) -> list[str]:
    for text in texts:
        preprocess_text(text)
    return [text for text in texts if text]

def has_matching_token(query_tokens: list[str], title_tokens: list[str], stop_word_token) -> bool:
    token_bool = any(
            query_token in title_token
            for query_token in query_tokens
            for title_token in title_tokens
            if (query_token not in stop_word_token) and (title_token not in stop_word_token)
        )
    return token_bool