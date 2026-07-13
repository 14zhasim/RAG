from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, STOPWORDS
import string
from nltk.stem import PorterStemmer
from pathlib import Path
import pickle

class InvertedIndex:
    def __init__(self):
        self.index = {} # map tokens to sets of document IDs
        self.docmap = {} # map document ID to full document object
    
    def __add_document(self, doc_id, text):
        tokens = preprocess_text(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
    
    def get_documents(self, term):
        if term in self.index:
            doc_id_list = list(self.index.get(term, set()))
            doc_id_list.sort()
            return doc_id_list

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self):
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        with open(cache_dir / "index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open(cache_dir / "docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

stemmer = PorterStemmer()

def build_command() -> int:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    return inverted_index.get_documents("merida")[0]

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    stop_words = STOPWORDS
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

def tokenize_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text_tokens = text.split()
    return [token for token in text_tokens if token]

def preprocess_text(text: str) -> list[str]:
    return [stemmer.stem(token) for token in tokenize_text(text)]

def preprocess_list(texts: list[str]) -> list[str]:
    tokens = []
    for text in texts:
        tokens.extend(tokenize_text(text))
    return tokens

def has_matching_token(query_tokens: list[str], title_tokens: list[str], stop_word_token) -> bool:
    token_bool = any(
            query_token in title_token
            for query_token in query_tokens
            for title_token in title_tokens
            if (query_token not in stop_word_token) and (title_token not in stop_word_token)
        )
    return token_bool
