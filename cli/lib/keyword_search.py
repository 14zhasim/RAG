from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, STOPWORDS
import string
from nltk.stem import PorterStemmer
from pathlib import Path
import pickle
from collections import Counter

class InvertedIndex:
    def __init__(self):
        self.index = {} # map tokens to sets of document IDs
        self.docmap = {} # map document ID to full document object
        self.term_frequencies = {}
    
    def __add_document(self, doc_id, text):
        tokens = preprocess_text(text)
        self.term_frequencies[doc_id] = Counter()
        
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
    
    def get_documents(self, term):
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

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
        
        with open(cache_dir / "term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
    
    def load(self):
        cache_dir = Path("cache")

        with open(cache_dir / "index.pkl", "rb") as f:        
            self.index = pickle.load(f)

        with open(cache_dir / "docmap.pkl", "rb") as f:        
            self.docmap = pickle.load(f)

        with open(cache_dir / "term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)

stemmer = PorterStemmer()

def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Search index not found. Run the build command first.")
        raise SystemExit(1)

    results = []
    seen_doc_ids = set()
    processed_stop_words = preprocess_list(STOPWORDS)
    processed_query = preprocess_text(query)
    
    for q in processed_query:
        if q in processed_stop_words:
            continue

        for doc_id in inverted_index.get_documents(q):
            if doc_id in seen_doc_ids:
                continue

            results.append(inverted_index.docmap[doc_id])
            seen_doc_ids.add(doc_id)

            if len(results) >= limit:
                return results

    return results

def tf_command(doc_id: int, term: str) -> int:
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Search index not found. Run the build command first.")
        raise SystemExit(1)

    token = tokenize_term(term)
    return inverted_index.get_tf(doc_id, token)

def tokenize_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text_tokens = text.split()
    return [token for token in text_tokens if token]

def tokenize_term(term: str) -> str:
    tokens = preprocess_text(term)
    if len(tokens) != 1:
        raise ValueError("Term must tokenize to exactly one token")
    return tokens[0]

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
