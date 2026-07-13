from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, STOPWORDS
import math
import string
from nltk.stem import PorterStemmer
from pathlib import Path
import pickle
from collections import Counter, defaultdict

CACHE_DIR = Path("cache")

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set) # map tokens to sets of document IDs
        self.docmap = {} # map document ID to full document object
        self.term_frequencies = defaultdict(Counter)
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap_path = CACHE_DIR / "docmap.pkl"
        self.tf_path = CACHE_DIR / "term_frequencies.pkl"
    
    def __add_document(self, doc_id, text):
        tokens = preprocess_text(text)
        
        for token in tokens:
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
        CACHE_DIR.mkdir(exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
    
    def load(self):
        with open(self.index_path, "rb") as f:        
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:        
            self.docmap = pickle.load(f)

        with open(self.tf_path, "rb") as f:
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

def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Search index not found. Run the build command first.")
        raise SystemExit(1)

    token = tokenize_term(term)
    return calculate_idf(inverted_index, token)

def calculate_idf(inverted_index: InvertedIndex, token: str) -> float:
    total_doc_count = len(inverted_index.docmap)
    term_match_doc_count = len(inverted_index.get_documents(token))
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

def calculate_tfidf(inverted_index: InvertedIndex, doc_id: int, token: str) -> float:
    tf = inverted_index.get_tf(doc_id, token)
    idf = calculate_idf(inverted_index, token)
    return tf * idf

def tfidf_command(doc_id: int, term: str) -> float:
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Search index not found. Run the build command first.")
        raise SystemExit(1)
    
    token = tokenize_term(term)
    return calculate_tfidf(inverted_index, doc_id, token)


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
