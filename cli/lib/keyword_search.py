import os
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, STOPWORDS, BM25_K1, BM25_B
import math
import string
from nltk.stem import PorterStemmer
from pathlib import Path
import pickle
from collections import Counter, defaultdict
from itertools import islice

CACHE_DIR = Path("cache")

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set) # map tokens to sets of document IDs
        self.docmap = {} # map document ID to full document object
        self.term_frequencies = defaultdict(Counter) #map doc ID to a map of words and each of their counts
        self.doc_lengths = {} #map each document to its document length (in terms of number of words)
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap_path = CACHE_DIR / "docmap.pkl"
        self.tf_path = CACHE_DIR / "term_frequencies.pkl"
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
    
    def __add_document(self, doc_id, text):
        tokens = preprocess_text(text)
        
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
    
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values())/len(self.doc_lengths)
    
    def get_documents(self, term):
        return sorted(self.index.get(term, set()))

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

        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)    
    
    def load(self):
        with open(self.index_path, "rb") as f:        
            self.index = pickle.load(f)

        with open(self.docmap_path, "rb") as f:        
            self.docmap = pickle.load(f)

        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)
    
    def get_bm25_idf(self, term: str) -> float:
        df = len(self.index[term])
        N = len(self.docmap)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: str, term: str):
        bm_tf = self.get_bm25_tf(doc_id, term)
        bm_idf = self.get_bm25_idf(term)
        return bm_tf * bm_idf
    
    def bm25_search(self, query: str, limit: int):
        query_tokens = preprocess_text(query)
        query_scores = {}

        for doc_id in self.docmap:
            bm25_score = 0
            for token in query_tokens:
                bm25_score += self.bm25(doc_id, token)
            query_scores[doc_id] = bm25_score
        
        sorted_query_scores = dict(sorted(query_scores.items(), key = lambda item: item[1], reverse=True))

        result = dict(islice(sorted_query_scores.items(), limit))

        return result

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

def tfidf_command(doc_id: int, term: str) -> float:
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Search index not found. Run the build command first.")
        raise SystemExit(1)
    
    token = tokenize_term(term)
    return calculate_tfidf(inverted_index, doc_id, token)

def calculate_tfidf(inverted_index: InvertedIndex, doc_id: int, token: str) -> float:
    tf = inverted_index.get_tf(doc_id, token)
    idf = calculate_idf(inverted_index, token)
    return tf * idf

def bm25_idf_command(term: str):
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Search index not found. Run the build command first.")
        raise SystemExit(1)

    token = tokenize_term(term)
    return inverted_index.get_bm25_idf(token)        

def bm25_tf_command(doc_id, term: str, k1=BM25_K1, b=BM25_B):
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Search index not found. Run the build command first.")
        raise SystemExit(1)
    
    token = tokenize_term(term)
    return inverted_index.get_bm25_tf(doc_id, token, k1, b)

def bm25_command(query: str, limit: int):
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Search index not found. Run the build command first.")
        raise SystemExit(1)
    
    return inverted_index.bm25_search(query, limit)

def get_movie_name_command(doc_id):
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Search index not found. Run the build command first.")
        raise SystemExit(1)
    
    return inverted_index.docmap[doc_id]['title']

###

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
    stop_words = set(preprocess_list(STOPWORDS))
    return [
        stemmer.stem(token)
        for token in tokenize_text(text)
        if token not in stop_words
    ]

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
