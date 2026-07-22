from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import CACHE_DIR, load_movies
import os
import re

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None #for each movie, a list of embeddings of movie title and description
        self.documents = None #list of document objects
        self.document_map = {} #map doc id to their object

    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError('no text provided to embed')
        embedding = self.model.encode([text]) 
        #So embedding after that line isn't a single vector — it's a 2D array with 
        # shape (1, 384) (1 row, 384 dimensions). To get the actual single vector out 
        # (shape (384,)), you need to grab the first (and only) row, which is what embedding[0] does.
        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        movie_strings = []
        for doc in documents:
            self.document_map[doc['id']] = doc            
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(CACHE_DIR / 'movie_embeddings.npy', self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc            
        if os.path.exists(CACHE_DIR / 'movie_embeddings.npy'):
            self.embeddings = np.load(CACHE_DIR / 'movie_embeddings.npy')
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
            else:
                self.build_embeddings(documents)
        else:
            self.build_embeddings(documents)
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        
        search_results = []

        for i in range(len(self.embeddings)):
            similarity_score = cosine_similarity(query_embedding, self.embeddings[i])
            document = self.documents[i]
            search_results.append((similarity_score, document))

        search_results.sort(key=lambda result: result[0], reverse=True)

        return [
            {
                "score": similarity_score,
                "title": document["title"],
                "description": document["description"],
            }
            for similarity_score, document in search_results[:limit]
        ]


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(query, limit=5):
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies) 
    result = semantic_search.search(query, limit)
    for i in range(len(result)):
        print(f"{i+1}. {result[i]["title"]} (score: {result[i]["score"]}) \n" +
             f"\t{result[i]["description"]}")
        