from sentence_transformers import SentenceTransformer
import numpy as np
from .search_utils import CACHE_DIR, load_movies
import os
import json

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

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
        np.save(CACHE_DIR / 'movie_embeddings.npy', self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc            
        if os.path.exists(CACHE_DIR / 'movie_embeddings.npy'):
            self.embeddings = np.load(CACHE_DIR / 'movie_embeddings.npy')
            if len(self.embeddings == len(self.documents)):
                return self.embeddings
            else:
                self.build_embeddings(documents)
        else:
            self.build_embeddings(documents)

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