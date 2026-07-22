import numpy as np
from .search_utils import CACHE_DIR, load_movies
import re
from .semantic_search import SemanticSearch
import json
import os

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_size = 4
        self.overlap = 1

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.document_map = {}
        self.documents = documents
        chunk_list = []
        chunk_metadata = []
        
        for i, doc in enumerate(documents):
            self.document_map[doc['id']] = doc
            description = doc.get('description', "")
            if not description or not description.strip():
                continue

            chunks = semantic_chunk_text(doc['description'], self.chunk_size, self.overlap)
            chunk_list.extend(chunks)

            total_chunks = len(chunks)
            for j in range(len(chunks)):
                chunk_metadata.append({
                    'movie_idx': i,
                    'chunk_idx': j,
                    'total_chunks': total_chunks
                })
        
        self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        CACHE_DIR.mkdir(exist_ok=True)
        np.save(CACHE_DIR / 'chunk_embeddings.npy', self.chunk_embeddings)
        
        with open(CACHE_DIR / 'chunk_metadata.json', "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(chunk_list)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.document_map = {}
        self.documents = documents
        
        for document in documents:
            self.document_map[document['id']] = document
        
        if os.path.exists(CACHE_DIR / 'chunk_embeddings.npy') and os.path.exists(CACHE_DIR / 'chunk_metadata.json'):
            self.chunk_embeddings = np.load(CACHE_DIR / 'chunk_embeddings.npy')

            with open(CACHE_DIR / "chunk_metadata.json", "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

        
def chunking(text_list, chunk_size, overlap):
    start = 0
    chunks = []

    while start < len(text_list):
        chunk_words = text_list[start: start + chunk_size]

        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(' '.join(chunk_words))
        start = start - overlap + chunk_size

    return chunks

def chunk_text(text, chunk_size, overlap):
    text_list = text.split(' ')
    chunks = chunking(text_list, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")
    return chunks

def semantic_chunk_text(text, chunk_size, overlap):
    #split input text into individual *sentences* using RegEx
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = chunking(sentences, chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")
    return chunks

def embed_chunks():
    chunked_semantic_search = ChunkedSemanticSearch()
    movies = load_movies()
    try:
        return chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    except FileNotFoundError:
        print("Chunk embeddings not found. Run the build command first, or make sure entered valid documents to chunk.")
        raise SystemExit(1)

