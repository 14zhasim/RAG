import numpy as np
from .search_utils import CACHE_DIR, load_movies, format_search_result
import re
import json
import os
from .semantic_search import SemanticSearch, cosine_similarity

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
        
    def search_chunks(self, query: str, limit: int = 10):
        semantic_search = SemanticSearch()
        if self.chunk_embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        query_embedding = semantic_search.generate_embedding(query)

        chunk_score = []
            
        for i in range(len(self.chunk_embeddings)):
            similarity_score = cosine_similarity(query_embedding, self.chunk_embeddings[i])

            chunk_score.append({
                "chunk_idx": self.chunk_metadata[i]['chunk_idx'],
                "movie_idx": self.chunk_metadata[i]['movie_idx'],
                "total_chunks": self.chunk_metadata[i]['total_chunks'],
                "score": similarity_score
            })
        
        movie_chunk_score = {}
        movie_chunk_metadata = {}

        for score in chunk_score:
            if score['movie_idx'] not in movie_chunk_score or movie_chunk_score[score['movie_idx']] < score['score']:
                movie_chunk_score[score['movie_idx']] = score['score']
                movie_chunk_metadata[score['movie_idx']] = {
                    "chunk_idx": score["chunk_idx"],
                    "total_chunks": score["total_chunks"],
                }

        sorted_score = sorted(
            movie_chunk_score.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            format_search_result(
                self.documents[movie_idx]["id"],
                self.documents[movie_idx]["title"],
                self.documents[movie_idx]["description"],
                score,
                movie_chunk_metadata[movie_idx],
            )
            for movie_idx, score in sorted_score[:limit]
        ]
        
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
        embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies)
        print(f"Generated {len(embeddings)} chunked embeddings")
    except FileNotFoundError:
        print("Chunk embeddings not found. Run the build command first, or make sure entered valid documents to chunk.")
        raise SystemExit(1)

def search_chunked_command(query, limit):
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    results = chunked_semantic_search.search_chunks(query, limit)

    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")
