import os

from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.inverted_index = InvertedIndex()
        if not os.path.exists(self.inverted_index.index_path):
            self.inverted_index.build()
            self.inverted_index.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.inverted_index.load()
        return self.inverted_index.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_score(*scores):
    score_list = list(scores)
    
    if not score_list:
        return []

    if min(score_list) == max(score_list):
        return [1.0] * len(score_list)
    
    max_score = max(score_list)
    min_score = min(score_list)
    return [(score - min_score) / (max_score - min_score) for score in score_list]
