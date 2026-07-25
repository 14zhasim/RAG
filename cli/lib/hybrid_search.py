import os

from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from collections import defaultdict

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
        result_pool_size = limit * 500

        bm25_results = self._bm25_search(query, result_pool_size)
        semantic_results = self.semantic_search.search_chunks(query, result_pool_size)
        ranked_results = combine_search_results(
            bm25_results,
            semantic_results,
            self.inverted_index.docmap,
            self.semantic_search.document_map,
            alpha,
        )

        return ranked_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        result_pool_size = 500 * limit
        bm25_results = self._bm25_search(query, result_pool_size)
        semantic_results = self.semantic_search.search_chunks(query, result_pool_size)
        combined_rrf_results = combine_rrf_results(
            bm25_results,
            semantic_results,
            self.inverted_index.docmap,
            self.semantic_search.document_map,
            k
        )

        return combined_rrf_results[:limit]


def normalize_score(*scores):
    score_list = list(scores)
    
    if not score_list:
        return []

    if min(score_list) == max(score_list):
        return [1.0] * len(score_list)
    
    max_score = max(score_list)
    min_score = min(score_list)
    return [(score - min_score) / (max_score - min_score) for score in score_list]

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_bm25_results(bm25_results):
    normalized_scores = normalize_score(*bm25_results.values())
    return zip(bm25_results.items(), normalized_scores)

def normalize_semantic_results(semantic_results):
    normalized_scores = normalize_score(*[result["score"] for result in semantic_results])
    return zip(semantic_results, normalized_scores)

def combine_search_results(
    bm25_results,
    semantic_results,
    bm25_docmap,
    semantic_docmap,
    alpha: float = 0.5,
):
    combined_results = defaultdict(dict)

    for (doc_id, _score), normalized_score in normalize_bm25_results(bm25_results):
        combined_results[doc_id]["document"] = bm25_docmap[doc_id]
        combined_results[doc_id]["bm25_score"] = normalized_score

    for result, normalized_score in normalize_semantic_results(semantic_results):
        doc_id = result["id"]
        combined_results[doc_id]["document"] = semantic_docmap[doc_id]
        combined_results[doc_id]["semantic_score"] = max(
            combined_results[doc_id].get("semantic_score", 0.0),
            normalized_score, #If this document appears more than once in semantic results, keep the highest normalized semantic score.
        )

    hybrid_results = []
    for result in combined_results.values():
        result["bm25_score"] = result.get("bm25_score", 0.0)
        result["semantic_score"] = result.get("semantic_score", 0.0)
        result["hybrid_score"] = hybrid_score(
            result["bm25_score"],
            result["semantic_score"],
            alpha,
        )
        hybrid_results.append(result)

    sorted_results = sorted(
            hybrid_results,
            key=lambda result: result["hybrid_score"],
            reverse=True,
        )
    
    return sorted_results

def calculate_rrf(rank: int, k: int) -> float:
    return 1 / (k + rank)

def combine_rrf_results(
    bm25_results,
    semantic_results,
    bm25_docmap,
    semantic_docmap,
    k
):
    combined_results = defaultdict(dict)

    for idx, doc_id in enumerate(bm25_results, start=1):
        combined_results[doc_id]["document"] = bm25_docmap[doc_id]
        combined_results[doc_id]["bm25_rank"] = idx
        rrf_score = calculate_rrf(idx, k)
        combined_results[doc_id]["rrf_score"] = combined_results[doc_id].get("rrf_score", 0) + rrf_score
    
    for idx, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        combined_results[doc_id]["document"] = semantic_docmap[doc_id]
        combined_results[doc_id]["semantic_rank"] = idx

        rrf_score = calculate_rrf(idx, k)
        combined_results[doc_id]["rrf_score"] = combined_results[doc_id].get("rrf_score", 0) + rrf_score
    
    for result in combined_results.values():
        result["bm25_rank"] = result.get("bm25_rank", "n/a")
        result["semantic_rank"] = result.get("semantic_rank", "n/a")
    
    return sorted(
        combined_results.values(),
        key=lambda result: result["rrf_score"],
        reverse=True,
    )
