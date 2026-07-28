import argparse
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies, load_test, RRF_SEARCH_PARAMETER

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
    type=int,
    default=5,
    help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    movies = load_movies()
    tests = load_test()

    hybrid_search = HybridSearch(movies)

    print(f"k={limit}")

    for test in tests:
        relevant_result_count = 0

        query = test['query']
        answers = test['relevant_docs']
        results = hybrid_search.rrf_search(query, RRF_SEARCH_PARAMETER, limit)
        result_titles = [result['document']['title'] for result in results]

        for title in result_titles:
            if title in answers: 
                relevant_result_count += 1

        precision = relevant_result_count / len(results)
        recall = relevant_result_count / len(answers)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print("  - Retrieved:", ', '.join(result_titles))
        print("  - Relevant:", ', '.join(answers))

        





if __name__ == "__main__":
    main()
