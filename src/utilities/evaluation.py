import dataclasses
from typing import Type, Tuple, Mapping, Sequence, Dict, List, NamedTuple, Any

import numpy as np

import src.utilities.scores as sc


DocumentIndices: Type = np.ndarray[int]


@dataclasses.dataclass
class Result:
    query_id: str
    recall_by_k_prime: Dict[int, float]
    ground_truth: DocumentIndices

    idx_to_doc_id: Dict[int, str]
    """
    Dictionary that maps document indices to their actual id in the original dataset
    """


ResultsByK: Type = Dict[int, List[Result]]


class ScoresPair(NamedTuple):
    dense: sc.Scores
    sparse: sc.Scores


def get_dataset_results(
        scores_by_query_id: Mapping[Any, ScoresPair],
        k_values: Sequence[int],
        idx_to_doc_id: Mapping[int, Any],
) -> ResultsByK:
    """
    Tests retrieval system for different values of k and returns a set of results for each such k.

    :param scores_by_query_id: map that contains pair of scores keyed by query id
    :param k_values: set of values of k to run the retrieval system on
    :param idx_to_doc_id: map that contains actual document ids
        (i.e. from the original dataset) keyed by document indices
    :return:
    """

    results_by_k: ResultsByK = {}

    for k in k_values:
        results: List[Result] = []

        # For each query, get the top documents from the sparse and dense system
        #   and calculate the recall with respect to the ground truth as a function of k'
        for q_id, scores in scores_by_query_id.items():
            dense, sparse = scores
            dense = normalize_scores(scores=dense)
            sparse = normalize_scores(scores=sparse)

            ground_truth = get_ground_truth(sparse_scores=sparse, dense_scores=dense, k=k)

            n_docs = len(sparse)
            recall_by_k_prime: Dict[int, float] = {}
            for k_prime in range(k, n_docs):
                approx_top_k = get_approximate_top_k(sparse_scores=sparse, dense_scores=dense, k=k, k_prime=k_prime)
                recall = get_recall(ground_truth_doc_ids=ground_truth, approximate_top_doc_ids=approx_top_k)

                recall_by_k_prime[k_prime] = recall

            query_result = Result(
                query_id=q_id,
                recall_by_k_prime=recall_by_k_prime,
                ground_truth=ground_truth,
                idx_to_doc_id={idx: idx_to_doc_id[idx] for idx in ground_truth}
            )
            results.append(query_result)

        results_by_k[k] = results

    return results_by_k


def get_ground_truth(
        sparse_scores: sc.Scores,
        dense_scores: sc.Scores,
        k: int
) -> DocumentIndices:
    merged_scores = sparse_scores + dense_scores

    return sc.get_top_k_indexes(merged_scores, k=k)


def get_approximate_top_k(
        sparse_scores: sc.Scores,
        dense_scores: sc.Scores,
        k: int,
        k_prime: int
) -> DocumentIndices:
    """
    Calculate approximate top k as the top k documents in the set union of the
        top k' dense and top k' sparse document scores

    :param sparse_scores:
    :param dense_scores:
    :param k:
    :param k_prime:
    :return:
    """

    assert k_prime >= k, "k_prime must be >= k"

    top_k_prime_sparse_indexes = sc.get_top_k_indexes(sparse_scores, k=k_prime)
    top_k_prime_dense_indexes = sc.get_top_k_indexes(dense_scores, k=k_prime)

    top_k_prime_union_indexes = np.union1d(top_k_prime_dense_indexes, top_k_prime_sparse_indexes)
    top_k_union_indexes = get_ground_truth(
        sparse_scores=sparse_scores[top_k_prime_union_indexes],
        dense_scores=dense_scores[top_k_prime_union_indexes],
        k=k
    )

    return top_k_prime_union_indexes[top_k_union_indexes]


def get_recall(ground_truth_doc_ids: DocumentIndices, approximate_top_doc_ids: DocumentIndices) -> float:
    return (len(np.intersect1d(ground_truth_doc_ids, approximate_top_doc_ids))) / len(ground_truth_doc_ids)


def normalize_scores(scores: sc.Scores) -> sc.Scores:
    """
    Returns a normalized vector of scores, where the maximum score is 1.

    :param scores:
    :return:
    """

    return scores / np.max(scores)
