import dataclasses
from typing import Type, Mapping, Sequence, Dict, List, NamedTuple, Any
import tqdm
import numpy as np
import matplotlib.pyplot as plt

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
        step_sizes: List[int]
) -> ResultsByK:
    """
    Tests retrieval system for different values of k and returns a set of results for each such k
    :param scores_by_query_id: map that contains pair of scores keyed by query id
    :param k_values: set of values of k to run the retrieval system on
    :param idx_to_doc_id: map that contains actual document ids
        (i.e. from the original dataset) keyed by document indices
    :param step_sizes: list of step sizes to use when iterating over the range of k_prime values
    :return:
    """

    results_by_k: ResultsByK = {}

    for k in k_values:
        results = []  # List[Result]

        # For each query, get the top documents from the sparse and dense system
        #   and calculate the recall with respect to the ground truth as a function of k'
        for q_id, scores in tqdm.tqdm(scores_by_query_id.items(), desc=f"Processing k={k}", unit="query"):
            dense, sparse = scores
            dense = normalize_scores(scores=dense)
            sparse = normalize_scores(scores=sparse)

            ground_truth = get_ground_truth(sparse_scores=sparse, dense_scores=dense, k=k)

            n_docs = len(sparse)
            recall_by_k_prime = {}  # Dict[int, float]

            # Iterate over the range of k_prime values with gradually increasing step size
            k_prime_values = []
            step_index = 0
            next_k_prime = k
            while next_k_prime <= n_docs:
                k_prime_values.append(next_k_prime)
                next_k_prime += step_sizes[step_index]
                step_index = min(step_index + 1, len(step_sizes) - 1)

            # Calculate recall for each k_prime value
            for k_prime in k_prime_values:
                approx_top_k = get_approximate_top_k(sparse_scores=sparse, dense_scores=dense, k=k, k_prime=k_prime)
                recall = get_recall(ground_truth_doc_ids=ground_truth, approximate_top_doc_ids=approx_top_k)
                recall_by_k_prime[k_prime] = recall

            # use a dictionary comprehension to create idx_to_doc_id
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


def plot_mean_recall_vs_k_prime(results_by_k: ResultsByK, split_plots: bool = False) -> None:
    """
    Plots the mean recall values for each k_prime value across all queries, for each k value in the
    results_by_k dictionary
    :param results_by_k: A dictionary where the keys are the k values and the values are lists of Result.
        Each Result contains the recall values for a single query at various values of k_prime
    :param split_plots: If True, creates a separate subplot for each k value. Plots all on a single plot otherwise.
    """
    # create a figure with subplots if separate_plots is True
    n_plots = len(results_by_k)
    n_cols = 3
    n_rows = (n_plots + 2) // n_cols
    if split_plots:
        fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(12, 4 * n_rows))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#18BFD0']

    # loop through results_by_k dictionary
    for i, (k_val, results) in enumerate(results_by_k.items()):
        # dictionary to store the recall values for each query at each k_prime value
        recall_vals_by_k_prime = {}
        # loop through results for each query
        for result in results:
            recall_by_k_prime = result.recall_by_k_prime
            # loop through recall values for each k_prime and add them to the dictionary for this query
            for k_prime, recall in recall_by_k_prime.items():
                if k_prime not in recall_vals_by_k_prime:
                    recall_vals_by_k_prime[k_prime] = []
                recall_vals_by_k_prime[k_prime].append(recall)

        # compute mean recall values at each k_prime value for all queries
        mean_recall_vals_by_k_prime = {k_prime: np.mean(recall_vals) for k_prime, recall_vals in
                                       recall_vals_by_k_prime.items()}

        # find the index of the first k_prime value where recall is 1
        max_k_prime = max(mean_recall_vals_by_k_prime.keys())
        max_recall = max(mean_recall_vals_by_k_prime.values())
        first_k_prime_with_max_recall = next((k_prime for k_prime in range(1, max_k_prime + 1) if
                                              k_prime not in mean_recall_vals_by_k_prime or
                                              mean_recall_vals_by_k_prime[k_prime] < max_recall), max_k_prime + 1)
        last_k_prime = min(first_k_prime_with_max_recall + 10, max_k_prime + 1)

        # plot the mean recall values for each k_prime value
        if split_plots:
            row_num = i // n_cols
            col_num = i % n_cols
            axs[row_num, col_num].plot(list(mean_recall_vals_by_k_prime.keys())[:last_k_prime],
                                       list(mean_recall_vals_by_k_prime.values())[:last_k_prime],
                                       label=f"k={k_val}",
                                       color=colors[i % len(colors)])
            # set axis labels and title for this subplot
            axs[row_num, col_num].set_xlabel("k_prime")
            axs[row_num, col_num].set_ylabel("mean recall")
            axs[row_num, col_num].set_title(f"Mean Recall vs k_prime for k={k_val}")
            # set y-axis limit from 0 to 1.05 for this subplot
            axs[row_num, col_num].set_ylim([0, 1.05])
            # add legend for this subplot
            axs[row_num, col_num].legend()
        else:
            plt.plot(list(mean_recall_vals_by_k_prime.keys()),
                     list(mean_recall_vals_by_k_prime.values()),
                     label=f"k={k_val}",
                     color=colors[i % len(colors)])

    plt.xlabel("k_prime")
    plt.ylabel("mean recall")
    plt.title("Mean Recall vs k_prime for all k values")
    plt.ylim([0, 1.05])
    plt.xlim([-500, 20000])

    # show the plot/subplots
    if split_plots:
        # remove unused subplots
        for i in range(n_plots, n_rows * n_cols):
            axs[i // n_cols, i % n_cols].remove()
        # adjust spacing between subplots
        fig.tight_layout()
        plt.show()
    else:
        plt.legend()
        plt.show()
