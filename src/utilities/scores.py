from typing import Type, List, Protocol

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import src.utilities.tokenization as tok

Scores: Type = np.ndarray[float]
SparseEmbeddings: Type = np.ndarray[float]
DenseEmbedding: Type = torch.Tensor


def get_dense_embeddings_scores(
        query: str,
        docs: tok.Documents
) -> Scores:
    """
    Returns an array with #docs entries, each representing the score of that
        doc for the provided query.
    The provided text is not tokenized because the underlying pre-trained transformer
        handles text tokenization by itself

    :param query: query to calculate the scores against
    :param docs: document to calculate the scores for
    :return: array of scores, one for each document
    """

    # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#usage-sentence-transformers
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embeddings: DenseEmbedding = model.encode(
        query,
        convert_to_numpy=False,
        convert_to_tensor=False
    )
    doc_embeddings: List[DenseEmbedding] = model.encode(
        tok.compact_documents(docs),
        convert_to_numpy=False,
        convert_to_tensor=False
    )

    scores = np.zeros(len(doc_embeddings))
    for i, d_embs in enumerate(doc_embeddings):
        score = torch.dot(query_embeddings, d_embs).item()

        scores[i] = score

    return scores


class DocumentScorer(Protocol):
    def __init__(self, corpus: tok.TokenizedDocuments, **kwargs):
        pass

    def get_scores(self, query: tok.Tokens) -> Scores:
        pass


def get_sparse_embeddings_scores(
        query: tok.TokenizedText,
        scorer: DocumentScorer
) -> Scores:
    """
    Returns an array with #docs entries, each representing the score of that
        doc for the provided query.

    :param query: query to calculate the scores against
    :param scorer: document scorer, defaults to a bm25-based scorer
    :return: array of scores, one for each document
    """

    return scorer.get_scores(query=query.tokens)


def get_top_k_indexes(a: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-a)[:k]
