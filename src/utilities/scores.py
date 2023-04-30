from typing import Type, List, Protocol

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import src.utilities.tokenization as tkn

Scores: Type = np.ndarray[float]
SparseEmbeddings: Type = np.ndarray[float]
DenseEmbedding: Type = torch.Tensor


class DocumentScorer(Protocol):
    def __init__(self, **kwargs):
        pass

    def get_scores(self, query: tkn.Tokens) -> Scores:
        pass


def get_sparse_scores(
        query: tkn.TokenizedText,
        scorer: DocumentScorer
) -> Scores:
    """
    Returns an array with #docs entries, each representing the score of that
        doc for the provided query
    :param query: query to be scored against
    :param scorer: document scorer used to score the query against the set of documents
    :return: array of scores, one for each document
    """
    return scorer.get_scores(query=query.tokens)


def get_dense_doc_embeddings(
        docs: tkn.Documents
) -> List[DenseEmbedding]:
    """
    Documents embeddings using pre-trained model
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#usage-sentence-transformers
    :param docs: document to be scored
    :return:
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    doc_embeddings: List[DenseEmbedding] = model.encode(list(tkn.compact_documents(docs)),
                                                        convert_to_numpy=False,
                                                        convert_to_tensor=False)
    return doc_embeddings


def get_dense_scores(
        query: str,
        doc_embeddings: List[DenseEmbedding]
) -> Scores:
    """
    Returns an array with the score of the documents for the given query.
    The given query is encoded into a dense embedding using the pre-trained model
    https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#usage-sentence-transformers
    :param query: query to be scored against
    :param doc_embeddings: embedded documents
    :return: array of scores, one for each document
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_embeddings: DenseEmbedding = model.encode(query, convert_to_numpy=False, convert_to_tensor=False)
    scores = np.zeros(len(doc_embeddings))

    for i, d_embeds in enumerate(doc_embeddings):
        score = torch.dot(query_embeddings, d_embeds).item()
        scores[i] = score

    return scores
