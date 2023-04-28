import string
from typing import Sequence, Type, Protocol, Dict, Tuple, NamedTuple

import nltk
from transformers import BertTokenizer
from tqdm import tqdm


Document: Type = Dict[str, str]
"""
{
    title: <title>
    text: <text>
}
"""

Documents: Type = Dict[str, Document]
"""
  {
    doc_id: {
      title: <title>
      text: <text>
    },
  }
"""

Queries: Type = Dict[str, str]
"""
  {
    query_id: <text>,
  }
"""

GroundTruth: Type = Dict[str, Dict[str, int]]
"""
  {
    query_id: {
      doc_id: <score>,
    },
  }
"""

Dataset: Type = Tuple[Documents, Queries, GroundTruth]

Tokens: Type = Sequence[str]


class TokenizedText(NamedTuple):
    text_id: str
    tokens: Tokens


TokenizedDocuments: Type = Sequence[TokenizedText]
TokenizedQueries: Type = Sequence[TokenizedText]


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> Tokens:
        pass


SUBWORD_TOKENIZER: Tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# https://huggingface.co/docs/transformers/tokenizer_summary#subword-tokenization


def get_tokenized_documents(
        docs: Documents
) -> TokenizedDocuments:
    tokenizer = SUBWORD_TOKENIZER

    return [
        TokenizedText(text_id=doc_id, tokens=tokenize(compact_document(doc), tokenizer))
        for doc_id, doc in tqdm(docs.items(), desc="Tokenizing documents")
    ]
# Each TokenizedText object contains a document ID and a list of tokenized words for that document.
# The function uses the get_tokenizer function to get a tokenizer based on the tokenization_type,
# and then applies that tokenizer to each document in docs using the tokenize function.


def get_tokenized_queries(
        queries: Queries
) -> TokenizedQueries:
    tokenizer = SUBWORD_TOKENIZER

    return [
        TokenizedText(text_id=q_id, tokens=tokenize(q_text, tokenizer))
        for q_id, q_text in tqdm(queries.items(), desc="Tokenizing queries")
    ]


def compact_documents(docs: Documents) -> Sequence[str]:
    """
    Transforms each document into a single string, where "title" and "text" fields
        are "compacted" into a single text body
    :param docs: documents to process
    :return: compacted documents: each document is now a single string instead of a dictionary
    """

    return [compact_document(d) for d in docs.values()]


def compact_document(doc: Document) -> str:
    return f"{doc['title']} {doc['text']}"


def tokenize(text: str, tokenizer: Tokenizer) -> Tokens:
    tokens = tokenizer.tokenize(text)
    tokens = remove_punctuation(remove_stopwords(tokens))

    return tokens


# Note: Python's set() has O(1) membership checking
nltk.download("stopwords")  # https://pythonspot.com/nltk-stop-words/
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
PUNCTUATION = set([c for c in string.punctuation])


def remove_stopwords(tokens: Tokens) -> Tokens:
    return [t for t in tokens if t not in STOPWORDS]


def remove_punctuation(tokens: Tokens) -> Tokens:
    return [t for t in tokens if t not in PUNCTUATION]
