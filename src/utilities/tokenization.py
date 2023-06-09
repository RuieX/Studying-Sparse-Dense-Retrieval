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
    """
    The function get a bert-base-uncased BertTokenizer,
    and applies the tokenizer to each document in docs using the tokenize function.
    :param docs:
    :return:
    """
    tokenizer = SUBWORD_TOKENIZER

    return [
        # Each TokenizedText object contains a document ID and a list of tokenized words for that document.
        TokenizedText(text_id=doc_id, tokens=tokenize(compact_document(doc), tokenizer))
        for doc_id, doc in tqdm(docs.items(), desc="Tokenizing documents")
    ]


def get_tokenized_queries(
        queries: Queries
) -> TokenizedQueries:
    """
    The function get a bert-base-uncased BertTokenizer,
    and applies the tokenizer to each query in queries using the tokenize function.
    :param queries:
    :return:
    """
    tokenizer = SUBWORD_TOKENIZER

    return [
        # Each TokenizedText object contains a document ID and a list of tokenized words for that document.
        TokenizedText(text_id=q_id, tokens=tokenize(q_text, tokenizer))
        for q_id, q_text in tqdm(queries.items(), desc="Tokenizing queries")
    ]


def compact_documents(docs: Documents) -> Sequence[str]:
    """
    Transforms each document into a compact form
    :param docs:
    :return: sequence of compacted documents
    """
    return [compact_document(d) for d in docs.values()]


def compact_document(doc: Document) -> str:
    """
    Compact form of the document
    :param doc:
    :return: string representation of the document where the title and text fields are concatenated.
    """
    return f"{doc['title']} {doc['text']}"


def tokenize(text: str, tokenizer: Tokenizer) -> Tokens:
    tokens = tokenizer.tokenize(text)
    tokens = remove_punctuation(remove_stopwords(tokens))
    return tokens


nltk.download("stopwords")  # https://pythonspot.com/nltk-stop-words/
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
PUNCTUATION = set([c for c in string.punctuation])


def remove_stopwords(tokens: Tokens) -> Tokens:
    return [t for t in tokens if t not in STOPWORDS]


def remove_punctuation(tokens: Tokens) -> Tokens:
    return [t for t in tokens if t not in PUNCTUATION]
