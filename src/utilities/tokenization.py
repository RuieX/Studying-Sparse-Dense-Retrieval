import enum
import string
from typing import Sequence, Type, Protocol, Dict, Tuple, NamedTuple

import nltk
from nltk import RegexpTokenizer
from transformers import BertTokenizer
from tqdm import tqdm


# Some typedefs to better grasp how the dataset is actually structured

GroundTruth: Type = Dict[str, Dict[str, int]]
"""
Example::

  {
    query_id: {
      doc_id: score,
    },
  }

Note: this won't be used in actuality because we will calculate our own ground truth
"""

Queries: Type = Dict[str, str]
"""
Example::

  {
    query_id: text,
  }

"""

Document: Type = Dict[str, str]
"""
Example::

{
    title: <title>
    text: <text>
}

"""

Documents: Type = Dict[str, Document]
"""
Example::

  {
    doc_id: {
      title: <title>
      text: <text>
    },
  }

"""

Dataset: Type = Tuple[Documents, Queries, GroundTruth]


class TokenizationType(enum.Enum):
    Word = 0
    """
    Simple word tokenization: sentences are split based on whitespaces and punctuation.
    Punctuation is removed
    """

    Subword = 1
    """
    Tokenization technique were common words are kept as a whole, but uncommon terms are split into
    more frequent sub-words. This helps to reduce vocabulary size
    """


Tokens: Type = Sequence[str]


class TokenizedText(NamedTuple):
    text_id: str
    tokens: Tokens


# Some typedef utils
TokenizedDocuments: Type = Sequence[TokenizedText]
TokenizedQueries: Type = Sequence[TokenizedText]


def get_tokenized_documents(
        docs: Documents,
        tokenization_type=TokenizationType.Subword
) -> TokenizedDocuments:
    tokenizer = get_tokenizer(tokenization_type)

    return [
        TokenizedText(text_id=doc_id, tokens=tokenize(compact_document(doc), tokenizer))
        for doc_id, doc in tqdm(docs.items(), desc="Tokenizing documents")
    ]
# Each TokenizedText object contains a document ID and a list of tokenized words for that document.
# The function uses the get_tokenizer function to get a tokenizer based on the tokenization_type,
# and then applies that tokenizer to each document in docs using the tokenize function.


# TODO no cleaning?
def get_tokenized_queries(
        queries: Queries,
        tokenization_type=TokenizationType.Subword
) -> TokenizedQueries:
    tokenizer = get_tokenizer(tokenization_type)

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


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> Tokens:
        pass


WORD_TOKENIZER: Tokenizer = RegexpTokenizer(r'\w+')

SUBWORD_TOKENIZER: Tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# https://huggingface.co/docs/transformers/tokenizer_summary#subword-tokenization


def get_tokenizer(tokenization_type: TokenizationType) -> Tokenizer:
    match tokenization_type:
        case TokenizationType.Word:
            return WORD_TOKENIZER
        case TokenizationType.Subword:
            return SUBWORD_TOKENIZER
        case _:
            raise Exception(f"Undefined tokenizer for {tokenization_type}")


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
