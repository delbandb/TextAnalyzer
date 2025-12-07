# elvarein/tfidf.py
import math
from collections import Counter
from typing import List, Tuple, Dict

def build_doc_term_counts(texts: List[str], tokenize_fn) -> Tuple[List[List[str]], List[Counter], List[int]]:
    """
    Input:
      - texts: list of raw document strings
      - tokenize_fn: function that returns token list for a text
    Returns:
      - doc_tokens: list of token lists
      - doc_counts: list of Counter objects (f_d(w))
      - doc_lengths: list of N_d (total tokens per doc)
    """
    doc_tokens = [tokenize_fn(t) for t in texts]
    doc_counts = [Counter(toks) for toks in doc_tokens]
    doc_lengths = [sum(c.values()) for c in doc_counts]
    return doc_tokens, doc_counts, doc_lengths

def compute_df(doc_counts: List[Counter]) -> Counter:
    """
    Document frequency df(w): number of documents where w appears at least once.
    """
    df = Counter()
    for c in doc_counts:
        for term in c.keys():
            df[term] += 1
    return df

def compute_idf(df: Counter, D: int) -> Dict[str, float]:
    """
    Smoothed idf:
      idf(w) = log( 1 + D / (1 + df(w)) )
    Returns dict term -> idf value.
    """
    idf = {}
    for term, dfv in df.items():
        idf[term] = math.log(1.0 + (D / (1.0 + dfv)))
    return idf

def compute_tfidf_per_doc(doc_counts: List[Counter], doc_lengths: List[int], idf: Dict[str, float]) -> List[Dict[str, float]]:
    """
    Compute tf-idf per document:
      tf_norm = f_d(w) / N_d
      tfidf = tf_norm * idf(w)
    Returns list of dicts (one dict per document).
    """
    tfidf_docs = []
    for c, N in zip(doc_counts, doc_lengths):
        tfidf = {}
        for term, freq in c.items():
            tf_norm = freq / N if N > 0 else 0.0
            tfidf[term] = tf_norm * idf.get(term, 0.0)
        tfidf_docs.append(tfidf)
    return tfidf_docs

def top_k(d: Dict[str, float], k: int = 10):
    """
    Return top-k items from a dict sorted by value descending.
    """
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
