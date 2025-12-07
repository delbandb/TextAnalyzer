# elvarein/cooccurrence.py
from collections import defaultdict, Counter
from typing import List, Dict

def build_cooccurrence(token_lists: List[List[str]], window: int = 3) -> Dict[str, Counter]:
    """
    Build a co-occurrence mapping: token -> Counter(neighbor -> count)
    For each token at position i, we look at tokens in [i-window, i+window], j != i.
    Returns a dict where each key is a token and value is a Counter of neighbors.
    """
    co = defaultdict(Counter)
    for tokens in token_lists:
        n = len(tokens)
        for i, u in enumerate(tokens):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                v = tokens[j]
                if v == u:
                    continue
                co[u][v] += 1
    return co

def symmetrize_cooccurrence(co: Dict[str, Counter]) -> Dict[str, Counter]:
    """
    Convert directed co-occurrence counts into symmetric counts.
    Result: for each unordered pair (u,v), both co_sym[u][v] and co_sym[v][u]
    contain the sum C[u,v] + C[v,u].
    """
    co_sym = defaultdict(Counter)
    for u, neigh in co.items():
        for v, w in neigh.items():
            co_sym[u][v] += w
            co_sym[v][u] += w
    return co_sym

def top_neighbors(co: Dict[str, Counter], token: str, k: int = 10):
    """
    Return the top-k neighbor list for a token (list of (neighbor, count)).
    If token not present, returns empty list.
    """
    if token not in co:
        return []
    return co[token].most_common(k)
