# elvarein/engine.py
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

from elvarein.io_ import load_texts, save_json
from elvarein.tokenizer import preprocess_corpus, simple_tokenize
from elvarein.tfidf import (
    build_doc_term_counts,
    compute_df,
    compute_idf,
    compute_tfidf_per_doc,
)
from elvarein.cooccurrence import (
    build_cooccurrence,
    symmetrize_cooccurrence,
)
from elvarein.graphify import (
    make_graph,
    compute_degree,
    compute_eigenvector,
    compute_communities,
    save_graph_png,
)


def _dictify_counter_map(cm: Dict[str, Counter]) -> Dict[str, Dict[str, int]]:
    """
    Convert a mapping token -> Counter(neighbor->count) into plain dicts for JSON.
    """
    return {k: dict(v) for k, v in cm.items()}


def _safe_cast_floats(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all numeric-like values are native python types (float/int).
    """
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out


class Engine:
    """
    Engine orchestrates the full pipeline:
      - load texts
      - preprocess (tokenize, stopwords, lemmatize)
      - compute TF / DF / IDF / TF-IDF
      - compute co-occurrence
      - build graph and compute metrics
      - save artifacts
    """

    def __init__(
        self,
        data_dir: Path,
        window: int = 3,
        min_edge_weight: int = 1,
        top_k_tokens: int = 20,
        extra_stopwords: List[str] = None,
        dynamic_min_doc_prop: float = 0.0,
    ):
        self.data_dir = Path(data_dir)
        self.window = window
        self.min_edge_weight = min_edge_weight
        self.top_k_tokens = top_k_tokens
        self.extra_stopwords = extra_stopwords or []
        # default dynamic_min_doc_prop is 0.0 (disabled) to avoid removing too many tokens on small corpora
        self.dynamic_min_doc_prop = float(dynamic_min_doc_prop)

    def run(self, save_outputs: bool = True) -> Dict[str, Any]:
        print("Loading texts...")
        names, texts = load_texts(self.data_dir)

        if not texts:
            raise RuntimeError(f"No .txt files found in {self.data_dir}. Aborting.")

        # -----------------------
        # Preprocessing / tokenization
        # -----------------------
        print("Preprocessing corpus (tokenization + lemmatization + stopword removal)...")
        token_lists = preprocess_corpus(
            texts,
            extra_stopwords=self.extra_stopwords,
            do_lemmatize=True,
            min_token_len=2,
            dynamic_min_doc_prop=self.dynamic_min_doc_prop,
            dynamic_top_k=0,
        )

        # Defensive: if preprocess removed all tokens, fallback to minimal tokenization
        empty_docs = all(len(toks) == 0 for toks in token_lists)
        if empty_docs:
            print("Warning: preprocessing removed all tokens. Falling back to simple tokenization.")
            token_lists = [simple_tokenize(t) for t in texts]

        # Build document counts from token_lists (these are used for TF/DF/IDF)
        doc_tokens = token_lists
        doc_counts = [Counter(toks) for toks in doc_tokens]
        doc_lengths = [sum(c.values()) for c in doc_counts]

        # -----------------------
        # TF / DF / IDF / TF-IDF
        # -----------------------
        print("Computing TF/DF/IDF...")
        df = compute_df(doc_counts)
        idf = compute_idf(df, len(doc_counts))
        tfidf_docs = compute_tfidf_per_doc(doc_counts, doc_lengths, idf)

        # -----------------------
        # Co-occurrence
        # -----------------------
        print("Building co-occurrence...")
        co = build_cooccurrence(token_lists, window=self.window)
        co_sym = symmetrize_cooccurrence(co)

        # -----------------------
        # Graph construction & metrics
        # -----------------------
        print("Constructing graph...")
        G = make_graph(co_sym, min_edge_weight=self.min_edge_weight, symmetrize=False)

        print("Computing graph metrics...")
        deg: Dict[str, int] = {}
        eig: Dict[str, float] = {}
        comm: Dict[str, int] = {}

        if G.number_of_nodes() == 0:
            print("Warning: graph is empty — skipping degree/eigenvector/community calculations.")
        else:
            deg = compute_degree(G)
            # eigenvector and community detection can fail for very small graphs; wrap in try/except
            try:
                eig = compute_eigenvector(G)
            except Exception as e:
                print("Warning: eigenvector centrality failed:", e)
                eig = {}
            try:
                comm = compute_communities(G)
            except Exception as e:
                print("Warning: community detection failed:", e)
                comm = {}

        # -----------------------
        # Save artifacts (safe JSON conversions)
        # -----------------------
        if save_outputs:
            print("Saving artifacts...")
            # df is Counter -> convert to dict
            save_json("symbols.json", dict(df))
            # cooccurrence: token -> Counter -> dict
            save_json("cooccurrence.json", _dictify_counter_map(co_sym))
            # tfidf docs are list of dicts (term->float) -> ensure floats
            save_json("tfidf.json", [{k: float(v) for k, v in td.items()} for td in tfidf_docs])
            # degree (int), eigenvector (float), communities (int)
            save_json("degree.json", {k: int(v) for k, v in deg.items()})
            save_json("eigenvector.json", _safe_cast_floats(eig))
            save_json("communities.json", {k: int(v) for k, v in comm.items()})
            # graph image
            try:
                Path("docs").mkdir(parents=True, exist_ok=True)
                save_graph_png(G, filename="docs/graph.png")
            except Exception as e:
                print("Warning: saving graph image failed:", e)

        # -----------------------
        # Print short summaries (helpful during development/demo)
        # -----------------------
        print("\nDone.")
        # Print top TF-IDF tokens for first doc (if available)
        if tfidf_docs and len(tfidf_docs) > 0:
            top_doc1 = sorted(tfidf_docs[0].items(), key=lambda x: x[1], reverse=True)[: min(self.top_k_tokens, 20)]
            print("\nTop TF-IDF (Doc 1):")
            for term, score in top_doc1:
                print(f"  {term:12} {score:.6f}")

        if deg:
            top_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[: min(self.top_k_tokens, 20)]
            print("\nDegree centrality (top):")
            for term, score in top_deg:
                print(f"  {term:12} {score}")

        # -----------------------
        # Return rich object for programmatic use
        # -----------------------
        return {
            "names": names,
            "df": dict(df),
            "idf": idf,
            "tfidf": tfidf_docs,
            "co": _dictify_counter_map(co_sym),
            "graph": G,
            "degree": deg,
            "eigenvector": eig,
            "communities": comm,
        }

