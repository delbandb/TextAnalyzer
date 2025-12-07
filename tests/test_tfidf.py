# tests/test_tfidf.py
from pathlib import Path
import sys

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elvarein.io_ import load_texts
from elvarein.tokenizer import simple_tokenize
from elvarein.tfidf import build_doc_term_counts, compute_df, compute_idf, compute_tfidf_per_doc, top_k
import json

DATA_DIR = Path("tests/example_texts")

def run_test():
    names, texts = load_texts(DATA_DIR)
    print("Documents:", names)
    doc_tokens, doc_counts, doc_lengths = build_doc_term_counts(texts, simple_tokenize)
    print("Doc lengths:", doc_lengths)
    print("Doc counts (per doc):")
    for i, c in enumerate(doc_counts, 1):
        print(f"  Doc{i}:", c.most_common())
    df = compute_df(doc_counts)
    print("Document frequency df():", dict(df))
    idf = compute_idf(df, len(doc_counts))
    print("IDF:", {k: round(v,6) for k,v in idf.items()})
    tfidf_docs = compute_tfidf_per_doc(doc_counts, doc_lengths, idf)
    for i, tfidf in enumerate(tfidf_docs, 1):
        print(f"\nTop TF-IDF Doc{i}:")
        for term,score in top_k(tfidf, 10):
            print(f"  {term:12} {score:.6f}  tf={doc_counts[i-1][term]}")

if __name__ == "__main__":
    run_test()
