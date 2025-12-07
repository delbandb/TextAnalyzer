# tests/test_cooccurrence.py
from pathlib import Path
import sys
# ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elvarein.io_ import load_texts
from elvarein.tokenizer import simple_tokenize
from elvarein.cooccurrence import build_cooccurrence, symmetrize_cooccurrence, top_neighbors

DATA_DIR = Path("tests/example_texts")

def run_test():
    names, texts = load_texts(DATA_DIR)
    token_lists = [simple_tokenize(t) for t in texts]
    print("Loaded docs:", names)
    print("Token lists sample:")
    for i, toks in enumerate(token_lists, 1):
        print(f" Doc{i}:", toks[:40])
    co = build_cooccurrence(token_lists, window=1)
    print("\nCo-occurrence (partial):")
    # print top neighbors for a few tokens we expect
    for token in ["mirror", "serpent", "gate", "coin", "the"]:
        print(f"  {token} -> {top_neighbors(co, token, 10)}")
    print("\nSymmetrized neighbors (mirror):")
    co_sym = symmetrize_cooccurrence(co)
    print(f"  mirror -> {top_neighbors(co_sym, 'mirror', 10)}")

if __name__ == "__main__":
    run_test()
