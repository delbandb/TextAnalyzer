# tests/test_graph.py
from pathlib import Path
import sys

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from elvarein.io_ import load_texts
from elvarein.tokenizer import simple_tokenize
from elvarein.cooccurrence import build_cooccurrence, symmetrize_cooccurrence
from elvarein.graphify import make_graph, compute_degree, compute_eigenvector, compute_communities, save_graph_png


DATA_DIR = Path("tests/example_texts")

def run_test():
    names, texts = load_texts(DATA_DIR)
    token_lists = [simple_tokenize(t) for t in texts]

    co = build_cooccurrence(token_lists, window=1)
    co_sym = symmetrize_cooccurrence(co)

    print("Building graph...")
    G = make_graph(co_sym, min_edge_weight=1, symmetrize=False)
    print("Nodes:", list(G.nodes())[:20])
    print("Edges (first 10):", list(G.edges(data=True))[:10])

    print("\nWeighted degree:")
    deg = compute_degree(G)
    for k, v in list(deg.items())[:10]:
        print(f"  {k:12} {v}")

    print("\nEigenvector centrality:")
    eig = compute_eigenvector(G)
    for k, v in list(eig.items())[:10]:
        print(f"  {k:12} {v:.4f}")

    print("\nCommunities:")
    comm = compute_communities(G)
    print(comm)

    print("\nSaving graph.png...")
    save_graph_png(G, filename="docs/graph.png")
    print("Graph saved to docs/graph.png")

if __name__ == "__main__":
    run_test()
