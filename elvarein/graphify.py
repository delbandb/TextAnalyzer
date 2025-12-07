# elvarein/graphify.py
import networkx as nx
from typing import Dict
from collections import Counter

def make_graph(co: Dict[str, Counter], min_edge_weight: int = 1, symmetrize: bool = True):
    """
    Convert co-occurrence dict into a weighted undirected graph.
    
    Nodes: tokens
    Edges: weight proportional to co-occurrence

    If symmetrize=True:
      weight(u, v) = co[u][v] + co[v][u]
    Else:
      weight(u, v) = co[u][v]
    """
    G = nx.Graph()

    # Build edges
    for u, neigh in co.items():
        for v, w in neigh.items():
            if w < min_edge_weight:
                continue
            if symmetrize:
                # Add weight symmetrically
                if G.has_edge(u, v):
                    G[u][v]["weight"] += w
                else:
                    G.add_edge(u, v, weight=w)
            else:
                # Directed-like approach
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=w)

    return G


def compute_degree(G):
    """
    Return weighted degree centrality: sum of weights adjacent to each node.
    """
    return {node: sum(data["weight"] for _, data in G[node].items())
            for node in G.nodes()}


def compute_eigenvector(G):
    """
    Eigenvector centrality (weights included).
    """
    return nx.eigenvector_centrality_numpy(G, weight="weight")


def compute_communities(G):
    """
    Community detection (Louvain method).
    """
    import community  # from python-louvain
    return community.best_partition(G, weight="weight")


def save_graph_png(G, filename="graph.png", dpi=300):
    """
    Save a simple visualization of the graph.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.5, weight="weight")
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in weights])
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
