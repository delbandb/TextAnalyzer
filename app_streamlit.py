# app_streamlit.py

import streamlit as st
from pathlib import Path
import shutil
import json
import html
from datetime import datetime
import streamlit.components.v1 as components
import networkx as nx

# Importar engine 
from elvarein.engine import Engine


# Page config & styling
st.set_page_config(page_title="ELVAREIN — Symbolic Analyzer", layout="wide")
st.markdown(
    """
    <style>
      .stApp { background-color: #0f1720; color: #e6eef8; }
      .stSidebar { background-color: #071018; }
      .title { font-size:28px; font-weight:700; color:#ffffff; }
      .subtitle { color:#b7c6d9; margin-bottom:18px; }
      .card { background:#0b1220; padding:12px; border-radius:8px; }
      .small-muted { color:#94a3b8; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">📘 ELVAREIN — Symbolic Text Analysis Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">TF-IDF, co-occurrence and interactive symbol graph (vis.js)</div>', unsafe_allow_html=True)


# Sidebar (Engine controls + uploads)
st.sidebar.header("⚙️ Engine Settings")

corpus_choice = st.sidebar.selectbox("Corpus to analyze (folder)", ["tests/example_texts", "texts"])
uploaded_files = st.sidebar.file_uploader("Or upload .txt files (multiple)", type=["txt"], accept_multiple_files=True)

if st.sidebar.button("Clear uploads"):
    u = Path("uploads")
    if u.exists():
        shutil.rmtree(u)
        st.sidebar.success("Uploads cleared.")
    else:
        st.sidebar.info("No uploads found.")

co_window = st.sidebar.slider("Co-occurrence window size", 1, 6, 2)
dynamic_prop = st.sidebar.slider("Dynamic stopword threshold (doc freq fraction)", 0.0, 1.0, 0.0, 0.05)
min_weight = st.sidebar.number_input("Minimum edge weight (graph)", min_value=1, max_value=10, value=1)
top_k_tokens = st.sidebar.number_input("Top tokens to show", min_value=5, max_value=200, value=40)

st.sidebar.markdown("### Extra stopwords (comma-separated)")
extra_sw_input = st.sidebar.text_area(label=" ", value="chapter, chapterid, page, scene, 200", height=80, label_visibility="collapsed")
extra_sw = [w.strip().lower() for w in extra_sw_input.split(",") if w.strip()]

st.sidebar.markdown("---")
st.sidebar.markdown("### Graph render options")
fig_w = st.sidebar.slider("Figure width (px)", 800, 2000, 1400, step=100)
fig_h = st.sidebar.slider("Figure height (px)", 600, 1600, 1000, step=100)
node_size = st.sidebar.slider("Node size scaling", 50, 1200, 300, step=10)
label_font = st.sidebar.slider("Label font size", 6, 18, 10, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("Interactive graph (vis.js)")
use_interactive = st.sidebar.checkbox("Use interactive graph (vis.js)", value=True)
max_nodes_interactive = st.sidebar.slider("Max nodes for interactive graph", 10, 500, 120, step=10)
include_neighbors = st.sidebar.checkbox("Include 1-hop neighbors of top nodes", value=True)

run_btn = st.sidebar.button("Run analysis")

# Save uploaded files to uploads if there is any
uploads_dir = Path("uploads")
use_uploads = False
if uploaded_files:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    # clear previous
    for old in uploads_dir.glob("*"):
        try:
            old.unlink()
        except Exception:
            pass
    saved = []
    for up in uploaded_files:
        target = uploads_dir / Path(up.name).name
        data = up.getvalue()
        text = data.decode("utf-8", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
        target.write_text(text, encoding="utf-8")
        saved.append(target.name)
    if saved:
        use_uploads = True
        st.sidebar.success(f"Saved {len(saved)} uploaded file(s).")


# build a small interactive subgraph

def make_interactive_subgraph(G: nx.Graph, max_nodes: int, include_neighbors: bool) -> nx.Graph:
    deg = dict(G.degree(weight="weight"))
    ordered = sorted(deg.keys(), key=lambda k: deg[k], reverse=True)
    core = ordered[:max_nodes]
    sub_nodes = set(core)
    if include_neighbors:
        for n in core:
            sub_nodes.update(set(G.neighbors(n)))
    subG = G.subgraph(sub_nodes).copy()
    return subG


# vis.js renderer ( using  CDN)

def render_visjs_graph(subG, width_px=1200, height_px=800, node_size=300, label_font=10, physics=True):
    nodes = []
    edges = []
    deg = dict(subG.degree(weight="weight"))
    for n, data in subG.nodes(data=True):
        nodes.append({
            "id": str(n),
            "label": str(n),
            "value": deg.get(n, 1),
            "title": f"{html.escape(str(n))} (deg={deg.get(n,0)})"
        })
    for u, v, data in subG.edges(data=True):
        w = data.get("weight", 1)
        edges.append({
            "from": str(u),
            "to": str(v),
            "value": w,
            "title": f"weight={w}"
        })

    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)

    options = {
        "nodes": {
            "shape": "dot",
            "scaling": {"min": 8, "max": 50, "label": {"min": 6, "max": 20}},
            "font": {"size": label_font}
        },
        "edges": {"smooth": {"type": "continuous"}, "scaling": {"min": 1, "max": 10}},
        "physics": {
            "barnesHut": {"gravitationalConstant": -20000, "springLength": 200, "springConstant": 0.001},
            "minVelocity": 0.5
        } if physics else {"enabled": False},
        "interaction": {"hover": True, "navigationButtons": True, "zoomView": True}
    }
    options_json = json.dumps(options)

    html_template = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Elvarein visjs graph</title>
      <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.js"></script>
      <link href="https://unpkg.com/vis-network@9.1.2/dist/vis-network.min.css" rel="stylesheet" type="text/css" />
      <style>
        body {{ margin: 0; padding: 0; background: #ffffff; }}
        #network {{ width: 100%; height: {height_px}px; border-radius: 6px; }}
      </style>
    </head>
    <body>
      <div id="network"></div>
      <script type="text/javascript">
        const nodes = new vis.DataSet({nodes_json});
        const edges = new vis.DataSet({edges_json});
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {options_json};
        const network = new vis.Network(container, data, options);

        network.on("click", function(params) {{
          if (params.nodes.length) {{
            const id = params.nodes[0];
            const n = nodes.get(id);
            const title = n && n.title ? n.title : id;
            alert("Node: " + title);
          }}
        }});
      </script>
    </body>
    </html>
    """
    return html_template


# Run analysis when button pressed
if run_btn:
    st.info("Running ELVAREIN engine... please wait.")
    # choose data dir (uploads override folder)
    data_dir_path = uploads_dir if use_uploads else Path(corpus_choice)

    # Instantiate engine - pass parameters the  Engine accepts.
    # If Engine signature supports extra_stopwords/dynamic_min_doc_prop, keep them;
    # otherwise Engine will ignore unknown kwargs (adjust if needed).
    try:
        engine = Engine(
            data_dir=data_dir_path,
            window=co_window,
            min_edge_weight=min_weight,
            top_k_tokens=top_k_tokens
        )
    except TypeError:
        # fallback if Engine has different signature
        engine = Engine(data_dir=data_dir_path, window=co_window, min_edge_weight=min_weight)

    # If engine has attributes for extra stopwords or dynamic prop, set them (best-effort)
    if hasattr(engine, "extra_stopwords"):
        try:
            engine.extra_stopwords = extra_sw
        except Exception:
            pass
    if hasattr(engine, "dynamic_min_doc_prop"):
        try:
            engine.dynamic_min_doc_prop = float(dynamic_prop)
        except Exception:
            pass

    try:
        results = engine.run(save_outputs=True)
    except Exception as e:
        st.error(f"Engine run failed: {e}")
        st.stop()

    st.success("Analysis complete.")

    # Layout: left column for tables, right column for graph
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.header("🔍 Top TF-IDF ")
        tfidf_docs = results.get("tfidf", [])
        if tfidf_docs:
            doc1 = tfidf_docs[0]
            sorted_doc1 = sorted(doc1.items(), key=lambda x: x[1], reverse=True)[:top_k_tokens]
            st.table({"Token": [t for t, _ in sorted_doc1], "TF-IDF": [round(s, 6) for _, s in sorted_doc1]})
        else:
            st.warning("No TF-IDF results found.")

        st.header("📡 Top Degree Centrality")
        deg = results.get("degree", {})
        if deg:
            sorted_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:top_k_tokens]
            st.table({"Token": [t for t, _ in sorted_deg], "Degree": [s for _, s in sorted_deg]})
        else:
            st.warning("No degree results found.")

    with right_col:
        st.header("🕸️ Symbol Graph (Interactive)")

        G = results.get("graph")
        if G is None or G.number_of_nodes() == 0:
            st.warning("Graph is empty. Nothing to show.")
        else:
            subG = make_interactive_subgraph(G, max_nodes_interactive, include_neighbors)
            if use_interactive:
                vis_html = render_visjs_graph(subG, width_px=fig_w, height_px=fig_h, node_size=node_size, label_font=label_font)
                components.html(vis_html, height=fig_h + 40, scrolling=True)
            else:
                # fallback: static image if generated by engine (docs/graph.png)
                img_path = Path("docs/graph.png")
                if img_path.exists():
                    st.image(str(img_path), use_column_width=True)
                else:
                    st.warning("Static graph image not found. Re-run engine or enable interactive graph.")

    # Raw JSON outputs expander
    with st.expander("📄 Raw JSON outputs (saved files)"):
        for key in ["df", "tfidf", "co", "degree", "eigenvector", "communities"]:
            if key in results:
                st.subheader(key)
                try:
                    st.json(results[key])
                except Exception:
                    st.write(results[key])

else:
    st.info("Configure parameters and click **Run analysis** (or upload files).")
