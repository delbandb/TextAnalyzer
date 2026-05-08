# TextAnalyzer

TextAnalyzer is a Python NLP and graph-analysis project built for my TFG. It analyzes a text corpus from *Elvarein*, a fantasy novel I wrote, and turns the writing into measurable structures: important terms, co-occurring concepts, central nodes, and thematic communities.

The goal is not generic sentiment analysis. The goal is closer to literary mapping: which words and symbols are structurally important, which ideas appear together, and how the text forms clusters of meaning.

## What It Does

- Loads a folder of `.txt` documents as a corpus.
- Tokenizes and preprocesses text with configurable stopwords.
- Calculates document frequency, inverse document frequency, and TF-IDF scores.
- Builds co-occurrence windows to find terms that appear near each other.
- Converts co-occurrence data into a weighted graph.
- Calculates weighted degree centrality and eigenvector centrality.
- Detects communities with the Louvain method.
- Saves JSON artifacts for later analysis.
- Generates a static graph image.
- Provides a Streamlit interface with upload support and an interactive graph view.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| NLP | NLTK |
| Math/data | numpy |
| Graph analysis | NetworkX, python-louvain |
| Visualization | matplotlib, Streamlit, vis-network |
| Outputs | JSON, PNG, Streamlit UI |

## Project Structure

```text
TextAnalyzer/
|-- elvarein/
|   |-- engine.py        # Orchestrates the full analysis pipeline
|   |-- tokenizer.py     # Tokenization and preprocessing
|   |-- tfidf.py         # TF-IDF calculations
|   |-- cooccurrence.py  # Co-occurrence matrix logic
|   |-- graphify.py      # Graph construction and metrics
|   |-- io_.py           # Text loading and JSON writing
|   `-- config.py        # Project configuration
|-- texts/               # Corpus chapters
|-- tests/               # Manual test scripts and example texts
|-- docs/                # Generated/debug artifacts
|-- app_streamlit.py     # Streamlit visual explorer
|-- run_engine.py        # CLI-style pipeline runner
|-- requirements.txt
`-- README.md
```

## Pipeline

```text
.txt files
  -> tokenization and preprocessing
  -> TF-IDF scoring
  -> co-occurrence windows
  -> weighted graph
  -> centrality and community detection
  -> JSON outputs and visual graph
```

## Run Locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python run_engine.py
```

On macOS or Linux, replace `.venv/Scripts/python` with `.venv/bin/python`.

## Run the Streamlit Explorer

```bash
streamlit run app_streamlit.py
```

The Streamlit app lets you choose the corpus folder, upload text files, adjust co-occurrence settings, and inspect the resulting graph interactively.

## Generated Outputs

Running the engine can generate:

- `symbols.json` for document-frequency style symbol counts.
- `tfidf.json` for per-document TF-IDF scores.
- `cooccurrence.json` for term-neighbor relationships.
- `degree.json` for weighted degree centrality.
- `eigenvector.json` for eigenvector centrality.
- `communities.json` for Louvain community assignments.
- `docs/graph.png` for a static graph visualization.

## Manual Test Scripts

The `tests/` folder contains small scripts for checking the TF-IDF, co-occurrence, and graph pieces with example texts:

```bash
python tests/test_tfidf.py
python tests/test_cooccurrence.py
python tests/test_graph.py
```

## Why This Project Matters

This project combines writing, data analysis, and graph theory in one portfolio piece. It shows that I can take an abstract question, design a pipeline around it, and produce outputs that are inspectable rather than magical. It also gives me a strong interview story because the domain is personal, but the techniques are transferable to document analysis, knowledge mapping, and text exploration.

## What I Would Improve Next

- Convert the manual test scripts into automated pytest tests.
- Add exportable charts and CSV summaries from the Streamlit app.
- Add named-entity recognition for characters, places, and organizations.
- Add better Spanish/English language configuration.
- Package the engine so it can be reused with any text corpus.
