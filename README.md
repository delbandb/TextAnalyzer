# TextAnalyzer

A literary analysis engine built for my TFG. It runs a graph-based NLP pipeline 
over the text of Elvarein — a fantasy novel I wrote — to map narrative structure, 
character relationships, and thematic communities.

I built this because I wanted to understand my own book analytically. Not sentiment 
analysis. Something closer to: which concepts are structurally central to the story, 
which characters cluster together, and how themes connect across the full text.

## What it does

- TF-IDF scoring to identify the most significant terms across the corpus
- Co-occurrence matrix to find words and concepts that appear near each other
- Graph centrality metrics (degree and eigenvector) to find the most connected nodes
- Community detection to group characters and themes that cluster together
- Symbol and thematic tagging across the narrative
- Streamlit interface to explore the results visually

## Stack

- Python 3.12
- NLTK / spaCy (NLP pipeline)
- NetworkX (graph construction and analysis)
- Streamlit (visual explorer)

## Structure
