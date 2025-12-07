# debug_tokens.py
from pathlib import Path
from elvarein.io_ import load_texts
from elvarein.tokenizer import preprocess_corpus, simple_tokenize
from elvarein.tfidf import build_doc_term_counts, compute_df
from elvarein.cooccurrence import build_cooccurrence
from collections import Counter
import json

DATA_DIR = Path("tests/example_texts")
names, texts = load_texts(DATA_DIR)
print("Loaded docs:", names)
print("Raw sample (first doc, first 300 chars):")
print(texts[0][:300])

# Option A: simple tokenize (no dynamic removal)
tokens_simple = [simple_tokenize(t) for t in texts]
print("\nSimple tokenization (no dynamic):")
for i, toks in enumerate(tokens_simple, 1):
    print(f" Doc{i} tokens ({len(toks)}): {toks[:40]}")

# Option B: preprocess_corpus (what Engine currently uses)
token_lists = preprocess_corpus(texts, extra_stopwords=["chapter","chapterid"], do_lemmatize=True, dynamic_min_doc_prop=0.5)
print("\nPreprocess_corpus result:")
for i, toks in enumerate(token_lists, 1):
    print(f" Doc{i} tokens ({len(toks)}): {toks[:40]}")

# show document frequencies and cooccurrence summary
doc_counts = [Counter(t) for t in token_lists]
df = compute_df(doc_counts)
print("\nDocument frequency (df) sample (first 30):")
print(dict(list(df.items())[:30]))
co = build_cooccurrence(token_lists, window=1)
print("\nCooccurrence keys (number of tokens with neighbors):", len(co.keys()))
print("Some keys:", list(co.keys())[:30])

# Save debug outputs
Path("docs").mkdir(exist_ok=True)
Path("docs/debug_tokens.json").write_text(json.dumps({
    "simple_first_doc_tokens": tokens_simple[0][:200],
    "preprocessed_first_doc_tokens": token_lists[0][:200],
    "df_sample": list(df.items())[:100],
    "co_keys_sample": list(co.keys())[:100]
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nDebug saved to docs/debug_tokens.json")
