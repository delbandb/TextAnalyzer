# run_engine.py
from pathlib import Path
from elvarein.engine import Engine

if __name__ == "__main__":

    # IMPORTANT:
    # Change data_dir to "texts" when you want to use your real chapters.
    #
    # Example:
    # data_dir=Path("texts")
    #
    # For tests, we keep using example_texts.
    engine = Engine(
        data_dir=Path("texts"),                 # my book texts
        window=2,                               # co-occurrence window size
        min_edge_weight=1,                      # ignore edges weaker than this
        top_k_tokens=20,                        # how many tokens to print
        extra_stopwords=["chapter", "chapterid", "behdadfar", "delband", "page", "scene"],
        dynamic_min_doc_prop=0.0                # safe default (no dynamic removal)
    )

    results = engine.run(save_outputs=True)

    print("\n==============================")
    print("      TOP TF-IDF     ")
    print("==============================")

    tfidf_docs = results.get("tfidf", [])
    if tfidf_docs and len(tfidf_docs) > 0:
        tfidf_doc1 = tfidf_docs[0]
        top_terms = sorted(tfidf_doc1.items(), key=lambda x: x[1], reverse=True)[:20]

        for term, score in top_terms:
            print(f"  {term:16} {score:.6f}")
    else:
        print("No TF-IDF results found.")

    print("\n==============================")
    print("   DEGREE CENTRALITY (Top)   ")
    print("==============================")

    deg = results.get("degree", {})
    if deg:
        top_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:20]
        for term, score in top_deg:
            print(f"  {term:16} {score}")
    else:
        print("No degree centrality results found.")

