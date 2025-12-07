# elvarein/tokenizer.py
import re
from collections import Counter
from typing import List, Set, Iterable, Tuple, Dict

# -------------------------
# Static stopwords (comprehensive)
# -------------------------
STOPWORDS: Set[str] = {
    # articles / pronouns / determiners
    "the","a","an","i","you","he","she","it","we","they","me","him","her","them",
    "my","your","his","their","our","yours","ours","hers","theirs",

    # contractions and common shortened tokens
    "i'm","you're","he's","she's","it's","we're","they're","that's","there's","isn't",
    "aren't","wasn't","weren't","don't","doesn't","didn't","won't","wouldn't","can't",
    "couldn't","shouldn't","haven't","hasn't","hadn't","i'll","you'll","he'll","she'll",
    "we'll","they'll","i'd","you'd","he'd","she'd","we'd","they'd","i've","you've",
    "we've","they've","t","s","m","ll","d","ve","re",

    # auxiliary + linking verbs
    "be","am","is","are","was","were","been","being","have","has","had","do","does",
    "did","doing","make","makes","made","get","gets","got","getting","go","goes",
    "went","gone","seem","seems","seemed","become","becomes","became",

    # modal verbs
    "can","could","should","would","may","might","must","shall","will",

    # prepositions
    "in","on","at","by","for","with","about","against","between","into","through",
    "during","before","after","above","below","to","from","up","down","over","under",
    "within","without","around","onto","upon","across","along","behind","beyond",

    # conjunctions
    "and","but","or","because","so","until","while","as","if","than","though",
    "although","nor","yet","whether","rather","since","once","unless",

    # adverbs & time words (common noise)
    "very","too","just","only","really","now","then","ever","never","always","sometimes",
    "often","usually","also","well","here","there","almost","quite","pretty","still",
    "already","soon","today","yesterday","tomorrow","ago","later","recently",

    # quantifiers
    "some","any","no","none","many","much","few","several","all","both","each","either",
    "neither","every","most","more","less","lot","lots","hundred","thousand",

    # generic filler nouns
    "thing","things","something","anything","nothing","everything",
    "people","person","one","ones","others","man","woman","men","women",

    # question words
    "what","who","which","whom","whose","when","where","why","how",

    # punctuation-noise (single letters often from contractions split)
    "’","`","'",

    # single letters and common OCR artifacts
    "x","y","z","u","v","w","q","p","o","n","r","k","j","h","g","f","e",

    # extra generic noise tokens
    "there","here","where","because","why","very","again","also","ok","okay","hey"
}

# -------------------------
# Lemmatizer: lightweight rule-based + small exceptions
# -------------------------
# Exceptions mapping where simple suffix rules would fail
_LEMMA_EXCEPTIONS: Dict[str, str] = {
    "felt": "feel",
    "went": "go",
    "gone": "go",
    "came": "come",
    "brought": "bring",
    "bought": "buy",
    "wasn": "be",
    "weren": "be",
    "isn": "be",
    "aren": "be",
    "hadn": "have",
    "hasn": "have",
    # Add domain-specific exceptions as needed
}

# Generic suffix rules (ordered). Keep conservative to avoid overstripping.
_LEMMA_RULES: List[Tuple[str, str]] = [
    ("ing", ""),  # running -> run (but careful: e.g., 'king' -> 'k')
    ("ed", ""),   # walked -> walk
    ("es", "e"),  # boxes -> box(e) rule is conservative: boxes -> boxe (we later handle)
    ("s", ""),    # snakes -> snake
]

# minimal safe check for returning a viable token
def _safe_root(token: str, root: str) -> str:
    if len(root) < 2:
        return token
    # avoid producing nonsense (e.g., 'king' -> 'k')
    if len(token) - len(root) >= 1 and len(root) >= 2:
        return root
    return token

def apply_lemma(token: str) -> str:
    """
    Apply a conservative lemmatization to a token:
     - first check exceptions mapping
     - then apply ordered suffix rules
     - finally normalize small artifacts like trailing 'e' from 'es' rule
    """
    if token in _LEMMA_EXCEPTIONS:
        return _LEMMA_EXCEPTIONS[token]

    for suf, repl in _LEMMA_RULES:
        if token.endswith(suf) and len(token) > len(suf) + 2:
            candidate = token[: -len(suf)] + repl
            # post-processing: fix 'boxe' -> 'box'
            if candidate.endswith("e") and not token.endswith("e"):
                candidate = candidate[:-1]
            return _safe_root(token, candidate)
    return token

# -------------------------
# Tokenization pipeline
# -------------------------
_TOKEN_RE = re.compile(r"[^a-záéíóúüñ0-9'\s\-]")  # allow letters, digits, apostrophes and hyphen

def simple_tokenize(text: str,
                    stopwords: Iterable[str] = None,
                    do_lemmatize: bool = True,
                    min_token_len: int = 2) -> List[str]:
    """
    Tokenize a text string into a cleaned list of tokens.
    - lowercases
    - removes non-letter characters (keeps accents, apostrophes, hyphens, digits)
    - splits on whitespace
    - optional stopword removal and lemmatization
    - filters short tokens by min_token_len
    """
    if not text:
        return []

    text = text.lower()
    # remove unwanted symbols
    text = _TOKEN_RE.sub(" ", text)
    raw = re.split(r"\s+", text)

    # prepare stopword set
    sw = set(STOPWORDS)
    if stopwords:
        sw = sw.union(set(w.lower() for w in stopwords))

    tokens: List[str] = []
    for t in raw:
        if not t:
            continue
        # remove leading/trailing apostrophes or hyphens
        t = t.strip("'-")
        if len(t) < min_token_len:
            continue
        if t in sw:
            continue
        if do_lemmatize:
            t = apply_lemma(t)
        if t in sw:  # re-check after lemmatization
            continue
        tokens.append(t)
    return tokens

# -------------------------
# Dynamic stopword detection across a corpus
# -------------------------
def detect_dynamic_stopwords(doc_token_lists: List[List[str]],
                             min_doc_proportion: float = 0.60,
                             top_k: int = 0) -> Set[str]:
    """
    Detect tokens that:
      - appear in more than min_doc_proportion fraction of documents (very common)
      OR
      - optionally return the top_k most frequent tokens across corpus (if top_k>0)
    Returns a set of tokens to be added to stopwords.
    """
    if not doc_token_lists:
        return set()

    D = len(doc_token_lists)
    df = Counter()
    corpus_freq = Counter()
    for toks in doc_token_lists:
        seen = set()
        for t in toks:
            corpus_freq[t] += 1
            if t not in seen:
                df[t] += 1
                seen.add(t)

    dyn = set()
    for term, dfv in df.items():
        if dfv / D >= min_doc_proportion:
            dyn.add(term)

    if top_k > 0:
        most = [t for t, _ in corpus_freq.most_common(top_k)]
        dyn.update(most)

    return dyn

# -------------------------
# Utility to apply preprocessing to entire corpus
# -------------------------
def preprocess_corpus(texts: List[str],
                      extra_stopwords: Iterable[str] = None,
                      do_lemmatize: bool = True,
                      min_token_len: int = 2,
                      dynamic_min_doc_prop: float = 0.0,
                      dynamic_top_k: int = 0) -> List[List[str]]:
    """
    Tokenize and preprocess a list of raw document texts.
    Parameters:
      - extra_stopwords: iterable of additional stopwords to add (domain-specific)
      - dynamic_min_doc_prop: if > 0, detect tokens present in that proportion of docs and remove them
      - dynamic_top_k: if >0, also remove top_k frequent tokens across corpus
    Returns:
      - token lists (one list of tokens per document)
    """
    # initial tokenization (no dynamic removals yet)
    base_sw = set(extra_stopwords) if extra_stopwords else set()
    token_lists = [simple_tokenize(t, stopwords=base_sw, do_lemmatize=do_lemmatize, min_token_len=min_token_len)
                   for t in texts]

    # detect dynamic stopwords if requested
    dynamic_sw = set()
    if dynamic_min_doc_prop > 0 or dynamic_top_k > 0:
        dynamic_sw = detect_dynamic_stopwords(token_lists, min_doc_proportion=dynamic_min_doc_prop, top_k=dynamic_top_k)

    final_sw = set(base_sw).union(dynamic_sw)
    # add static STOPWORDS too
    final_sw = final_sw.union(STOPWORDS)

    # re-tokenize applying the final stopword set
    cleaned = [simple_tokenize(" ".join(toks), stopwords=final_sw, do_lemmatize=do_lemmatize, min_token_len=min_token_len)
               for toks in token_lists]

    return cleaned

# -------------------------
# Small test when run directly
# -------------------------
if __name__ == "__main__":
    demo = [
        "The mirror showed a wheel and a book. The serpent slept beneath the sand.",
        "A coin fell near the gate. The serpent had chosen to walk the path."
    ]
    print("=== Raw tokenization ===")
    for i, t in enumerate(demo, 1):
        print(i, simple_tokenize(t, do_lemmatize=False))

    print("\n=== Lemmatized / stopword filtered tokenization ===")
    cleaned = preprocess_corpus(demo, extra_stopwords=["chose"], do_lemmatize=True)
    for i, toks in enumerate(cleaned, 1):
        print(i, toks)

    print("\n=== Dynamic stopwords example (min_doc_prop=0.5) ===")
    cleaned2 = preprocess_corpus(demo, do_lemmatize=True, dynamic_min_doc_prop=0.5)
    for i, toks in enumerate(cleaned2, 1):
        print(i, toks)

