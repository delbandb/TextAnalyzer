# elvarein/io_.py
from pathlib import Path
from typing import List, Tuple

def load_texts(folder: Path) -> Tuple[list, list]:
    """
    Load all .txt files from folder.
    Returns (names, texts)
    """
    folder = Path(folder)
    names = []
    texts = []
    if not folder.exists():
        raise FileNotFoundError(f"{folder} does not exist")
    for p in sorted(folder.glob("*.txt")):
        names.append(p.name)
        texts.append(p.read_text(encoding="utf-8"))
    return names, texts

def save_json(path, data):
    import json
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
