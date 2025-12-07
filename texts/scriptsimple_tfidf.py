from collections import Counter

# Términos que quiero  analizar manualmente
terms = ["path", "green", "road", "sand", "serpents", "lover", "balcony"]

# Archivos del corpus
files = ["chapter1.txt", "chapter2.txt", "chapter3.txt"]

# Función de tokenización simple
def tokenize(text):
    clean = ""
    for ch in text.lower():
        clean += ch if ch.isalnum() or ch.isspace() else " "
    return clean.split()


all_counts = []
doc_lengths = []

for filename in files:
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = tokenize(text)
    doc_lengths.append(len(tokens))

    counter = Counter(tokens)

    # Conteo de cada término específico
    term_counts = {t: counter.get(t, 0) for t in terms}
    all_counts.append(term_counts)


# Mostrar tabla estilo TFG

print("Término | C1 | C2 | C3")
print("------------------------")

for term in terms:
    c1 = all_counts[0][term]
    c2 = all_counts[1][term]
    c3 = all_counts[2][term]
    print(f"{term:8} | {c1:2d} | {c2:2d} | {c3:2d}")

print("\nTokens totales por capítulo:")
for i, n in enumerate(doc_lengths, start=1):
    print(f"C{i}: {n} tokens")
