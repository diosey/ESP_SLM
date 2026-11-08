# build_vocab_bigram.py
import json, os
from collections import Counter

CORPUS_PATH = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\Dataset\corpus_llanero_sin_glosarios.txt"
VOCAB_PATH  = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\vocab_bigram.json"

TARGET_VOCAB = 2000
MIN_NGRAM_COUNT = 20   # n-gramas con <20 apariciones se descartan
MAX_NGRAM_LEN = 4      # bigramas, trigramas, 4-gramas
SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]


def main():
    print("Leyendo corpus...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  {len(text):,} chars")

    # 1) Caracteres únicos (siempre se incluyen, para fallback)
    chars = sorted(set(text))
    print(f"  Caracteres únicos: {len(chars)}")

    # 2) N-gramas por longitud
    print("Contando n-gramas...")
    ngrams = {}
    for n in range(2, MAX_NGRAM_LEN + 1):
        c = Counter(text[i:i+n] for i in range(len(text) - n + 1))
        ngrams[n] = c
        print(f"  {n}-gramas únicos: {len(c):,}")

    # 3) Construir vocabulario: especiales + chars + n-gramas por frecuencia
    vocab = list(SPECIAL) + chars
    seen = set(vocab)

    # Candidatos: todos los n-gramas con count >= MIN_NGRAM_COUNT
    candidates = []
    for n in range(MAX_NGRAM_LEN, 1, -1):  # preferir n-gramas más largos
        for gram, cnt in ngrams[n].items():
            if cnt >= MIN_NGRAM_COUNT and gram not in seen:
                candidates.append((cnt, gram))
    candidates.sort(key=lambda x: -x[0])

    for cnt, gram in candidates:
        if len(vocab) >= TARGET_VOCAB:
            break
        vocab.append(gram)
        seen.add(gram)

    vocab_size = len(vocab)
    print(f"\nVocabulario final: {vocab_size} tokens")
    print(f"  Especiales: {len(SPECIAL)}")
    print(f"  Chars:      {len(chars)}")
    print(f"  N-gramas:   {vocab_size - len(SPECIAL) - len(chars)}")

    # 4) Verificar cobertura con greedy longest-match
    print("\nVerificando cobertura con greedy tokenizer...")
    vocab_set = set(vocab)
    max_len = max(len(g) for g in vocab)
    i, covered, unk = 0, 0, 0
    while i < len(text):
        matched = False
        for L in range(min(max_len, len(text) - i), 0, -1):
            if text[i:i+L] in vocab_set:
                i += L; covered += L; matched = True; break
        if not matched:
            i += 1; unk += 1
    coverage = covered / len(text) * 100
    print(f"  Cobertura: {coverage:.4f}%")
    print(f"  UNK: {unk:,}")

    # 5) Guardar
    data = {
        "special_tokens": SPECIAL,
        "chars": chars,
        "ngrams": [g for _, g in candidates][:vocab_size - len(SPECIAL) - len(chars)],
        "vocab": vocab,
        "vocab_size": vocab_size,
    }
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ {VOCAB_PATH} ({os.path.getsize(VOCAB_PATH)/1024:.1f} KB)")

    # Muestra del top de n-gramas
    print("\nTop 20 n-gramas:")
    for cnt, g in candidates[:20]:
        print(f"  {cnt:>7,}  {g!r}")


if __name__ == "__main__":
    main()
