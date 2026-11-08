# build_vocab.py
import json
import os
from collections import Counter

CORPUS_PATH = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\Dataset\corpus_llanero_sin_glosarios.txt"
VOCAB_PATH  = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\vocab.json"

# Caracteres que aparecen menos de MIN_COUNT veces se descartan (ruido)
MIN_COUNT = 5

# Tokens especiales (IDs fijos)
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

def main():
    print("=" * 60)
    print("Construyendo vocabulario char-level desde el corpus")
    print("=" * 60)

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\nCorpus: {len(text.encode('utf-8'))/1024/1024:.2f} MB")
    print(f"Caracteres: {len(text):,}")

    counter = Counter(text)
    print(f"Caracteres únicos: {len(counter)}")

    # Filtrar ruido (caracteres raros que aparecen pocas veces)
    kept = [(ch, n) for ch, n in counter.items() if n >= MIN_COUNT]
    dropped = [(ch, n) for ch, n in counter.items() if n < MIN_COUNT]

    print(f"\nDespués de filtrar (MIN_COUNT={MIN_COUNT}):")
    print(f"  Conservados: {len(kept)}")
    print(f"  Descartados: {len(dropped)}")
    if dropped:
        print(f"  Descartados (muestra): {[ch for ch, _ in dropped[:15]]}")

    # Ordenar alfabéticamente (estable entre ejecuciones)
    kept_chars = sorted([ch for ch, _ in kept])

    # Vocabulario final: especiales + caracteres
    vocab = SPECIAL_TOKENS + kept_chars
    vocab_size = len(vocab)

    # Verificación
    total_kept = sum(n for _, n in kept)
    coverage = total_kept / len(text) * 100
    print(f"\nVocabulario final: {vocab_size} tokens")
    print(f"  Especiales: {len(SPECIAL_TOKENS)}")
    print(f"  Caracteres: {len(kept_chars)}")
    print(f"  Cobertura del corpus: {coverage:.4f}%")

    # Mostrar los caracteres
    print(f"\nCaracteres del vocabulario:")
    for i in range(0, len(kept_chars), 20):
        chunk = kept_chars[i:i+20]
        print(f"  {''.join(chunk)!r}")

    # Guardar
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "special_tokens": SPECIAL_TOKENS,
            "chars": kept_chars,
            "vocab_size": vocab_size,
            "min_count": MIN_COUNT,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Guardado: {VOCAB_PATH}")

if __name__ == "__main__":
    main()
