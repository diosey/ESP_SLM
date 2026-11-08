"""
tokenizer.py — BigramTokenizer con greedy longest-match.
"""
import json, os

VOCAB_PATH = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\vocab_bigram.json"


class BigramTokenizer:
    PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

    def __init__(self, vocab_path=VOCAB_PATH):
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.vocab_size = len(self.vocab)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.id_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.max_len = max(len(t) for t in self.vocab)
        # Precomputar longitudes ordenadas descendente
        self.lengths = sorted(set(len(t) for t in self.vocab if t not in
                                  ["<pad>","<sos>","<eos>","<unk>"]), reverse=True)

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        n = len(text)
        while i < n:
            matched = False
            for L in self.lengths:
                if i + L <= n and text[i:i+L] in self.token_to_id:
                    ids.append(self.token_to_id[text[i:i+L]])
                    i += L
                    matched = True
                    break
            if not matched:
                ids.append(self.UNK_ID)
                i += 1
        if add_special_tokens:
            ids = [self.SOS_ID] + ids + [self.EOS_ID]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        parts = []
        for i in ids:
            if skip_special_tokens and i < 4:
                continue
            parts.append(self.id_to_token.get(i, "<unk>"))
        return "".join(parts)

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"BigramTokenizer(vocab_size={self.vocab_size}, max_len={self.max_len})"


# ======================================================================
# TEST
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BigramTokenizer — Verificación")
    print("=" * 60)

    tok = BigramTokenizer()
    print(f"{tok}\n")

    # Test round-trip
    for text in ["Había una vez un llanero.", "El pueblo de San Pedro", "¿Qué pasó? ¡Mucho!"]:
        ids = tok.encode(text)
        dec = tok.decode(ids)
        ok = "✅" if dec == text else "❌"
        print(f"[{ok}] '{text}'")
        print(f"     {len(text)} chars → {len(ids)} tokens (ratio {len(ids)/len(text):.2f})")
        print(f"     Decoded: '{dec}'")

    # Corpus completo
    corpus = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\Dataset\corpus_llanero_sin_glosarios.txt"
    if os.path.exists(corpus):
        print("\nCorpus completo:")
        with open(corpus, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tok.encode(text)
        unk = sum(1 for i in ids if i == tok.UNK_ID)
        print(f"  {len(text):,} chars → {len(ids):,} tokens")
        print(f"  Ratio: {len(ids)/len(text):.3f} tokens/char")
        print(f"  UNK: {unk:,} ({100*unk/len(ids):.4f}%)")
        seq_len = 65
        print(f"  Secuencias de 65 tokens: ~{len(ids)//seq_len:,}")

    print("\n" + "=" * 60)
