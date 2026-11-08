import json

VOCAB_JSON = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\vocab_bigram.json"
OUT_H      = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\bitnet-to-c\vocab_data.h"

with open(VOCAB_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

vocab = data["vocab"]
n = len(vocab)

def is_hex(b):
    return b in b'0123456789abcdefABCDEF'

def c_lit(tok_bytes):
    if not tok_bytes:
        return '""'
    parts = []
    for b in tok_bytes:
        if b == 0x22: parts.append('\\"')
        elif b == 0x5C: parts.append('\\\\')
        elif b == 0x0A: parts.append('\\n')
        elif b == 0x09: parts.append('\\t')
        elif b == 0x0D: parts.append('\\r')
        elif 0x20 <= b <= 0x7E: parts.append(chr(b))
        else: parts.append(f'\\x{b:02x}')

    # Romper secuencia si \xNN va seguido de un char hex
    result = '"'
    for i, p in enumerate(parts):
        result += p
        if p.startswith('\\x') and i + 1 < len(parts):
            nxt = parts[i+1]
            if nxt and nxt[0] in '0123456789abcdefABCDEF':
                result += '""'
    result += '"'
    return result

lines = []
lines.append("// Auto-generated.\n")
lines.append("#ifndef VOCAB_DATA_H\n#define VOCAB_DATA_H\n\n")
lines.append(f"#define VOCAB_N {n}\n\n")
lines.append(f"static const char* VOCAB_STRINGS[{n}] = {{\n")
for tok in vocab:
    lines.append(f"    {c_lit(tok.encode('utf-8'))},\n")
lines.append("};\n\n")
lines.append(f"static const int VOCAB_LENGTHS[{n}] = {{\n")
for tok in vocab:
    lines.append(f"    {len(tok.encode('utf-8'))},\n")
lines.append("};\n\n")
lines.append("#endif\n")

with open(OUT_H, "w", encoding="ascii") as f:
    f.write("".join(lines))

print(f"✅ {OUT_H}")
