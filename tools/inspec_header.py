# inspect_header.py
HEADER = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\bitnet-to-c\bitnet_weights.h"

with open(HEADER, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total líneas: {len(lines)}")
print(f"Línea más larga: {max(len(l) for l in lines):,} chars\n")

# Mostrar líneas 270-280 (donde GCC reporta el primer error)
print("=" * 70)
print("Líneas 270-280 (primer error de GCC en línea 273):")
print("=" * 70)
for i in range(269, 281):
    if i < len(lines):
        print(f"{i+1:5d}: {lines[i][:120]}")

# Buscar FINAL_NORM_gamma
print("\n" + "=" * 70)
print("Líneas alrededor de FINAL_NORM_gamma:")
print("=" * 70)
for i, line in enumerate(lines):
    if "FINAL_NORM_gamma" in line:
        for j in range(max(0, i-1), min(len(lines), i+4)):
            print(f"{j+1:5d}: {lines[j][:120]}")
        break

# Buscar LM_HEAD_weight
print("\n" + "=" * 70)
print("Líneas alrededor de LM_HEAD_weight:")
print("=" * 70)
for i, line in enumerate(lines):
    if "LM_HEAD_weight" in line:
        for j in range(max(0, i-1), min(len(lines), i+3)):
            print(f"{j+1:5d}: {lines[j][:120]}")
        break

# Verificar bytes exactos de la primera declaración de array
print("\n" + "=" * 70)
print("Bytes exactos de la línea de declaración de B0_NORM1_gamma:")
print("=" * 70)
for i, line in enumerate(lines):
    if "B0_NORM1_gamma" in line:
        print(f"Línea {i+1}: {repr(line)}")
        break
