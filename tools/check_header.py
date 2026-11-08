# check_lines.py
HEADER = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\bitnet-to-c\bitnet_weights.h"
with open(HEADER, "r", encoding="utf-8") as f:
    lines = f.readlines()

lengths = [(i, len(l)) for i, l in enumerate(lines)]
lengths.sort(key=lambda x: -x[1])

print(f"Total líneas: {len(lines)}")
print(f"Línea más larga: {lengths[0][1]:,} caracteres (línea {lengths[0][0]+1})")
print(f"Top 5 líneas más largas:")
for i, l in lengths[:5]:
    print(f"  Línea {i+1}: {l:,} chars")
