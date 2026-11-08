"""
clean_corpus.py
Analiza y limpia el corpus:
1. Elimina entradas no narrativas (ensayos, biografías, antologías, prólogos, glosarios).
2. Limpia el ruido interno de las entradas que se quedan (URLs, citas académicas, bibliografías).
"""
import os
import re

INPUT  = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\Dataset\corpus_llanero_sin_glosarios.txt"
OUTPUT = r"C:\Users\OPTIPLEX 5050\Desktop\IA\Modelo_01\Dataset\corpus_llanero_limpio.txt"

SEP = "\n\n" + "=" * 80 + "\n\n"

# === 1. Detección de entradas NO narrativas (por título) ===
NON_NARRATIVE_TITLE_KEYWORDS = [
    "antologia", "antología", "prologo", "prólogo", "estudio",
    "ensayo", "biografia", "biografía", "glosario",
    "tabla-de-contenido", "tabla de contenido",
    "leer-el-mundo", "leer el mundo",
    "compilacion", "compilación",
    "introduccion", "introducción",
    "presentacion", "presentación",
    "bibliografia", "bibliografía",
    "indice", "índice",
]

# === 2. Detección de contenido NO narrativo ===
# Citas académicas: (Autor, 1998) o (1998, ) o (Autor 1998, )
ACADEMIC_CITE_RE = re.compile(
    r"\([^)]{0,60}\b(1[89]\d{2}|20\d{2})\b[^)]{0,20}\)"
)
# Referencias a libros/revistas
BOOK_REF_RE = re.compile(
    r"\b(Revista|Boletín|Editorial|Imprenta|Tropykos|Visor|Unellez|UNELLEZ|"
    r"Ministerio de Educación|Instituto|Cátedra|Colección|Vol\.|N°|p\.\s*\d+|pp\.\s*\d+)",
    re.IGNORECASE,
)
# Encabezados de bibliografía
BIBLIO_HEADER_RE = re.compile(
    r"^\s*(BIBLIOGRAF[IÍ]A|REFERENCIAS|FUENTES CONSULTADAS|TABLA DE CONTENIDO)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# === 3. Limpieza interna (para entradas que se quedan) ===
NOISE_LINE_PATTERNS = [
    r"https?://\S+",
    r"www\.\S+",
    r"\S+\.blogspot\.com\S*",
    r"^\s*BIBLIOGRAF[IÍ]A\s*$",
    r"^\s*REFERENCIAS\s*$",
    r"^\s*FUENTES CONSULTADAS\s*$",
    r"^\s*TABLA DE CONTENIDO\s*$",
    r"^\s*GLOSARIO.*$",
    r"^\s*Publicado por.*$",
    r"^\s*Etiquetas:.*$",
    r"^\s*\d{1,2}:\d{2}\s*$",
    r"^\s*\d{1,2} de \w+ de \d{4}\s*$",
]
NOISE_LINE_RE = re.compile("|".join(NOISE_LINE_PATTERNS), re.IGNORECASE)

# Citas académicas inline: "(Lanz:1998)", "(Subero 1991, )", "(1997, )"
INLINE_CITE_RE = re.compile(
    r"\([^()]{0,80}?\b(?:1[89]\d{2}|20\d{2})\b[^()]{0,20}?\)"
)
# Refs tipo "Vigosky, L . Pensamiento y lenguaje. Obras escogidas"
BOOK_REF_INLINE_RE = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+,\s*[A-Z]\.\s*[A-Z][^.]+\.",  # Autor, X. Título.
)


def classify_entry(entry):
    """Devuelve (keep: bool, reason: str) para una entrada."""
    lines = entry.strip().split("\n")
    if not lines:
        return False, "vacía"

    title = lines[0].replace("###", "").strip()
    title_lower = title.lower()

    # 1) Título con palabra clave no narrativa
    for kw in NON_NARRATIVE_TITLE_KEYWORDS:
        if kw in title_lower:
            return False, f"título contiene '{kw}'"

    # 2) Contenido: % de líneas con citas académicas
    body = "\n".join(lines[1:])
    body_lines = [l for l in body.split("\n") if l.strip()]
    if body_lines:
        cite_lines = sum(1 for l in body_lines if ACADEMIC_CITE_RE.search(l))
        book_lines = sum(1 for l in body_lines if BOOK_REF_RE.search(l))
        n = len(body_lines)
        if cite_lines / n > 0.10:
            return False, f"{cite_lines}/{n} líneas con citas académicas ({100*cite_lines/n:.0f}%)"
        if book_lines / n > 0.25:
            return False, f"{book_lines}/{n} líneas con refs a libros/revistas ({100*book_lines/n:.0f}%)"

    # 3) Encabezado de bibliografía al final
    if BIBLIO_HEADER_RE.search(body):
        return False, "contiene sección BIBLIOGRAFÍA/REFERENCIAS"

    # 4) Muy corta (<200 chars) → descartar
    if len(body.strip()) < 200:
        return False, f"contenido muy corto ({len(body.strip())} chars)"

    return True, "narrativa"


def clean_entry_body(entry):
    """Limpia ruido interno de una entrada que se conserva."""
    lines = entry.split("\n")
    title = lines[0]
    body_lines = lines[1:]

    # Quitar líneas con URL/bibliografía/etc.
    cleaned = []
    for line in body_lines:
        if NOISE_LINE_RE.search(line):
            continue
        cleaned.append(line)
    body = "\n".join(cleaned)

    # Quitar citas académicas inline
    body = INLINE_CITE_RE.sub("", body)
    # Quitar refs a libros inline (conservador)
    body = BOOK_REF_INLINE_RE.sub("", body)

    # Normalizar espacios
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = body.strip()

    if not body:
        return None
    return title + "\n" + body


def main():
    print(f"Leyendo: {INPUT}")
    with open(INPUT, "r", encoding="utf-8") as f:
        text = f.read()
    original_mb = len(text.encode("utf-8")) / 1024 / 1024
    print(f"Tamaño original: {original_mb:.2f} MB\n")

    entries = [e.strip() for e in text.split(SEP) if e.strip()]
    print(f"Entradas totales: {len(entries)}\n")

    kept, removed = [], []
    for entry in entries:
        keep, reason = classify_entry(entry)
        if keep:
            kept.append(entry)
        else:
            title = entry.split("\n")[0].replace("###", "").strip()[:60]
            removed.append((title, reason, len(entry)))

    # === Preview de eliminados ===
    print("=" * 90)
    print(f"🗑️  ENTRADAS A ELIMINAR: {len(removed)}")
    print("=" * 90)
    for title, reason, size in removed:
        print(f"  [{size:>6,} chars] {title:<55s} → {reason}")

    print("\n" + "=" * 90)
    print(f"✅ ENTRADAS A CONSERVAR: {len(kept)}")
    print("=" * 90)
    for entry in kept[:10]:
        title = entry.split("\n")[0].replace("###", "").strip()[:70]
        print(f"  {title}")
    if len(kept) > 10:
        print(f"  ... y {len(kept) - 10} más")

    # === Cálculo de tamaño resultante ===
    cleaned_entries = []
    empty_after_clean = 0
    for e in kept:
        c = clean_entry_body(e)
        if c:
            cleaned_entries.append(c)
        else:
            empty_after_clean += 1

    result = SEP.join(cleaned_entries)
    result_mb = len(result.encode("utf-8")) / 1024 / 1024

    print("\n" + "=" * 90)
    print("📊 RESUMEN")
    print("=" * 90)
    print(f"  Entradas originales:       {len(entries):>6,}")
    print(f"  Eliminadas (no narrativa): {len(removed):>6,}")
    print(f"  Conservadas (narrativa):   {len(kept):>6,}")
    print(f"  Vacías tras limpieza:      {empty_after_clean:>6,}")
    print(f"  Entradas finales:          {len(cleaned_entries):>6,}")
    print(f"  Tamaño original:           {original_mb:.2f} MB")
    print(f"  Tamaño final:              {result_mb:.2f} MB")
    print(f"  Reducción:                 {100*(1 - result_mb/original_mb):.1f}%")

    # === Confirmar ===
    print("\n" + "=" * 90)
    resp = input(f"\n¿Escribir el corpus limpio ({result_mb:.2f} MB)? [s/N]: ").strip().lower()
    if resp != "s":
        print("❌ Cancelado. No se modificó nada.")
        return

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"\n✅ Guardado: {OUTPUT}")

    # Muestra
    print("\n📖 Primeras 5 líneas del corpus limpio:")
    for line in [l for l in result.split("\n") if l.strip()][:5]:
        print(f"  {line[:100]}")


if __name__ == "__main__":
    main()
