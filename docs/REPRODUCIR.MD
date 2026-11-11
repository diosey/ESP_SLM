# Guía de reproducción del proyecto BitNet ESP32

> Cómo reproducir el proyecto completo desde cero: desde el dataset
> original hasta un ESP32 generando texto de forma local.

**Requisito previo**: si ya tienes `checkpoints/ep20_val1.382.pt` y
`artifacts/bitnet_weights.bin`, puedes saltar directamente a la
[Guía rápida con Eclipse IDE](#guía-rápida-con-eclipse-ide) o a la
[Fase 7](#7-compilación-del-firmware).

---

## ⚡ Guía rápida con Eclipse IDE

> Para usuarios que ya tienen `esp32-bitnet-firmware.rar` y
> `bitnet_weights.bin` (2,872,912 bytes), y quieren compilar y flashear
> sin pasar por la línea de comandos.

**Requisitos**:
- Eclipse IDE con el plugin **ESP-IDF** de Espressif instalado.
- ESP32 conectado por USB.
- El archivo `bitnet_weights.bin` (2,872,912 bytes) disponible.

### Paso 1: Extraer el firmware

Extrae `ESP32_SLM_192.rar` a una carpeta local. El resultado es
un proyecto ESP-IDF completo, listo para importar. Estructura:

```
ESP32_SML/
├── CMakeLists.txt
├── partitions.csv
├── sdkconfig.defaults
└── main/
    ├── CMakeLists.txt
    ├── main.c
    ├── bitnet_inference.c / .h
    ├── tokenizer.c / .h
    ├── bitnet_weights.h
    ├── vocab_data.h
    ├── data/
    │   └── index.html
    └── weights/
        └── bitnet_weights.bin    ← 2,872,912 bytes
```

### Paso 2: Importar el proyecto en Eclipse

1. Abre Eclipse con el plugin ESP-IDF.
2. Ve a `File → Import → ESP-IDF → Existing IDF Project`.
3. Selecciona la carpeta extraída (`ESP32_SML/`).
4. Finaliza la importación.

### Paso 3: Flashear el binario de pesos

El binario de pesos va a la partición `weights` en el offset
`0x120000`. Es una partición **independiente** del firmware. El orden
respecto al firmware es indiferente: ambas escriben regiones distintas
de la flash.

1. Clic derecho sobre el proyecto en el explorador.
2. Menú **ESP-IDF** → **Write Binary Data to Flash**.
3. En el diálogo:
   - **Binary file**: `main/weights/bitnet_weights.bin`
   - **Offset**: `0x120000`
   - **COM port**: el del ESP32
4. Clic en **Flash**.

### Paso 4: Compilar el firmware

1. Clic en el icono del **martillo** (🔨) en la barra superior.
2. Espera a que termine la compilación.

**Salida esperada**:

```
Project build complete.
Binary size: <tamaño> bytes (<%> free in factory partition)
```

El tamaño exacto varía con la versión del compilador; verifica que no
aparezca `region overflowed` ni `section overflowed`.

### Paso 5: Flashear y ejecutar

1. Clic en el icono **verde** (▶️ Run) en la barra superior.
2. El IDE flashea:
   - Bootloader (`0x1000`)
   - Partition table (`0x8000`)
   - Firmware (`0x20000`)
   - **SPIFFS** (`index.html`) — solo si `FLASH_IN_PROJECT` está
     configurado en `main/CMakeLists.txt` (lo está por defecto).
3. Abre el **monitor serie** (`ESP-IDF → Monitor`).
4. Deberías ver:

```
I (xxx) BITNET: === BitNet ESP32 (chat web) ===
I (xxx) BITNET: SPIFFS: 2 KB / 110 KB
I (xxx) BITNET: weights: 2883584 bytes @ 0x120000 (bin: 2872912 bytes)
I (xxx) BITNET: WiFi AP OK: SSID=BitNet-ESP32, pass=bitnet1234
I (xxx) BITNET: Listo. Conectate a BitNet-ESP32 y abre http://192.168.4.1
```

### ⚠️ Notas importantes

- **Orden de flasheo**: pesos y firmware van a offsets distintos
  (`0x120000` vs `0x20000`), así que cualquier orden es válido.
- **Erase-flash borra el bin**: si ejecutas
  `Espressif → Erase Flash`, el binario de pesos **se elimina**. Deberás
  flashearlo de nuevo (Paso 3).
- **SPIFFS vacío**: si el monitor muestra `SPIFFS: 0 KB / 110 KB`, el
  `index.html` no se flasheó. Prueba
  `ESP-IDF → Flash SPIFFS Image` manualmente. Verifica también que
  `main/CMakeLists.txt` tenga:
  ```cmake
  spiffs_create_partition_image(spiffs data FLASH_IN_PROJECT)
  ```
- **`MAX_SEQ_LEN = 160`**: el firmware compila con `MAX_SEQ_LEN = 160`
  por límite de DRAM, aunque el `.bin` de pesos se exportó a 192. No
  hay que re-exportar: el binario de 2,872,912 bytes sigue siendo
  válido y solo se usan las primeras 160 filas de `pos_embed`.

---

## Índice

1. [Requisitos previos](#1-requisitos-previos)
2. [Preparación del entorno](#2-preparación-del-entorno)
3. [Dataset](#3-dataset)
4. [Tokenizador](#4-tokenizador)
5. [Entrenamiento en Colab](#5-entrenamiento-en-colab)
6. [Exportación a C](#6-exportación-a-c)
7. [Compilación del firmware](#7-compilación-del-firmware)
8. [Flasheo del ESP32](#8-flasheo-del-esp32)
9. [Verificación](#9-verificación)
10. [Problemas conocidos](#10-problemas-conocidos-y-soluciones)

---

## 1. Requisitos previos

### 1.1. Hardware

| Componente         | Especificación mínima                                    |
|--------------------|----------------------------------------------------------|
| PC de desarrollo   | Windows 10/11, 8 GB RAM, 50 GB libres                    |
| ESP32              | ESP32 original (rev 3.x), 4 MB Flash, **sin PSRAM**      |
| Cable USB          | USB de datos (no solo carga), que entregue ≥500 mA       |
| Fuente             | Puerto USB del PC o hub con alimentación                 |

### 1.2. Software

| Herramienta   | Versión | Dónde se instala              |
|---------------|---------|-------------------------------|
| Python        | 3.11+   | PC local                      |
| PyTorch       | 2.0+    | PC local y Colab              |
| ESP-IDF       | v5.3.1  | PC local                      |
| Eclipse IDE   | —       | PC local (opcional)           |
| Cuenta Google | —       | Para Colab y Drive            |

### 1.3. Instalación de ESP-IDF

1. Descarga el instalador de
   [ESP-IDF v5.3.1](https://docs.espressif.com/projects/esp-idf/en/v5.3.1/esp32/get-started/windows-setup.html).
2. Ejecuta el instalador y elige **"Install ESP-IDF and tools"**.
3. Al terminar, abre **"ESP-IDF 5.3 CMD"** desde el menú de inicio.
   Ese terminal tiene todas las variables de entorno configuradas.

Verifica la instalación:

```cmd
idf.py --version
```

Debe mostrar `ESP-IDF v5.3.1`.

### 1.4. Instalación de PyTorch

```cmd
pip install torch numpy
```

---

## 2. Preparación del entorno

### 2.1. Estructura esperada del proyecto

```
ESP_SLM/
├── README.MD
├── REPRODUCIR.md                    ← este archivo
├── esp32-bitnet-firmware.rar
├── docs/
│   └── ARQUITECTURA.MD
├── training/
│   ├── model_192.py
│   ├── tokenizer.py
│   ├── train_192.py
│   ├── export_weights_192.py
│   ├── export_vocab.py
│   ├── build_vocab_bigram.py
│   ├── vocab_bigram.json
│   └── ...
├── tools/
│   ├── check_header.py
│   └── inspec_header.py
├── bitnet-to-c/                     (motor C de referencia)
├── checkpoints/                     (ep20_val1.382.pt si lo tienes)
└── artifacts/                       (bitnet_weights.bin si lo tienes)
```

### 2.2. Verificar qué tienes

```cmd
cd C:\ruta\a\ESP_SLM
dir checkpoints
dir artifacts
```

| Situación                                     | Acción                       |
|-----------------------------------------------|------------------------------|
| Ambos archivos existen                        | Saltar a la **Guía rápida** o a la **Fase 7** |
| Falta el checkpoint pero existe el binario    | Reentrenar (Fases 3-6)       |
| Falta el binario pero existe el checkpoint    | Re-exportar (Fase 6)         |
| Faltan ambos                                  | Seguir todas las fases       |

---

## 3. Dataset

### 3.1. Descargar TinyStories en español

El dataset es `robrenaud/multilingual_tinystories` en HuggingFace.
Contiene 5 millones de historias en español (campo `story`) con
traducción al inglés (campo `translation`).

```python
!pip install huggingface_hub -q

from huggingface_hub import list_repo_files, hf_hub_download
import os

repo_id = "robrenaud/multilingual_tinystories"
all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
json_files = [f for f in all_files if f.endswith(".json")]

# Descargar los primeros 3 archivos (~1.4 GB, ~100K historias cada uno)
files_to_download = json_files[:3]
local_dir = "/content/dataset_subset"
os.makedirs(local_dir, exist_ok=True)

for f in files_to_download:
    print(f"Descargando {f}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=f,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )

total_size = sum(
    os.path.getsize(os.path.join(local_dir, f))
    for f in os.listdir(local_dir)
)
print(f"\nTotal descargado: {total_size / (1024**3):.2f} GB")
```

**Verificación**: deben aparecer 3 archivos `stories_00.json`,
`stories_01.json`, `stories_02.json` (~470 MB cada uno).

> **Nota**: con 1 solo archivo (~100K historias) es suficiente para
> empezar. Con 3 archivos se llega a ~80M tokens disponibles, aunque
> `MAX_SEQUENCES` limitará el número real usado en el entrenamiento.

---

## 4. Tokenizador

El tokenizador es a nivel de **bigramas/trigramas de caracteres** con
vocabulario de **2000 tokens**. Se construye una sola vez a partir del
corpus de entrenamiento.

### 4.1. Extraer el texto plano

```python
import json, glob

output_path = "/content/corpus_tinystories.txt"
all_stories = []

for fpath in sorted(glob.glob("/content/dataset_subset/*.json")):
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "story" in item:
            all_stories.append(item["story"])
    print(f"{fpath}: {len(all_stories):,} historias acumuladas")

with open(output_path, "w", encoding="utf-8") as f:
    for s in all_stories:
        f.write(s + "\n\n" + "=" * 80 + "\n\n")
print(f"✅ {output_path} ({len(all_stories):,} historias)")
```

### 4.2. Construir el vocabulario de bigramas

```python
!python build_vocab_bigram.py
```

**Salida esperada**:

```
Vocabulario final: 2000 tokens
  Especiales: 4
  Chars:      147
  N-gramas:   1849
Cobertura: 100.0000%
```

### 4.3. Verificar el tokenizador

```python
!python tokenizer.py
```

**Salida esperada**:

```
BigramTokenizer(vocab_size=2000, max_len=5)
[✅] 'Había una vez un llanero.'
[✅] 'El pueblo de San Pedro'
Corpus completo:
  ratio: 0.364 tokens/char
  UNK: 0
```

---

## 5. Entrenamiento en Colab

### 5.1. Preparar el tokenizado

```python
import json, pickle, glob
from tokenizer import BigramTokenizer

tok = BigramTokenizer()
all_ids = []

for fpath in sorted(glob.glob("/content/dataset_subset/*.json")):
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if "story" in item:
            all_ids.extend(tok.encode(item["story"]))
            all_ids.append(tok.EOS_ID)
    print(f"{fpath}: {len(all_ids):,} tokens")

with open("/content/corpus_tinystories_tokenized.pkl", "wb") as f:
    pickle.dump(all_ids, f)
print(f"✅ {len(all_ids):,} tokens → corpus_tinystories_tokenized.pkl")
```

Resultado: ~80M tokens en un `.pkl` de ~280 MB.

### 5.2. Configurar rutas en `train_192.py`

Edita `train_192.py` y ajusta las rutas:

```python
TOKENIZED_CORPUS = "/content/corpus_tinystories_tokenized.pkl"
OUTPUT_DIR = "/content/drive/MyDrive/checkpoints_PLE_192"
MAX_SEQUENCES = 300_000     # 300K secuencias × 192 tokens ≈ 57.6M tokens
MAX_SEQ_LEN = 192
```

> **Recomendación**: monta tu Drive primero con
> `from google.colab import drive; drive.mount('/content/drive')` para
> que los checkpoints persistan si la sesión se reinicia.

### 5.3. Verificar la GPU

```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

Debe mostrar `True` y `Tesla T4`. Si muestra `False`, ve a
`Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`.

### 5.4. Lanzar el entrenamiento

```python
!cd /content && python train_192.py
```

**Hiperparámetros** (por defecto en `train_192.py`):

| Parámetro        | Valor                       |
|------------------|-----------------------------|
| `batch_size`     | 64                          |
| `epochs`         | 20                          |
| `learning_rate`  | 3e-4                        |
| `weight_decay`   | 0.01                        |
| `grad_clip`      | 1.0                         |
| `val_split`      | 0.05                        |
| `MAX_SEQUENCES`  | 300,000 (~57.6M tokens)     |
| `MAX_SEQ_LEN`    | 192                         |

**Progreso esperado** (val loss):

| Época | Val Loss | PPL  |
|-------|----------|------|
| 1     | ~1.96    | ~7.1 |
| 5     | ~1.69    | ~5.4 |
| 10    | ~1.62    | ~5.0 |
| 15    | ~1.57    | ~4.8 |
| 20    | **1.3817**| **3.98** |

Cada época se guarda en `OUTPUT_DIR` como `epXX_valX.XXX.pt`.

### 5.5. Descargar el checkpoint

```python
from google.colab import files
files.download("/content/drive/MyDrive/checkpoints_PLE_192/ep20_val1.382.pt")
```

Guárdalo en `ESP_SLM/checkpoints/ep20_val1.382.pt`.

---

## 6. Exportación a C

Convierte el checkpoint de PyTorch en un binario cuantizado para el ESP32.

### 6.1. Ejecutar la exportación

```python
!python export_weights_192.py
```

**Salida esperada**:

```
Bin: /content/bitnet_weights.bin (2,872,912 bytes = 2.74 MB)
Header: /content/bitnet_weights.h (~8 KB)

Desglose:
  token_embed:    200 KB (INT4 + scales)
  pos_embed:       72 KB (FP16, 192 posiciones)
  PLE table:     1.83 MB (INT4 + scales)
  Ternarios (2b): 610 KB
  FP16 (biases):   55 KB
  ...
```

### 6.2. Verificar el binario

```python
import os
size = os.path.getsize("/content/bitnet_weights.bin")
assert size == 2872912, f"Tamaño inesperado: {size}"
print("✅ Tamaño correcto")
```

Debe ser **exactamente 2,872,912 bytes**. Cualquier otro valor indica
que el export falló, el checkpoint no es el correcto, o el
`MAX_SEQ_LEN` no es 192.

### 6.3. Descargar los artefactos

```python
from google.colab import files
files.download("/content/bitnet_weights.bin")
files.download("/content/bitnet_weights.h")
```

Guárdalos en:

| Archivo              | Destino                                        |
|----------------------|------------------------------------------------|
| `bitnet_weights.bin` | `ESP_SLM/artifacts/bitnet_weights.bin`         |
| `bitnet_weights.bin` | `ESP_SLM/bitnet-to-c/bitnet_weights.bin` (ref) |
| `bitnet_weights.h`   | `ESP_SLM/bitnet-to-c/bitnet_weights.h`         |

### 6.4. Validar el binario en PC (opcional)

```cmd
cd ESP_SLM\bitnet-to-c
gcc bitnet_inference.c tokenizer.c bitnet_main.c -o bitnet_test.exe -O2 -lm
bitnet_test.exe
```

En otra terminal:

```cmd
cd ESP_SLM\training
python diagnostico_192.py
```

Los top-5 logits de cada posición deben coincidir con error <0.5
(esperado por la cuantización INT4/INT8/2-bit y el KV cache INT8).

---

## 7. Compilación del firmware

### 7.1. Extraer el proyecto IDF

```cmd
mkdir C:\ESP32_work
cd C:\ESP32_work
REM Extraer ESP32_SML.rar con WinRAR o 7-Zip
```

**Estructura resultante**:

```
C:\ESP32_work\ESP32_SML\
├── CMakeLists.txt
├── partitions.csv
├── sdkconfig.defaults
└── main/
    ├── CMakeLists.txt
    ├── main.c
    ├── bitnet_inference.c / .h
    ├── tokenizer.c / .h
    ├── bitnet_weights.h
    ├── vocab_data.h
    ├── data/
    │   └── index.html
    └── weights/
        └── bitnet_weights.bin
```

### 7.2. Verificar el binario de pesos

El binario `main/weights/bitnet_weights.bin` ya viene incluido en el
`.rar` del firmware. Comprueba que su tamaño sea **2,872,912 bytes**:

```cmd
dir C:\ESP32_work\ESP32_SML\main\weights\bitnet_weights.bin
```

Si el tamaño es correcto, no hay que hacer nada más. Si por alguna
razón falta o está corrupto, cópialo desde
`ESP_SLM/artifacts/bitnet_weights.bin`:

```cmd
copy ESP_SLM\artifacts\bitnet_weights.bin C:\ESP32_work\ESP32_SML\main\weights\
```

### 7.3. Compilar

```cmd
cd C:\ESP32_work\ESP32_SML
idf.py set-target esp32
idf.py build
```

**Salida esperada**:

```
Project build complete.
Binary size: <tamaño> bytes (<%> free in factory partition)
```

Si aparece `region overflowed` o `section overflowed`, revisa la
sección de [problemas conocidos](#10-problemas-conocidos-y-soluciones).

---

## 8. Flasheo del ESP32

### 8.1. Detectar el puerto COM

```cmd
mode
```

Busca el puerto asignado (ej. `COM3`, `COM4`).

### 8.2. Flashear el firmware

```cmd
cd C:\ESP32_work\ESP32_SML
idf.py -p COM3 flash
```

Esto flashea:

| Componente       | Offset     |
|------------------|------------|
| Bootloader       | `0x1000`   |
| Partition table  | `0x8000`   |
| Firmware         | `0x20000`  |

### 8.3. Flashear SPIFFS (`index.html`)

```cmd
idf.py -p COM3 spiffs-flash
```

Si el comando no existe en tu IDF, alternativa manual:

```cmd
python %IDF_PATH%\components\spiffs\spiffsgen.py 0x20000 main\data spiffs.bin
esptool.py --chip esp32 -p COM3 write_flash 0x3E0000 spiffs.bin
```

**Verificación**: durante el arranque, el monitor debe mostrar
`SPIFFS: 2 KB / 110 KB`.

### 8.4. Flashear el binario de pesos

```cmd
esptool.py --chip esp32 -p COM3 -b 460800 write_flash ^
    0x120000 main\weights\bitnet_weights.bin
```

Escribe los 2,872,912 bytes en la partición `weights` (offset `0x120000`).

> **Importante**: el binario de pesos **no se re-flashea** al
> recompilar el firmware. Solo hay que repetir este paso si:
> - Se borra la flash entera (`idf.py erase-flash`).
> - Se cambia el binario por uno nuevo.
> - Se modifica la tabla de particiones.

> **Nota**: el orden entre firmware y pesos es indiferente (regiones
> distintas de la flash). Cualquiera de los dos órdenes funciona.

---

## 9. Verificación

### 9.1. Monitor serie

```cmd
idf.py -p COM3 monitor
```

**Log esperado**:

```
I (xxx) BITNET: === BitNet ESP32 (chat web) ===
I (xxx) BITNET: Heap al inicio: ~219 KB
I (xxx) BITNET: SPIFFS: 2 KB / 110 KB
I (xxx) BITNET: weights: 2883584 bytes @ 0x120000 (bin: 2872912 bytes)
I (xxx) BITNET: Pesos mapeados
I (xxx) BITNET: Heap tras init: ~135 KB
I (xxx) BITNET: Heap antes WiFi: ~130 KB
I (xxx) BITNET: WiFi AP OK: SSID=BitNet-ESP32, pass=bitnet1234, IP=192.168.4.1
I (xxx) BITNET: Heap final: ~88 KB
I (xxx) BITNET: Listo. Conectate a BitNet-ESP32 y abre http://192.168.4.1
```

### 9.2. Probar el chat

1. Desde tu móvil o PC, conéctate a la WiFi **`BitNet-ESP32`**
   (contraseña `bitnet1234`).
2. Abre **`http://192.168.4.1`** en el navegador.
3. Escribe un prompt (ej. `"Había una vez"`).
4. Pulsa **Enviar**. El texto aparecerá token a token durante
   ~60-75 s (o hasta que aparezca `<eos>`).

**Velocidad esperada**: ~2-2.5 tokens/s.

### 9.3. Logs durante la generación

```
I (xxx) BITNET: WS handshake
I (xxx) BITNET: Prompt: Había una vez (heap: 88 KB)
I (xxx) BITNET: Respuesta: ~150 tokens (heap: 88 KB, stack free: XXXX B)
```

Si el heap **baja** con cada petición, hay una fuga de memoria. Si se
mantiene estable en ~88 KB, todo correcto.

---

## 10. Problemas conocidos y soluciones

### 10.1. El ESP32 se resetea en bucle (`rst:0x3 SW_RESET`)

**Causa**: la calibración RF falla (`0x1102`) e intenta calibración
completa, que consume un pico de corriente.

**Solución**:

1. Verifica que `sdkconfig.defaults` tenga:
   ```
   CONFIG_ESP_TASK_WDT_TIMEOUT_S=30
   ```
2. Borra la flash y re-flashea:
   ```cmd
   idf.py -p COM3 erase-flash
   idf.py -p COM3 flash
   idf.py -p COM3 spiffs-flash
   esptool.py --chip esp32 -p COM3 write_flash 0x120000 main\weights\bitnet_weights.bin
   ```
3. Si sigue fallando, cambia el cable USB o usa un puerto con más corriente.

### 10.2. `SPIFFS: 0 KB / 110 KB` en el arranque

**Causa**: el `index.html` no se flasheó a SPIFFS.

**Solución**:

```cmd
idf.py -p COM3 spiffs-flash
```

### 10.3. Generación muy lenta (~1.5 tok/s en vez de 2-2.5)

**Causa**: el firmware se compiló con `-Os`.

**Solución**: en `menuconfig` → `Component config → Compiler optimization`
→ seleccionar **`Optimize for performance (-O2)`**. Recompilar y flashear.

### 10.4. `Guru Meditation Error` durante la generación

**Causa**: heap agotado o stack overflow.

**Solución**:

1. Verifica el heap libre en el log. Si es <50 KB, reduce el consumo:
   - `CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM = 4`
   - `CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM = 16`
2. Si el heap está bien, aumenta el stack del servidor HTTP:
   ```c
   httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
   cfg.stack_size = 8192;  // antes 4096
   ```

### 10.5. Caracteres raros en el monitor (`ï¿½ï¿½` en vez de `ñ` o `á`)

**Causa**: el monitor serie no interpreta UTF-8. **El modelo genera
UTF-8 correcto.** En el navegador se ven bien.

**Solución** (solo en el monitor):

```cmd
chcp 65001
idf.py -p COM3 monitor
```

### 10.6. `ModuleNotFoundError: No module named 'model_192'` en Colab

**Causa**: los scripts no están en `/content/`.

**Solución**:

```python
!cp /content/drive/MyDrive/*.py /content/
!cp /content/drive/MyDrive/vocab_bigram.json /content/
```

### 10.7. Error de calibración RF `0x1102` en cada arranque

**Causa**: no hay datos de calibración en NVS. La primera vez es normal.
Si ocurre **en cada arranque**, la calibración nunca termina de guardarse.

**Solución**: activar calibración parcial en `menuconfig`:

```
Component config → PHY → RF calibration → Partial calibration
```

### 10.8. Generación incoherente: solo repite "Habia una vez!" y se corta

**Causa**: bug en `attention()` del motor C. K y V se indexaban con
`off + d` (offset de la cabeza Q) aunque `num_kv_heads = 1`. Esto
producía lecturas fuera de fila para las cabezas 1-3 y contaminaba
posiciones >0. El motor C de versiones antiguas tiene este bug.

**Solución**: en `bitnet_inference.c`, función `attention()`, eliminar
`off` del índice de K y V (mantenerlo solo para Q):

```c
float kd = (float)krow[d] * sk_j;                        // antes: krow[off + d]
acc += buf_scores[j] * ((float)V_cache[l][j][d] * sv_j); // antes: [off + d]
```

**Verificación**: tras el fix, los logits de pos 0-4 deben coincidir
con PyTorch (error <0.5) y la generación debe producir español
coherente. Detalle en `ARQUITECTURA.MD §5.3`.

### 10.9. `region 'dram0_0_seg' overflowed by 5176 bytes`

**Causa**: con `MAX_SEQ_LEN = 192`, el KV cache INT8 (110 KB) + las
escalas + buffers + WiFi + HTTPD no caben en la DRAM útil del ESP32
original (~160 KB). El linker falla al enlazar `app-template.elf`.

**Solución**: en `main/bitnet_weights.h`:

```c
#define MAX_SEQ_LEN  160   // antes 192
```

El binario de pesos sigue siendo válido (`2,872,912` bytes): solo se
usan las primeras 160 filas de `pos_embed` en runtime. No hay que
re-exportar. Detalle en `ARQUITECTURA.MD §5.2`.

### 10.10. Relación entre el modelo (192) y el firmware (160)

**Aclaración**: el modelo se entrena con `max_seq_len = 192`, pero el
firmware se compila con `MAX_SEQ_LEN = 160` por límite de DRAM. No es
un error de configuración:

- El `.bin` exportado contiene `pos_embed` para 192 posiciones y todos
  los pesos. Sigue siendo el mismo archivo de **2,872,912 bytes**.
- El firmware solo usa las primeras 160 posiciones del `pos_embed` y
  reserva KV cache para 160 tokens.
- En la práctica, el modelo emite `<eos>` antes de llegar a 160, así
  que la pérdida de contexto no afecta a la calidad de las respuestas.

Si en el futuro se quiere exprimir hasta 192, habría que refactorizar
el motor para mover `buf_ple` y `logits` a heap (ver `ARQUITECTURA.MD
§5.2`), pero no es necesario para el funcionamiento actual.

---

## Resumen del pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  FASE 1-2  Preparar entorno (ESP-IDF, PyTorch, estructura)   │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  FASE 3    Descargar TinyStories español (3 JSON, ~1.4 GB)   │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  FASE 4    Construir tokenizador (vocab_bigram.json, 2000)   │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  FASE 5    Entrenar en Colab T4 (20 épocas)                  │
│            → ep20_val1.382.pt (27.3 MB, val_loss 1.3817)     │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  FASE 6    Exportar a C (INT4/INT8/2-bit)                    │
│            → bitnet_weights.bin (2.74 MB, 2,872,912 B)       │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  FASE 7    Compilar firmware ESP-IDF (idf.py build)          │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  FASE 8    Flashear (flash + spiffs-flash + weights)         │
└──────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  FASE 9    Verificar (WiFi AP + chat web a ~2-2.5 tok/s)     │
└──────────────────────────────────────────────────────────────┘
```

**Fin del pipeline.** Si llegas aquí, tienes un LLM de 6.8M de
parámetros generando texto de forma local en un ESP32 de 4 MB.