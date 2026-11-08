# Arquitectura técnica del proyecto

> Detalle de las decisiones de diseño, la implementación y las
> limitaciones del hardware que condicionan cada componente.

---

## Índice

1. [Modelo](#1-modelo)
2. [Tokenizador](#2-tokenizador)
3. [Entrenamiento](#3-entrenamiento)
4. [Exportación a C](#4-exportación-a-c)
5. [Motor de inferencia en C](#5-motor-de-inferencia-en-c)
6. [Firmware del ESP32](#6-firmware-del-esp32)
7. [Compilación y despliegue](#7-compilación-y-despliegue)
8. [Estructura del repositorio](#8-estructura-del-repositorio)

---

## 1. Modelo

### 1.1. Arquitectura general

Transformer decoder-only con **tres innovaciones** que permiten su
ejecución en un ESP32:

1. **Multi-Query Attention (MQA)**: todas las cabezas de Query comparten
   un único par Key/Value. Reduce el KV Cache en un factor de
   `num_heads` (4×) sin pérdida significativa de calidad.
2. **Per-Layer Embeddings (PLE)**: cada capa recibe una señal de
   identidad del token desde una tabla gigante en Flash. Esta tabla
   aporta la mayor parte de los parámetros del modelo (3.84M de 6.81M)
   pero se lee bajo demanda, sin ocupar SRAM.
3. **Pesos ternarios estilo BitNet**: los pesos de las capas densas
   (atención, FFN, LM head) se cuantizan a `{-1, 0, +1}` durante el
   entrenamiento mediante un Straight-Through Estimator. En inferencia
   se empaquetan a 2 bits (4 pesos por byte).

### 1.2. Bloques internos

Cada bloque del transformer sigue el patrón **Pre-Norm**:

```python
x = x + Attention(Norm1(x))
x = x + FFN(Norm2(x))
x = x + PLE_to_Residual(Norm_ple(PLE_slice[l]))
```

El PLE añade una **tercera conexión residual** después de la FFN, con
su propia normalización. Es la única modificación estructural respecto
a un transformer estándar.

### 1.3. Dimensiones de los tensores

| Tensor                    | Forma                     |
|---------------------------|---------------------------|
| `token_embed`             | `[2000 × 192]`            |
| `pos_embed`               | `[64 × 192]`              |
| `PLE_table`               | `[2000 × (6 × 320)]`      |
| `PLE_projection`          | `[192 × (6 × 320)]`       |
| `Q_proj` (por capa)       | `[192 × 192]`             |
| `K_proj` (por capa)       | `[192 × 48]` (MQA)        |
| `V_proj` (por capa)       | `[192 × 48]` (MQA)        |
| `O_proj` (por capa)       | `[192 × 192]`             |
| `FC1` (por capa)          | `[192 × 384]`             |
| `FC2` (por capa)          | `[384 × 192]`             |
| `PLE_residual` (por capa) | `[320 × 192]`             |
| `LM_head`                 | `[192 × 2000]`            |

---

## 2. Tokenizador

### 2.1. Diseño

Tokenizador basado en **n-gramas de caracteres** (bigramas, trigramas,
4-gramas) con un vocabulario de **2000 tokens**. Se construye una sola
vez a partir del corpus de entrenamiento.

**Composición del vocabulario**:

| Categoría    | Cantidad                                 |
|--------------|------------------------------------------|
| Especiales   | 4 (`<pad>`, `<sos>`, `<eos>`, `<unk>`)   |
| Caracteres   | 147                                      |
| N-gramas     | 1849                                     |
| **Total**    | **2000**                                 |

### 2.2. Codificación

Algoritmo **greedy longest-match**: para cada posición del texto, se
busca el n-grama más largo del vocabulario que coincida. Esto garantiza
la máxima compresión con las entradas disponibles.

**Métricas sobre el corpus de entrenamiento**:

| Métrica       | Valor                                            |
|---------------|--------------------------------------------------|
| Ratio         | 0.364 tokens por carácter (2.8× mejor que byte-level) |
| Cobertura     | 100% (todos los caracteres del corpus están en el vocabulario) |

### 2.3. Implementación en C

En el ESP32, el tokenizador se implementa como una **búsqueda lineal**
sobre las 2000 entradas de la tabla `VOCAB_STRINGS`/`VOCAB_LENGTHS`,
almacenada en Flash como parte del firmware. No requiere estructuras de
datos dinámicas.

---

## 3. Entrenamiento

### 3.1. Infraestructura

| Componente    | Valor                                                      |
|---------------|------------------------------------------------------------|
| Hardware      | NVIDIA Tesla T4 (16 GB VRAM) en Google Colab               |
| Framework     | PyTorch (versión compatible con Python 3.10+)              |
| Dataset       | 300,000 historias de TinyStories en español                |
| Secuencias    | 65 tokens (64 de entrada, 1 de target)                     |

### 3.2. Hiperparámetros

| Parámetro         | Valor                |
|-------------------|----------------------|
| `batch_size`      | 64                   |
| `epochs`          | 20                   |
| `learning_rate`   | 3e-4                 |
| `weight_decay`    | 0.01                 |
| `grad_clip`       | 1.0                  |
| `val_split`       | 0.05                 |
| `optimizer`       | AdamW                |
| `scheduler`       | CosineAnnealingLR    |

### 3.3. Cuantización durante el entrenamiento

La cuantización ternaria se aplica en el **forward** de cada `BitLinear`:

1. Se calcula `scale = mean(|W|)`.
2. Se obtiene `W_q = clamp(round(W / scale), -1, 1)`.
3. En el **forward**, se usa `W_q * scale`.
4. En el **backward**, se usa el gradiente original mediante
   `W + (W_q * scale - W).detach()` (STE).

El modelo nunca ve pesos de alta precisión durante el forward, pero el
backward los conserva para que los gradientes fluyan sin distorsión.

### 3.4. Métricas de convergencia

| Época | Val Loss | PPL    |
|-------|----------|--------|
| 1     | 1.960    | 7.10   |
| 5     | 1.689    | 5.41   |
| 10    | 1.616    | 5.03   |
| 15    | 1.573    | 4.82   |
| 20    | **1.537**| **4.65** |

El entrenamiento converge de forma estable sin *early stopping*.

---

## 4. Exportación a C

### 4.1. Formato del binario

El binario `bitnet_weights.bin` (**2,823,760 bytes**) contiene los pesos
serializados en un orden fijo. Cada componente se cuantiza de forma
independiente:

| Componente       | Formato       | Tamaño    |
|------------------|---------------|-----------|
| `token_embed`    | INT4 por fila | 188 KB    |
| `pos_embed`      | FP16          | 24 KB     |
| `PLE_table`      | INT4 por fila | 1.83 MB   |
| `PLE_scales`     | FP32          | 8 KB      |
| `PLE_projection` | Ternario 2b   | 92 KB     |
| Núcleo (6 capas) | Ternario 2b   | 518 KB    |
| LayerNorms       | FP16          | 40 KB     |
| Biases           | FP16          | 24 KB     |
| `LM_head`        | Ternario 2b   | 96 KB     |
| **Total**        |               | **2.69 MB** |

### 4.2. Empaquetado

| Formato         | Descripción                                                         |
|-----------------|---------------------------------------------------------------------|
| Ternario 2-bit  | 4 pesos por byte con códigos `{0: -1, 1: 0, 2: +1}`                 |
| INT4            | 2 valores por byte con *sign extension*                             |
| INT8            | 1 valor por byte con *scale* global por tensor o por fila           |

### 4.3. Estructura del header

El archivo `bitnet_weights.h` (6 KB) contiene:

- Definiciones de constantes de configuración del modelo.
- Offsets de cada componente dentro del binario.
- Tabla de tokens (`vocab_data.h`, generado por separado).

> **Nota**: el header **no contiene los pesos**, solo los metadatos
> necesarios para mapearlos.

---

## 5. Motor de inferencia en C

### 5.1. Componentes implementados

| Función              | Descripción                                                       |
|----------------------|-------------------------------------------------------------------|
| `ternary_matmul`     | Multiplicación matriz-vector con pesos ternarios empaquetados, sin *branches* en el bucle interno |
| `layer_norm`         | Con `gamma`/`beta` en FP16 y `eps = 1e-5`                         |
| `attention`          | MQA con KV Cache INT8 y *scale* por posición                      |
| `ffn`                | Dos proyecciones ternarias con GELU exacto (usando `erff`)        |
| `ple_lookup`         | Lectura de la tabla PLE desde Flash con desempaquetado INT4       |
| `bitnet_forward`     | Orquestación completa del forward pass                            |
| `bitnet_generate`    | Generación autorregresiva con *top-k* sampling                    |

### 5.2. KV Cache

Almacenado en SRAM como `int8_t` con un `scale` en FP32 por cada
(capa, posición):

```c
static int8_t K_cache[NUM_LAYERS][MAX_SEQ_LEN][KV_DIM];
static int8_t V_cache[NUM_LAYERS][MAX_SEQ_LEN][KV_DIM];
static float  K_scales[NUM_LAYERS][MAX_SEQ_LEN];
static float  V_scales[NUM_LAYERS][MAX_SEQ_LEN];
```

**Tamaño total para 64 tokens**:

| Componente               | Cálculo                    | Tamaño      |
|--------------------------|----------------------------|-------------|
| `K_cache` + `V_cache`    | 6 × 64 × 48                | 18,432 B    |
| `K_scales` + `V_scales`  | 6 × 64 × 4 × 2             | 3,072 B     |
| **Total KV Cache**       |                            | **~21 KB**  |
| Buffers adicionales      |                            | ~15 KB      |
| **Total SRAM motor**     |                            | **~36 KB**  |

El `scale` se calcula en cada escritura como `max(|K|) / 127`, dando la
máxima precisión posible para cada vector.

### 5.3. Validación con PyTorch

Los logits producidos por el motor C se compararon con los de PyTorch
para 5 posiciones consecutivas. Los resultados **coinciden en los IDs
del top-3** y los logits difieren en menos de 0.4 (esperado por la
cuantización INT4/INT8/2-bit).

**Ejemplo** (posición 0, token 4):

| PyTorch         | C               |
|-----------------|-----------------|
| 2 (11.354)      | 2 (11.171)      |
| 7 (9.840)       | 7 (9.792)       |
| 1956 (8.361)    | 1956 (7.960)    |

---

## 6. Firmware del ESP32

### 6.1. Características del microcontrolador

| Recurso   | Valor                              |
|-----------|------------------------------------|
| Chip      | ESP32 original (rev 3.1)           |
| CPU       | Dual-core Xtensa LX6, 240 MHz      |
| SRAM      | 520 KB                             |
| Flash     | 4 MB                               |
| PSRAM     | No                                 |
| WiFi      | 802.11 b/g/n                       |

### 6.2. Carga de pesos con `mmap`

El binario de pesos se almacena en la partición `weights` y se mapea en
el espacio de direcciones del procesador con `esp_partition_mmap`. Esto
permite **leer los pesos directamente desde Flash** sin copiarlos a
SRAM. La única penalización es la velocidad de acceso (~4× más lento
que SRAM), parcialmente mitigada por el caché de instrucciones y datos.

### 6.3. Servidor web

| Componente   | Configuración                                                    |
|--------------|------------------------------------------------------------------|
| **WiFi AP**  | SSID `BitNet-ESP32`, contraseña `bitnet1234`, IP `192.168.4.1`   |
| **HTTP**     | Sirve `index.html` desde SPIFFS en la ruta `/`                   |
| **WebSocket**| Canal bidireccional en `/ws`                                     |

**Protocolo WebSocket**:

| Byte(s)     | Significado                              |
|-------------|------------------------------------------|
| `0x01`      | Inicio de generación                     |
| UTF-8       | Tokens individuales conforme se generan  |
| `0x02`      | Fin de generación                        |

**Streaming**: el texto aparece en el navegador a medida que se genera,
sin esperar a que termine la secuencia completa.

### 6.4. Gestión de memoria

Durante la operación estable, el **heap libre se mantiene en ~143 KB**.
Este margen cubre:

- Stack de la tarea HTTP (~8 KB).
- Buffers de WebSocket (~2 KB).
- Picos transitorios de la pila durante la generación.

El firmware **no asigna memoria dinámica durante la inferencia**: todos
los buffers del motor son estáticos y se declaran en tiempo de
compilación.

### 6.5. Rendimiento medido

| Métrica                          | Valor      |
|----------------------------------|------------|
| Velocidad de generación          | ~3.2 tok/s |
| Latencia por token               | ~310 ms    |
| Tiempo de generación (50 tok)    | ~16 s      |
| Heap libre en operación          | 143 KB     |
| Uso de SRAM para KV Cache        | 36 KB      |
| Uso de SRAM para buffers         | ~15 KB     |
| Uso de Flash para pesos          | 2.69 MB    |
| Uso de Flash para firmware       | 867 KB     |

### 6.6. Limitaciones derivadas del hardware

| Limitación                     | Consecuencia                                                                 |
|--------------------------------|------------------------------------------------------------------------------|
| **Sin PSRAM**                  | El KV Cache y los buffers deben vivir en los 520 KB de SRAM, lo que limita `MAX_SEQ_LEN` a 64 tokens con el margen actual |
| **Ancho de banda de Flash**    | Las lecturas XIP limitan la velocidad del matmul. Los pesos se leen en cada token, por lo que la velocidad está acotada por el bus SPI (40 MHz en modo DIO) |
| **Precisión de `expf`/`erff`** | Las funciones matemáticas del ESP32 son ligeramente menos precisas que las de una CPU, lo que introduce diferencias de ~1e-4 respecto a PyTorch |
| **Sin aceleración vectorial**  | El ESP32 original no dispone de instrucciones SIMD; el ESP32-S3 (que sí las tiene) sería entre 2 y 3 veces más rápido con el mismo firmware |

---

## 7. Compilación y despliegue

### 7.1. Compilación

```bash
cd esp32-bitnet
idf.py set-target esp32
idf.py build
```

**Resultado esperado**:

| Archivo                | Tamaño             |
|------------------------|--------------------|
| `app-template.bin`     | 887,520 bytes (15% libre en la partición de 1 MB) |
| `partition-table.bin`  | 3 KB               |
| `bootloader.bin`       | 26 KB              |
| `spiffs.bin`           | 2 KB               |

### 7.2. Flasheo

```bash
# Firmware + bootloader + partition table
idf.py -p COMx flash

# Sistema de archivos web (index.html)
idf.py -p COMx spiffs-flash

# Pesos del modelo (partición independiente)
esptool.py --chip esp32 -p COMx -b 460800 write_flash \
    0x120000 main/weights/bitnet_weights.bin
```

> **Importante**: el binario de pesos **no se re-flashea** al recompilar
> el firmware, ya que ocupa una partición independiente y no se ve
> afectado por `idf.py flash`.

---

## 8. Estructura del repositorio

```
esp32-bitnet/
├── CMakeLists.txt
├── partitions.csv
├── sdkconfig.defaults
└── main/
    ├── CMakeLists.txt
    ├── main.c                  # app_main, WiFi, HTTP, WebSocket
    ├── bitnet_inference.c      # motor de inferencia
    ├── bitnet_inference.h
    ├── tokenizer.c             # tokenizador de n-gramas
    ├── tokenizer.h
    ├── bitnet_weights.h        # offsets del binario
    ├── vocab_data.h            # tabla de tokens
    ├── data/
    │   └── index.html          # interfaz web
    └── weights/
        └── bitnet_weights.bin  # pesos del modelo (2.69 MB)
```