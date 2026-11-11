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
7. [Estructura del repositorio](#7-estructura-del-repositorio)

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
   aporta la mayor parte de los parámetros del modelo (3.84M de 6.835M)
   pero se lee bajo demanda, sin ocupar SRAM.
3. **Pesos ternarios estilo BitNet**: los pesos de las capas densas
   (atención, FFN, LM head) se cuantizan a {-1, 0, +1} durante el
   entrenamiento mediante un Straight-Through Estimator. En inferencia
   se empaquetan a 2 bits (4 pesos por byte).

### 1.2. Bloques internos

Cada bloque del transformer sigue el patrón **Pre-Norm**:

    x = x + Attention(Norm1(x))
    x = x + FFN(Norm2(x))
    x = x + PLE_to_Residual(Norm_ple(PLE_slice[l]))

El PLE añade una **tercera conexión residual** después de la FFN, con
su propia normalización. Es la única modificación estructural respecto
a un transformer estándar.

### 1.3. Dimensiones de los tensores

| Tensor                    | Forma                     |
|---------------------------|---------------------------|
| `token_embed`             | `[2000 × 192]`            |
| `pos_embed`               | `[192 × 192]`             |
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

> `pos_embed` se entrena a 192 posiciones, pero en el ESP32 el firmware
> solo usa las primeras 160 (`MAX_SEQ_LEN = 160`, ver §5.2 y §6.6).

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
| Framework     | PyTorch (versión compatible con Python 3.11+)              |
| Dataset       | 300,000 historias de TinyStories en español                |
| Secuencias    | 193 tokens (192 de entrada, 1 de target)                   |

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

La corrida final usa secuencias de 192 tokens y ~300K historias
(~80.9M tokens tokenizados).

| Corrida                | Épocas | Val Loss final | PPL final |
|------------------------|--------|----------------|-----------|
| 64 tokens (referencia) | 20     | 1.537          | 4.65      |
| **192 tokens (final)** | 20     | **1.3817**     | **3.98**  |

El entrenamiento converge de forma estable sin *early stopping*.
Checkpoint final: `ep20_val1.382.pt` (27.3 MB).

---

## 4. Exportación a C

### 4.1. Formato del binario

El binario `bitnet_weights.bin` (**2,872,912 bytes** = **2.74 MB**)
contiene los pesos serializados en un orden fijo. Cada componente se
cuantiza de forma independiente:

| Componente       | Formato                      | Tamaño                       |
|------------------|------------------------------|------------------------------|
| `token_embed`    | INT4 por fila + scales FP32  | 200,000 B (192 KB + 8 KB)    |
| `pos_embed`      | FP16                         | 73,728 B (72 KB)             |
| `PLE_table`      | INT4 por fila + scales FP32  | 1,928,000 B (1.83 MB + 8 KB) |
| `PLE_projection` | Ternario 2b                  | 92,160 B (90 KB)             |
| Núcleo (6 capas) | Ternario 2b + LN + bias FP16 | 478,248 B (~467 KB)          |
| `final_norm`     | FP16                         | 768 B                        |
| `LM_head`        | Ternario 2b + bias FP16      | 100,004 B (~98 KB)           |
| **Total**        |                              | **2,872,912 B (2.74 MB)**    |

### 4.2. Empaquetado

| Formato         | Descripción                                                         |
|-----------------|---------------------------------------------------------------------|
| Ternario 2-bit  | 4 pesos por byte con códigos `{0: -1, 1: 0, 2: +1}`                 |
| INT4            | 2 valores por byte con *sign extension*                             |
| INT8            | 1 valor por byte con *scale* global por tensor o por fila           |

### 4.3. Estructura del header

El archivo `bitnet_weights.h` (~8 KB) contiene:

- Definiciones de constantes de configuración del modelo
  (`MAX_SEQ_LEN 192`, `WEIGHTS_BIN_SIZE 2872912`).
- Offsets de cada componente dentro del binario (`OFF_TOKEN_EMBED`,
  `OFF_POS_EMBED`, `OFF_PLE_TABLE`, `OFF_B0_*`… `OFF_LM_HEAD_*`).
- Tabla de tokens (`vocab_data.h`, generado por separado).

> **Nota**: el header **no contiene los pesos**, solo los metadatos
> necesarios para mapearlos.

---

## 5. Motor de inferencia en C

### 5.1. Componentes implementados

| Función                  | Descripción                                                       |
|--------------------------|-------------------------------------------------------------------|
| `ternary_matmul`         | Multiplicación matriz-vector con pesos ternarios empaquetados, sin *branches* en el bucle interno |
| `layer_norm`             | Con `gamma`/`beta` en FP16 y `eps = 1e-5`                         |
| `attention`              | MQA con KV Cache INT8 y *scale* por posición                      |
| `ffn`                    | Dos proyecciones ternarias con GELU exacto (usando `erff`)        |
| `ple_lookup`             | Lectura de la tabla PLE desde Flash con desempaquetado INT4       |
| `bitnet_forward`         | Orquestación completa del forward pass                            |
| `bitnet_generate`        | Generación autorregresiva con *top-k* sampling                    |
| `bitnet_generate_stream` | Igual que `bitnet_generate` pero con callback por token (WebSocket) |

### 5.2. KV Cache

Almacenado en SRAM como `int8_t` con un `scale` en FP32 por cada
(capa, posición):

    static int8_t K_cache[NUM_LAYERS][MAX_SEQ_LEN][KV_DIM];
    static int8_t V_cache[NUM_LAYERS][MAX_SEQ_LEN][KV_DIM];
    static float  K_scales[NUM_LAYERS][MAX_SEQ_LEN];
    static float  V_scales[NUM_LAYERS][MAX_SEQ_LEN];

**Tamaño total para `MAX_SEQ_LEN = 160`** (el valor que usa el ESP32):

| Componente               | Cálculo                    | Tamaño       |
|--------------------------|----------------------------|--------------|
| `K_cache` + `V_cache`    | 6 × 160 × 48 × 2           | 92,160 B     |
| `K_scales` + `V_scales`  | 6 × 160 × 4 × 2            | 7,680 B      |
| **Total KV Cache**       |                            | **~97.5 KB** |
| Buffers adicionales      | `buf_x`, `buf_q`, `buf_ple[1920]`, `logits[2000]`, etc. | ~22 KB |
| **Total SRAM motor**     |                            | **~120 KB**  |

El `scale` se calcula en cada escritura como `max(|K|) / 127`, dando la
máxima precisión posible para cada vector.

> **Por qué 160 y no 192**: con `MAX_SEQ_LEN = 192` el linker falla con
> `dram0_0_seg overflowed by 5176 bytes`. El KV Cache INT8 a 192 tokens
> (110 KB) + escalas + buffers + WiFi + HTTPD no caben en los ~160 KB
> de DRAM útil del ESP32 original. Bajar a 160 libera ~19.5 KB de `.bss`
> y deja ~88 KB de heap libre en operación. El modelo se entrena a 192,
> pero el `<eos>` aparece antes de 160 en la práctica, así que la
> pérdida de contexto es irrelevante.

### 5.3. Validación con PyTorch

Los logits producidos por el motor C se compararon con los de PyTorch
para 5 posiciones consecutivas con `TOKENS = {4, 10, 20, 30, 40}`.
Los resultados **coinciden en los IDs del top-5** y los logits difieren
en menos de 0.5 (esperado por la cuantización INT4/INT8/2-bit y el KV
Cache INT8).

**Ejemplo** (posición 0, token 4):

| PyTorch    | C          |
|------------|------------|
| 2 (10.887) | 2 (10.851) |
| 7 (9.503)  | 7 (9.559)  |
| 5 (8.768)  | 5 (8.844)  |

> **Bug corregido durante la validación (MQA)**: en la primera versión
> del motor, `attention()` indexaba K y V con `off + d` (offset de la
> cabeza Q) aunque `num_kv_heads = 1`. Esto provocaba lecturas fuera de
> fila para `h ≥ 1`, contaminando posiciones >0 y cortando la generación
> tras 1-2 tokens ("Habia una vez!"). El fix fue eliminar el `off` del
> índice de K/V (mantenerlo solo para Q):
>
>     float kd = (float)krow[d] * sk_j;                        // antes: krow[off + d]
>     acc += buf_scores[j] * ((float)V_cache[l][j][d] * sv_j); // antes: [off + d]
>
> Con este fix, los logits coinciden con PyTorch en pos 0–4 y la
> generación produce español coherente en el ESP32.

---

## 6. Firmware del ESP32

### 6.1. Características del microcontrolador

| Recurso | Valor                         |
|---------|-------------------------------|
| Chip    | ESP32 original (rev 3.1)      |
| CPU     | Dual-core Xtensa LX6, 240 MHz |
| SRAM    | 520 KB                        |
| Flash   | 4 MB                          |
| PSRAM   | No                            |
| WiFi    | 802.11 b/g/n                  |

### 6.2. Carga de pesos con `mmap`

El binario de pesos se almacena en la partición `weights` (offset
`0x120000`, 2.75 MB) y se mapea en el espacio de direcciones del
procesador con `esp_partition_mmap`. Esto permite **leer los pesos
directamente desde Flash** sin copiarlos a SRAM. La única penalización
es la velocidad de acceso (~4× más lento que SRAM), parcialmente
mitigada por el caché de instrucciones y datos.

### 6.3. Servidor web

| Componente   | Configuración                                                    |
|--------------|------------------------------------------------------------------|
| **WiFi AP**  | SSID `BitNet-ESP32`, contraseña `bitnet1234`, IP `192.168.4.1`   |
| **HTTP**     | Sirve `index.html` desde SPIFFS en la ruta `/`                   |
| **WebSocket**| Canal bidireccional en `/ws`                                     |

**Protocolo WebSocket**:

| Byte(s) | Significado                             |
|---------|-----------------------------------------|
| `0x01`  | Inicio de generación                    |
| UTF-8   | Tokens individuales conforme se generan |
| `0x02`  | Fin de generación                       |

**Streaming**: el texto aparece en el navegador a medida que se genera,
sin esperar a que termine la secuencia completa. El callback
`ws_token_cb` decodifica cada token con `VOCAB_STRINGS`/`VOCAB_LENGTHS`
y lo envía como frame WebSocket.

### 6.4. Gestión de memoria

Durante la operación estable, el **heap libre se mantiene en ~88 KB**.
Este margen cubre:

- Stack de la tarea HTTP (~8 KB).
- Buffers de WebSocket (~2 KB).
- Picos transitorios de la pila durante la generación.

El firmware **no asigna memoria dinámica durante la inferencia**: todos
los buffers del motor son estáticos y se declaran en tiempo de
compilación. `out_ids[MAX_SEQ_LEN]` e `ids[128]` viven en `.bss` del
task del servidor.

### 6.5. Rendimiento medido

| Métrica                          | Valor        |
|----------------------------------|--------------|
| Velocidad de generación          | ~2–2.5 tok/s |
| Latencia por token               | ~400–500 ms  |
| Tiempo de generación (~150 tok)  | ~60–75 s     |
| Heap libre en operación          | ~88 KB       |
| Uso de SRAM para KV Cache        | ~97.5 KB     |
| Uso de SRAM para buffers         | ~22 KB       |
| Uso de Flash para pesos          | 2.74 MB      |
| Uso de Flash para firmware       | ~867 KB      |

### 6.6. Limitaciones derivadas del hardware

| Limitación                     | Consecuencia                                                                 |
|--------------------------------|------------------------------------------------------------------------------|
| **Sin PSRAM**                  | El KV Cache y los buffers deben vivir en los 520 KB de SRAM. Con `MAX_SEQ_LEN = 192` la `.bss` desborda la DRAM (`overflowed by 5176 bytes`); con 160 cabe con ~88 KB de heap libre. Los 32 tokens restantes del entrenamiento no se usan en runtime |
| **Ancho de banda de Flash**    | Las lecturas XIP limitan la velocidad del matmul. Los pesos se leen en cada token, por lo que la velocidad está acotada por el bus SPI (40 MHz en modo DIO) |
| **Precisión de `expf`/`erff`** | Las funciones matemáticas del ESP32 son ligeramente menos precisas que las de una CPU, lo que introduce diferencias de ~1e-4 respecto a PyTorch |
| **Sin aceleración vectorial**  | El ESP32 original no dispone de instrucciones SIMD; el ESP32-S3 (que sí las tiene) sería entre 2 y 3 veces más rápido con el mismo firmware |
| **Sin PSRAM → KV INT8 fijo**  | El KV Cache se mantiene en INT8 (no INT4) porque reducir a INT4 liberaría ~54 KB pero introduciría ruido de precisión en la atención y riesgo de bugs de empaquetado. Con 88 KB de heap libre, no hay necesidad |

---

## 7. Estructura del repositorio

    ESP32_SML/
    ├── CMakeLists.txt
    ├── partitions.csv
    ├── sdkconfig.defaults
    └── main/
        ├── CMakeLists.txt
        ├── main.c                  # app_main, WiFi, HTTP, WebSocket
        ├── bitnet_inference.c      # motor de inferencia (MQA fix aplicado)
        ├── bitnet_inference.h
        ├── tokenizer.c             # tokenizador de n-gramas
        ├── tokenizer.h
        ├── bitnet_weights.h        # offsets del binario (MAX_SEQ_LEN 192)
        ├── vocab_data.h            # tabla de tokens
        ├── data/
        │   └── index.html          # interfaz web
        └── weights/
            └── bitnet_weights.bin  # pesos del modelo (2.74 MB, 2,872,912 B)