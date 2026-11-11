# BitNet ESP32 — LLM de 6.8M parámetros en un ESP32

> Implementación completa de un modelo de lenguaje (Small Language
> Model) que corre íntegramente en un ESP32 original (520 KB SRAM,
> 4 MB Flash), sin PSRAM y sin conexión a internet. El modelo se
> entrena en una GPU en la nube, se cuantiza agresivamente y se ejecuta
> en el microcontrolador con un motor de inferencia escrito en C.

---

## Índice

1. [Características principales](#1-características-principales)
2. [Parámetros del modelo](#2-parámetros-del-modelo)
3. [Resultados del entrenamiento](#3-resultados-del-entrenamiento)
4. [Resultados en el ESP32](#4-resultados-en-el-esp32)
5. [Estructura de particiones](#5-estructura-de-particiones)
6. [Flujo completo del proyecto](#6-flujo-completo-del-proyecto)
7. [Problemas encontrados y resueltos](#7-problemas-encontrados-y-resueltos)
8. [Requisitos para reproducir el proyecto](#8-requisitos-para-reproducir-el-proyecto)
9. [Limitaciones conocidas](#9-limitaciones-conocidas)

---

## 1. Características principales

- **Modelo**: Transformer decoder-only de 6.8M de parámetros con
  Multi-Query Attention (MQA), Per-Layer Embeddings (PLE) y pesos
  ternarios estilo BitNet.
- **Tokenizador**: bigramas/trigramas de caracteres, vocabulario de
  2000 tokens.
- **Inferencia local**: el ESP32 genera texto sin depender de ningún
  servicio externo.
- **Interfaz**: página web servida por el propio ESP32, accesible vía
  WiFi AP. Comunicación por WebSocket con streaming token a token.
- **Cuantización**: núcleo BitNet a 2 bits empaquetados, tabla PLE a
  INT4, embeddings a INT4, biases y normalizaciones a FP16, KV Cache a
  INT8.
- **Sin asignación dinámica de pesos**: los 2.74 MB del modelo viven en
  Flash y se mapean con `esp_partition_mmap` (cero copia a SRAM).
- **Aislamiento del firmware**: el binario de pesos va en una partición
  independiente; el firmware (~867 KB) y la web (SPIFFS, ~2.2 KB) van en
  sus propias particiones.

---

## 2. Parámetros del modelo

| Componente             | Valor                                |
|------------------------|--------------------------------------|
| Arquitectura           | Transformer decoder-only, Pre-Norm   |
| Parámetros totales     | **6,835,472**                        |
| `vocab_size`           | 2000                                 |
| `embed_dim`            | 192                                  |
| `num_heads` (Q)        | 4                                    |
| `num_kv_heads`         | 1 (MQA)                              |
| `head_dim`             | 48                                   |
| `num_layers`           | 6                                    |
| `ff_dim`               | 384                                  |
| `max_seq_len` (entrenado) | 192                               |
| `max_seq_len` (ESP32)  | 160 (limitado por DRAM)              |
| `ple_hidden_dim`       | 320                                  |
| Parámetros en PLE      | 3,840,000                            |
| Parámetros ternarios   | 2,558,976                            |

---

## 3. Resultados del entrenamiento

| Métrica                      | Valor                                            |
|------------------------------|--------------------------------------------------|
| Dataset                      | TinyStories español (`robrenaud/multilingual_tinystories`) |
| Historias usadas             | 300,000 (~80.9M tokens tokenizados)              |
| Épocas                       | 20                                               |
| Loss de validación final     | **1.3817**                                       |
| Perplejidad (PPL)            | **3.98**                                         |
| Tiempo total de entrenamiento| ~110 minutos en T4 (Colab)                       |
| Velocidad                    | ~5.5 min/época                                   |

---

## 4. Resultados en el ESP32

| Métrica                          | Valor                                          |
|----------------------------------|------------------------------------------------|
| Velocidad de generación          | **~2–2.5 tokens/s**                            |
| Heap libre en operación          | **~88 KB**                                     |
| Consumo de Flash del modelo      | 2.74 MB de los 4 MB totales                    |
| Firmware compilado               | ~867 KB (15% libre en la partición de 1 MB)    |
| KV Cache en SRAM                 | ~97.5 KB (INT8, 160 tokens)                    |
| Tiempo de inferencia de un prompt| ~60–75 segundos para ~150 tokens               |

---

## 5. Estructura de particiones

| Partición   | Offset     | Tamaño   | Contenido                        |
|-------------|------------|----------|----------------------------------|
| `nvs`       | `0x9000`   | 20 KB    | Credenciales y config de WiFi    |
| `otadata`   | `0xe000`   | 8 KB     | Metadata de OTA                  |
| `phy_init`  | `0x10000`  | 4 KB     | Calibración de RF                |
| `factory`   | `0x20000`  | 1 MB     | Firmware                         |
| `weights`   | `0x120000` | 2.75 MB  | Binario de pesos (mmap, 2,872,912 B) |
| `spiffs`    | `0x3E0000` | 128 KB   | `index.html`                     |

---

## 6. Flujo completo del proyecto

1. **Dataset**: descarga y filtrado de TinyStories en español.
2. **Tokenizador**: extracción de n-gramas del corpus, vocabulario de 2000.
3. **Entrenamiento**: 20 épocas en Colab (T4), con cuantización ternaria
   mediante STE (Straight-Through Estimator).
4. **Exportación**: conversión del checkpoint de PyTorch a un binario con
   cuantización específica por componente.
5. **Motor C**: implementación del forward pass (atención, FFN,
   LayerNorm, PLE, KV Cache) sin dependencias externas.
6. **Firmware**: WiFi AP, servidor HTTP, WebSocket, integración del
   motor y de la interfaz web.

---

## 7. Problemas encontrados y resueltos

| Problema                                        | Causa                                                              | Solución                                                       |
|-------------------------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------|
| Dataset inicial insuficiente                    | El corpus llanero original (6.68 MB) hacía inviable la tabla PLE (PPL 52.82) | Migración a TinyStories en español (80M tokens disponibles, 80.9M tokenizados) |
| Ruido académico en el corpus                    | Referencias bibliográficas y URLs contaminaban el texto            | Eliminación de entradas no narrativas                          |
| Tabla PLE no aprendía con pocos datos           | Requería al menos 10 tokens por parámetro; con 6.68 MB no se alcanzaba | Aumento del corpus a ~80.9M tokens                         |
| Cuantización de la tabla PLE                    | Pasó de FP16 (1.57 MB) a INT4 (1.83 MB empaquetada)                | Necesario para permitir el vocabulario real                    |
| Velocidad baja del firmware (`-Os`)             | `ternary_matmul` a 1.85 tok/s con *branches*                       | Cambio a `-O2` + eliminación de *branches* del bucle interno   |
| Timeout del Task Watchdog                       | La calibración RF y el matmul excedían los 5 s por defecto         | Subida del timeout a 30 s                                      |
| Calibración RF fallida (`0x1102`)               | Sin datos de calibración en NVS, se intentaba calibración completa y el sistema reiniciaba | Forzar calibración parcial + borrar flash antes del primer boot |
| Encoding del monitor serie                      | Los caracteres UTF-8 (ñ, á) se mostraban como `ï¿½`                 | El modelo genera UTF-8 correcto; el problema es del terminal, no del firmware |
| `hex escape sequence out of range` en `vocab_data.h` | Literales C con `\xc3\xb1` seguidos de caracteres hexadecimales ambiguos | Añadir separadores `""` entre secuencias de escape              |
| Bug MQA en `attention()`                        | K/V se indexaban con `off + d` (offset de cabeza Q), pero `NUM_KV_HEADS=1` → lectura fuera de fila para `h ≥ 1`, contaminando pos>0 | Cambiar `krow[off + d]` → `krow[d]` y `V_cache[l][j][off + d]` → `V_cache[l][j][d]` |
| Overflow de DRAM con `MAX_SEQ_LEN=192`          | `dram0_0_seg overflowed by 5176 bytes`; el KV Cache INT8 a 192 tokens + buffers no caben en la DRAM del ESP32 original | Bajar `MAX_SEQ_LEN` a 160 en `bitnet_weights.h` (ahorro de ~19.5 KB); el modelo se entrena a 192 pero se ejecuta a 160 |
| Fine-tuning fallido                             | Intentos de especializar en poesía con BitLoRA (val_loss 7.5, azar) y LoRA estándar (val_loss 3.75, peor que base) | El modelo base es demasiado pequeño para absorber un dominio nuevo sin destruir la coherencia. Se usa el checkpoint base sin fine-tuning |

---

## 8. Requisitos para reproducir el proyecto

| Componente         | Requisito                                                       |
|--------------------|-----------------------------------------------------------------|
| Entrenamiento      | Cuenta de Google Colab (GPU T4), Python 3.11+, PyTorch           |
| Exportación        | Python 3.11+ con PyTorch y NumPy                                |
| Firmware           | ESP-IDF v5.3.1, ESP32 original (sin PSRAM, 4 MB Flash)          |
| Hardware           | Cable USB de datos, fuente que entregue ≥500 mA durante la calibración RF |

---

## 9. Limitaciones conocidas

| Limitación                                                | Detalle                                                              |
|-----------------------------------------------------------|----------------------------------------------------------------------|
| Longitud máxima de contexto                               | 160 tokens (prompt + generación) en el ESP32; el modelo base se entrena a 192 |
| Coherencia en posiciones avanzadas                        | Se degrada en las últimas posiciones (limitación intrínseca de ~7M params) |
| Dominio del modelo                                        | Español genérico; no está especializado en ningún dominio particular |
| Determinismo                                              | Las respuestas son deterministas para la misma semilla; se re-siembra `srand` en cada petición |
| Sin fine-tuning                                           | Los intentos de especialización (poesía) fallaron; se conserva el modelo base |