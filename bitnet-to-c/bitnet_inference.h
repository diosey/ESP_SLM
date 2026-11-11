#ifndef BITNET_INFERENCE_H
#define BITNET_INFERENCE_H
#include <stdint.h>
#include <stddef.h>
#include "bitnet_weights.h"

typedef struct {
    const uint8_t* token_embed;        // INT4 empaquetado (2 valores/byte)
    const float*   token_embed_scales; // V floats (scale por fila)
    const uint16_t* pos_embed;         // FP16

    const uint8_t* ple_table;          // INT4 empaquetado: V filas × PLE_ROW_BYTES
    const float*   ple_scales;         // V floats
    const uint8_t* ple_proj_weight;    // Ternario 2-bit
    float          ple_proj_scale;

    struct {
        const uint16_t* norm1_gamma;
        const uint16_t* norm1_beta;
        const uint8_t* q_weight;  float q_scale;  const uint16_t* q_bias;
        const uint8_t* k_weight;  float k_scale;  const uint16_t* k_bias;
        const uint8_t* v_weight;  float v_scale;  const uint16_t* v_bias;
        const uint8_t* o_weight;  float o_scale;  const uint16_t* o_bias;
        const uint16_t* norm2_gamma;
        const uint16_t* norm2_beta;
        const uint8_t* fc1_weight; float fc1_scale; const uint16_t* fc1_bias;
        const uint8_t* fc2_weight; float fc2_scale; const uint16_t* fc2_bias;
        const uint8_t* ple_res_weight; float ple_res_scale;
        const uint16_t* ple_norm_gamma;
        const uint16_t* ple_norm_beta;
    } layers[NUM_LAYERS];

    const uint16_t* final_norm_gamma;
    const uint16_t* final_norm_beta;
    const uint8_t* lm_head_weight;  float lm_head_scale;
    const uint16_t* lm_head_bias;
} Weights;

// Callback para streaming de tokens (usado por el WebSocket del firmware)
typedef void (*bitnet_token_cb)(int token_id, void* user_data);

void bitnet_init(void);
void bitnet_forward(const Weights* w, int token_id, int pos, float* logits_out);

int bitnet_generate(const Weights* w, const int* prompt, int plen,
                    int max_new, float temp, int* out);

int bitnet_generate_stream(const Weights* w, const int* prompt, int plen,
                           int max_new, float temp, int* out,
                           bitnet_token_cb cb, void* user_data);

// Solo para PC (no se compila en ESP32)
#ifndef ESP_PLATFORM
int  weights_load_pc(const char* path, Weights* w, void** buf_out);
void weights_free_pc(void* buf);
#endif

#endif