// Auto-generated. Do not edit.
#ifndef BITNET_LAYERS_H
#define BITNET_LAYERS_H

#include <stdint.h>
#include "bitnet_weights.h"

typedef struct {
    const float* norm1_gamma; const float* norm1_beta;
    const int8_t* q_weight; float q_scale; const float* q_bias;
    const int8_t* k_weight; float k_scale; const float* k_bias;
    const int8_t* v_weight; float v_scale; const float* v_bias;
    const int8_t* out_weight; float out_scale; const float* out_bias;
    const float* norm2_gamma; const float* norm2_beta;
    const int8_t* fc1_weight; float fc1_scale; const float* fc1_bias;
    const int8_t* fc2_weight; float fc2_scale; const float* fc2_bias;
} LayerWeights;

extern const LayerWeights LAYERS[5];

#endif
