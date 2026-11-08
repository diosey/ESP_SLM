#include "bitnet_inference.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// === KV Cache INT8 con scale dinámico por (capa, token) ===
static int8_t  K_cache[NUM_LAYERS][MAX_SEQ_LEN][KV_DIM];
static int8_t  V_cache[NUM_LAYERS][MAX_SEQ_LEN][KV_DIM];
static float   K_scales[NUM_LAYERS][MAX_SEQ_LEN];
static float   V_scales[NUM_LAYERS][MAX_SEQ_LEN];

// Buffers temporales
static float buf_x[EMBED_DIM];
static float buf_norm[EMBED_DIM];
static float buf_q[EMBED_DIM];        // 4 heads × HEAD_DIM
static float buf_k[KV_DIM];
static float buf_v[KV_DIM];
static float buf_heads[EMBED_DIM];
static float buf_attn[EMBED_DIM];
static float buf_hidden[FF_DIM];
static float buf_ffn[EMBED_DIM];
static float buf_ple[PLE_ROW_VALUES]; // L*PLE
static float buf_ple_proj[PLE_ROW_VALUES];
static float buf_ple_out[EMBED_DIM];
static float buf_scores[MAX_SEQ_LEN];
static float buf_ffn_act[EMBED_DIM];

void bitnet_init(void) {
    memset(K_cache, 0, sizeof(K_cache));
    memset(V_cache, 0, sizeof(V_cache));
    memset(K_scales, 0, sizeof(K_scales));
    memset(V_scales, 0, sizeof(V_scales));
}

// === FP16 → FP32 ===
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = h & 0x03FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x0400)) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float ret;
    memcpy(&ret, &f, 4);
    return ret;
}

// === Desempaquetado 2-bit ternario (códigos: -1→0, 0→1, +1→2) ===
static inline float unpack_ternary(uint8_t byte, int idx) {
    uint8_t code = (byte >> (idx * 2)) & 0x03;
    return (float)((int)code - 1);  // 0→-1, 1→0, 2→+1, 3→+2 (no usado)
}


// === Desempaquetado INT4 con sign extend ===
static inline int unpack_int4(uint8_t byte, int idx) {
    int v = (idx % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
    if (v >= 8) v -= 16;
    return v;
}






// === Matmul ternario empaquetado ===
static void ternary_matmul(const uint8_t* W, float scale, const uint16_t* bias,
                           const float* x, float* y, int out_dim, int in_dim) {
    int in_packed = (in_dim + 3) / 4;
    for (int i = 0; i < out_dim; i++) {
        const uint8_t* row = &W[i * in_packed];
        float acc = 0.0f;
        for (int j = 0; j < in_packed; j++) {
            uint8_t b = row[j];
            int base = j * 4;
            float w0 = unpack_ternary(b, 0);
            float w1 = unpack_ternary(b, 1);
            float w2 = unpack_ternary(b, 2);
            float w3 = unpack_ternary(b, 3);
            if (w0 != 0 && base + 0 < in_dim) acc += w0 * x[base + 0];
            if (w1 != 0 && base + 1 < in_dim) acc += w1 * x[base + 1];
            if (w2 != 0 && base + 2 < in_dim) acc += w2 * x[base + 2];
            if (w3 != 0 && base + 3 < in_dim) acc += w3 * x[base + 3];
        }
        float b = bias ? fp16_to_fp32(bias[i]) : 0.0f;
        y[i] = acc * scale + b;
    }
}

// === LayerNorm ===
static void layer_norm(const float* x, const uint16_t* g16, const uint16_t* b16,
                       float* y, int d) {
    float m = 0;
    for (int i = 0; i < d; i++) m += x[i];
    m /= d;
    float v = 0;
    for (int i = 0; i < d; i++) { float t = x[i] - m; v += t * t; }
    v /= d;
    float inv = 1.0f / sqrtf(v + 1e-5f);
    for (int i = 0; i < d; i++) {
        float g = fp16_to_fp32(g16[i]);
        float b = fp16_to_fp32(b16[i]);
        y[i] = g * (x[i] - m) * inv + b;
    }
}

// === Softmax in-place ===
static void softmax_ip(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

// === GELU exacto (PyTorch F.gelu) ===
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067812f));
}

// === Guardar K/V en cache INT8 con scale dinámico ===
static void kv_store(int l, int pos, const float* k, const float* v) {
    float mk = 1e-8f, mv = 1e-8f;
    for (int i = 0; i < KV_DIM; i++) {
        float ak = fabsf(k[i]), av = fabsf(v[i]);
        if (ak > mk) mk = ak;
        if (av > mv) mv = av;
    }
    float sk = mk / 127.0f, sv = mv / 127.0f;
    K_scales[l][pos] = sk;
    V_scales[l][pos] = sv;
    for (int i = 0; i < KV_DIM; i++) {
        float qk = roundf(k[i] / sk);
        float qv = roundf(v[i] / sv);
        if (qk >  127) qk =  127; if (qk < -127) qk = -127;
        if (qv >  127) qv =  127; if (qv < -127) qv = -127;
        K_cache[l][pos][i] = (int8_t)qk;
        V_cache[l][pos][i] = (int8_t)qv;
    }
}

// === Atención MQA con KV cache INT8 ===
static void attention(int l, int pos, const Weights* w, const float* xn, float* out) {
    // Proyecciones Q, K, V
    ternary_matmul(w->layers[l].q_weight, w->layers[l].q_scale, w->layers[l].q_bias,
                   xn, buf_q, EMBED_DIM, EMBED_DIM);
    ternary_matmul(w->layers[l].k_weight, w->layers[l].k_scale, w->layers[l].k_bias,
                   xn, buf_k, KV_DIM, EMBED_DIM);
    ternary_matmul(w->layers[l].v_weight, w->layers[l].v_scale, w->layers[l].v_bias,
                   xn, buf_v, KV_DIM, EMBED_DIM);

    kv_store(l, pos, buf_k, buf_v);

    const float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int h = 0; h < NUM_HEADS; h++) {
        int off = h * HEAD_DIM;
        // Scores contra tokens 0..pos
        for (int j = 0; j <= pos; j++) {
            float s = 0.0f;
            float sk_j = K_scales[l][j];
            const int8_t* krow = K_cache[l][j];
            for (int d = 0; d < HEAD_DIM; d++) {
                float kd = (float)krow[d] * sk_j;
                s += buf_q[off + d] * kd;
            }
            buf_scores[j] = s * attn_scale;
        }
        softmax_ip(buf_scores, pos + 1);
        // Suma ponderada de V
        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int j = 0; j <= pos; j++) {
                float sv_j = V_scales[l][j];
                acc += buf_scores[j] * ((float)V_cache[l][j][d] * sv_j);
            }
            buf_heads[off + d] = acc;
        }
    }
    // Output projection
    ternary_matmul(w->layers[l].o_weight, w->layers[l].o_scale, w->layers[l].o_bias,
                   buf_heads, out, EMBED_DIM, EMBED_DIM);
}

// === FFN ===
static void ffn(int l, const Weights* w, const float* xn, float* out) {
    ternary_matmul(w->layers[l].fc1_weight, w->layers[l].fc1_scale, w->layers[l].fc1_bias,
                   xn, buf_hidden, FF_DIM, EMBED_DIM);
    for (int i = 0; i < FF_DIM; i++) buf_hidden[i] = gelu(buf_hidden[i]);
    ternary_matmul(w->layers[l].fc2_weight, w->layers[l].fc2_scale, w->layers[l].fc2_bias,
                   buf_hidden, out, EMBED_DIM, FF_DIM);
}

// === Lookup PLE desde tabla INT4 ===
static void ple_lookup(const Weights* w, int token_id) {
    const uint8_t* row = w->ple_table + (size_t)token_id * PLE_ROW_BYTES;
    float sc = w->ple_scales[token_id];
    // Cada byte: 2 valores INT4 (low nibble, high nibble)
    for (int i = 0; i < PLE_ROW_VALUES; i++) {
        uint8_t b = row[i / 2];
        int v = (i % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
        if (v >= 8) v -= 16;  // sign extend INT4
        buf_ple[i] = (float)v * sc;
    }
}

void bitnet_forward(const Weights* w, int token_id, int pos, float* logits_out) {
    // 1) Embedding base: token (INT4) + pos (FP16)
    int row_bytes = TOKEN_EMBED_ROW_BYTES;
    const uint8_t* te_row = w->token_embed + (size_t)token_id * row_bytes;
    float te_scale = w->token_embed_scales[token_id];
    for (int d = 0; d < EMBED_DIM; d++) {
        int v = unpack_int4(te_row[d / 2], d);
        float te = (float)v * te_scale;
        float pe = fp16_to_fp32(w->pos_embed[pos * EMBED_DIM + d]);
        buf_x[d] = te + pe;
    }
	
    // 2) PLE: tabla + proyección
    ple_lookup(w, token_id);
    ternary_matmul(w->ple_proj_weight, w->ple_proj_scale, NULL,
                   buf_x, buf_ple_proj, PLE_ROW_VALUES, EMBED_DIM);
    const float ple_scale = sqrtf((float)PLE_HIDDEN_DIM);
    for (int i = 0; i < PLE_ROW_VALUES; i++) {
        buf_ple[i] = (buf_ple[i] + buf_ple_proj[i]) * ple_scale;
    }

    // 3) Bloques
    for (int l = 0; l < NUM_LAYERS; l++) {
        // Atención + residual
        layer_norm(buf_x, w->layers[l].norm1_gamma, w->layers[l].norm1_beta,
                   buf_norm, EMBED_DIM);
        attention(l, pos, w, buf_norm, buf_attn);
        for (int d = 0; d < EMBED_DIM; d++) buf_x[d] += buf_attn[d];

        // FFN + residual
        layer_norm(buf_x, w->layers[l].norm2_gamma, w->layers[l].norm2_beta,
                   buf_norm, EMBED_DIM);
        ffn(l, w, buf_norm, buf_ffn);
        for (int d = 0; d < EMBED_DIM; d++) buf_x[d] += buf_ffn[d];

        // PLE slice + residual (con norm)
        const float* ple_slice = &buf_ple[l * PLE_HIDDEN_DIM];
        ternary_matmul(w->layers[l].ple_res_weight, w->layers[l].ple_res_scale, NULL,
                       ple_slice, buf_ple_out, EMBED_DIM, PLE_HIDDEN_DIM);
        // ple_norm
        float m = 0;
        for (int d = 0; d < EMBED_DIM; d++) m += buf_ple_out[d];
        m /= EMBED_DIM;
        float v = 0;
        for (int d = 0; d < EMBED_DIM; d++) { float t = buf_ple_out[d] - m; v += t * t; }
        v /= EMBED_DIM;
        float inv = 1.0f / sqrtf(v + 1e-5f);
        for (int d = 0; d < EMBED_DIM; d++) {
            float g = fp16_to_fp32(w->layers[l].ple_norm_gamma[d]);
            float b = fp16_to_fp32(w->layers[l].ple_norm_beta[d]);
            buf_ple_out[d] = g * (buf_ple_out[d] - m) * inv + b;
            buf_x[d] += buf_ple_out[d];
        }
    }

    // 4) Final norm
    layer_norm(buf_x, w->final_norm_gamma, w->final_norm_beta, buf_norm, EMBED_DIM);

    // 5) LM head
    ternary_matmul(w->lm_head_weight, w->lm_head_scale, w->lm_head_bias,
                   buf_norm, logits_out, VOCAB_SIZE, EMBED_DIM);
}

int bitnet_generate(const Weights* w, const int* prompt, int plen,
                    int max_new, float temp, int* out) {
    bitnet_init();
    static float logits[VOCAB_SIZE];

    for (int i = 0; i < plen; i++) {
        out[i] = prompt[i];
        bitnet_forward(w, prompt[i], i, logits);
    }
    int total = plen;
    const int TOP_K = 50;

    for (int t = 0; t < max_new; t++) {
        int nid;
        if (temp <= 0) {
            nid = 0;
            for (int i = 1; i < VOCAB_SIZE; i++)
                if (logits[i] > logits[nid]) nid = i;
        } else {
            // Aplicar temperatura
            for (int i = 0; i < VOCAB_SIZE; i++) logits[i] /= temp;
            // Encontrar umbral top-k
            float kth = -1e30f;
            for (int k = 0; k < TOP_K; k++) {
                float mx = -1e30f;
                for (int i = 0; i < VOCAB_SIZE; i++)
                    if (logits[i] > mx && logits[i] < kth + 1e-6f) mx = logits[i];
                // Alternativa simple: usar kth como el k-ésimo mayor
                kth = mx;
                if (kth == -1e30f) break;
            }
            // Más simple: ordenar aproximado no vale. Usar un enfoque directo:
            float top_vals[TOP_K];
            for (int i = 0; i < TOP_K; i++) top_vals[i] = -1e30f;
            for (int i = 0; i < VOCAB_SIZE; i++) {
                for (int k = 0; k < TOP_K; k++) {
                    if (logits[i] > top_vals[k]) {
                        for (int m = TOP_K - 1; m > k; m--) top_vals[m] = top_vals[m-1];
                        top_vals[k] = logits[i];
                        break;
                    }
                }
            }
            float threshold = top_vals[TOP_K - 1];
            // Enmascarar tokens fuera del top-k
            for (int i = 0; i < VOCAB_SIZE; i++)
                if (logits[i] < threshold) logits[i] = -1e30f;
            softmax_ip(logits, VOCAB_SIZE);
            float r = (float)rand() / RAND_MAX, c = 0;
            nid = VOCAB_SIZE - 1;
            for (int i = 0; i < VOCAB_SIZE; i++) {
                c += logits[i];
                if (r <= c) { nid = i; break; }
            }
        }
        out[total++] = nid;
        if (nid == 2 || total >= MAX_SEQ_LEN) break;
        bitnet_forward(w, nid, total - 1, logits);
    }
    return total;
}


// === Cargar desde PC ===
static const uint8_t* P(const void* base, size_t off) { return (const uint8_t*)base + off; }

#define LAYER_SIZE 79708  // bytes por bloque (ver export_weights.py)

int weights_load_pc(const char* path, Weights* w, void** buf_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz != WEIGHTS_BIN_SIZE) {
        fprintf(stderr, "Tamaño %ld != %d\n", sz, WEIGHTS_BIN_SIZE);
        fclose(f); return -1;
    }
    uint8_t* buf = malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fclose(f); free(buf); return -1; }
    fclose(f);


    w->token_embed = (const uint8_t*)P(buf, OFF_TOKEN_EMBED);
    w->token_embed_scales = (const float*)P(buf, OFF_TOKEN_EMBED_SCALES);
	
	
	
	
	
    w->pos_embed = (const uint16_t*)P(buf, OFF_POS_EMBED);
    w->ple_table = (const uint8_t*)P(buf, OFF_PLE_TABLE);
    w->ple_scales = (const float*)P(buf, OFF_PLE_SCALES);
    w->ple_proj_weight = (const uint8_t*)P(buf, OFF_PLE_PROJ_WEIGHT);
    w->ple_proj_scale = *(const float*)P(buf, OFF_PLE_PROJ_SCALE);

    for (int i = 0; i < NUM_LAYERS; i++) {
        size_t base = (size_t)i * LAYER_SIZE;
        w->layers[i].norm1_gamma = (const uint16_t*)P(buf, OFF_B0_NORM1_GAMMA + base);
        w->layers[i].norm1_beta  = (const uint16_t*)P(buf, OFF_B0_NORM1_BETA  + base);
        w->layers[i].q_weight = (const uint8_t*)P(buf, OFF_B0_Q_WEIGHT + base);
        w->layers[i].q_scale  = *(const float*)P(buf, OFF_B0_Q_SCALE + base);
        w->layers[i].q_bias   = (const uint16_t*)P(buf, OFF_B0_Q_BIAS + base);
        w->layers[i].k_weight = (const uint8_t*)P(buf, OFF_B0_K_WEIGHT + base);
        w->layers[i].k_scale  = *(const float*)P(buf, OFF_B0_K_SCALE + base);
        w->layers[i].k_bias   = (const uint16_t*)P(buf, OFF_B0_K_BIAS + base);
        w->layers[i].v_weight = (const uint8_t*)P(buf, OFF_B0_V_WEIGHT + base);
        w->layers[i].v_scale  = *(const float*)P(buf, OFF_B0_V_SCALE + base);
        w->layers[i].v_bias   = (const uint16_t*)P(buf, OFF_B0_V_BIAS + base);
        w->layers[i].o_weight = (const uint8_t*)P(buf, OFF_B0_O_WEIGHT + base);
        w->layers[i].o_scale  = *(const float*)P(buf, OFF_B0_O_SCALE + base);
        w->layers[i].o_bias   = (const uint16_t*)P(buf, OFF_B0_O_BIAS + base);
        w->layers[i].norm2_gamma = (const uint16_t*)P(buf, OFF_B0_NORM2_GAMMA + base);
        w->layers[i].norm2_beta  = (const uint16_t*)P(buf, OFF_B0_NORM2_BETA  + base);
        w->layers[i].fc1_weight = (const uint8_t*)P(buf, OFF_B0_FC1_WEIGHT + base);
        w->layers[i].fc1_scale  = *(const float*)P(buf, OFF_B0_FC1_SCALE + base);
        w->layers[i].fc1_bias   = (const uint16_t*)P(buf, OFF_B0_FC1_BIAS + base);
        w->layers[i].fc2_weight = (const uint8_t*)P(buf, OFF_B0_FC2_WEIGHT + base);
        w->layers[i].fc2_scale  = *(const float*)P(buf, OFF_B0_FC2_SCALE + base);
        w->layers[i].fc2_bias   = (const uint16_t*)P(buf, OFF_B0_FC2_BIAS + base);
        w->layers[i].ple_res_weight = (const uint8_t*)P(buf, OFF_B0_PLE_RES_WEIGHT + base);
        w->layers[i].ple_res_scale  = *(const float*)P(buf, OFF_B0_PLE_RES_SCALE + base);
        w->layers[i].ple_norm_gamma = (const uint16_t*)P(buf, OFF_B0_PLE_NORM_GAMMA + base);
        w->layers[i].ple_norm_beta  = (const uint16_t*)P(buf, OFF_B0_PLE_NORM_BETA + base);
    }

    w->final_norm_gamma = (const uint16_t*)P(buf, OFF_FINAL_NORM_GAMMA);
    w->final_norm_beta  = (const uint16_t*)P(buf, OFF_FINAL_NORM_BETA);
    w->lm_head_weight = (const uint8_t*)P(buf, OFF_LM_HEAD_WEIGHT);
    w->lm_head_scale  = *(const float*)P(buf, OFF_LM_HEAD_SCALE);
    w->lm_head_bias   = (const uint16_t*)P(buf, OFF_LM_HEAD_BIAS);

    *buf_out = buf;
    return 0;
}

void weights_free_pc(void* buf) { free(buf); }
