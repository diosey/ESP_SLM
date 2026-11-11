#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "bitnet_inference.h"
#include "tokenizer.h"

// fp16_to_fp32 es static en bitnet_inference.c; reimplementación local para el dump.
static float fp16_to_fp32_local(uint16_t h) {
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
    float ret; memcpy(&ret, &f, 4); return ret;
}

static void print_top5(const float* logits) {
    int idx[5] = {-1,-1,-1,-1,-1};
    float val[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
    for (int i = 0; i < VOCAB_SIZE; i++) {
        for (int m = 0; m < 5; m++) {
            if (logits[i] > val[m]) {
                for (int j = 4; j > m; j--) { val[j]=val[j-1]; idx[j]=idx[j-1]; }
                val[m] = logits[i]; idx[m] = i;
                break;
            }
        }
    }
    for (int m = 0; m < 5; m++) {
        printf("#%d=%d(%+.4f)%s", m+1, idx[m], val[m], m < 4 ? " " : "");
    }
}

int main(void) {
    srand(42);
    printf("BitNet C - diagnostico 192 tokens\n");
    printf("V=%d E=%d L=%d H=%d F=%d S=%d PLE=%d\n\n",
           VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, FF_DIM, MAX_SEQ_LEN, PLE_HIDDEN_DIM);

    Weights w;
    void* buf = NULL;
    if (weights_load_pc("bitnet_weights.bin", &w, &buf) != 0) {
        fprintf(stderr, "No se pudo cargar el .bin\n");
        return 1;
    }

    // ===== pos_embed sanity (dims 0..3 en pos 0,1,50,100,191) =====
    printf("=== pos_embed (FP16 -> FP32), dims 0..3 ===\n");
    int probes[] = {0, 1, 50, 100, 191};
    for (int i = 0; i < 5; i++) {
        int p = probes[i];
        printf("pos %3d: bin = [", p);
        for (int d = 0; d < 4; d++) {
            float v = fp16_to_fp32_local(w.pos_embed[p * EMBED_DIM + d]);
            printf("%+.5f%s", v, d < 3 ? " " : "");
        }
        printf("]\n");
    }
    printf("\n");

    // ===== Logits top-5 en pos 0..4 (debe coincidir con diagnostico_192.py sección 5) =====
    printf("=== Logits (top-5) con TOKENS = {4, 10, 20, 30, 40} ===\n");
    bitnet_init();
    static const int TOKENS[] = {4, 10, 20, 30, 40};
    static float logits[VOCAB_SIZE];
    for (int p = 0; p < 5; p++) {
        bitnet_forward(&w, TOKENS[p], p, logits);
        printf("pos %d: ", p);
        print_top5(logits);
        printf("\n");
    }

    // ===== Generación =====
    printf("\n=== Generacion ===\n");
    const char* prompts[] = {"Habia una vez", "El nino", "Un dia", "La nina estaba"};
    for (int pi = 0; pi < 4; pi++) {
        const char* prompt = prompts[pi];
        int ids[64];
        int n_ids = tokenizer_encode(prompt, ids, 64);

        int out_ids[MAX_SEQ_LEN];
        int max_new = MAX_SEQ_LEN - n_ids - 1;
        int total = bitnet_generate(&w, ids, n_ids, max_new, 0.8f, out_ids);

        char out_text[2048];
        tokenizer_decode(out_ids, total, out_text, sizeof(out_text));

        printf("\n----------------------------------------\n");
        printf("Prompt: \"%s\" (%d tokens)\n", prompt, n_ids);
        printf("Salida: %s\n", out_text);
    }

    weights_free_pc(buf);
    return 0;
}