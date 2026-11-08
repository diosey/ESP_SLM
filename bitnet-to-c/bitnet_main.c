#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "bitnet_inference.h"
#include "tokenizer.h"

int main(void) {
    srand(42);

    printf("BitNet C - logits + generacion\n");
    printf("V=%d E=%d L=%d H=%d F=%d S=%d PLE=%d\n\n",
           VOCAB_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, FF_DIM, MAX_SEQ_LEN, PLE_HIDDEN_DIM);

    Weights w;
    void* buf = NULL;
    if (weights_load_pc("bitnet_weights.bin", &w, &buf) != 0) {
        fprintf(stderr, "No se pudo cargar el .bin\n");
        return 1;
    }

    // === 1) Verificacion de logits ===
    bitnet_init();
    static const int TOKENS[] = {4, 10, 20, 30, 40};
    for (int p = 0; p < 5; p++) {
        static float logits[VOCAB_SIZE];
        bitnet_forward(&w, TOKENS[p], p, logits);
        int idx[3] = {-1,-1,-1};
        float val[3] = {-1e30f,-1e30f,-1e30f};
        for (int i = 0; i < VOCAB_SIZE; i++) {
            for (int k = 0; k < 3; k++) {
                if (logits[i] > val[k]) {
                    for (int m = 2; m > k; m--) { val[m]=val[m-1]; idx[m]=idx[m-1]; }
                    val[k]=logits[i]; idx[k]=i;
                    break;
                }
            }
        }
        printf("pos %d: #1=%d(%.3f) #2=%d(%.3f) #3=%d(%.3f)\n",
               p, idx[0], val[0], idx[1], val[1], idx[2], val[2]);
    }

    // === 2) Generacion ===
    printf("\n=== Generacion ===\n");
    const char* prompts[] = {"Habia una vez", "El nino", "Un dia", "La nina estaba"};
    for (int pi = 0; pi < 4; pi++) {
        const char* prompt = prompts[pi];
        int ids[64];
        int n_ids = tokenizer_encode(prompt, ids, 64);

        int out_ids[64];
        int total = bitnet_generate(&w, ids, n_ids, 100, 0.8f, out_ids);

        char out_text[2048];
        tokenizer_decode(out_ids, total, out_text, sizeof(out_text));

        printf("\n----------------------------------------\n");
        printf("Prompt: \"%s\" (%d tokens)\n", prompt, n_ids);
        printf("Salida: %s\n", out_text);
    }

    weights_free_pc(buf);
    return 0;
}