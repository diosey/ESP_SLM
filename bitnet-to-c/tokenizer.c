#include "tokenizer.h"
#include <string.h>

int tokenizer_encode(const char* text, int* out_ids, int max_ids) {
    int text_len = (int)strlen(text);
    int i = 0, out_n = 0;

    while (i < text_len && out_n < max_ids) {
        int best_id = -1, best_len = 0;
        // Buscar el token más largo que coincida (greedy)
        for (int tid = 4; tid < VOCAB_N; tid++) {
            int tlen = VOCAB_LENGTHS[tid];
            if (tlen > best_len && i + tlen <= text_len) {
                if (memcmp(text + i, VOCAB_STRINGS[tid], tlen) == 0) {
                    best_id = tid;
                    best_len = tlen;
                }
            }
        }
        if (best_id >= 0) {
            out_ids[out_n++] = best_id;
            i += best_len;
        } else {
            out_ids[out_n++] = TOK_UNK;
            i++;
        }
    }
    return out_n;
}

int tokenizer_decode(const int* ids, int n, char* out_text, int max_len) {
    int out = 0;
    for (int i = 0; i < n && out < max_len - 1; i++) {
        int id = ids[i];
        if (id < 4 || id >= VOCAB_N) continue;
        int tlen = VOCAB_LENGTHS[id];
        if (out + tlen >= max_len) break;
        memcpy(out_text + out, VOCAB_STRINGS[id], tlen);
        out += tlen;
    }
    out_text[out] = '\0';
    return out;
}