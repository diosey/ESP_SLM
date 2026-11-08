#ifndef TOKENIZER_H
#define TOKENIZER_H
#include "vocab_data.h"

#define TOK_PAD 0
#define TOK_SOS 1
#define TOK_EOS 2
#define TOK_UNK 3

int tokenizer_encode(const char* text, int* out_ids, int max_ids);
int tokenizer_decode(const int* ids, int n, char* out_text, int max_len);

#endif