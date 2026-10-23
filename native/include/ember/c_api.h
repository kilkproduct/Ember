#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define EMBER_API __declspec(dllexport)
#else
#define EMBER_API __attribute__((visibility("default")))
#endif

typedef struct ember_engine_s ember_engine_t;

EMBER_API ember_engine_t* ember_engine_create(void);
EMBER_API void ember_engine_destroy(ember_engine_t* engine);

EMBER_API int ember_engine_load(
    ember_engine_t* engine,
    const char* model_path,
    const char* tokenizer_path
);

EMBER_API const char* ember_engine_last_error(const ember_engine_t* engine);
EMBER_API int ember_engine_vocab_size(const ember_engine_t* engine);
EMBER_API int ember_engine_block_size(const ember_engine_t* engine);
EMBER_API int ember_engine_is_ready(const ember_engine_t* engine);
EMBER_API void ember_engine_cancel(ember_engine_t* engine);

EMBER_API int ember_engine_encode(
    ember_engine_t* engine,
    const char* utf8_text,
    size_t text_len,
    int32_t* out_ids,
    size_t max_ids
);

EMBER_API int ember_engine_decode(
    ember_engine_t* engine,
    const int32_t* ids,
    size_t num_ids,
    char* out_utf8,
    size_t max_bytes
);

EMBER_API int ember_engine_forward_logits(
    ember_engine_t* engine,
    const int32_t* ids,
    size_t seq_len,
    float* out_logits,
    size_t out_floats
);

EMBER_API int ember_engine_generate(
    ember_engine_t* engine,
    const char* prompt_utf8,
    size_t prompt_len,
    int max_new_tokens,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty,
    uint64_t seed,
    int stop_on_eos,
    int32_t* out_ids,
    size_t max_ids,
    char* out_utf8,
    size_t max_bytes
);

#ifdef __cplusplus
}
#endif
