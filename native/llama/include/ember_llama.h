/*
 * Ember native inference — llama.cpp backend.
 *
 * This is the *only* production text-generation path in Ember: tokenization,
 * prompt evaluation, sampling and decoding all run inside llama.cpp on the
 * device. No network, no server process, no Python per token.
 *
 * The C ABI keeps ownership clear across language boundaries:
 *   - handles are opaque pointers created/destroyed here,
 *   - all strings cross the boundary as UTF-8 with explicit lengths,
 *   - generation results are returned as a compact JSON document,
 *   - errors are reported as a code + message, never as an exception.
 *
 * Pinned llama.cpp revision: see third_party/llama.cpp.version.
 */
#ifndef EMBER_LLAMA_H
#define EMBER_LLAMA_H

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

typedef struct ember_llama_s ember_llama_t;

/* Error codes returned by every entry point (0 == success). */
enum ember_llama_error {
    EMBER_LLAMA_OK                  =  0,
    EMBER_LLAMA_ERR_ARGS            = -1,  /* null handle / null pointer argument       */
    EMBER_LLAMA_ERR_NOT_INITIALIZED = -2,  /* llama_backend_init failed                 */
    EMBER_LLAMA_ERR_NOT_LOADED      = -3,  /* no model is loaded                        */
    EMBER_LLAMA_ERR_MODEL_NOT_FOUND = -4,  /* file missing or unreadable                */
    EMBER_LLAMA_ERR_MODEL_INVALID   = -5,  /* not a GGUF file / corrupt / truncated      */
    EMBER_LLAMA_ERR_MODEL_INCOMPAT  = -6,  /* GGUF but an architecture this build rejects */
    EMBER_LLAMA_ERR_CONTEXT         = -7,  /* could not allocate the generation context  */
    EMBER_LLAMA_ERR_OUT_OF_MEMORY   = -8,  /* allocation failed (model too large)        */
    EMBER_LLAMA_ERR_DECODE          = -9,  /* llama_decode reported a failure            */
    EMBER_LLAMA_ERR_TOKENIZE        = -10, /* prompt tokenization failed                 */
    EMBER_LLAMA_ERR_CANCELLED       = -11, /* generation was cancelled by the caller     */
    EMBER_LLAMA_ERR_BUFFER          = -12, /* caller buffer too small for the result     */
    EMBER_LLAMA_ERR_INTERNAL        = -99, /* unexpected internal failure                */
};

/* Engine/context configuration. Zero fields mean "use a conservative default". */
typedef struct ember_llama_config {
    int32_t n_ctx;          /* context length in tokens (clamped to the model's training ctx) */
    int32_t n_batch;        /* logical batch size for prompt evaluation                       */
    int32_t n_threads;      /* generation threads; 0 = auto (capped for phones)               */
    int32_t n_gpu_layers;   /* 0 = CPU only (the default; GPU offload is opt-in)              */
    int32_t use_mmap;       /* 1 = mmap weights (lower RSS), 0 = read into RAM                */
    int32_t check_tensors;  /* 1 = validate tensor data while loading                         */
    int32_t n_ubatch;       /* physical microbatch; 0 = min(batch, 64) */
} ember_llama_config_t;

/* Sampling configuration for one generation request. */
typedef struct ember_llama_gen_params {
    int32_t max_tokens;      /* hard cap on generated tokens                                  */
    float   temperature;     /* <= 0 selects greedy decoding                                  */
    int32_t top_k;           /* 0 disables top-k                                              */
    float   top_p;           /* 1.0 disables nucleus sampling                                 */
    float   repeat_penalty;  /* 1.0 disables the repetition penalty                           */
    int32_t repeat_last_n;   /* window for the repetition penalty                             */
    uint32_t seed;           /* 0xFFFFFFFF = nondeterministic                                 */
    int32_t stop_on_eos;     /* stop when the model emits an end-of-generation token          */
    int32_t prepend_bos;     /* prepend <bos> to match Ember's training streams               */
} ember_llama_gen_params_t;

/* Called after each generated token with the newly decodable UTF-8 text. */
typedef void (*ember_llama_token_cb)(
    void* user_data, const char* utf8_piece, int32_t piece_len, int32_t token_id, int32_t n_generated);

/* Called while the model file is being loaded, progress in [0, 1]. */
typedef void (*ember_llama_progress_cb)(void* user_data, float progress);

EMBER_API ember_llama_t* ember_llama_create(void);
EMBER_API void           ember_llama_destroy(ember_llama_t* engine);

/* Load (or replace) the model. Returns EMBER_LLAMA_OK on success. */
EMBER_API int ember_llama_load(
    ember_llama_t* engine,
    const char* model_path,
    const ember_llama_config_t* config,
    ember_llama_progress_cb progress_cb,
    void* progress_user_data);

EMBER_API int  ember_llama_is_loaded(const ember_llama_t* engine);
EMBER_API void ember_llama_unload(ember_llama_t* engine);

EMBER_API int         ember_llama_error_code(const ember_llama_t* engine);
EMBER_API const char* ember_llama_last_error(const ember_llama_t* engine);
EMBER_API const char* ember_llama_error_name(int code);

/* Model facts as JSON: arch, name, vocab, context, size, params, quantization… */
EMBER_API int ember_llama_model_info(const ember_llama_t* engine, char* json_out, size_t max_bytes);

/* Tokenizer access (used by tests and by the Python trainer for parity checks). */
EMBER_API int ember_llama_tokenize(
    ember_llama_t* engine, const char* text, size_t text_len, int32_t add_special,
    int32_t* out_ids, size_t max_ids);

EMBER_API int ember_llama_detokenize(
    ember_llama_t* engine, const int32_t* ids, size_t n_ids, int32_t remove_special,
    char* out_text, size_t max_bytes);

/* Verification API: evaluates a sequence and returns its last-position logits. */
EMBER_API int ember_llama_logits(ember_llama_t*, const int32_t*, size_t, float*, size_t);

/* Generate. Returns EMBER_LLAMA_OK (or EMBER_LLAMA_ERR_CANCELLED, which still
 * fills json_out with the partial result) and writes a JSON summary:
 * {"ok":bool,"text":str,"tokenIds":[int],"promptTokens":int,"newTokens":int,
 *  "stoppedBy":"eos|length|cancelled|context|error","warning":str|null,
 *  "ms":double,"promptMs":double,"genMs":double,"tokensPerSec":double,
 *  "seed":int,"nCtx":int,"error":str|null}
 */
EMBER_API int ember_llama_generate(
    ember_llama_t* engine,
    const char* prompt_utf8,
    size_t prompt_len,
    const ember_llama_gen_params_t* params,
    ember_llama_token_cb token_cb,
    void* token_user_data,
    char* json_out,
    size_t max_json);

/* Ask an in-flight generation to stop. Safe from any thread, never blocks. */
EMBER_API void ember_llama_cancel(ember_llama_t* engine);

/* Recent llama.cpp log lines (diagnostics kept out of the user-facing UI). */
EMBER_API int ember_llama_logs(char* out_json, size_t max_bytes);

/* llama.cpp build/system description, for the app's diagnostics screen. */
EMBER_API const char* ember_llama_backend_info(void);
/* JSON evidence generated from the compiled upstream registry + model factory. */
EMBER_API const char* ember_llama_capabilities(void);

#ifdef __cplusplus
}
#endif

#endif /* EMBER_LLAMA_H */
