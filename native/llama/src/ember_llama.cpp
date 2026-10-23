/*
 * Ember native inference engine — llama.cpp implementation.
 *
 * Replaces the previous hand-written inference engine (custom .emb loader,
 * hand-rolled attention/KV cache, custom sampler). All generation now runs
 * inside llama.cpp: its GGUF loader, its tokenizer, its graph evaluation and
 * its sampler chain.
 *
 * Pinned llama.cpp revision: see third_party/llama.cpp.version.
 */
#include "ember_llama.h"

#include "llama.h"
#include "gguf.h"
#include <fstream>
#include <limits>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <mutex>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// Admission is conservative arithmetic, not an allocation guarantee. On Linux /
// Android the kernel reports reclaimable system memory; unknown hosts defer to
// llama.cpp's allocation errors. No device RAM capacity is hard-coded.
uint64_t available_memory_bytes() {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f,line)) {
        if (line.rfind("MemAvailable:",0)==0) {
            std::istringstream v(line.substr(13)); uint64_t kib=0; v>>kib;
            return kib*1024;
        }
    }
    return 0;
}

// Read only metadata and tensor descriptors, never the tensor payload.
uint64_t admission_bytes(const char* path, int requested_ctx, int batch) {
    gguf_init_params params{};params.no_alloc=true;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> g(gguf_init_from_file(path,params),gguf_free);
    if (!g) return 0; // authoritative loader will produce the format error
    auto str=[&](const char* key)->std::string {
        auto i=gguf_find_key(g.get(),key);
        return i>=0 && gguf_get_kv_type(g.get(),i)==GGUF_TYPE_STRING ? gguf_get_val_str(g.get(),i) : "";
    };
    auto integer=[&](const std::string& key,uint64_t fallback)->uint64_t {
        auto i=gguf_find_key(g.get(),key.c_str());if(i<0)return fallback;
        if(gguf_get_kv_type(g.get(),i)==GGUF_TYPE_UINT32)return gguf_get_val_u32(g.get(),i);
        if(gguf_get_kv_type(g.get(),i)==GGUF_TYPE_INT32)return std::max(0,gguf_get_val_i32(g.get(),i));
        return fallback;
    };
    const auto arch=str("general.architecture");
    if(arch!="llama" && arch!="qwen2" && arch!="qwen3")return 0;
    const auto c=integer(arch+".embedding_length",0),heads=integer(arch+".attention.head_count",0);
    const auto layers=integer(arch+".block_count",0),kv=integer(arch+".attention.head_count_kv",heads);
    if(!c || !heads || !layers || c>65536 || heads>1024 || layers>1024 || kv>1024)return 0;
    const auto kd=integer(arch+".attention.key_length",c/heads),vd=integer(arch+".attention.value_length",kd);
    if(kd>65536 || vd>65536)return 0;
    const auto ctx=std::min(uint64_t(requested_ctx),integer(arch+".context_length",requested_ctx));
    uint64_t weights=0;
    for(int64_t i=0;i<gguf_get_n_tensors(g.get());++i)weights+=gguf_get_tensor_size(g.get(),i);
    // F32 K/V, intermediate tensors/logits, graph, tokenizer and runtime reserve.
    return weights+ctx*layers*kv*(kd+vd)*4+uint64_t(batch)*c*layers*16+96ULL*1024*1024;
}

// ---------------------------------------------------------------------------
// Process-wide llama.cpp state
// ---------------------------------------------------------------------------

std::once_flag g_backend_once;
bool g_backend_ok = false;

std::mutex g_log_mu;
std::deque<std::string> g_logs;
constexpr size_t kMaxLogs = 60;
// Error details from the upstream loader on the calling thread. Keep the
// actual tensor/tokenizer/backend failure, not just a generic wrapper error.
thread_local std::string g_load_errors;

void llama_log_sink(enum ggml_log_level level, const char* text, void* /*user*/) {
    if (!text) return;
    std::string line(text);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (line.empty()) return;
    if (level == GGML_LOG_LEVEL_ERROR) {
        g_load_errors += line + "\n";
        if (g_load_errors.size() > 8192) g_load_errors.erase(0, g_load_errors.size()-8192);
    }
    std::lock_guard<std::mutex> lock(g_log_mu);
    g_logs.emplace_back(line);
    if (g_logs.size() > kMaxLogs) {
        g_logs.pop_front();
    }
    (void) level;
}

void init_backend_once() {
    llama_log_set(llama_log_sink, nullptr);
    llama_backend_init();
    g_backend_ok = true;
}

bool ensure_backend() {
    std::call_once(g_backend_once, init_backend_once);
    return g_backend_ok;
}

int conservative_thread_count() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) return 2;
    int t = static_cast<int>(hw) - 1;
    return std::max(1, std::min(4, t));
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    static const char* kHex = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(kHex[(c >> 4) & 0xF]);
                    out.push_back(kHex[c & 0xF]);
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

std::string model_meta(const llama_model* model, const char* key) {
    char buf[512];
    int32_t n = llama_model_meta_val_str(model, key, buf, sizeof(buf));
    if (n < 0) return {};
    if (n < static_cast<int32_t>(sizeof(buf))) return std::string(buf,n);
    std::vector<char> full(static_cast<size_t>(n)+1);
    llama_model_meta_val_str(model,key,full.data(),full.size());
    return std::string(full.data(),n);
}

/* Length of the longest prefix of `s` that ends on a UTF-8 character boundary. */
size_t complete_utf8_prefix(const std::string& s) {
    size_t n = s.size();
    if (n == 0) return 0;
    // Walk back over a possible incomplete trailing sequence (max 3 bytes).
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        unsigned char c = static_cast<unsigned char>(s[n - back]);
        if ((c & 0xC0) != 0x80) {
            // c is a lead byte (or ASCII): expected length of this character
            int want = 1;
            if ((c & 0x80) == 0x00) want = 1;
            else if ((c & 0xE0) == 0xC0) want = 2;
            else if ((c & 0xF0) == 0xE0) want = 3;
            else if ((c & 0xF8) == 0xF0) want = 4;
            else return n;  // invalid byte: let the caller emit it verbatim
            if (static_cast<int>(back) < want) {
                return n - back;  // incomplete character: hold it back
            }
            return n;
        }
    }
    return n;
}

struct ProgressTrampoline {
    ember_llama_progress_cb cb;
    void* user;
    std::atomic<bool>* cancel;
};

bool progress_thunk(float progress, void* user_data) {
    auto* t = static_cast<ProgressTrampoline*>(user_data);
    if (t->cancel && t->cancel->load(std::memory_order_relaxed)) {
        return false;  // abort the load
    }
    if (t->cb) {
        t->cb(t->user, progress);
    }
    return true;
}

/* Write a structured error document so callers always receive parseable JSON,
 * even on the early-return paths (no model loaded, bad arguments). */
void write_error_json(char* json_out, size_t max_bytes, int code, const std::string& message) {
    if (!json_out || max_bytes == 0) return;
    std::ostringstream oss;
    oss << "{\"ok\":false,\"text\":\"\",\"tokenIds\":[],\"newTokens\":0,\"promptTokens\":0"
        << ",\"stoppedBy\":\"error\",\"warning\":null"
        << ",\"error\":\"" << json_escape(message) << "\""
        << ",\"errorCode\":" << code
        << ",\"errorName\":\"" << ember_llama_error_name(code) << "\""
        << ",\"ms\":0,\"promptMs\":0,\"genMs\":0,\"tokensPerSec\":0,\"seed\":0,\"nCtx\":0}";
    const std::string payload = oss.str();
    if (payload.size() + 1 <= max_bytes) {
        std::memcpy(json_out, payload.data(), payload.size());
        json_out[payload.size()] = '\0';
        return;
    }
    const std::string tiny = "{\"ok\":false,\"text\":\"\",\"newTokens\":0,\"error\":\"result buffer too small\"}";
    const size_t n = std::min(tiny.size(), max_bytes - 1);
    std::memcpy(json_out, tiny.data(), n);
    json_out[n] = '\0';
}

}  // namespace

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

struct ember_llama_s {
    mutable std::mutex mu;
    std::atomic<bool> cancel{false};

    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;

    ember_llama_config cfg{};
    int32_t requested_ctx = 0;
    int32_t n_ctx = 0;
    int32_t n_batch = 0;
    std::vector<llama_logit_bias> suppress;  // control tokens that must never be emitted
    std::string model_path;

    // Mutable so const entry points (info/tokenize/detokenize) can still record
    // a structured error for the caller to read.
    mutable int code = EMBER_LLAMA_OK;
    mutable std::string err;

    void set_error(int c, std::string message) const {
        code = c;
        err = std::move(message);
    }
    void clear_error() const {
        code = EMBER_LLAMA_OK;
        err.clear();
    }
    bool loaded() const { return model != nullptr && ctx != nullptr && vocab != nullptr; }
};

namespace {

void release_locked(ember_llama_s* e) {
    if (e->ctx) {
        llama_free(e->ctx);
        e->ctx = nullptr;
    }
    if (e->model) {
        llama_model_free(e->model);
        e->model = nullptr;
    }
    e->vocab = nullptr;
    e->suppress.clear();
    e->model_path.clear();
    e->n_ctx = 0;
}

/* Build a sampler chain for one generation request. */
llama_sampler* build_sampler_chain(
    const ember_llama_s* e,
    const ember_llama_gen_params_t& p
) {
    llama_sampler* chain = llama_sampler_chain_init(llama_sampler_chain_default_params());

    // Control tokens (pad/bos/doc/unk, exported as GGUF CONTROL type) must never
    // be emitted as text; end-of-generation stays sampleable.
    auto suppressed = e->suppress;
    if (!p.stop_on_eos) {
        for (int id = 0; id < llama_vocab_n_tokens(e->vocab); ++id)
            if (llama_vocab_is_eog(e->vocab, id)) suppressed.push_back({id, -INFINITY});
    }
    if (!suppressed.empty()) {
        llama_sampler_chain_add(
            chain,
            llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(e->vocab),
                static_cast<int32_t>(suppressed.size()),
                suppressed.data()));
    }

    if (p.repeat_penalty != 1.0f && p.repeat_last_n > 0) {
        llama_sampler_chain_add(
            chain,
            llama_sampler_init_penalties(
                llama_vocab_n_tokens(e->vocab),
                p.repeat_last_n,
                p.repeat_penalty,
                0.0f,   // penalty_freq
                0.0f)); // penalty_present
    }

    if (p.top_k > 1) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(p.top_k));
    }
    if (p.top_p > 0.0f && p.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(p.top_p, 1));
    }

    if (p.temperature <= 0.0f || p.top_k == 1) {
        // Deterministic greedy decoding.
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(p.temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(p.seed));
    }
    return chain;
}

}  // namespace

extern "C" {

EMBER_API ember_llama_t* ember_llama_create(void) {
    try {
        if (!ensure_backend()) return nullptr;
        return new ember_llama_s();
    } catch (...) {
        return nullptr;
    }
}

EMBER_API void ember_llama_destroy(ember_llama_t* engine) {
    if (!engine) return;
    {
        std::lock_guard<std::mutex> lock(engine->mu);
        release_locked(engine);
    }
    delete engine;
}

EMBER_API int ember_llama_load(
    ember_llama_t* engine,
    const char* model_path,
    const ember_llama_config_t* config,
    ember_llama_progress_cb progress_cb,
    void* progress_user_data
) try {
    if (!engine || !model_path) return EMBER_LLAMA_ERR_ARGS;
    if (!ensure_backend()) {
        engine->set_error(EMBER_LLAMA_ERR_NOT_INITIALIZED, "llama.cpp backend failed to initialize.");
        return engine->code;
    }

    std::lock_guard<std::mutex> lock(engine->mu);
    engine->cancel.store(false, std::memory_order_relaxed);
    release_locked(engine);
    engine->clear_error();
    g_load_errors.clear();

    // Validate the file exists before handing it to llama.cpp so the message is
    // honest and local ("file missing") instead of a generic load failure.
    {
        FILE* probe = std::fopen(model_path, "rb");
        if (!probe) {
            engine->set_error(EMBER_LLAMA_ERR_MODEL_NOT_FOUND,
                              std::string("Model file not found or not readable. Import the local GGUF file again."));
            return engine->code;
        }
        unsigned char magic[4] = {0, 0, 0, 0};
        size_t got = std::fread(magic, 1, sizeof(magic), probe);
        std::fseek(probe, 0, SEEK_END);
        long size = std::ftell(probe);
        std::fclose(probe);
        if (got < 4 || size < 1024) {
            engine->set_error(EMBER_LLAMA_ERR_MODEL_INVALID,
                              std::string("Not a usable GGUF model (too small or unreadable)."));
            return engine->code;
        }
        if (!(magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F')) {
            engine->set_error(EMBER_LLAMA_ERR_MODEL_INVALID,
                              std::string("Not a GGUF file (missing GGUF magic)") +
                              ". Ember training checkpoints (.emb) and JSON brains must be exported to GGUF first.");
            return engine->code;
        }
    }

    ember_llama_config cfg{};
    if (config) cfg = *config;
    if (cfg.n_ctx <= 0) cfg.n_ctx = 512;
    if (cfg.n_batch <= 0) cfg.n_batch = 256;
    if (cfg.n_ubatch <= 0) cfg.n_ubatch = std::min(cfg.n_batch,64);
    if (cfg.n_ubatch>cfg.n_batch || cfg.n_gpu_layers != 0 || cfg.n_ctx>8192 || cfg.n_batch>2048 || cfg.n_threads>8 || cfg.n_threads<0) {
        engine->set_error(EMBER_LLAMA_ERR_ARGS,"CPU-only build: require microbatch <= batch <= 2048, context <= 8192, GPU layers = 0");
        return engine->code;
    }
    if (cfg.n_threads <= 0) cfg.n_threads = conservative_thread_count();
    if (cfg.n_gpu_layers < 0) cfg.n_gpu_layers = 0;

    engine->requested_ctx=cfg.n_ctx;
    const uint64_t required=admission_bytes(model_path,cfg.n_ctx,cfg.n_batch);
    const uint64_t available=available_memory_bytes();
    if(required && available && required>available*8/10) {
        engine->set_error(EMBER_LLAMA_ERR_OUT_OF_MEMORY,
            "Model + F32 KV/context working-set estimate " + std::to_string(required/(1024*1024)) +
            " MiB exceeds 80% of currently available system memory (" + std::to_string(available/(1024*1024)) +
            " MiB). Choose a smaller quantized model, context or batch.");
        return engine->code;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = cfg.n_gpu_layers;
    mp.check_tensors = (cfg.check_tensors != 0);
    mp.load_mode = (cfg.use_mmap == 0) ? LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_MMAP;

    ProgressTrampoline tramp{progress_cb, progress_user_data, &engine->cancel};
    // Always register the cancellation hook, even without a visual progress
    // listener (e.g. JNI). Otherwise Stop cannot interrupt a large model load.
    mp.progress_callback = progress_thunk;
    mp.progress_callback_user_data = &tramp;

    std::unique_ptr<llama_model, decltype(&llama_model_free)> model_owner(
        llama_model_load_from_file(model_path, mp), llama_model_free);
    llama_model* model = model_owner.get();
    if (!model) {
        if (engine->cancel.load(std::memory_order_relaxed)) {
            engine->set_error(EMBER_LLAMA_ERR_CANCELLED, "Model loading was cancelled.");
        } else {
            engine->set_error(EMBER_LLAMA_ERR_MODEL_INVALID,
                              std::string("llama.cpp could not load this local model") +
                              ". The file may be corrupt, truncated, quantized in an unsupported way, "
                              "or use an architecture this build does not include. Upstream: " + g_load_errors);
        }
        return engine->code;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int32_t n_ctx_train = llama_model_n_ctx_train(model);
    if (n_ctx_train > 0 && cfg.n_ctx > n_ctx_train) {
        cfg.n_ctx = n_ctx_train;  // never exceed the trained context
    }

    llama_context_params cp = llama_context_default_params();
    // F32 KV avoids silently reducing the training/export parity to fp16.
    // Small on-device vessels are cheap; larger imported models may expose KV
    // quantization in a future performance profile.
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.n_ctx = static_cast<uint32_t>(cfg.n_ctx);
    cp.n_batch = static_cast<uint32_t>(cfg.n_batch);
    cp.n_ubatch = static_cast<uint32_t>(cfg.n_ubatch);
    cp.n_threads = cfg.n_threads;
    cp.n_threads_batch = cfg.n_threads;

    std::unique_ptr<llama_context, decltype(&llama_free)> ctx_owner(
        llama_init_from_model(model, cp), llama_free);
    llama_context* ctx = ctx_owner.get();
    if (!ctx) {
        engine->set_error(EMBER_LLAMA_ERR_CONTEXT,
                          "Could not allocate a generation context for n_ctx=" + std::to_string(cfg.n_ctx) +
                          ". Try a shorter context length — the model plus KV cache does not fit in memory.");
        return engine->code;
    }

    // Tokens that must never be emitted as text (GGUF CONTROL type), except the
    // end-of-generation token which has to stay sampleable.
    std::vector<llama_logit_bias> suppress;
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    const llama_token eot = llama_vocab_eot(vocab);
    for (int32_t id = 0; id < n_vocab; ++id) {
        if (!llama_vocab_is_control(vocab, id)) continue;
        if (id == eos || id == eot) continue;
        if (llama_vocab_is_eog(vocab, id)) continue;
        suppress.push_back(llama_logit_bias{id, -INFINITY});
    }

    engine->model = model_owner.release();
    engine->ctx = ctx_owner.release();
    engine->vocab = vocab;
    engine->cfg = cfg;
    // llama_n_ctx() can be rounded up past the trained context. For a learned
    // positional embedding model (gpt2) that would index rows beyond
    // position_embd, so the generation limit is clamped to what was trained.
    const int32_t runtime_ctx = static_cast<int32_t>(llama_n_ctx(ctx));
    engine->n_ctx = (n_ctx_train > 0) ? std::min(runtime_ctx, n_ctx_train) : runtime_ctx;
    engine->n_batch = cfg.n_batch;
    engine->suppress = std::move(suppress);
    engine->model_path = model_path;
    return EMBER_LLAMA_OK;
} catch (const std::exception& exc) {
    if (engine) engine->set_error(EMBER_LLAMA_ERR_INTERNAL, exc.what());
    return EMBER_LLAMA_ERR_INTERNAL;
} catch (...) {
    return EMBER_LLAMA_ERR_INTERNAL;
}

EMBER_API int ember_llama_is_loaded(const ember_llama_t* engine) {
    if (!engine) return 0;
    std::lock_guard<std::mutex> lock(engine->mu);
    return engine->loaded() ? 1 : 0;
}

EMBER_API void ember_llama_unload(ember_llama_t* engine) {
    if (!engine) return;
    std::lock_guard<std::mutex> lock(engine->mu);
    engine->cancel.store(true, std::memory_order_relaxed);
    release_locked(engine);
    engine->clear_error();
}

EMBER_API int ember_llama_error_code(const ember_llama_t* engine) {
    if (!engine) return EMBER_LLAMA_ERR_ARGS;
    std::lock_guard<std::mutex> lock(engine->mu);
    return engine->code;
}

EMBER_API const char* ember_llama_last_error(const ember_llama_t* engine) {
    if (!engine) return "null engine handle";
    std::lock_guard<std::mutex> lock(engine->mu);
    return engine->err.c_str();
}

EMBER_API const char* ember_llama_error_name(int code) {
    switch (code) {
        case EMBER_LLAMA_OK: return "ok";
        case EMBER_LLAMA_ERR_ARGS: return "invalid-arguments";
        case EMBER_LLAMA_ERR_NOT_INITIALIZED: return "backend-not-initialized";
        case EMBER_LLAMA_ERR_NOT_LOADED: return "model-not-loaded";
        case EMBER_LLAMA_ERR_MODEL_NOT_FOUND: return "model-not-found";
        case EMBER_LLAMA_ERR_MODEL_INVALID: return "model-invalid";
        case EMBER_LLAMA_ERR_MODEL_INCOMPAT: return "model-incompatible";
        case EMBER_LLAMA_ERR_CONTEXT: return "context-allocation-failed";
        case EMBER_LLAMA_ERR_OUT_OF_MEMORY: return "out-of-memory";
        case EMBER_LLAMA_ERR_DECODE: return "decode-failed";
        case EMBER_LLAMA_ERR_TOKENIZE: return "tokenize-failed";
        case EMBER_LLAMA_ERR_CANCELLED: return "cancelled";
        case EMBER_LLAMA_ERR_BUFFER: return "buffer-too-small";
        default: return "internal-error";
    }
}

EMBER_API int ember_llama_model_info(const ember_llama_t* engine, char* json_out, size_t max_bytes) {
    if (!engine || !json_out || max_bytes == 0) return EMBER_LLAMA_ERR_ARGS;
    std::lock_guard<std::mutex> lock(engine->mu);
    if (!engine->loaded()) {
        engine->set_error(EMBER_LLAMA_ERR_NOT_LOADED, "No model is loaded.");
        return engine->code;
    }
    char desc[256] = {0};
    llama_model_desc(engine->model, desc, sizeof(desc));

    std::ostringstream oss;
    oss << "{"
        << "\"loaded\":true"
        << ",\"path\":" << "\"" << json_escape(engine->model_path) << "\""
        << ",\"desc\":" << "\"" << json_escape(desc) << "\""
        << ",\"arch\":" << "\"" << json_escape(model_meta(engine->model, "general.architecture")) << "\""
        << ",\"name\":" << "\"" << json_escape(model_meta(engine->model, "general.name")) << "\""
        << ",\"emberFormat\":" << "\"" << json_escape(model_meta(engine->model, "ember.format_version")) << "\""
        << ",\"vocabSize\":" << llama_vocab_n_tokens(engine->vocab)
        << ",\"parameters\":" << llama_model_n_params(engine->model)
        << ",\"microBatch\":" << engine->cfg.n_ubatch
        << ",\"hasChatTemplate\":" << (llama_model_chat_template(engine->model,nullptr) ? "true":"false")
        << ",\"nCtxTrain\":" << llama_model_n_ctx_train(engine->model)
        << ",\"nCtx\":" << engine->n_ctx
        << ",\"nBatch\":" << engine->n_batch
        << ",\"nThreads\":" << engine->cfg.n_threads
        << ",\"nGpuLayers\":" << engine->cfg.n_gpu_layers
        << ",\"modelBytes\":" << llama_model_size(engine->model)
        << ",\"nParams\":" << llama_model_n_params(engine->model)
        << ",\"bosTokenId\":" << llama_vocab_bos(engine->vocab)
        << ",\"eosTokenId\":" << llama_vocab_eos(engine->vocab)
        << ",\"padTokenId\":" << llama_vocab_pad(engine->vocab)
        << ",\"addBos\":" << (llama_vocab_get_add_bos(engine->vocab) ? "true" : "false")
        << ",\"addEos\":" << (llama_vocab_get_add_eos(engine->vocab) ? "true" : "false")
        << ",\"suppressedTokens\":" << engine->suppress.size()
        << "}";
    const std::string payload = oss.str();
    if (payload.size() + 1 > max_bytes) return EMBER_LLAMA_ERR_BUFFER;
    std::memcpy(json_out, payload.data(), payload.size());
    json_out[payload.size()] = '\0';
    return EMBER_LLAMA_OK;
}

EMBER_API int ember_llama_tokenize(
    ember_llama_t* engine, const char* text, size_t text_len, int32_t add_special,
    int32_t* out_ids, size_t max_ids
) {
    if (!engine || !text) return EMBER_LLAMA_ERR_ARGS;
    std::lock_guard<std::mutex> lock(engine->mu);
    if (!engine->loaded()) {
        engine->set_error(EMBER_LLAMA_ERR_NOT_LOADED, "No model is loaded.");
        return engine->code;
    }
    const int32_t needed = -llama_tokenize(
        engine->vocab, text, static_cast<int32_t>(text_len), nullptr, 0, (add_special & 1) != 0, (add_special & 2) != 0);
    if (needed <= 0) return 0;
    if (out_ids && max_ids >= static_cast<size_t>(needed)) {
        int32_t n = llama_tokenize(
            engine->vocab, text, static_cast<int32_t>(text_len), out_ids,
            static_cast<int32_t>(max_ids), (add_special & 1) != 0, (add_special & 2) != 0);
        if (n < 0) {
            engine->set_error(EMBER_LLAMA_ERR_TOKENIZE, "Tokenization failed (buffer too small).");
            return engine->code;
        }
        return n;
    }
    // Report the required size so callers can allocate and retry.
    return needed;
}

EMBER_API int ember_llama_detokenize(
    ember_llama_t* engine, const int32_t* ids, size_t n_ids, int32_t remove_special,
    char* out_text, size_t max_bytes
) {
    if (!engine || (!ids && n_ids > 0)) return EMBER_LLAMA_ERR_ARGS;
    std::lock_guard<std::mutex> lock(engine->mu);
    if (!engine->loaded()) {
        engine->set_error(EMBER_LLAMA_ERR_NOT_LOADED, "No model is loaded.");
        return engine->code;
    }
    if (n_ids == 0) {
        if (out_text && max_bytes > 0) out_text[0] = '\0';
        return 0;
    }
    const int32_t needed = -llama_detokenize(
        engine->vocab, ids, static_cast<int32_t>(n_ids), nullptr, 0, remove_special != 0, false);
    if (needed <= 0) {
        if (out_text && max_bytes > 0) out_text[0] = '\0';
        return 0;
    }
    if (!out_text || max_bytes < static_cast<size_t>(needed) + 1) {
        return needed;  // caller must retry with a bigger buffer
    }
    int32_t n = llama_detokenize(
        engine->vocab, ids, static_cast<int32_t>(n_ids), out_text,
        static_cast<int32_t>(max_bytes), remove_special != 0, false);
    if (n < 0) return -n;
    return n;
}

EMBER_API int ember_llama_logits(ember_llama_t* e, const int32_t* ids, size_t n, float* out, size_t cap) {
    if (!e || !ids || !out || n == 0) return EMBER_LLAMA_ERR_ARGS;
    std::lock_guard<std::mutex> lock(e->mu);
    if (!e->loaded()) return EMBER_LLAMA_ERR_NOT_LOADED;
    const int v = llama_vocab_n_tokens(e->vocab);
    if (cap < static_cast<size_t>(v) || n > static_cast<size_t>(e->n_ctx)) return EMBER_LLAMA_ERR_BUFFER;
    std::vector<llama_token> tokens(ids, ids+n);
    for (auto id : tokens) if (id < 0 || id >= v) return EMBER_LLAMA_ERR_ARGS;
    llama_memory_clear(llama_get_memory(e->ctx), true);
    for (size_t i = 0; i < n; i += e->n_batch) {
        auto batch = llama_batch_get_one(tokens.data()+i, static_cast<int>(std::min(n-i, size_t(e->n_batch))));
        if (llama_decode(e->ctx, batch)) return EMBER_LLAMA_ERR_DECODE;
    }
    const float* logits = llama_get_logits_ith(e->ctx, -1);
    if (!logits || !std::all_of(logits, logits+v, [](float x) { return std::isfinite(x); })) {
        e->set_error(EMBER_LLAMA_ERR_DECODE, "The model produced non-finite logits.");
        return EMBER_LLAMA_ERR_DECODE;
    }
    std::memcpy(out, logits, v*sizeof(float));
    return 0;
}

EMBER_API int ember_llama_generate(
    ember_llama_t* engine,
    const char* prompt_utf8,
    size_t prompt_len,
    const ember_llama_gen_params_t* params,
    ember_llama_token_cb token_cb,
    void* token_user_data,
    char* json_out,
    size_t max_json
) try {
    if (!engine || !json_out || max_json == 0) return EMBER_LLAMA_ERR_ARGS;
    std::lock_guard<std::mutex> lock(engine->mu);
    if (!engine->loaded()) {
        engine->set_error(EMBER_LLAMA_ERR_NOT_LOADED,
                          "No model is loaded. Import or export a GGUF model first.");
        write_error_json(json_out, max_json, engine->code, engine->err);
        return engine->code;
    }

    engine->cancel.store(false, std::memory_order_relaxed);
    engine->clear_error();

    ember_llama_gen_params p{};
    if (params) {
        p = *params;
    } else {
        p.max_tokens = 128;
        p.temperature = 0.8f;
        p.top_k = 40;
        p.top_p = 0.95f;
        p.repeat_penalty = 1.1f;
        p.repeat_last_n = 64;
        p.seed = LLAMA_DEFAULT_SEED;
        p.stop_on_eos = 1;
        p.prepend_bos = 1;
    }
    // Resolve llama.cpp's random sentinel ourselves so the returned seed is
    // replayable; returning UINT32_MAX would misleadingly randomize again.
    if (p.seed == LLAMA_DEFAULT_SEED) {
        p.seed = std::random_device{}();
        if (p.seed == LLAMA_DEFAULT_SEED) --p.seed;
    }
    if (p.max_tokens < 0) p.max_tokens = 0;
    if (p.max_tokens > 8192) p.max_tokens = 8192;
    if (!std::isfinite(p.temperature)) p.temperature = 0.0f;
    if (!std::isfinite(p.top_p)) p.top_p = 1.0f;
    if (!std::isfinite(p.repeat_penalty) || p.repeat_penalty <= 0.0f) p.repeat_penalty = 1.0f;
    if (p.repeat_last_n < 0) p.repeat_last_n = 0;

    const auto t_start = std::chrono::steady_clock::now();
    std::string warning;
    if (engine->requested_ctx != engine->n_ctx) warning="Requested context was clamped to the model's native training limit. ";
    std::string stopped_by = "length";
    int result_code = EMBER_LLAMA_OK;

    std::string formatted_prompt;
    bool templated=false;
    const char* chat_template=llama_model_chat_template(engine->model,nullptr);
    if (chat_template && *chat_template && model_meta(engine->model,"ember.format_version").empty()) {
        std::string content(prompt_utf8 ? prompt_utf8 : "",prompt_len);
        llama_chat_message message{"user",content.c_str()};
        int32_t needed=llama_chat_apply_template(chat_template,&message,1,true,nullptr,0);
        if (needed<0) {
            engine->set_error(EMBER_LLAMA_ERR_MODEL_INCOMPAT,"This embedded chat template is not supported by pinned llama.cpp's built-in formatter.");
            write_error_json(json_out,max_json,engine->code,engine->err);
            return engine->code;
        }
        std::vector<char> formatted(static_cast<size_t>(needed)+1);
        llama_chat_apply_template(chat_template,&message,1,true,formatted.data(),formatted.size());
        formatted_prompt.assign(formatted.data(),needed);
        prompt_utf8=formatted_prompt.data();prompt_len=formatted_prompt.size();templated=true;
    }
    // ---- 1. tokenize the prompt -------------------------------------------
    std::vector<llama_token> prompt_ids;
    if (prompt_utf8 && prompt_len > 0) {
        int32_t needed = -llama_tokenize(
            engine->vocab, prompt_utf8, static_cast<int32_t>(prompt_len), nullptr, 0, false, templated);
        if (needed < 0) needed = 0;
        if (needed > 0) {
            prompt_ids.resize(static_cast<size_t>(needed) + 1);
            int32_t n = llama_tokenize(
                engine->vocab, prompt_utf8, static_cast<int32_t>(prompt_len), prompt_ids.data(),
                static_cast<int32_t>(prompt_ids.size()), false, templated);
            if (n < 0) {
                engine->set_error(EMBER_LLAMA_ERR_TOKENIZE, "Prompt tokenization failed.");
                write_error_json(json_out, max_json, engine->code, engine->err);
                return engine->code;
            }
            prompt_ids.resize(static_cast<size_t>(n));
        } else {
            prompt_ids.clear();
        }
    }

    // ---- 2. prepend BOS (Ember's training streams start with <bos>) -------
    if (p.prepend_bos && !templated && (!model_meta(engine->model,"ember.format_version").empty() || llama_vocab_get_add_bos(engine->vocab))) {
        const llama_token bos = llama_vocab_bos(engine->vocab);
        if (bos != LLAMA_TOKEN_NULL) {
            prompt_ids.insert(prompt_ids.begin(), bos);
        }
    }

    // ---- 3. clamp to the context window -----------------------------------
    const int32_t n_ctx = engine->n_ctx;
    const int32_t room = std::max(1, n_ctx - 1);
    const size_t prompt_tokens = prompt_ids.size();
    if (static_cast<int32_t>(prompt_ids.size()) > room) {
        std::ostringstream w;
        w << "Prompt (" << prompt_ids.size() << " tokens) exceeded the context window (" << n_ctx
          << "); kept the most recent " << room << " tokens.";
        warning += w.str();
        prompt_ids.erase(prompt_ids.begin(), prompt_ids.end() - room);
    }

    // ---- 4. evaluate the prompt -------------------------------------------
    llama_memory_clear(llama_get_memory(engine->ctx), true);

    double prompt_ms = 0.0;
    const auto t_prompt = std::chrono::steady_clock::now();
    size_t decoded = 0;
    while (decoded < prompt_ids.size()) {
        if (engine->cancel.load(std::memory_order_relaxed)) {
            stopped_by = "cancelled";
            result_code = EMBER_LLAMA_ERR_CANCELLED;
            break;
        }
        const size_t chunk = std::min<size_t>(static_cast<size_t>(engine->n_batch), prompt_ids.size() - decoded);
        llama_batch batch = llama_batch_get_one(prompt_ids.data() + decoded, static_cast<int32_t>(chunk));
        const int32_t rc = llama_decode(engine->ctx, batch);
        if (rc != 0) {
            std::ostringstream msg;
            msg << "Prompt evaluation failed (llama_decode returned " << rc << ").";
            if (rc == 1) {
                msg << " The context is full — reduce the context length or the prompt.";
            } else if (rc == 2) {
                msg << " Loading was aborted.";
            }
            engine->set_error(EMBER_LLAMA_ERR_DECODE, msg.str());
            stopped_by = "error";
            result_code = EMBER_LLAMA_ERR_DECODE;
            break;
        }
        decoded += chunk;
    }
    prompt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prompt).count();

    // ---- 5. sample tokens --------------------------------------------------
    std::vector<llama_token> generated;
    generated.reserve(static_cast<size_t>(std::max<int32_t>(0, p.max_tokens)));
    size_t sent_bytes = 0;
    llama_sampler* chain = nullptr;

    if (result_code == EMBER_LLAMA_OK && !prompt_ids.empty() && p.max_tokens > 0) {
        chain = build_sampler_chain(engine, p);
        // Repetition history includes the prompt once, then sample() accepts each
        // generated token once. Never double-accept the sampled token.
        for (auto token : prompt_ids) llama_sampler_accept(chain, token);
        int32_t n_past = static_cast<int32_t>(prompt_ids.size());

        for (int32_t i = 0; i < p.max_tokens; ++i) {
            if (engine->cancel.load(std::memory_order_relaxed)) {
                stopped_by = "cancelled";
                result_code = EMBER_LLAMA_ERR_CANCELLED;
                break;
            }
            if (n_past >= n_ctx) {
                if(warning.find("oldest tokens")==std::string::npos)warning+=" Context filled; oldest tokens were removed to continue generation.";
                // Learned GPT-2 positions cannot be shifted as if they were RoPE.
                // Re-evaluate the newest context with positions reset to zero.
                std::vector<llama_token> window(prompt_ids);
                window.insert(window.end(), generated.begin(), generated.end());
                if (window.size() > static_cast<size_t>(n_ctx))
                    window.erase(window.begin(), window.end() - n_ctx);
                llama_memory_clear(llama_get_memory(engine->ctx), true);
                for (size_t off = 0; off < window.size(); off += engine->n_batch) {
                    auto batch = llama_batch_get_one(window.data()+off,
                        static_cast<int>(std::min(window.size()-off, size_t(engine->n_batch))));
                    if (llama_decode(engine->ctx, batch)) {
                        stopped_by = "error";
                        result_code = EMBER_LLAMA_ERR_DECODE;
                        break;
                    }
                }
                if (result_code != 0) break;
                n_past = n_ctx;
            }

            const float* scores = llama_get_logits_ith(engine->ctx, -1);
            const int vocab_size = llama_vocab_n_tokens(engine->vocab);
            if (!scores || !std::all_of(scores, scores + vocab_size, [](float v) { return std::isfinite(v); })) {
                engine->set_error(EMBER_LLAMA_ERR_DECODE,
                    "The model produced non-finite logits. Retrain or select a different GGUF model.");
                stopped_by = "error";
                result_code = EMBER_LLAMA_ERR_DECODE;
                break;
            }
            const llama_token next = llama_sampler_sample(chain, engine->ctx, -1);
            // llama_sampler_sample already accepts the token in this pinned API.

            if (p.stop_on_eos && llama_vocab_is_eog(engine->vocab, next)) {
                stopped_by = "eos";
                break;
            }

            generated.push_back(next);

            // Stream the newly complete UTF-8 text to the caller.
            if (token_cb && !generated.empty()) {
                const int32_t need = -llama_detokenize(
                    engine->vocab, generated.data(), static_cast<int32_t>(generated.size()),
                    nullptr, 0, true, false);
                if (need > 0) {
                    std::string full(static_cast<size_t>(need) + 1, '\0');
                    int32_t wrote = llama_detokenize(
                        engine->vocab, generated.data(), static_cast<int32_t>(generated.size()),
                        full.data(), static_cast<int32_t>(full.size()), true, false);
                    if (wrote >= 0) {
                        full.resize(static_cast<size_t>(wrote));
                        const size_t safe = complete_utf8_prefix(full);
                        if (safe > sent_bytes) {
                            const std::string piece = full.substr(sent_bytes, safe - sent_bytes);
                            sent_bytes = safe;
                            token_cb(token_user_data, piece.data(), static_cast<int32_t>(piece.size()),
                                    next, static_cast<int32_t>(generated.size()));
                        }
                    }
                }
            }

            // Stop before decoding a token we will never sample from.
            if (i + 1 >= p.max_tokens) break;

            // If full, the next iteration rebuilds the sliding window, including
            // this newly generated token. Never index beyond learned positions.
            if (n_past >= n_ctx - 1) { n_past = n_ctx; continue; }
            llama_batch step = llama_batch_get_one(&generated.back(), 1);
            const int32_t rc = llama_decode(engine->ctx, step);
            if (rc != 0) {
                engine->set_error(EMBER_LLAMA_ERR_DECODE,
                                  "Token evaluation failed (llama_decode returned " + std::to_string(rc) + ").");
                stopped_by = "error";
                result_code = EMBER_LLAMA_ERR_DECODE;
                break;
            }
            ++n_past;
        }
    } else if (p.max_tokens == 0) {
        stopped_by = "length";
    }

    if (chain) {
        llama_sampler_free(chain);
        chain = nullptr;
    }

    // ---- 6. final text -----------------------------------------------------
    std::string text;
    if (!generated.empty()) {
        int32_t need = -llama_detokenize(
            engine->vocab, generated.data(), static_cast<int32_t>(generated.size()), nullptr, 0, true, false);
        if (need < 0) need = 0;
        if (need > 0) {
            text.resize(static_cast<size_t>(need) + 1, '\0');
            int32_t wrote = llama_detokenize(
                engine->vocab, generated.data(), static_cast<int32_t>(generated.size()),
                text.data(), static_cast<int32_t>(text.size()), true, false);
            if (wrote >= 0) {
                text.resize(static_cast<size_t>(wrote));
            } else {
                text.clear();
            }
        }
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double total_ms = std::max(0.001, std::chrono::duration<double, std::milli>(t_end - t_start).count());
    const double gen_ms = std::max(0.0, total_ms - prompt_ms);
    const double tps = generated.empty() ? 0.0 : (static_cast<double>(generated.size()) / (gen_ms / 1000.0));

    // ---- 7. JSON result ----------------------------------------------------
    std::ostringstream oss;
    oss << "{"
        << "\"ok\":" << (result_code == EMBER_LLAMA_OK || result_code == EMBER_LLAMA_ERR_CANCELLED ? "true" : "false")
        << ",\"text\":\"" << json_escape(text) << "\""
        << ",\"tokenIds\":[";
    for (size_t i = 0; i < generated.size(); ++i) {
        if (i) oss << ",";
        oss << generated[i];
    }
    oss << "]"
        << ",\"promptTokens\":" << prompt_tokens
        << ",\"newTokens\":" << generated.size()
        << ",\"stoppedBy\":\"" << json_escape(stopped_by) << "\""
        << ",\"warning\":" << (warning.empty() ? "null" : ("\"" + json_escape(warning) + "\""))
        << ",\"error\":" << (engine->err.empty() ? "null" : ("\"" + json_escape(engine->err) + "\""))
        << ",\"errorCode\":" << result_code
        << ",\"errorName\":\"" << ember_llama_error_name(result_code) << "\""
        << ",\"ms\":" << total_ms
        << ",\"promptMs\":" << prompt_ms
        << ",\"genMs\":" << gen_ms
        << ",\"tokensPerSec\":" << tps
        << ",\"seed\":" << p.seed
        << ",\"nCtx\":" << n_ctx
        << "}";

    const std::string payload = oss.str();
    if (payload.size() + 1 <= max_json) {
        std::memcpy(json_out, payload.data(), payload.size());
        json_out[payload.size()] = '\0';
    } else {
        // Never overflow the caller's buffer: return a minimal valid document.
        const std::string tiny = "{\"ok\":false,\"text\":\"\",\"newTokens\":0,\"error\":\"result buffer too small\"}";
        std::memcpy(json_out, tiny.data(), std::min(tiny.size(), max_json - 1));
        json_out[std::min(tiny.size(), max_json - 1)] = '\0';
        return EMBER_LLAMA_ERR_BUFFER;
    }
    return result_code;
} catch (const std::exception& exc) {
    if (engine) engine->set_error(EMBER_LLAMA_ERR_INTERNAL, exc.what());
    write_error_json(json_out, max_json, EMBER_LLAMA_ERR_INTERNAL, "Native inference failed; see diagnostics.");
    return EMBER_LLAMA_ERR_INTERNAL;
} catch (...) {
    return EMBER_LLAMA_ERR_INTERNAL;
}

EMBER_API void ember_llama_cancel(ember_llama_t* engine) {
    if (!engine) return;
    // Atomic and lock-free on purpose: cancellation must work from the UI thread
    // while a generation holds the engine mutex.
    engine->cancel.store(true, std::memory_order_relaxed);
}

EMBER_API int ember_llama_logs(char* out_json, size_t max_bytes) {
    if (!out_json || max_bytes == 0) return EMBER_LLAMA_ERR_ARGS;
    std::ostringstream oss;
    oss << "[";
    {
        std::lock_guard<std::mutex> lock(g_log_mu);
        bool first = true;
        for (const auto& line : g_logs) {
            if (!first) oss << ",";
            first = false;
            oss << "\"" << json_escape(line) << "\"";
        }
    }
    oss << "]";
    const std::string payload = oss.str();
    if (payload.size() + 1 > max_bytes) return EMBER_LLAMA_ERR_BUFFER;
    std::memcpy(out_json, payload.data(), payload.size());
    out_json[payload.size()] = '\0';
    return EMBER_LLAMA_OK;
}

EMBER_API const char* ember_llama_backend_info(void) {
    static std::string info;
    static std::once_flag once;
    std::call_once(once, []() {
        const char* sys = llama_print_system_info();
        info = std::string("llama.cpp ") + (sys ? sys : "(unknown)");
    });
    return info.c_str();
}

}  // extern "C"
