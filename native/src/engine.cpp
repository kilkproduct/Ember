#include "ember/engine.hpp"
#include "ember/c_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <mutex>
#include <unordered_set>

namespace ember {

int32_t select_next_token(
    const float* logits_f32,
    int32_t vocab_size,
    const std::vector<int32_t>& recent_ids,
    const GenerateOptions& opts,
    SplitMix64& rng
) {
    std::vector<double> logits(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) {
        double v = static_cast<double>(logits_f32[i]);
        logits[i] = std::isfinite(v) ? v : -1e30;
    }

    // Mask non-text special tokens identically to Python sampler
    for (int32_t sp : {kSpecialPad, kSpecialBos, kSpecialDoc, kSpecialUnk}) {
        if (sp < vocab_size) logits[sp] = -1e30;
    }
    if (!opts.stop_on_eos && kSpecialEos < vocab_size) {
        logits[kSpecialEos] = -1e30;
    }

    // Repetition penalty over recent window
    if (std::isfinite(opts.repetition_penalty) &&
        opts.repetition_penalty > 0.0f &&
        std::fabs(static_cast<double>(opts.repetition_penalty) - 1.0) > 1e-6 &&
        !recent_ids.empty()) {
        std::unordered_set<int32_t> seen;
        size_t start = recent_ids.size() > 128 ? recent_ids.size() - 128 : 0;
        for (size_t i = start; i < recent_ids.size(); ++i) {
            int32_t tid = recent_ids[i];
            if (tid >= 0 && tid < vocab_size) seen.insert(tid);
        }
        const double rep = static_cast<double>(opts.repetition_penalty);
        for (int32_t tid : seen) {
            if (logits[tid] > 0.0) {
                logits[tid] /= rep;
            } else {
                logits[tid] *= rep;
            }
        }
    }

    // Deterministic greedy decode
    if (!std::isfinite(opts.temperature) || opts.temperature <= 1e-6f || opts.top_k == 1) {
        int32_t best_id = 0;
        double best_val = logits[0];
        for (int32_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_id = i;
            }
        }
        return best_id;
    }

    const double inv_temp = 1.0 / static_cast<double>(opts.temperature);
    for (int32_t i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_temp;
    }

    std::vector<int32_t> order(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) order[i] = i;
    int32_t k = (opts.top_k > 0 && opts.top_k < vocab_size) ? opts.top_k : vocab_size;

    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int32_t a, int32_t b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;
    });
    order.resize(k);

    double max_l = logits[order[0]];
    std::vector<double> probs(k);
    double sum_p = 0.0;
    for (int32_t i = 0; i < k; ++i) {
        probs[i] = std::exp(logits[order[i]] - max_l);
        sum_p += probs[i];
    }
    if (!std::isfinite(sum_p) || sum_p <= 0.0) {
        return order[0];
    }
    for (int32_t i = 0; i < k; ++i) {
        probs[i] /= sum_p;
    }

    // Top-p nucleus filter
    if (std::isfinite(opts.top_p) && opts.top_p > 0.0f && opts.top_p < 1.0f) {
        double csum = 0.0;
        int32_t cutoff = k;
        for (int32_t i = 0; i < k; ++i) {
            csum += probs[i];
            if (csum >= static_cast<double>(opts.top_p)) {
                cutoff = i + 1;
                break;
            }
        }
        cutoff = std::max<int32_t>(1, std::min<int32_t>(k, cutoff));
        order.resize(cutoff);
        probs.resize(cutoff);
        double renorm = 0.0;
        for (double p : probs) renorm += p;
        for (double& p : probs) p /= renorm;
    }

    double u = rng.next_f64();
    double acc = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        acc += probs[i];
        if (u < acc) {
            return order[i];
        }
    }
    return order.back();
}

// =========================================================================
// Engine
// =========================================================================

bool Engine::load(const std::string& model_path, const std::string& tokenizer_path) {
    std::lock_guard<std::mutex> lock(mu_);
    BPETokenizer next_tok;
    std::string err;
    if (!next_tok.load_from_file(tokenizer_path, err)) {
        last_error_ = err;
        return false;
    }
    GPTModel next_model;
    if (!next_model.load_from_file(model_path, err)) {
        last_error_ = err;
        return false;
    }
    if (next_tok.vocab_size() != next_model.config().vocab_size) {
        last_error_ = "Tokenizer vocab_size (" + std::to_string(next_tok.vocab_size()) +
                      ") does not match model vocab_size (" + std::to_string(next_model.config().vocab_size) + ").";
        return false;
    }
    tokenizer_ = std::move(next_tok);
    model_ = std::move(next_model);
    last_error_.clear();
    return true;
}

void Engine::unload() {
    std::lock_guard<std::mutex> lock(mu_);
    model_ = GPTModel{};
    tokenizer_ = BPETokenizer{};
    last_error_.clear();
}

bool Engine::is_ready() const {
    std::lock_guard<std::mutex> lock(mu_);
    return model_.ready() && tokenizer_.ready();
}

int32_t Engine::vocab_size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return model_.ready() ? model_.config().vocab_size : 0;
}

int32_t Engine::block_size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return model_.ready() ? model_.config().block_size : 0;
}

std::string Engine::last_error() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_error_;
}

bool Engine::encode(const uint8_t* utf8, size_t len, std::vector<int32_t>& out_ids) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!tokenizer_.ready()) {
        last_error_ = "Tokenizer is not loaded.";
        return false;
    }
    out_ids = tokenizer_.encode(utf8, len);
    return true;
}

bool Engine::decode(const int32_t* ids, size_t n, std::string& out_utf8) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!tokenizer_.ready()) {
        last_error_ = "Tokenizer is not loaded.";
        return false;
    }
    out_utf8 = tokenizer_.decode(ids, n, true);
    return true;
}

bool Engine::forward_logits(const int32_t* ids, size_t seq_len, std::vector<float>& out_logits) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!model_.ready()) {
        last_error_ = "Model is not loaded.";
        return false;
    }
    out_logits.resize(seq_len * static_cast<size_t>(model_.config().vocab_size));
    std::string err;
    if (!model_.forward_sequence(ids, static_cast<int32_t>(seq_len), out_logits.data(), err)) {
        last_error_ = err;
        return false;
    }
    return true;
}

bool Engine::generate(
    const uint8_t* prompt_utf8,
    size_t prompt_len,
    const GenerateOptions& opts,
    std::vector<int32_t>& out_ids,
    GenerateOutput& out_summary
) {
    std::lock_guard<std::mutex> lock(mu_);
    cancel_requested_.store(false, std::memory_order_relaxed);
    out_ids.clear();
    out_summary = GenerateOutput{};
    out_summary.seed = opts.seed;

    if (!model_.ready() || !tokenizer_.ready()) {
        last_error_ = "No vessel is loaded in native C++ inference engine. Cast or load a model first.";
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    SplitMix64 rng(opts.seed);

    std::vector<int32_t> prompt_ids;
    if (prompt_utf8 != nullptr && prompt_len > 0) {
        prompt_ids = tokenizer_.encode(prompt_utf8, prompt_len);
    }
    out_summary.prompt_tokens = static_cast<int32_t>(prompt_ids.size());

    int32_t max_new = std::clamp<int32_t>(opts.max_new_tokens, 0, 8192);
    std::vector<int32_t> context;
    context.reserve(prompt_ids.size() + static_cast<size_t>(max_new) + 2);
    context.push_back(kSpecialBos);
    context.insert(context.end(), prompt_ids.begin(), prompt_ids.end());

    const int32_t B = model_.config().block_size;
    const int32_t V = model_.config().vocab_size;

    if (static_cast<int32_t>(context.size()) > B) {
        out_summary.warning = "Prompt (" + std::to_string(context.size()) +
                              " tokens) exceeded context window (" + std::to_string(B) +
                              "); kept the most recent " + std::to_string(B) + " tokens.";
        context.erase(context.begin(), context.end() - B);
    }

    if (max_new == 0) {
        out_summary.stopped_by = "length";
        return true;
    }

    std::vector<float> logits(V);
    std::string err;

    // Initial prefill into KV cache
    int32_t kv_len = static_cast<int32_t>(context.size());
    if (!model_.prefill_last(context.data(), kv_len, logits.data(), err)) {
        last_error_ = err;
        return false;
    }

    for (int32_t step = 0; step < max_new; ++step) {
        if (cancel_requested_.load(std::memory_order_relaxed)) {
            out_summary.stopped_by = "cancelled";
            break;
        }

        // Build recent window identically to Python sampler (generated[-64:] + prompt_ids[-64:])
        std::vector<int32_t> recent;
        size_t g_start = out_ids.size() > 64 ? out_ids.size() - 64 : 0;
        size_t p_start = prompt_ids.size() > 64 ? prompt_ids.size() - 64 : 0;
        recent.insert(recent.end(), out_ids.begin() + g_start, out_ids.end());
        recent.insert(recent.end(), prompt_ids.begin() + p_start, prompt_ids.end());

        int32_t next_id = select_next_token(logits.data(), V, recent, opts, rng);
        if (opts.stop_on_eos && next_id == kSpecialEos) {
            out_summary.stopped_by = "eos";
            break;
        }

        out_ids.push_back(next_id);
        context.push_back(next_id);

        if (step + 1 < max_new) {
            if (kv_len < B) {
                // Fast O(pos) incremental KV-cached decode step
                if (!model_.step_cached(next_id, kv_len, logits.data(), err)) {
                    last_error_ = err;
                    return false;
                }
                ++kv_len;
            } else {
                // Sliding context window reached capacity: keep newest B tokens and refresh KV cache
                const int32_t* window_ptr = context.data() + (context.size() - B);
                if (!model_.prefill_last(window_ptr, B, logits.data(), err)) {
                    last_error_ = err;
                    return false;
                }
                kv_len = B;
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ms < 0.001) ms = 0.001;

    out_summary.new_tokens = static_cast<int32_t>(out_ids.size());
    out_summary.elapsed_ms = ms;
    out_summary.tokens_per_sec = out_ids.empty() ? 0.0 : (static_cast<double>(out_ids.size()) / (ms / 1000.0));
    out_summary.text = tokenizer_.decode(out_ids.data(), out_ids.size(), true);
    return true;
}

}  // namespace ember

// =========================================================================
// C ABI Exports (all exceptions caught across the C boundary)
// =========================================================================

struct ember_engine_s {
    ember::Engine impl;
    // Guards both the cached error string and the last generation summary, so
    // readers never observe a torn value while another thread is generating.
    mutable std::mutex aux_mu;
    mutable std::string cached_err;
    ember::GenerateOutput last_summary;
};

namespace {

void set_c_err(const ember_engine_t* engine, const std::string& msg) {
    if (!engine) return;
    std::lock_guard<std::mutex> lk(engine->aux_mu);
    engine->cached_err = msg;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
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

}  // namespace

extern "C" {

ember_engine_t* ember_engine_create(void) {
    try {
        return new ember_engine_s();
    } catch (...) {
        return nullptr;
    }
}

void ember_engine_destroy(ember_engine_t* engine) {
    delete engine;
}

int ember_engine_load(ember_engine_t* engine, const char* model_path, const char* tokenizer_path) {
    if (!engine) return -1;
    if (!model_path || !tokenizer_path) {
        set_c_err(engine, "Null model or tokenizer path.");
        return -1;
    }
    try {
        if (!engine->impl.load(model_path, tokenizer_path)) {
            set_c_err(engine, engine->impl.last_error());
            return -1;
        }
        set_c_err(engine, {});
        return 0;
    } catch (const std::exception& ex) {
        set_c_err(engine, std::string("Native exception in load: ") + ex.what());
        return -1;
    } catch (...) {
        set_c_err(engine, "Unknown native exception in load.");
        return -1;
    }
}

const char* ember_engine_last_error(const ember_engine_t* engine) {
    if (!engine) return "null engine handle";
    std::lock_guard<std::mutex> lk(engine->aux_mu);
    std::string impl_err = engine->impl.last_error();
    if (!impl_err.empty()) {
        engine->cached_err = std::move(impl_err);
    }
    return engine->cached_err.c_str();
}

int ember_engine_last_generate_summary(ember_engine_t* engine, char* out_json, size_t max_bytes) {
    if (!engine || !out_json || max_bytes == 0) return -1;
    std::string payload;
    {
        std::lock_guard<std::mutex> lk(engine->aux_mu);
        const ember::GenerateOutput& s = engine->last_summary;
        payload = "{\"warning\":\"" + json_escape(s.warning) + "\""
                + ",\"stoppedBy\":\"" + json_escape(s.stopped_by) + "\""
                + ",\"promptTokens\":" + std::to_string(s.prompt_tokens)
                + ",\"newTokens\":" + std::to_string(s.new_tokens)
                + ",\"seed\":" + std::to_string(s.seed)
                + ",\"elapsedMs\":" + std::to_string(s.elapsed_ms)
                + ",\"tokensPerSec\":" + std::to_string(s.tokens_per_sec)
                + "}";
    }
    if (payload.size() + 1 > max_bytes) return -1;
    std::memcpy(out_json, payload.data(), payload.size());
    out_json[payload.size()] = '\0';
    return static_cast<int>(payload.size());
}

int ember_engine_vocab_size(const ember_engine_t* engine) {
    return engine ? engine->impl.vocab_size() : 0;
}

int ember_engine_block_size(const ember_engine_t* engine) {
    return engine ? engine->impl.block_size() : 0;
}

int ember_engine_is_ready(const ember_engine_t* engine) {
    return (engine && engine->impl.is_ready()) ? 1 : 0;
}

void ember_engine_cancel(ember_engine_t* engine) {
    if (engine) engine->impl.cancel();
}

int ember_engine_encode(
    ember_engine_t* engine,
    const char* utf8_text,
    size_t text_len,
    int32_t* out_ids,
    size_t max_ids
) {
    if (!engine) return -1;
    try {
        std::vector<int32_t> ids;
        if (!engine->impl.encode(reinterpret_cast<const uint8_t*>(utf8_text ? utf8_text : ""), text_len, ids)) {
            set_c_err(engine, engine->impl.last_error());
            return -1;
        }
        if (out_ids && max_ids > 0 && !ids.empty()) {
            size_t copy_n = std::min(ids.size(), max_ids);
            std::memcpy(out_ids, ids.data(), copy_n * sizeof(int32_t));
        }
        return static_cast<int>(ids.size());
    } catch (const std::exception& ex) {
        set_c_err(engine, std::string("Native exception in encode: ") + ex.what());
        return -1;
    } catch (...) {
        set_c_err(engine, "Unknown native exception in encode.");
        return -1;
    }
}

int ember_engine_decode(
    ember_engine_t* engine,
    const int32_t* ids,
    size_t num_ids,
    char* out_utf8,
    size_t max_bytes
) {
    if (!engine) return -1;
    try {
        std::string text;
        if (!engine->impl.decode(ids, num_ids, text)) {
            set_c_err(engine, engine->impl.last_error());
            return -1;
        }
        if (out_utf8 && max_bytes > 0) {
            size_t copy_n = std::min(text.size(), max_bytes - 1);
            if (copy_n > 0) {
                std::memcpy(out_utf8, text.data(), copy_n);
            }
            out_utf8[copy_n] = '\0';
        }
        return static_cast<int>(text.size());
    } catch (const std::exception& ex) {
        set_c_err(engine, std::string("Native exception in decode: ") + ex.what());
        return -1;
    } catch (...) {
        set_c_err(engine, "Unknown native exception in decode.");
        return -1;
    }
}

int ember_engine_forward_logits(
    ember_engine_t* engine,
    const int32_t* ids,
    size_t seq_len,
    float* out_logits,
    size_t out_floats
) {
    if (!engine || !ids || !out_logits) return -1;
    try {
        std::vector<float> logits;
        if (!engine->impl.forward_logits(ids, seq_len, logits)) {
            set_c_err(engine, engine->impl.last_error());
            return -1;
        }
        if (logits.size() > out_floats) {
            set_c_err(engine, "Output logits buffer too small.");
            return -2;
        }
        std::memcpy(out_logits, logits.data(), logits.size() * sizeof(float));
        return 0;
    } catch (const std::exception& ex) {
        set_c_err(engine, std::string("Native exception in forward_logits: ") + ex.what());
        return -1;
    } catch (...) {
        set_c_err(engine, "Unknown native exception in forward_logits.");
        return -1;
    }
}

int ember_engine_generate(
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
) {
    if (!engine) return -1;
    try {
        ember::GenerateOptions opts;
        opts.max_new_tokens = max_new_tokens;
        opts.temperature = temperature;
        opts.top_k = top_k;
        opts.top_p = top_p;
        opts.repetition_penalty = repetition_penalty;
        opts.seed = seed;
        opts.stop_on_eos = (stop_on_eos != 0);

        std::vector<int32_t> ids;
        ember::GenerateOutput summary;
        if (!engine->impl.generate(
                reinterpret_cast<const uint8_t*>(prompt_utf8 ? prompt_utf8 : ""),
                prompt_len,
                opts,
                ids,
                summary)) {
            set_c_err(engine, engine->impl.last_error());
            return -1;
        }
        {
            std::lock_guard<std::mutex> lk(engine->aux_mu);
            engine->last_summary = summary;
        }
        if (out_ids && max_ids > 0 && !ids.empty()) {
            size_t copy_ids = std::min(ids.size(), max_ids);
            std::memcpy(out_ids, ids.data(), copy_ids * sizeof(int32_t));
        }
        if (out_utf8 && max_bytes > 0) {
            size_t copy_txt = std::min(summary.text.size(), max_bytes - 1);
            if (copy_txt > 0) {
                std::memcpy(out_utf8, summary.text.data(), copy_txt);
            }
            out_utf8[copy_txt] = '\0';
        }
        return static_cast<int>(ids.size());
    } catch (const std::exception& ex) {
        set_c_err(engine, std::string("Native exception in generate: ") + ex.what());
        return -1;
    } catch (...) {
        set_c_err(engine, "Unknown native exception in generate.");
        return -1;
    }
}

}  // extern "C"
