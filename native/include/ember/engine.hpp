#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ember/spec.hpp"

namespace ember {

class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t next_u64() {
        state_ += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    double next_f64() {
        return static_cast<double>(next_u64() >> 11) * (1.0 / static_cast<double>(1ULL << 53));
    }

private:
    uint64_t state_;
};

std::vector<std::string> pretokenize_bytes(const uint8_t* data, size_t len);

class BPETokenizer {
public:
    BPETokenizer() = default;
    // The encode cache is guarded by a mutex, which is neither copyable nor
    // movable, so the transfer operations are written explicitly: they move the
    // vocabulary/merge tables and the cache contents, and leave a fresh mutex
    // behind. `Engine::load` relies on move-assignment to swap in a validated
    // tokenizer without ever exposing a half-loaded one to readers.
    BPETokenizer(const BPETokenizer&) = delete;
    BPETokenizer& operator=(const BPETokenizer&) = delete;
    BPETokenizer(BPETokenizer&& other) noexcept;
    BPETokenizer& operator=(BPETokenizer&& other) noexcept;

    bool load_from_json(const std::string& json_text, std::string& error);
    bool load_from_file(const std::string& path, std::string& error);

    int32_t vocab_size() const { return static_cast<int32_t>(vocab_.size()); }
    bool ready() const { return !vocab_.empty(); }

    std::vector<int32_t> encode(const uint8_t* data, size_t len) const;
    std::vector<int32_t> encode_str(const std::string& s) const {
        return encode(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }
    std::string decode(const int32_t* ids, size_t n, bool skip_special = true) const;

private:
    std::vector<int32_t> encode_chunk(const std::string& chunk) const;

    std::vector<std::string> vocab_;  // raw byte sequences
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> pair_ranks_;
    mutable std::mutex cache_mu_;
    mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

struct BlockWeights {
    std::vector<float> ln1_g;
    std::vector<float> ln1_b;
    std::vector<float> qkv_w_t;   // [3*C, C] transposed for contiguous SIMD dot
    std::vector<float> qkv_b;     // [3*C]
    std::vector<float> proj_w_t;  // [C, C] transposed
    std::vector<float> proj_b;    // [C]
    std::vector<float> ln2_g;
    std::vector<float> ln2_b;
    std::vector<float> fc_w_t;    // [F, C] transposed
    std::vector<float> fc_b;      // [F]
    std::vector<float> mproj_w_t; // [C, F] transposed
    std::vector<float> mproj_b;   // [C]
};

class GPTModel {
public:
    bool load_from_bytes(const uint8_t* data, size_t len, std::string& error);
    bool load_from_file(const std::string& path, std::string& error);

    const ModelConfig& config() const { return cfg_; }
    bool ready() const { return ready_; }

    // Full sequence forward pass returning [T, V] float32 logits (also fills KV cache)
    bool forward_sequence(const int32_t* ids, int32_t T, float* out_logits, std::string& error);

    // Prefill KV cache on [0..T-1] and compute only the last position logits [V]
    bool prefill_last(const int32_t* ids, int32_t T, float* out_last_logits, std::string& error);

    // Single-step KV-cached incremental decode at position `pos` (0 <= pos < block_size)
    bool step_cached(int32_t token_id, int32_t pos, float* out_logits, std::string& error);

private:
    void forward_token_at(int32_t token_id, int32_t pos, float* out_logits);

    ModelConfig cfg_;
    bool ready_ = false;
    std::vector<float> wte_;       // [V, C] row-major (already contiguous per vocab row!)
    std::vector<float> wpe_;       // [block_size, C]
    std::vector<BlockWeights> blocks_;
    std::vector<float> lnf_g_;
    std::vector<float> lnf_b_;
    std::vector<float> lm_head_;   // [V, C] when !tie_embeddings

    // Pre-allocated KV cache and working buffers (zero heap churn per token)
    std::vector<float> k_cache_;   // [n_layer * block_size * C]
    std::vector<float> v_cache_;   // [n_layer * block_size * C]
    std::vector<float> work_x_;
    std::vector<float> work_xmid_;
    std::vector<float> work_ln_;
    std::vector<float> work_qkv_;
    std::vector<float> work_att_out_;
    std::vector<float> work_proj_;
    std::vector<float> work_fc_;
    std::vector<float> work_scores_;
};

int32_t select_next_token(
    const float* logits,
    int32_t vocab_size,
    const std::vector<int32_t>& recent_ids,
    const GenerateOptions& opts,
    SplitMix64& rng
);

class Engine {
public:
    Engine() = default;

    bool load(const std::string& model_path, const std::string& tokenizer_path);
    void unload();
    bool is_ready() const;
    int32_t vocab_size() const;
    int32_t block_size() const;
    std::string last_error() const;

    void cancel() { cancel_requested_.store(true, std::memory_order_relaxed); }

    bool encode(const uint8_t* utf8, size_t len, std::vector<int32_t>& out_ids);
    bool decode(const int32_t* ids, size_t n, std::string& out_utf8);
    bool forward_logits(const int32_t* ids, size_t seq_len, std::vector<float>& out_logits);

    bool generate(
        const uint8_t* prompt_utf8,
        size_t prompt_len,
        const GenerateOptions& opts,
        std::vector<int32_t>& out_ids,
        GenerateOutput& out_summary
    );

private:
    mutable std::mutex mu_;
    std::atomic<bool> cancel_requested_{false};
    GPTModel model_;
    BPETokenizer tokenizer_;
    std::string last_error_;
};

}  // namespace ember
