#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ember {

constexpr char kFormatMagic[4] = {'E', 'M', 'B', '3'};
constexpr uint32_t kFormatVersion = 3;
constexpr uint32_t kDtypeFloat32 = 0;
constexpr size_t kHeaderAlign = 16;

constexpr int32_t kBaseVocab = 256;
constexpr int32_t kSpecialPad = 256;
constexpr int32_t kSpecialBos = 257;
constexpr int32_t kSpecialEos = 258;
constexpr int32_t kSpecialDoc = 259;
constexpr int32_t kSpecialUnk = 260;
constexpr int32_t kFirstMergeId = 261;

constexpr float kNormEps = 1e-5f;
constexpr float kMaskNeg = -1e9f;

struct ModelConfig {
    int32_t n_layer = 2;
    int32_t n_head = 2;
    int32_t n_embd = 64;
    int32_t block_size = 64;
    int32_t vocab_size = 1500;
    float ffn_mult = 4.0f;
    bool tie_embeddings = true;
    float norm_eps = kNormEps;

    int32_t head_dim() const { return n_embd / n_head; }
    int32_t ffn_dim() const;
    std::string validate() const;
};

struct GenerateOptions {
    int32_t max_new_tokens = 80;
    float temperature = 0.9f;
    int32_t top_k = 40;
    float top_p = 0.95f;
    float repetition_penalty = 1.15f;
    uint64_t seed = 42;
    bool stop_on_eos = true;
};

struct GenerateOutput {
    std::string text;
    std::string warning;
    std::string stopped_by = "length";
    int32_t prompt_tokens = 0;
    int32_t new_tokens = 0;
    double elapsed_ms = 0.0;
    double tokens_per_sec = 0.0;
    uint64_t seed = 0;
};

}  // namespace ember
