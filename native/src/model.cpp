#include "ember/engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "json_min.hpp"

namespace ember {

namespace {

constexpr int32_t kMaxLayers = 64;
constexpr int32_t kMaxHeads = 64;
constexpr int32_t kMaxEmbd = 4096;
constexpr int32_t kMaxBlockSize = 8192;
constexpr int32_t kMaxVocabSize = 262144;
constexpr uint32_t kMaxTensors = 2048;

const std::array<uint32_t, 256>& crc32_table() {
    static const std::array<uint32_t, 256> table = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1U) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
        return t;
    }();
    return table;
}

uint32_t crc32_ieee(const uint8_t* data, size_t len) {
    const auto& table = crc32_table();
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < len; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFU] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFU;
}

inline uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline float dot_f32(const float* __restrict a, const float* __restrict b, int n) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
        sum1 = vfmaq_f32(sum1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
#else
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
#endif
}

// Compute y[o] = dot(W_T[o], x) + bias[o] for o in 0..out_dim-1
inline void linear_vec(
    const float* __restrict w_t,
    const float* __restrict bias,
    const float* __restrict x,
    float* __restrict y,
    int in_dim,
    int out_dim
) {
    for (int o = 0; o < out_dim; ++o) {
        float sum = dot_f32(w_t + static_cast<size_t>(o) * in_dim, x, in_dim);
        y[o] = bias ? (sum + bias[o]) : sum;
    }
}

inline void layernorm_vec(
    const float* __restrict x,
    const float* __restrict gain,
    const float* __restrict bias,
    float* __restrict y,
    int n,
    float eps
) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += static_cast<double>(x[i]);
    mean /= static_cast<double>(n);

    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = static_cast<double>(x[i]) - mean;
        var += d * d;
    }
    var /= static_cast<double>(n);

    const float m = static_cast<float>(mean);
    const float rstd = 1.0f / std::sqrt(static_cast<float>(var) + eps);
    for (int i = 0; i < n; ++i) {
        y[i] = (x[i] - m) * rstd * gain[i] + bias[i];
    }
}

inline void gelu_inplace(float* x, int n) {
    constexpr float kC = 0.7978845608028654f;
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        float inner = kC * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

// Transpose [rows, cols] -> [cols, rows] once at load time
std::vector<float> transpose_2d(const std::vector<float>& src, int rows, int cols) {
    std::vector<float> dst(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            dst[static_cast<size_t>(c) * rows + r] = src[static_cast<size_t>(r) * cols + c];
        }
    }
    return dst;
}

}  // namespace

int32_t ModelConfig::ffn_dim() const {
    float mult = (std::isfinite(ffn_mult) && ffn_mult > 0.0f) ? ffn_mult : 4.0f;
    double scaled = static_cast<double>(n_embd) * static_cast<double>(mult);
    if (!std::isfinite(scaled) || scaled > static_cast<double>(kMaxEmbd * 16)) {
        return kMaxEmbd * 16;
    }
    int32_t d = static_cast<int32_t>(std::llround(scaled));
    return d < 1 ? 1 : d;
}

std::string ModelConfig::validate() const {
    if (n_layer < 1 || n_layer > kMaxLayers) return "n_layer must be in 1..64";
    if (n_head < 1 || n_head > kMaxHeads) return "n_head must be in 1..64";
    if (n_embd < 1 || n_embd > kMaxEmbd) return "n_embd must be in 1..4096";
    if (block_size < 1 || block_size > kMaxBlockSize) return "block_size must be in 1..8192";
    if (vocab_size < kFirstMergeId + 1 || vocab_size > kMaxVocabSize) {
        return "vocab_size must be in 262..262144";
    }
    if (n_embd % n_head != 0) return "n_embd must be divisible by n_head";
    if (!std::isfinite(ffn_mult) || ffn_mult < 0.25f || ffn_mult > 16.0f) {
        return "ffn_mult must be finite and in 0.25..16.0";
    }
    if (!std::isfinite(norm_eps) || norm_eps <= 0.0f || norm_eps > 1.0f) {
        return "norm_eps must be finite and in (0, 1]";
    }
    return {};
}

bool GPTModel::load_from_file(const std::string& path, std::string& error) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        error = "Could not open model file: " + path;
        return false;
    }
    std::streamsize sz = in.tellg();
    if (sz < 20) {
        error = "Model file is too small or empty.";
        return false;
    }
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    if (!in.read(reinterpret_cast<char*>(buf.data()), sz)) {
        error = "Failed reading model bytes from disk.";
        return false;
    }
    return load_from_bytes(buf.data(), buf.size(), error);
}

bool GPTModel::load_from_bytes(const uint8_t* data, size_t len, std::string& error) {
    ready_ = false;
    if (!data || len < 20) {
        error = "File is too small to be an Ember v3 model.";
        return false;
    }
    if (std::memcmp(data, kFormatMagic, 4) != 0) {
        if (data[0] == '{' || data[0] == '[') {
            error = "Legacy JSON brain detected. Old TensorFlow.js brains cannot be executed by native C++ inference — recast a v3 vessel.";
        } else {
            error = "Invalid model file magic (expected EMB3).";
        }
        return false;
    }

    const size_t body_end = len - 4;
    uint32_t expected_crc = read_u32_le(data + body_end);
    uint32_t actual_crc = crc32_ieee(data, body_end);
    if (expected_crc != actual_crc) {
        error = "Model file checksum mismatch (corrupt or truncated .emb file).";
        return false;
    }

    uint32_t version = read_u32_le(data + 4);
    uint32_t header_len = read_u32_le(data + 8);
    if (version != kFormatVersion) {
        error = "Unsupported model format version " + std::to_string(version) + " (expected 3).";
        return false;
    }
    if (header_len > body_end - 16) {
        error = "Truncated model header.";
        return false;
    }

    ModelConfig parsed_cfg;
    try {
        std::string hdr_str(reinterpret_cast<const char*>(data + 12), header_len);
        auto hdr = json::parse(hdr_str);
        if (hdr.type != json::Type::Object) {
            error = "Model header JSON must be an object.";
            return false;
        }
        if (hdr.str_or("arch", "ember-gpt") != "ember-gpt") {
            error = "Unsupported model architecture in header.";
            return false;
        }
        parsed_cfg.n_layer = static_cast<int32_t>(hdr.int_or("n_layer", 0));
        parsed_cfg.n_head = static_cast<int32_t>(hdr.int_or("n_head", 0));
        parsed_cfg.n_embd = static_cast<int32_t>(hdr.int_or("n_embd", 0));
        parsed_cfg.block_size = static_cast<int32_t>(hdr.int_or("block_size", 0));
        parsed_cfg.vocab_size = static_cast<int32_t>(hdr.int_or("vocab_size", 0));
        parsed_cfg.ffn_mult = static_cast<float>(hdr.double_or("ffn_mult", 4.0));
        parsed_cfg.tie_embeddings = hdr.bool_or("tie_embeddings", true);
        parsed_cfg.norm_eps = static_cast<float>(hdr.double_or("norm_eps", kNormEps));
    } catch (const std::exception& ex) {
        error = std::string("Malformed model JSON header: ") + ex.what();
        return false;
    }

    std::string cfg_err = parsed_cfg.validate();
    if (!cfg_err.empty()) {
        error = "Invalid model config: " + cfg_err;
        return false;
    }

    size_t off = 12 + static_cast<size_t>(header_len);
    uint32_t tensor_count = read_u32_le(data + off);
    off += 4;
    if (tensor_count > kMaxTensors) {
        error = "Excessive tensor count in model file (" + std::to_string(tensor_count) + ").";
        return false;
    }

    struct RawTensor {
        std::vector<int32_t> shape;
        std::vector<float> values;
    };
    std::unordered_map<std::string, RawTensor> tensors;

    for (uint32_t ti = 0; ti < tensor_count; ++ti) {
        if (body_end - off < 4) {
            error = "Truncated tensor entry.";
            return false;
        }
        uint32_t name_len = read_u32_le(data + off);
        off += 4;
        if (name_len > 256 || body_end - off < static_cast<size_t>(name_len) + 8) {
            error = "Truncated or invalid tensor metadata.";
            return false;
        }
        std::string name(reinterpret_cast<const char*>(data + off), name_len);
        off += name_len;
        uint32_t dtype = read_u32_le(data + off);
        uint32_t ndim = read_u32_le(data + off + 4);
        off += 8;
        if (dtype != kDtypeFloat32) {
            error = "Unsupported tensor dtype for " + name;
            return false;
        }
        if (ndim < 1 || ndim > 8 || body_end - off < static_cast<size_t>(ndim) * 4) {
            error = "Invalid tensor shape dimensions for " + name;
            return false;
        }

        size_t meta_bytes = 8 + 4 * static_cast<size_t>(ndim);
        size_t pad = (kHeaderAlign - (meta_bytes % kHeaderAlign)) % kHeaderAlign;
        size_t header_tail = static_cast<size_t>(ndim) * 4 + pad;
        if (body_end - off < header_tail) {
            error = "Truncated tensor alignment padding for " + name;
            return false;
        }
        const size_t max_floats_remaining = (body_end - off - header_tail) / sizeof(float);

        RawTensor rt;
        size_t numel = 1;
        for (uint32_t di = 0; di < ndim; ++di) {
            uint32_t dim = read_u32_le(data + off);
            off += 4;
            if (dim == 0 || dim > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
                numel > max_floats_remaining / static_cast<size_t>(dim)) {
                error = "Tensor shape overflow or truncated payload for " + name;
                return false;
            }
            rt.shape.push_back(static_cast<int32_t>(dim));
            numel *= static_cast<size_t>(dim);
        }
        off += pad;

        size_t nbytes = numel * sizeof(float);
        if (body_end - off < nbytes) {
            error = "Truncated tensor payload for " + name;
            return false;
        }
        rt.values.resize(numel);
        std::memcpy(rt.values.data(), data + off, nbytes);
        off += nbytes;

        for (size_t vi = 0; vi < numel; ++vi) {
            if (!std::isfinite(rt.values[vi])) {
                error = "Corrupt non-finite weight value (NaN/Inf) in tensor: " + name;
                return false;
            }
        }
        if (!tensors.emplace(std::move(name), std::move(rt)).second) {
            error = "Duplicate tensor entry in model file.";
            return false;
        }
    }

    auto take = [&](const std::string& name, const std::vector<int32_t>& expected_shape, std::vector<float>& dst) -> bool {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            error = "Missing required tensor: " + name;
            return false;
        }
        if (it->second.shape != expected_shape) {
            error = "Shape mismatch on tensor: " + name;
            return false;
        }
        dst = std::move(it->second.values);
        return true;
    };

    const int32_t C = parsed_cfg.n_embd;
    const int32_t F = parsed_cfg.ffn_dim();
    const int32_t V = parsed_cfg.vocab_size;
    const int32_t T = parsed_cfg.block_size;

    if (!take("wte", {V, C}, wte_)) return false;
    if (!take("wpe", {T, C}, wpe_)) return false;
    if (!take("ln_f.gain", {C}, lnf_g_)) return false;
    if (!take("ln_f.bias", {C}, lnf_b_)) return false;
    if (!parsed_cfg.tie_embeddings) {
        if (!take("lm_head.weight", {V, C}, lm_head_)) return false;
    } else {
        lm_head_.clear();
    }

    blocks_.assign(parsed_cfg.n_layer, BlockWeights{});
    for (int32_t l = 0; l < parsed_cfg.n_layer; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        BlockWeights& bw = blocks_[l];
        std::vector<float> tmp;
        if (!take(p + "ln1.gain", {C}, bw.ln1_g)) return false;
        if (!take(p + "ln1.bias", {C}, bw.ln1_b)) return false;
        if (!take(p + "attn.qkv.weight", {C, 3 * C}, tmp)) return false;
        bw.qkv_w_t = transpose_2d(tmp, C, 3 * C);
        if (!take(p + "attn.qkv.bias", {3 * C}, bw.qkv_b)) return false;
        if (!take(p + "attn.proj.weight", {C, C}, tmp)) return false;
        bw.proj_w_t = transpose_2d(tmp, C, C);
        if (!take(p + "attn.proj.bias", {C}, bw.proj_b)) return false;
        if (!take(p + "ln2.gain", {C}, bw.ln2_g)) return false;
        if (!take(p + "ln2.bias", {C}, bw.ln2_b)) return false;
        if (!take(p + "mlp.fc.weight", {C, F}, tmp)) return false;
        bw.fc_w_t = transpose_2d(tmp, C, F);
        if (!take(p + "mlp.fc.bias", {F}, bw.fc_b)) return false;
        if (!take(p + "mlp.proj.weight", {F, C}, tmp)) return false;
        bw.mproj_w_t = transpose_2d(tmp, F, C);
        if (!take(p + "mlp.proj.bias", {C}, bw.mproj_b)) return false;
    }

    cfg_ = parsed_cfg;
    const size_t kv_size = static_cast<size_t>(cfg_.n_layer) * static_cast<size_t>(T) * static_cast<size_t>(C);
    k_cache_.assign(kv_size, 0.0f);
    v_cache_.assign(kv_size, 0.0f);
    work_x_.resize(C);
    work_xmid_.resize(C);
    work_ln_.resize(C);
    work_qkv_.resize(3 * C);
    work_att_out_.resize(C);
    work_proj_.resize(C);
    work_fc_.resize(F);
    work_scores_.resize(T);

    ready_ = true;
    return true;
}

void GPTModel::forward_token_at(int32_t token_id, int32_t pos, float* out_logits) {
    const int32_t C = cfg_.n_embd;
    const int32_t H = cfg_.n_head;
    const int32_t D = cfg_.head_dim();
    const int32_t F = cfg_.ffn_dim();
    const int32_t T = cfg_.block_size;
    const int32_t V = cfg_.vocab_size;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));

    const float* emb_tok = wte_.data() + static_cast<size_t>(token_id) * C;
    const float* emb_pos = wpe_.data() + static_cast<size_t>(pos) * C;
    for (int32_t i = 0; i < C; ++i) {
        work_x_[i] = emb_tok[i] + emb_pos[i];
    }

    for (int32_t l = 0; l < cfg_.n_layer; ++l) {
        const BlockWeights& bw = blocks_[l];
        layernorm_vec(work_x_.data(), bw.ln1_g.data(), bw.ln1_b.data(), work_ln_.data(), C, cfg_.norm_eps);
        linear_vec(bw.qkv_w_t.data(), bw.qkv_b.data(), work_ln_.data(), work_qkv_.data(), C, 3 * C);

        const float* q_all = work_qkv_.data();
        const float* k_all = work_qkv_.data() + C;
        const float* v_all = work_qkv_.data() + 2 * C;

        float* layer_k = k_cache_.data() + static_cast<size_t>(l) * T * C;
        float* layer_v = v_cache_.data() + static_cast<size_t>(l) * T * C;
        std::memcpy(layer_k + static_cast<size_t>(pos) * C, k_all, static_cast<size_t>(C) * sizeof(float));
        std::memcpy(layer_v + static_cast<size_t>(pos) * C, v_all, static_cast<size_t>(C) * sizeof(float));

        for (int32_t h = 0; h < H; ++h) {
            const float* q_h = q_all + h * D;
            float max_score = -1e30f;
            for (int32_t s = 0; s <= pos; ++s) {
                const float* k_hs = layer_k + static_cast<size_t>(s) * C + h * D;
                float sc = dot_f32(q_h, k_hs, D) * inv_sqrt_d;
                work_scores_[s] = sc;
                if (sc > max_score) max_score = sc;
            }
            float sum_exp = 0.0f;
            for (int32_t s = 0; s <= pos; ++s) {
                float e = std::exp(work_scores_[s] - max_score);
                work_scores_[s] = e;
                sum_exp += e;
            }
            const float inv_sum = 1.0f / sum_exp;
            float* out_h = work_att_out_.data() + h * D;
            std::fill(out_h, out_h + D, 0.0f);
            for (int32_t s = 0; s <= pos; ++s) {
                const float w = work_scores_[s] * inv_sum;
                const float* v_hs = layer_v + static_cast<size_t>(s) * C + h * D;
                for (int32_t d = 0; d < D; ++d) {
                    out_h[d] += w * v_hs[d];
                }
            }
        }

        linear_vec(bw.proj_w_t.data(), bw.proj_b.data(), work_att_out_.data(), work_proj_.data(), C, C);
        for (int32_t i = 0; i < C; ++i) {
            work_xmid_[i] = work_x_[i] + work_proj_[i];
        }

        layernorm_vec(work_xmid_.data(), bw.ln2_g.data(), bw.ln2_b.data(), work_ln_.data(), C, cfg_.norm_eps);
        linear_vec(bw.fc_w_t.data(), bw.fc_b.data(), work_ln_.data(), work_fc_.data(), C, F);
        gelu_inplace(work_fc_.data(), F);
        linear_vec(bw.mproj_w_t.data(), bw.mproj_b.data(), work_fc_.data(), work_proj_.data(), F, C);
        for (int32_t i = 0; i < C; ++i) {
            work_x_[i] = work_xmid_[i] + work_proj_[i];
        }
    }

    if (out_logits != nullptr) {
        layernorm_vec(work_x_.data(), lnf_g_.data(), lnf_b_.data(), work_ln_.data(), C, cfg_.norm_eps);
        const float* head_w = cfg_.tie_embeddings ? wte_.data() : lm_head_.data();
        linear_vec(head_w, nullptr, work_ln_.data(), out_logits, C, V);
    }
}

bool GPTModel::forward_sequence(const int32_t* ids, int32_t T, float* out_logits, std::string& error) {
    if (!ready_) {
        error = "Model is not loaded.";
        return false;
    }
    if (!ids || !out_logits) {
        error = "Null input/output buffer passed to forward_sequence.";
        return false;
    }
    if (T < 1 || T > cfg_.block_size) {
        error = "Sequence length " + std::to_string(T) + " is outside 1.." + std::to_string(cfg_.block_size);
        return false;
    }
    for (int32_t t = 0; t < T; ++t) {
        int32_t id = ids[t];
        if (id < 0 || id >= cfg_.vocab_size) {
            error = "Token ID " + std::to_string(id) + " out of range [0, " + std::to_string(cfg_.vocab_size) + ").";
            return false;
        }
        forward_token_at(id, t, out_logits + static_cast<size_t>(t) * cfg_.vocab_size);
    }
    return true;
}

bool GPTModel::prefill_last(const int32_t* ids, int32_t T, float* out_last_logits, std::string& error) {
    if (!ready_) {
        error = "Model is not loaded.";
        return false;
    }
    if (!ids || !out_last_logits) {
        error = "Null input/output buffer passed to prefill_last.";
        return false;
    }
    if (T < 1 || T > cfg_.block_size) {
        error = "Prefill length out of range.";
        return false;
    }
    for (int32_t t = 0; t < T; ++t) {
        int32_t id = ids[t];
        if (id < 0 || id >= cfg_.vocab_size) {
            error = "Token ID out of range.";
            return false;
        }
        forward_token_at(id, t, (t == T - 1) ? out_last_logits : nullptr);
    }
    return true;
}

bool GPTModel::step_cached(int32_t token_id, int32_t pos, float* out_logits, std::string& error) {
    if (!ready_) {
        error = "Model is not loaded.";
        return false;
    }
    if (!out_logits) {
        error = "Null output buffer passed to step_cached.";
        return false;
    }
    if (pos < 0 || pos >= cfg_.block_size) {
        error = "Cached decode position out of range.";
        return false;
    }
    if (token_id < 0 || token_id >= cfg_.vocab_size) {
        error = "Token ID out of range.";
        return false;
    }
    forward_token_at(token_id, pos, out_logits);
    return true;
}

}  // namespace ember
