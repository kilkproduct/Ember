#include "ember/engine.hpp"

#include <fstream>
#include <limits>
#include <sstream>

#include "json_min.hpp"

namespace ember {

namespace {

constexpr int kClassSpace = 0;
constexpr int kClassWord = 1;
constexpr int kClassPunct = 2;

inline int byte_class(uint8_t b) {
    if (b == 0x20 || (b >= 0x09 && b <= 0x0D)) {
        return kClassSpace;
    }
    if ((b >= 0x30 && b <= 0x39) ||
        (b >= 0x41 && b <= 0x5A) ||
        (b >= 0x61 && b <= 0x7A) ||
        b == 0x5F ||
        b >= 0x80) {
        return kClassWord;
    }
    return kClassPunct;
}

inline std::string make_pair_key(const std::string& a, const std::string& b) {
    uint32_t la = static_cast<uint32_t>(a.size());
    std::string key;
    key.reserve(4 + a.size() + b.size());
    key.push_back(static_cast<char>((la >> 24) & 0xFF));
    key.push_back(static_cast<char>((la >> 16) & 0xFF));
    key.push_back(static_cast<char>((la >> 8) & 0xFF));
    key.push_back(static_cast<char>(la & 0xFF));
    key.append(a);
    key.append(b);
    return key;
}

}  // namespace

std::vector<std::string> pretokenize_bytes(const uint8_t* data, size_t len) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < len) {
        int cls = byte_class(data[i]);
        if (cls == kClassSpace) {
            size_t j = i;
            while (j < len && byte_class(data[j]) == kClassSpace) {
                ++j;
            }
            if (j < len) {
                for (size_t k = i; k + 1 < j; ++k) {
                    out.emplace_back(reinterpret_cast<const char*>(data + k), 1);
                }
                size_t start = j - 1;
                int cls2 = byte_class(data[j]);
                size_t k = j;
                while (k < len && byte_class(data[k]) == cls2) {
                    ++k;
                }
                out.emplace_back(reinterpret_cast<const char*>(data + start), k - start);
                i = k;
            } else {
                for (size_t k = i; k < j; ++k) {
                    out.emplace_back(reinterpret_cast<const char*>(data + k), 1);
                }
                i = j;
            }
        } else {
            size_t j = i;
            while (j < len && byte_class(data[j]) == cls) {
                ++j;
            }
            out.emplace_back(reinterpret_cast<const char*>(data + i), j - i);
            i = j;
        }
    }
    return out;
}

bool BPETokenizer::load_from_file(const std::string& path, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Could not open tokenizer file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return load_from_json(ss.str(), error);
}

bool BPETokenizer::load_from_json(const std::string& json_text, std::string& error) {
    try {
        auto root = json::parse(json_text);
        if (root.type != json::Type::Object) {
            error = "Tokenizer file must be a JSON object.";
            return false;
        }
        int64_t ver = root.int_or("version", 0);
        if (ver != 3) {
            error = "Tokenizer version " + std::to_string(ver) + " is incompatible with Ember v3.";
            return false;
        }
        const auto& specials = root.at("specials");
        if (specials.int_or("pad", -1) != kSpecialPad ||
            specials.int_or("bos", -1) != kSpecialBos ||
            specials.int_or("eos", -1) != kSpecialEos ||
            specials.int_or("doc", -1) != kSpecialDoc ||
            specials.int_or("unk", -1) != kSpecialUnk) {
            error = "Tokenizer special token IDs do not match Ember v3 specification.";
            return false;
        }
        const auto& v_arr = root.at("vocab");
        if (v_arr.type != json::Type::Array || static_cast<int32_t>(v_arr.arr.size()) < kFirstMergeId) {
            error = "Tokenizer vocabulary is incomplete (< 261 entries).";
            return false;
        }

        vocab_.clear();
        token_to_id_.clear();
        pair_ranks_.clear();
        cache_.clear();

        vocab_.reserve(v_arr.arr.size());
        for (size_t i = 0; i < v_arr.arr.size(); ++i) {
            const std::string& raw = v_arr.arr[i].latin1;
            vocab_.push_back(raw);
            token_to_id_[raw] = static_cast<int32_t>(i);
        }

        const auto& m_arr = root.at("merges");
        if (m_arr.type == json::Type::Array) {
            for (size_t i = 0; i < m_arr.arr.size(); ++i) {
                const auto& pair = m_arr.arr[i];
                if (pair.type == json::Type::Array && pair.arr.size() == 2) {
                    pair_ranks_[make_pair_key(pair.arr[0].latin1, pair.arr[1].latin1)] =
                        static_cast<int32_t>(i);
                }
            }
        }
        return true;
    } catch (const std::exception& ex) {
        error = std::string("Malformed tokenizer JSON: ") + ex.what();
        return false;
    }
}

std::vector<int32_t> BPETokenizer::encode_chunk(const std::string& chunk) const {
    auto it_cache = cache_.find(chunk);
    if (it_cache != cache_.end()) {
        return it_cache->second;
    }

    std::vector<std::string> word;
    word.reserve(chunk.size());
    for (size_t i = 0; i < chunk.size(); ++i) {
        word.emplace_back(chunk.data() + i, 1);
    }

    while (word.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        std::string best_first;
        std::string best_second;
        bool found = false;

        for (size_t i = 0; i + 1 < word.size(); ++i) {
            auto it = pair_ranks_.find(make_pair_key(word[i], word[i + 1]));
            if (it != pair_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_first = word[i];
                best_second = word[i + 1];
                found = true;
            }
        }
        if (!found) {
            break;
        }

        std::vector<std::string> merged;
        merged.reserve(word.size());
        size_t i = 0;
        while (i < word.size()) {
            if (i + 1 < word.size() && word[i] == best_first && word[i + 1] == best_second) {
                merged.push_back(best_first + best_second);
                i += 2;
            } else {
                merged.push_back(std::move(word[i]));
                i += 1;
            }
        }
        word.swap(merged);
    }

    std::vector<int32_t> ids;
    ids.reserve(word.size());
    for (const auto& sym : word) {
        auto it = token_to_id_.find(sym);
        ids.push_back(it != token_to_id_.end() ? it->second : kSpecialUnk);
    }

    if (cache_.size() < 65536) {
        cache_.emplace(chunk, ids);
    }
    return ids;
}

std::vector<int32_t> BPETokenizer::encode(const uint8_t* data, size_t len) const {
    std::vector<int32_t> out;
    if (len == 0 || !ready()) {
        return out;
    }
    auto chunks = pretokenize_bytes(data, len);
    for (const auto& chunk : chunks) {
        auto sub = encode_chunk(chunk);
        out.insert(out.end(), sub.begin(), sub.end());
    }
    return out;
}

std::string BPETokenizer::decode(const int32_t* ids, size_t n, bool skip_special) const {
    std::string out;
    for (size_t i = 0; i < n; ++i) {
        int32_t id = ids[i];
        if (id < 0 || id >= static_cast<int32_t>(vocab_.size())) {
            continue;
        }
        if (skip_special && id >= kBaseVocab && id < kFirstMergeId) {
            continue;
        }
        out.append(vocab_[id]);
    }
    return out;
}

}  // namespace ember
