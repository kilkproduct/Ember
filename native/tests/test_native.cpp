#include "ember/c_api.h"
#include "ember/engine.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

uint32_t crc32_buf(const std::vector<uint8_t>& buf) {
    uint32_t crc = 0xFFFFFFFFU;
    for (uint8_t b : buf) {
        crc ^= b;
        for (int k = 0; k < 8; ++k) {
            crc = (crc & 1U) ? (0xEDB88320U ^ (crc >> 1)) : (crc >> 1);
        }
    }
    return crc ^ 0xFFFFFFFFU;
}

void push_u32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFFU));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFU));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFU));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFU));
}

void finalize_emb(std::vector<uint8_t>& out) {
    uint32_t crc = crc32_buf(out);
    push_u32(out, crc);
}

void test_pretokenizer_rules() {
    std::string sample = "  hello,   world!\nEmoji: \xF0\x9F\x94\xA5\xF0\x9F\x8E\x89   ";
    auto chunks = ember::pretokenize_bytes(reinterpret_cast<const uint8_t*>(sample.data()), sample.size());
    assert(!chunks.empty());
    assert(chunks[0] == " ");
    assert(chunks[1] == " hello");
    assert(chunks[2] == ",");

    // Concatenation of all pre-tokenizer chunks must always equal original bytes
    std::string rebuilt;
    for (const auto& c : chunks) rebuilt += c;
    assert(rebuilt == sample);
}

void test_malformed_overflow_and_nan_model_rejection() {
    ember::GPTModel model;
    std::string err;

    // 1. Empty / tiny
    uint8_t tiny[4] = {'E', 'M', 'B', '3'};
    assert(!model.load_from_bytes(tiny, sizeof(tiny), err));
    assert(!err.empty());

    // 2. Legacy JSON brain clearly identified
    std::string legacy = "{\"format\":\"ember-brain-v2\",\"weights\":{}}";
    assert(!model.load_from_bytes(reinterpret_cast<const uint8_t*>(legacy.data()), legacy.size(), err));
    assert(err.find("Legacy JSON brain") != std::string::npos);

    // 3. Adversarial integer-overflow dimensions inside a valid-CRC32 .emb container
    //    dims = [65536, 65536, 4] -> wraps uint64 product to 2^64 on unguarded multiplication!
    {
        std::vector<uint8_t> adv = {'E', 'M', 'B', '3'};
        push_u32(adv, 3);  // version
        std::string hdr = "{\"arch\":\"ember-gpt\",\"n_layer\":2,\"n_head\":2,\"n_embd\":64,\"block_size\":64,\"vocab_size\":300}";
        push_u32(adv, static_cast<uint32_t>(hdr.size()));
        adv.insert(adv.end(), hdr.begin(), hdr.end());
        push_u32(adv, 1);  // tensor_count
        std::string name = "wte";
        push_u32(adv, static_cast<uint32_t>(name.size()));
        adv.insert(adv.end(), name.begin(), name.end());
        push_u32(adv, 0);  // float32
        push_u32(adv, 3);  // ndim = 3
        push_u32(adv, 65536U);
        push_u32(adv, 65536U);
        push_u32(adv, 4U);
        // meta_bytes = 8 + 12 = 20 -> pad = 12
        adv.insert(adv.end(), 12, 0);
        finalize_emb(adv);

        err.clear();
        assert(!model.load_from_bytes(adv.data(), adv.size(), err));
        assert(err.find("overflow") != std::string::npos || err.find("truncated") != std::string::npos);
        assert(!model.ready());
    }

    // 4. NaN weight inside a valid-CRC32 .emb container must be rejected cleanly on load
    {
        std::vector<uint8_t> nan_emb = {'E', 'M', 'B', '3'};
        push_u32(nan_emb, 3);
        std::string hdr = "{\"arch\":\"ember-gpt\",\"n_layer\":2,\"n_head\":2,\"n_embd\":16,\"block_size\":8,\"vocab_size\":264}";
        push_u32(nan_emb, static_cast<uint32_t>(hdr.size()));
        nan_emb.insert(nan_emb.end(), hdr.begin(), hdr.end());
        push_u32(nan_emb, 1);
        std::string name = "wte";
        push_u32(nan_emb, static_cast<uint32_t>(name.size()));
        nan_emb.insert(nan_emb.end(), name.begin(), name.end());
        push_u32(nan_emb, 0);
        push_u32(nan_emb, 2);
        push_u32(nan_emb, 264U);
        push_u32(nan_emb, 16U);
        // meta_bytes = 8 + 8 = 16 -> pad = 0
        std::vector<float> vals(264 * 16, 0.01f);
        vals[7] = std::numeric_limits<float>::quiet_NaN();
        const auto* raw_vals = reinterpret_cast<const uint8_t*>(vals.data());
        nan_emb.insert(nan_emb.end(), raw_vals, raw_vals + vals.size() * sizeof(float));
        finalize_emb(nan_emb);

        err.clear();
        assert(!model.load_from_bytes(nan_emb.data(), nan_emb.size(), err));
        assert(err.find("non-finite") != std::string::npos);
        assert(!model.ready());
    }
}

void test_utf8_invalid_byte_decoding_safety() {
    // Build minimal valid v3 tokenizer JSON (261 entries: 256 raw bytes + 5 specials)
    std::ostringstream oss;
    oss << "{\"version\":3,\"specials\":{\"pad\":256,\"bos\":257,\"eos\":258,\"doc\":259,\"unk\":260},\"merges\":[],\"vocab\":[";
    const std::string hex = "0123456789abcdef";
    for (int i = 0; i < 256; ++i) {
        if (i > 0) oss << ",";
        oss << "\"\\u00" << hex[(i >> 4) & 0xF] << hex[i & 0xF] << "\"";
    }
    oss << ",\"<pad>\",\"<bos>\",\"<eos>\",\"<doc>\",\"<unk>\",\"ab\"]}";

    ember::BPETokenizer tok;
    std::string err;
    assert(tok.load_from_json(oss.str(), err));

    // Lone 0x80, truncated 4-byte emoji [0xF0, 0x9F, 0x94] + 'A', surrogate [0xED, 0xA0, 0x80], embedded NUL
    int32_t seq[] = {0x80, 0xF0, 0x9F, 0x94, 'A', 0xED, 0xA0, 0x80, 0x00, 'Z'};
    std::string decoded = tok.decode(seq, sizeof(seq) / sizeof(seq[0]), true);
    // Expected: \uFFFD + \uFFFD + 'A' + \uFFFD + \uFFFD + \uFFFD + '\0' + 'Z' = 5*3 + 3 = 18 bytes
    std::string expected = "\xEF\xBF\xBD\xEF\xBF\xBD" "A" "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD" + std::string("\0Z", 2);
    assert(decoded == expected);
}

std::string minimal_tokenizer_json() {
    std::ostringstream oss;
    oss << "{\"version\":3,\"specials\":{\"pad\":256,\"bos\":257,\"eos\":258,\"doc\":259,\"unk\":260},"
        << "\"merges\":[[\"a\",\"b\"],[\"ab\",\"c\"]],\"vocab\":[";
    const std::string hex = "0123456789abcdef";
    for (int i = 0; i < 256; ++i) {
        if (i > 0) oss << ",";
        oss << "\"\\u00" << hex[(i >> 4) & 0xF] << hex[i & 0xF] << "\"";
    }
    oss << ",\"<pad>\",\"<bos>\",\"<eos>\",\"<doc>\",\"<unk>\",\"ab\",\"abc\"]}";
    return oss.str();
}

void test_concurrent_encode_cache_is_thread_safe() {
    // `BPETokenizer::encode_chunk` memoizes into a mutable cache. One tokenizer is
    // shared by the JNI generation thread and the Python ctypes bridge, so that
    // cache must be guarded: an unsynchronized rehash corrupts the hash table and
    // crashes mid-generation (verified with ThreadSanitizer).
    //
    // The workload deliberately encodes strings that are NOT cached yet, so every
    // call takes the contended *write* path. Pre-warming the cache would make this
    // test read-only and blind to the race.
    const std::string json = minimal_tokenizer_json();
    ember::BPETokenizer tok;
    std::string err;
    // Explicit check (not assert) so the tokenizer is still loaded when the suite
    // is compiled with -DNDEBUG; an empty tokenizer would silently skip the race.
    if (!tok.load_from_json(json, err)) {
        std::cerr << "concurrency test could not load tokenizer: " << err << "\n";
        std::abort();
    }

    std::atomic<int> mismatches{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&, t]() {
            // An independent tokenizer gives the reference answer without touching
            // the shared instance under test.
            ember::BPETokenizer reference;
            std::string ref_err;
            if (!reference.load_from_json(json, ref_err)) {
                mismatches.fetch_add(1);
                return;
            }
            for (int i = 0; i < 400; ++i) {
                std::string s = "abc abc hello world " + std::to_string(t) + "-" + std::to_string(i);
                if (tok.encode_str(s) != reference.encode_str(s)) {
                    mismatches.fetch_add(1);
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    assert(mismatches.load() == 0);

    // The shared tokenizer must still be usable and correct after the contention.
    ember::BPETokenizer reference;
    std::string ref_err;
    if (!reference.load_from_json(json, ref_err)) {
        std::cerr << "concurrency test could not load reference tokenizer: " << ref_err << "\n";
        std::abort();
    }
    assert(tok.encode_str("abc world abc") == reference.encode_str("abc world abc"));
}

void test_c_api_lifecycle_and_errors() {
    ember_engine_t* eng = ember_engine_create();
    assert(eng != nullptr);
    assert(ember_engine_is_ready(eng) == 0);
    assert(ember_engine_load(eng, "/nonexistent/model.emb", "/nonexistent/tok.json") != 0);
    std::string err = ember_engine_last_error(eng);
    assert(!err.empty());

    // Calling generate before loading must fail cleanly without crashing
    int32_t ids[16];
    char buf[128];
    assert(ember_engine_generate(eng, "hi", 2, 10, 0.0f, 1, 1.0f, 1.0f, 42, 1, ids, 16, buf, 128) < 0);

    // The summary accessor must be safe on a fresh/failed handle and reject a
    // buffer that is too small instead of writing past it.
    char summary[512];
    int slen = ember_engine_last_generate_summary(eng, summary, sizeof(summary));
    assert(slen > 0);
    assert(std::string(summary).find("\"stoppedBy\":") != std::string::npos);
    assert(ember_engine_last_generate_summary(eng, summary, 4) == -1);
    assert(ember_engine_last_generate_summary(nullptr, summary, sizeof(summary)) == -1);
    assert(ember_engine_last_generate_summary(eng, nullptr, sizeof(summary)) == -1);

    ember_engine_destroy(eng);
}

void test_splitmix64_determinism() {
    ember::SplitMix64 a(123456789ULL);
    ember::SplitMix64 b(123456789ULL);
    for (int i = 0; i < 100; ++i) {
        assert(a.next_u64() == b.next_u64());
    }
    double u = a.next_f64();
    assert(u >= 0.0 && u < 1.0);
}

}  // namespace

int main(int argc, char** argv) {
    test_pretokenizer_rules();
    test_malformed_overflow_and_nan_model_rejection();
    test_utf8_invalid_byte_decoding_safety();
    test_concurrent_encode_cache_is_thread_safe();
    test_c_api_lifecycle_and_errors();
    test_splitmix64_determinism();

    // Optional fixture parity + concurrent cancellation test if model + tokenizer paths passed on CLI
    if (argc == 3) {
        ember_engine_t* eng = ember_engine_create();
        int rc = ember_engine_load(eng, argv[1], argv[2]);
        if (rc != 0) {
            std::cerr << "Fixture load failed: " << ember_engine_last_error(eng) << "\n";
            ember_engine_destroy(eng);
            return 1;
        }
        assert(ember_engine_is_ready(eng) == 1);
        assert(ember_engine_vocab_size(eng) > 261);

        // Repeated generation + reload cleanup test
        for (int iter = 0; iter < 5; ++iter) {
            assert(ember_engine_load(eng, argv[1], argv[2]) == 0);
            int32_t out_ids[32];
            char out_txt[512];
            int n = ember_engine_generate(
                eng,
                "Alice looked at the stars",
                25,
                20,
                0.0f,
                1,
                1.0f,
                1.1f,
                99ULL,
                1,
                out_ids,
                32,
                out_txt,
                sizeof(out_txt)
            );
            assert(n == 20);

            // The generate call returns only a token count; the summary accessor
            // is how callers learn the stop reason and any truncation warning.
            char summary[1024];
            int slen = ember_engine_last_generate_summary(eng, summary, sizeof(summary));
            assert(slen > 0);
            std::string sjson(summary);
            assert(sjson.find("\"stoppedBy\":\"length\"") != std::string::npos);
            assert(sjson.find("\"newTokens\":20") != std::string::npos);
            assert(sjson.find("\"promptTokens\":") != std::string::npos);
            assert(sjson.find("\"warning\":\"") != std::string::npos);

            // A prompt longer than the context window must be reported, not silently cut.
            std::string long_prompt(4000, 'a');
            int n2 = ember_engine_generate(
                eng,
                long_prompt.data(),
                long_prompt.size(),
                4,
                0.0f,
                1,
                1.0f,
                1.0f,
                5ULL,
                1,
                out_ids,
                32,
                out_txt,
                sizeof(out_txt)
            );
            assert(n2 == 4);
            slen = ember_engine_last_generate_summary(eng, summary, sizeof(summary));
            assert(slen > 0);
            sjson = std::string(summary);
            if (ember_engine_block_size(eng) < 1000) {
                assert(sjson.find("exceeded context window") != std::string::npos);
            }
        }

        // A too-small summary buffer must fail cleanly instead of overflowing.
        char tiny[8];
        assert(ember_engine_last_generate_summary(eng, tiny, sizeof(tiny)) == -1);

        // Thread-safe mid-flight cancellation test
        std::atomic<bool> started{false};
        std::thread canceller([&]() {
            while (!started.load()) {
                std::this_thread::yield();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ember_engine_cancel(eng);
        });
        started.store(true);
        std::vector<int32_t> big_ids(4096);
        std::vector<char> big_txt(16384);
        int cancelled_n = ember_engine_generate(
            eng,
            "Alice",
            5,
            4096,
            0.8f,
            20,
            0.9f,
            1.1f,
            7ULL,
            0,
            big_ids.data(),
            big_ids.size(),
            big_txt.data(),
            big_txt.size()
        );
        canceller.join();
        assert(cancelled_n >= 0 && cancelled_n < 4096);

        ember_engine_destroy(eng);
    }

    std::cout << "All C++ native inference unit tests passed.\n";
    return 0;
}
