#include "ember/c_api.h"
#include "ember/engine.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

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

void test_malformed_model_rejection() {
    ember::GPTModel model;
    std::string err;

    // Empty / tiny
    uint8_t tiny[4] = {'E', 'M', 'B', '3'};
    assert(!model.load_from_bytes(tiny, sizeof(tiny), err));
    assert(!err.empty());

    // Legacy JSON brain clearly identified
    std::string legacy = "{\"format\":\"ember-brain-v2\",\"weights\":{}}";
    assert(!model.load_from_bytes(reinterpret_cast<const uint8_t*>(legacy.data()), legacy.size(), err));
    assert(err.find("Legacy JSON brain") != std::string::npos);

    // Bad magic
    std::string garbage(64, 'X');
    assert(!model.load_from_bytes(reinterpret_cast<const uint8_t*>(garbage.data()), garbage.size(), err));
    assert(!model.ready());
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
    test_malformed_model_rejection();
    test_c_api_lifecycle_and_errors();
    test_splitmix64_determinism();

    // Optional fixture parity run if model + tokenizer paths passed on CLI
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
            assert(n >= 0);
        }
        ember_engine_destroy(eng);
    }

    std::cout << "All C++ native inference unit tests passed.\n";
    return 0;
}
