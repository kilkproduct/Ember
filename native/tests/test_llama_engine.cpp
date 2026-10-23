/*
 * Native engine tests for the llama.cpp-backed inference path.
 *
 * Usage:
 *   test_llama_engine [model.gguf] [not-a-gguf-file]
 *
 * Without arguments it still verifies the failure paths (missing file, non-GGUF
 * bytes, no model loaded, buffer handling). With a real GGUF fixture — produced
 * by `scripts/make_test_gguf.py` from Ember's own exporter — it verifies load,
 * tokenizer round-trips, deterministic generation, cancellation, repeated
 * load/unload cycles and resource cleanup (run under ASan/LSan in CI).
 */
#include "ember_llama.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <future>
#include <thread>
#include <vector>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

/* Extract the value of "text":"..." (unescaping \" and \\) from a result payload. */
std::string json_text_field(const std::string& payload) {
    const std::string key = "\"text\":\"";
    const size_t start = payload.find(key);
    if (start == std::string::npos) return {};
    std::string out;
    for (size_t i = start + key.size(); i < payload.size(); ++i) {
        const char c = payload[i];
        if (c == '\\' && i + 1 < payload.size()) {
            const char n = payload[++i];
            switch (n) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                default: out.push_back(n);
            }
            continue;
        }
        if (c == '"') break;
        out.push_back(c);
    }
    return out;
}

/* Extract the raw "tokenIds":[...] array text. */
std::string json_token_ids_field(const std::string& payload) {
    const std::string key = "\"tokenIds\":[";
    const size_t start = payload.find(key);
    if (start == std::string::npos) return {};
    const size_t end = payload.find(']', start);
    if (end == std::string::npos) return {};
    return payload.substr(start + key.size(), end - start - key.size());
}

void test_failure_paths_without_a_model() {
    ember_llama_t* engine = ember_llama_create();
    assert(engine != nullptr);
    assert(ember_llama_is_loaded(engine) == 0);
    assert(ember_llama_error_code(engine) == EMBER_LLAMA_OK);

    // Generating without a model must fail cleanly, not crash.
    std::vector<char> out(64 * 1024);
    ember_llama_gen_params_t p{};
    p.max_tokens = 8;
    p.temperature = 0.0f;
    p.top_k = 1;
    p.stop_on_eos = 1;
    p.prepend_bos = 1;
    p.seed = 1234;
    int rc = ember_llama_generate(engine, "hello", 5, &p, nullptr, nullptr, out.data(), out.size());
    assert(rc == EMBER_LLAMA_ERR_NOT_LOADED);
    std::string payload(out.data());
    assert(contains(payload, "\"ok\":false"));
    assert(contains(payload, "model-not-loaded"));

    // Tokenizer access also requires a loaded model.
    int32_t ids[16];
    assert(ember_llama_tokenize(engine, "hello", 5, 0, ids, 16) == EMBER_LLAMA_ERR_NOT_LOADED);
    char text[64];
    assert(ember_llama_detokenize(engine, ids, 1, 1, text, sizeof(text)) == EMBER_LLAMA_ERR_NOT_LOADED);

    // A missing model file is reported as "not found", not as a generic failure.
    rc = ember_llama_load(engine, "/nonexistent/ember-model.gguf", nullptr, nullptr, nullptr);
    assert(rc == EMBER_LLAMA_ERR_MODEL_NOT_FOUND);
    assert(contains(ember_llama_last_error(engine), "not found"));
    assert(std::string(ember_llama_error_name(rc)) == "model-not-found");

    // Null arguments are rejected.
    assert(ember_llama_load(nullptr, "/tmp/x.gguf", nullptr, nullptr, nullptr) == EMBER_LLAMA_ERR_ARGS);
    assert(ember_llama_load(engine, nullptr, nullptr, nullptr, nullptr) == EMBER_LLAMA_ERR_ARGS);
    assert(ember_llama_generate(nullptr, "hi", 2, &p, nullptr, nullptr, out.data(), out.size()) == EMBER_LLAMA_ERR_ARGS);
    assert(ember_llama_model_info(engine, out.data(), out.size()) == EMBER_LLAMA_ERR_NOT_LOADED);

    // Cancelling an idle engine is a no-op, and unloading too.
    ember_llama_cancel(engine);
    ember_llama_unload(engine);
    assert(ember_llama_is_loaded(engine) == 0);

    // Logs are always a JSON array, even before any model was loaded.
    std::vector<char> logs(64 * 1024);
    assert(ember_llama_logs(logs.data(), logs.size()) == EMBER_LLAMA_OK);
    assert(logs[0] == '[');
    assert(ember_llama_backend_info() != nullptr);

    ember_llama_destroy(engine);
    // Destroying twice / null handles must be safe.
    ember_llama_destroy(nullptr);
    ember_llama_cancel(nullptr);
    ember_llama_unload(nullptr);
}

void test_non_gguf_file_is_rejected(const std::string& path) {
    ember_llama_t* engine = ember_llama_create();
    assert(engine != nullptr);
    const int rc = ember_llama_load(engine, path.c_str(), nullptr, nullptr, nullptr);
    assert(rc == EMBER_LLAMA_ERR_MODEL_INVALID);
    const std::string err = ember_llama_last_error(engine);
    assert(!err.empty());
    // The message must explain what to do instead of dumping a native trace.
    assert(contains(err, "GGUF") || contains(err, "gguf"));
    ember_llama_destroy(engine);
}

void test_too_small_file_is_rejected(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    out << "GGUF";  // right magic, absurdly short
    out.close();
    ember_llama_t* engine = ember_llama_create();
    assert(ember_llama_load(engine, path.c_str(), nullptr, nullptr, nullptr) == EMBER_LLAMA_ERR_MODEL_INVALID);
    ember_llama_destroy(engine);
    std::remove(path.c_str());
}

struct CancelState {
    ember_llama_t* engine = nullptr;
    std::atomic<int> tokens{0};
    std::atomic<bool> signalled{false};
    std::atomic<int> cancel_after{0};   // 0 = do not cancel from inside the loop
    std::promise<void>* started = nullptr;
};

void cancel_cb(void* user, const char* /*piece*/, int32_t /*len*/, int32_t /*token*/, int32_t n_generated) {
    auto* s = static_cast<CancelState*>(user);
    s->tokens.store(n_generated);
    if (s->started && !s->signalled.exchange(true)) {
        s->started->set_value();   // tell the waiting thread that generation is in flight
    }
    if (s->cancel_after > 0 && n_generated >= s->cancel_after) {
        ember_llama_cancel(s->engine);
    }
}

struct StreamState {
    std::string text;
    int pieces = 0;
    int last_token = -1;
    std::atomic<bool> stop_after{false};
};

void stream_cb(void* user, const char* piece, int32_t len, int32_t token_id, int32_t n_generated) {
    auto* s = static_cast<StreamState*>(user);
    s->text.append(piece, static_cast<size_t>(len));
    s->pieces = n_generated;
    s->last_token = token_id;
}

void test_model_lifecycle_and_generation(const std::string& model_path) {
    ember_llama_t* engine = ember_llama_create();
    assert(engine != nullptr);

    // A conservative mobile configuration.
    ember_llama_config_t cfg{};
    cfg.n_ctx = 128;
    cfg.n_batch = 64;
    cfg.n_threads = 2;
    cfg.n_gpu_layers = 0;
    cfg.use_mmap = 1;
    cfg.check_tensors = 1;

    std::atomic<int> progress_calls{0};
    struct ProgressCtx { std::atomic<int>* calls; } pctx{&progress_calls};
    auto progress_cb = [](void* user, float p) {
        auto* c = static_cast<ProgressCtx*>(user);
        if (p >= 0.0f && p <= 1.0f) c->calls->fetch_add(1);
    };

    const int rc = ember_llama_load(engine, model_path.c_str(), &cfg, progress_cb, &pctx);
    if (rc != EMBER_LLAMA_OK) {
        std::cerr << "load failed: " << ember_llama_last_error(engine) << "\n";
    }
    assert(rc == EMBER_LLAMA_OK);
    assert(ember_llama_is_loaded(engine) == 1);

    // ---- model info -------------------------------------------------------
    std::vector<char> info(8192);
    assert(ember_llama_model_info(engine, info.data(), info.size()) == EMBER_LLAMA_OK);
    const std::string info_json(info.data());
    assert(contains(info_json, "\"arch\":\"gpt2\""));
    assert(contains(info_json, "\"loaded\":true"));
    // The fixture is trained with a 64-token context; asking for 128 must be
    // clamped down, because a learned-positional model would otherwise index
    // position rows past the end of position_embd.weight.
    assert(contains(info_json, "\"nCtxTrain\":64"));
    assert(contains(info_json, "\"nCtx\":64"));
    assert(contains(info_json, "\"vocabSize\":"));
    assert(contains(info_json, "\"eosTokenId\":"));
    assert(contains(info_json, "\"suppressedTokens\":"));

    // ---- tokenizer round trip --------------------------------------------
    const std::string sample = "The tide rises and the tide falls. It's a test—1234 \xF0\x9F\x94\xA5";
    int32_t ids[512];
    int32_t n_ids = ember_llama_tokenize(engine, sample.c_str(), sample.size(), 0, ids, 512);
    assert(n_ids > 0);
    std::vector<char> back(4096);
    int32_t n_chars = ember_llama_detokenize(engine, ids, static_cast<size_t>(n_ids), 1, back.data(), back.size());
    assert(n_chars > 0);
    const std::string round_trip(back.data(), static_cast<size_t>(n_chars));
    if (round_trip != sample) {
        std::cerr << "round trip mismatch:\n  in : " << sample << "\n  out: " << round_trip << "\n";
    }
    assert(round_trip == sample);

    // Ember exports `tokenizer.ggml.add_bos_token = false` and prepends <bos>
    // explicitly during generation, so add_special must NOT insert a second BOS.
    int32_t ids_special[512];
    int32_t n_special = ember_llama_tokenize(engine, sample.c_str(), sample.size(), 1, ids_special, 512);
    assert(n_special == n_ids);
    assert(contains(info_json, "\"addBos\":false"));
    assert(contains(info_json, "\"addEos\":false"));

    // ---- deterministic greedy generation ----------------------------------
    ember_llama_gen_params_t p{};
    p.max_tokens = 24;
    p.temperature = 0.0f;   // greedy
    p.top_k = 1;
    p.top_p = 1.0f;
    p.repeat_penalty = 1.0f;
    p.repeat_last_n = 0;
    p.seed = 4242;
    p.stop_on_eos = 0;
    p.prepend_bos = 1;

    std::vector<char> out(256 * 1024);
    assert(ember_llama_generate(engine, "The tide", 8, &p, nullptr, nullptr, out.data(), out.size()) == EMBER_LLAMA_OK);
    const std::string first(out.data());
    assert(contains(first, "\"ok\":true"));
    assert(contains(first, "\"newTokens\":24"));
    assert(contains(first, "\"stoppedBy\":\"length\""));
    assert(contains(first, "\"promptTokens\":"));

    // Same prompt + greedy decoding must reproduce the same tokens and text
    // exactly. Timing fields legitimately differ, so compare only the output.
    assert(ember_llama_generate(engine, "The tide", 8, &p, nullptr, nullptr, out.data(), out.size()) == EMBER_LLAMA_OK);
    const std::string second(out.data());
    if (json_text_field(first) != json_text_field(second) ||
        json_token_ids_field(first) != json_token_ids_field(second)) {
        std::cerr << "greedy generation is not deterministic:\n  a: " << json_text_field(first)
                  << "\n  b: " << json_text_field(second) << "\n";
    }
    assert(json_text_field(first) == json_text_field(second));
    assert(json_token_ids_field(first) == json_token_ids_field(second));
    assert(!json_token_ids_field(first).empty());

    // ---- streaming ---------------------------------------------------------
    StreamState stream;
    assert(ember_llama_generate(engine, "The tide", 8, &p, stream_cb, &stream, out.data(), out.size()) ==
           EMBER_LLAMA_OK);
    assert(stream.pieces > 0);
    const std::string streamed_json(out.data());
    // The streamed pieces must reconstruct the same text as the final result.
    const std::string streamed = stream.text;
    assert(!streamed.empty());
    assert(contains(streamed_json, "\"ok\":true"));

    // ---- zero-token request is a valid no-op -------------------------------
    ember_llama_gen_params_t zero = p;
    zero.max_tokens = 0;
    assert(ember_llama_generate(engine, "The tide", 8, &zero, nullptr, nullptr, out.data(), out.size()) ==
           EMBER_LLAMA_OK);
    assert(contains(std::string(out.data()), "\"newTokens\":0"));

    // ---- oversized prompt is truncated with a warning -----------------------
    const std::string huge(4000, 'a');
    assert(ember_llama_generate(engine, huge.c_str(), huge.size(), &p, nullptr, nullptr, out.data(), out.size()) ==
           EMBER_LLAMA_OK);
    const std::string truncated(out.data());
    assert(contains(truncated, "exceeded the context window"));

    // ---- output buffer too small -------------------------------------------
    char tiny[64];
    assert(ember_llama_generate(engine, "The tide", 8, &p, nullptr, nullptr, tiny, sizeof(tiny)) ==
           EMBER_LLAMA_ERR_BUFFER);
    assert(contains(std::string(tiny), "\"ok\":false"));

    // ---- cancellation from inside the generation loop -----------------------
    // Deterministic: the callback cancels once six tokens exist, so the run is
    // guaranteed to be in flight (a sleep-based cancel races a tiny model that
    // can finish, or hit the context limit, before the timer fires).
    ember_llama_gen_params_t sampling = p;
    sampling.max_tokens = 4096;
    sampling.temperature = 0.8f;   // exercises the dist sampler path too
    sampling.top_k = 40;
    sampling.top_p = 0.95f;
    sampling.repeat_penalty = 1.1f;
    sampling.repeat_last_n = 64;

    CancelState inner{engine, {}, {}, 6, nullptr};
    assert(ember_llama_generate(engine, "The tide rises", 14, &sampling, cancel_cb, &inner,
                                out.data(), out.size()) == EMBER_LLAMA_ERR_CANCELLED);
    const std::string inner_cancelled(out.data());
    if (!contains(inner_cancelled, "\"stoppedBy\":\"cancelled\"")) {
        std::cerr << "in-loop cancellation not reported: " << inner_cancelled.substr(0, 200) << "\n";
    }
    assert(contains(inner_cancelled, "\"stoppedBy\":\"cancelled\""));
    assert(contains(inner_cancelled, "\"ok\":true"));  // a cancelled run is a normal result
    assert(inner.tokens.load() >= 6);

    // ---- cancellation from another thread (the UI "Stop" path) --------------
    std::promise<void> started;
    CancelState cross{engine, {}, {}, 0, &started};
    std::vector<char> long_out(512 * 1024);
    std::atomic<bool> finished{false};
    std::thread worker([&]() {
        ember_llama_generate(engine, "The tide rises", 14, &sampling, cancel_cb, &cross,
                             long_out.data(), long_out.size());
        finished.store(true);
    });
    started.get_future().wait();      // generation has produced its first token
    ember_llama_cancel(engine);       // cancel from the "UI" thread
    worker.join();
    assert(finished.load());
    const std::string cancelled(long_out.data());
    if (!contains(cancelled, "\"stoppedBy\":\"cancelled\"")) {
        std::cerr << "cross-thread cancellation not reported: " << cancelled.substr(0, 200) << "\n";
    }
    assert(contains(cancelled, "\"stoppedBy\":\"cancelled\""));

    // The engine stays usable after a cancelled run.
    assert(ember_llama_generate(engine, "The tide", 8, &p, nullptr, nullptr, out.data(), out.size()) == EMBER_LLAMA_OK);
    assert(contains(std::string(out.data()), "\"ok\":true"));

    // ---- repeated load/unload cycles ---------------------------------------
    for (int i = 0; i < 3; ++i) {
        ember_llama_unload(engine);
        assert(ember_llama_is_loaded(engine) == 0);
        assert(ember_llama_load(engine, model_path.c_str(), &cfg, nullptr, nullptr) == EMBER_LLAMA_OK);
        assert(ember_llama_is_loaded(engine) == 1);
        assert(ember_llama_generate(engine, "Stars rise", 10, &p, nullptr, nullptr, out.data(), out.size()) ==
               EMBER_LLAMA_OK);
    }

    ember_llama_unload(engine);
    ember_llama_destroy(engine);
}

}  // namespace

int main(int argc, char** argv) {
    test_failure_paths_without_a_model();
    std::cout << "  failure paths: ok\n";

    if (argc >= 3) {
        test_non_gguf_file_is_rejected(argv[2]);
        std::cout << "  non-GGUF rejection: ok\n";
    }

    if (argc >= 2) {
        const std::string tmp = std::string(argv[1]) + ".tiny";
        test_too_small_file_is_rejected(tmp);
        std::cout << "  truncated-file rejection: ok\n";
        test_model_lifecycle_and_generation(argv[1]);
        std::cout << "  model lifecycle + generation: ok\n";
    } else {
        std::cout << "  (no GGUF fixture supplied; skipped model tests)\n";
    }

    std::cout << "All llama.cpp native inference tests passed.\n";
    return 0;
}
