/*
 * JNI bridge between the Ember Android app (Kotlin) and the llama.cpp engine.
 *
 * Ownership rules, deliberately strict:
 *   - the native handle is created/destroyed only from Kotlin, never cached
 *     across a destroy;
 *   - strings cross the boundary as UTF-8 byte arrays (never NewStringUTF, which
 *     mangles 4-byte UTF-8 such as emoji);
 *   - the streaming listener is held as a global ref only for the duration of one
 *     generation and released before returning, so a destroyed UI component can
 *     never be called back;
 *   - every entry point catches C++ exceptions and reports them as JSON errors
 *     instead of unwinding into the JVM.
 */
#if defined(__ANDROID__)

#include <jni.h>

#include <exception>
#include <string>
#include <vector>

#include "ember_llama.h"

namespace {

std::string jbytes_to_utf8(JNIEnv* env, jbyteArray arr) {
    if (!arr) return {};
    const jsize len = env->GetArrayLength(arr);
    if (len <= 0) return {};
    std::string out(static_cast<size_t>(len), '\0');
    env->GetByteArrayRegion(arr, 0, len, reinterpret_cast<jbyte*>(out.data()));
    return out;
}

jbyteArray utf8_to_jbytes(JNIEnv* env, const std::string& s) {
    jbyteArray out = env->NewByteArray(static_cast<jsize>(s.size()));
    if (!out) return nullptr;
    if (!s.empty()) {
        env->SetByteArrayRegion(out, 0, static_cast<jsize>(s.size()), reinterpret_cast<const jbyte*>(s.data()));
    }
    return out;
}

jbyteArray error_json(JNIEnv* env, int code, const std::string& message) {
    std::string escaped;
    escaped.reserve(message.size() + 8);
    for (char c : message) {
        if (c == '"' || c == '\\') escaped.push_back('\\');
        if (static_cast<unsigned char>(c) < 0x20) {
            if (c == '\n') escaped += "\\n";
            else if (c == '\r') escaped += "\\r";
            else if (c == '\t') escaped += "\\t";
            continue;
        }
        escaped.push_back(c);
    }
    const std::string payload = std::string("{\"ok\":false,\"errorCode\":") + std::to_string(code) +
                                ",\"errorName\":\"" + ember_llama_error_name(code) +
                                "\",\"error\":\"" + escaped + "\"}";
    return utf8_to_jbytes(env, payload);
}

struct ListenerContext {
    JNIEnv* env = nullptr;
    jobject listener = nullptr;  // global ref
    jmethodID method = nullptr;
    bool stop = false;
    bool broken = false;
};

struct WrappedListener {
    ListenerContext* ctx;
    ember_llama_t* engine;
};

void token_thunk(void* user_data, const char* piece, int32_t piece_len, int32_t token_id, int32_t n_generated) {
    auto* ctx = static_cast<ListenerContext*>(user_data);
    if (!ctx || !ctx->listener || !ctx->method || ctx->broken) return;
    JNIEnv* env = ctx->env;
    jbyteArray bytes = env->NewByteArray(piece_len);
    if (!bytes) {
        ctx->broken = true;
        return;
    }
    if (piece_len > 0) {
        env->SetByteArrayRegion(bytes, 0, piece_len, reinterpret_cast<const jbyte*>(piece));
    }
    const jboolean keep_going = env->CallBooleanMethod(
        ctx->listener, ctx->method, bytes, static_cast<jint>(token_id), static_cast<jint>(n_generated));
    env->DeleteLocalRef(bytes);
    if (env->ExceptionCheck()) {
        // The listener threw (for example a detached UI): stop generating rather
        // than crashing the app, and let the JVM see the exception afterwards.
        ctx->stop = true;
        ctx->broken = true;
        return;
    }
    if (!keep_going) {
        ctx->stop = true;
    }
}

/* Streams a token to Kotlin and honours a "stop generating" answer from the UI. */
void wrapped_token_cb(void* user_data, const char* piece, int32_t piece_len, int32_t token_id, int32_t n_generated) {
    auto* w = static_cast<WrappedListener*>(user_data);
    token_thunk(w->ctx, piece, piece_len, token_id, n_generated);
    if (w->ctx->stop) {
        ember_llama_cancel(w->engine);
    }
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_kilkproduct_ember_backend_LlamaInference_nativeCreate(JNIEnv*, jobject) {
    try {
        return reinterpret_cast<jlong>(ember_llama_create());
    } catch (...) {
        return 0;
    }
}

JNIEXPORT void JNICALL
Java_com_kilkproduct_ember_backend_LlamaInference_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    ember_llama_destroy(reinterpret_cast<ember_llama_t*>(handle));
}

JNIEXPORT jbyteArray JNICALL
Java_com_kilkproduct_ember_backend_LlamaInference_nativeLoad(
    JNIEnv* env, jobject, jlong handle, jbyteArray pathUtf8,
    jint nCtx, jint nBatch, jint nThreads, jint nGpuLayers, jboolean useMmap, jboolean checkTensors, jint nUbatch
) {
    try {
        auto* engine = reinterpret_cast<ember_llama_t*>(handle);
        if (!engine) return error_json(env, EMBER_LLAMA_ERR_ARGS, "Native engine handle is null.");
        const std::string path = jbytes_to_utf8(env, pathUtf8);
        if (path.empty()) return error_json(env, EMBER_LLAMA_ERR_ARGS, "Model path is empty.");

        ember_llama_config_t cfg{};
        cfg.n_ctx = nCtx;
        cfg.n_batch = nBatch;
        cfg.n_ubatch = nUbatch;
        cfg.n_threads = nThreads;
        cfg.n_gpu_layers = nGpuLayers;
        cfg.use_mmap = useMmap ? 1 : 0;
        cfg.check_tensors = checkTensors ? 1 : 0;

        const int rc = ember_llama_load(engine, path.c_str(), &cfg, nullptr, nullptr);
        if (rc != EMBER_LLAMA_OK) {
            return error_json(env, rc, ember_llama_last_error(engine));
        }

        char info[4096];
        if (ember_llama_model_info(engine, info, sizeof(info)) != EMBER_LLAMA_OK) {
            return error_json(env, ember_llama_error_code(engine), ember_llama_last_error(engine));
        }
        return utf8_to_jbytes(env, std::string("{\"ok\":true,\"model\":") + info + "}");
    } catch (const std::exception& ex) {
        return error_json(env, EMBER_LLAMA_ERR_INTERNAL, std::string("Native exception in load: ") + ex.what());
    } catch (...) {
        return error_json(env, EMBER_LLAMA_ERR_INTERNAL, "Unknown native exception in load.");
    }
}

JNIEXPORT jbyteArray JNICALL
Java_com_kilkproduct_ember_backend_LlamaInference_nativeModelInfo(JNIEnv* env, jobject, jlong handle) {
    try {
        auto* engine = reinterpret_cast<ember_llama_t*>(handle);
        if (!engine) return error_json(env, EMBER_LLAMA_ERR_ARGS, "Native engine handle is null.");
        char info[4096];
        const int rc = ember_llama_model_info(engine, info, sizeof(info));
        if (rc != EMBER_LLAMA_OK) return error_json(env, rc, ember_llama_last_error(engine));
        return utf8_to_jbytes(env, std::string("{\"ok\":true,\"model\":") + info + "}");
    } catch (const std::exception& ex) {
        return error_json(env, EMBER_LLAMA_ERR_INTERNAL, std::string("Native exception: ") + ex.what());
    } catch (...) {
        return error_json(env, EMBER_LLAMA_ERR_INTERNAL, "Unknown native exception.");
    }
}

JNIEXPORT void JNICALL
Java_com_kilkproduct_ember_backend_LlamaInference_nativeUnload(JNIEnv*, jobject, jlong handle) {
    ember_llama_unload(reinterpret_cast<ember_llama_t*>(handle));
}

JNIEXPORT jboolean JNICALL
Java_com_kilkproduct_ember_backend_LlamaInference_nativeIsLoaded(JNIEnv*, jobject, jlong handle) {
    return ember_llama_is_loaded(reinterpret_cast<ember_llama_t*>(handle)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_kilkproduct_ember_backend_LlamaInference_nativeCancel(JNIEnv*, jobject, jlong handle) {
    ember_llama_cancel(reinterpret_cast<ember_llama_t*>(handle));
}

JNIEXPORT jbyteArray JNICALL
Java_com_kilkproduct_ember_backend_LlamaInference_nativeGenerate(
    JNIEnv* env, jobject, jlong handle, jbyteArray promptUtf8,
    jint maxTokens, jfloat temperature, jint topK, jfloat topP,
    jfloat repeatPenalty, jint repeatLastN, jlong seed,
    jboolean stopOnEos, jboolean prependBos, jobject listener
) {
    try {
        auto* engine = reinterpret_cast<ember_llama_t*>(handle);
        if (!engine) return error_json(env, EMBER_LLAMA_ERR_ARGS, "Native engine handle is null.");

        const std::string prompt = jbytes_to_utf8(env, promptUtf8);

        ember_llama_gen_params_t p{};
        p.max_tokens = maxTokens;
        p.temperature = temperature;
        p.top_k = topK;
        p.top_p = topP;
        p.repeat_penalty = repeatPenalty;
        p.repeat_last_n = repeatLastN;
        p.seed = static_cast<uint32_t>(seed);
        p.stop_on_eos = stopOnEos ? 1 : 0;
        p.prepend_bos = prependBos ? 1 : 0;

        ListenerContext ctx;
        ember_llama_token_cb cb = nullptr;
        jobject global_listener = nullptr;
        if (listener) {
            jclass cls = env->GetObjectClass(listener);
            if (cls) {
                ctx.method = env->GetMethodID(cls, "onToken", "([BII)Z");
                env->DeleteLocalRef(cls);
            }
            if (ctx.method) {
                global_listener = env->NewGlobalRef(listener);
                ctx.env = env;
                ctx.listener = global_listener;
                cb = token_thunk;
            }
        }

        // The streaming listener can stop generation by returning false; the
        // wrapper converts that into the engine's atomic cancel flag.
        std::vector<char> out(256 * 1024);
        int rc = EMBER_LLAMA_OK;
        if (cb) {
            WrappedListener wrapped{&ctx, engine};
            rc = ember_llama_generate(engine, prompt.data(), prompt.size(), &p,
                                      wrapped_token_cb, &wrapped, out.data(), out.size());
        } else {
            rc = ember_llama_generate(engine, prompt.data(), prompt.size(), &p,
                                      nullptr, nullptr, out.data(), out.size());
        }

        if (global_listener) {
            env->DeleteGlobalRef(global_listener);
        }
        if (ctx.broken) {
            // Clear a pending listener exception before making any more JNI calls;
            // return a structured failure after the global reference is released.
            if (env->ExceptionCheck()) env->ExceptionClear();
            return error_json(env, EMBER_LLAMA_ERR_INTERNAL, "Token listener failed during generation.");
        }

        // rc is carried inside the JSON payload (errorCode / errorName / stoppedBy),
        // so Kotlin always sees one structured result object.
        (void) rc;
        return utf8_to_jbytes(env, std::string(out.data()));
    } catch (const std::exception& ex) {
        return error_json(env, EMBER_LLAMA_ERR_INTERNAL, std::string("Native exception in generate: ") + ex.what());
    } catch (...) {
        return error_json(env, EMBER_LLAMA_ERR_INTERNAL, "Unknown native exception in generate.");
    }
}

JNIEXPORT jbyteArray JNICALL
Java_com_kilkproduct_ember_backend_LlamaInference_nativeLogs(JNIEnv* env, jobject) {
    try {
        std::vector<char> buf(64 * 1024);
        const int rc = ember_llama_logs(buf.data(), buf.size());
        if (rc != EMBER_LLAMA_OK) return utf8_to_jbytes(env, "[]");
        return utf8_to_jbytes(env, std::string(buf.data()));
    } catch (...) {
        return utf8_to_jbytes(env, "[]");
    }
}

JNIEXPORT jbyteArray JNICALL
Java_com_kilkproduct_ember_backend_LlamaInference_nativeBackendInfo(JNIEnv* env, jobject) {
    const char* info = ember_llama_backend_info();
    return utf8_to_jbytes(env, std::string("{\"ok\":true,\"backend\":\"") + (info ? info : "") + "\"}");
}

}  // extern "C"

#endif  // __ANDROID__
