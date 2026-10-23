#if defined(__ANDROID__) || defined(EMBER_BUILD_JNI)

#include <jni.h>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "ember/engine.hpp"

namespace {

std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 0x20) {
                    oss << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c) << std::dec;
                } else {
                    oss << static_cast<char>(c);
                }
        }
    }
    return oss.str();
}

std::string jbytes_to_std(JNIEnv* env, jbyteArray arr) {
    if (!arr) return {};
    jsize len = env->GetArrayLength(arr);
    if (len <= 0) return {};
    std::string out(static_cast<size_t>(len), '\0');
    env->GetByteArrayRegion(arr, 0, len, reinterpret_cast<jbyte*>(out.data()));
    return out;
}

jbyteArray std_to_jbytes(JNIEnv* env, const std::string& s) {
    jbyteArray out = env->NewByteArray(static_cast<jsize>(s.size()));
    if (!out) return nullptr;
    if (!s.empty()) {
        env->SetByteArrayRegion(out, 0, static_cast<jsize>(s.size()), reinterpret_cast<const jbyte*>(s.data()));
    }
    return out;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_kilkproduct_ember_backend_NativeInference_nativeCreate(JNIEnv*, jobject) {
    try {
        return reinterpret_cast<jlong>(new ember::Engine());
    } catch (...) {
        return 0;
    }
}

JNIEXPORT void JNICALL
Java_com_kilkproduct_ember_backend_NativeInference_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    auto* eng = reinterpret_cast<ember::Engine*>(handle);
    delete eng;
}

JNIEXPORT jbyteArray JNICALL
Java_com_kilkproduct_ember_backend_NativeInference_nativeLoad(
    JNIEnv* env,
    jobject,
    jlong handle,
    jbyteArray modelPathUtf8,
    jbyteArray tokPathUtf8
) {
    try {
        auto* eng = reinterpret_cast<ember::Engine*>(handle);
        if (!eng) {
            return std_to_jbytes(env, "{\"ok\":false,\"error\":\"Native inference handle is null.\"}");
        }
        std::string mpath = jbytes_to_std(env, modelPathUtf8);
        std::string tpath = jbytes_to_std(env, tokPathUtf8);
        if (!eng->load(mpath, tpath)) {
            return std_to_jbytes(env, "{\"ok\":false,\"error\":\"" + json_escape(eng->last_error()) + "\"}");
        }
        std::ostringstream oss;
        oss << "{\"ok\":true,\"vocabSize\":" << eng->vocab_size()
            << ",\"blockSize\":" << eng->block_size() << "}";
        return std_to_jbytes(env, oss.str());
    } catch (const std::exception& ex) {
        return std_to_jbytes(env, "{\"ok\":false,\"error\":\"" + json_escape(std::string("Native exception in load: ") + ex.what()) + "\"}");
    } catch (...) {
        return std_to_jbytes(env, "{\"ok\":false,\"error\":\"Unknown native exception in load.\"}");
    }
}

JNIEXPORT void JNICALL
Java_com_kilkproduct_ember_backend_NativeInference_nativeCancel(JNIEnv*, jobject, jlong handle) {
    auto* eng = reinterpret_cast<ember::Engine*>(handle);
    if (eng) eng->cancel();
}

JNIEXPORT jboolean JNICALL
Java_com_kilkproduct_ember_backend_NativeInference_nativeIsReady(JNIEnv*, jobject, jlong handle) {
    auto* eng = reinterpret_cast<ember::Engine*>(handle);
    return (eng && eng->is_ready()) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jbyteArray JNICALL
Java_com_kilkproduct_ember_backend_NativeInference_nativeGenerate(
    JNIEnv* env,
    jobject,
    jlong handle,
    jbyteArray promptUtf8,
    jint maxNewTokens,
    jfloat temperature,
    jint topK,
    jfloat topP,
    jfloat repetitionPenalty,
    jlong seed,
    jboolean stopOnEos
) {
    try {
        auto* eng = reinterpret_cast<ember::Engine*>(handle);
        if (!eng) {
            return std_to_jbytes(env, "{\"ok\":false,\"error\":\"Native inference engine is not initialized.\"}");
        }
        std::string prompt = jbytes_to_std(env, promptUtf8);
        ember::GenerateOptions opts;
        opts.max_new_tokens = static_cast<int32_t>(maxNewTokens);
        opts.temperature = temperature;
        opts.top_k = static_cast<int32_t>(topK);
        opts.top_p = topP;
        opts.repetition_penalty = repetitionPenalty;
        opts.seed = static_cast<uint64_t>(seed);
        opts.stop_on_eos = (stopOnEos == JNI_TRUE);

        std::vector<int32_t> ids;
        ember::GenerateOutput summary;
        if (!eng->generate(
                reinterpret_cast<const uint8_t*>(prompt.data()),
                prompt.size(),
                opts,
                ids,
                summary)) {
            return std_to_jbytes(env, "{\"ok\":false,\"error\":\"" + json_escape(eng->last_error()) + "\"}");
        }

        std::ostringstream meta;
        meta << summary.new_tokens << " tokens · "
             << std::fixed << std::setprecision(1) << summary.tokens_per_sec
             << " tok/s · C++ native";

        std::ostringstream oss;
        oss << "{\"ok\":true"
            << ",\"text\":\"" << json_escape(summary.text) << "\""
            << ",\"meta\":\"" << json_escape(meta.str()) << "\""
            << ",\"newTokens\":" << summary.new_tokens
            << ",\"promptTokens\":" << summary.prompt_tokens
            << ",\"tokensPerSec\":" << std::fixed << std::setprecision(3) << summary.tokens_per_sec
            << ",\"ms\":" << std::fixed << std::setprecision(3) << summary.elapsed_ms
            << ",\"seed\":" << static_cast<int64_t>(opts.seed)
            << ",\"stoppedBy\":\"" << json_escape(summary.stopped_by) << "\"";
        if (!summary.warning.empty()) {
            oss << ",\"warning\":\"" << json_escape(summary.warning) << "\"";
        }
        oss << "}";
        return std_to_jbytes(env, oss.str());
    } catch (const std::exception& ex) {
        return std_to_jbytes(env, "{\"ok\":false,\"error\":\"" + json_escape(std::string("Native exception in generate: ") + ex.what()) + "\"}");
    } catch (...) {
        return std_to_jbytes(env, "{\"ok\":false,\"error\":\"Unknown native exception in generate.\"}");
    }
}

}  // extern "C"

#endif
