cmake_minimum_required(VERSION 3.22.1)
project(ember_llama LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Ember native inference: llama.cpp backend
#
# Builds a single shared library (`libember_llama.so`) that statically links
# llama.cpp + ggml, so the APK ships exactly one Ember native library and no
# separate libllama/libggml*.so to keep in sync.
#
# The pinned llama.cpp source is fetched by `scripts/fetch-llama-cpp.sh` into
# `third_party/llama.cpp` (see third_party/llama.cpp.version for the exact
# revision + checksum). Point EMBER_LLAMA_CPP_DIR somewhere else for a local
# checkout.
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(EMBER_BUILD_LLAMA_TESTS "Build the host llama.cpp engine tests" OFF)

set(EMBER_LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp"
    CACHE PATH "Path to the pinned llama.cpp source tree")

if (NOT EXISTS "${EMBER_LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp source not found at ${EMBER_LLAMA_CPP_DIR}.\n"
        "Run: scripts/fetch-llama-cpp.sh   (downloads the pinned revision listed in "
        "third_party/llama.cpp.version and verifies its SHA-256)")
endif()

# llama.cpp build configuration.
# Library-only build: no examples, tools, server, tests or the unified binary.
set(LLAMA_STANDALONE      OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP       OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD      OFF CACHE BOOL "" FORCE)
# Inference is local-only: no TLS, no HTTP client, no model downloads.
set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)
# Static so everything is folded into libember_llama.so.
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
# ggml options from llama.cpp docs/android.md: these are required for
# cross-compilation and avoid dependencies the Android NDK does not provide.
set(GGML_NATIVE           OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP           OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE        OFF CACHE BOOL "" FORCE)
set(GGML_RPC              OFF CACHE BOOL "" FORCE)
set(GGML_CUDA             OFF CACHE BOOL "" FORCE)
set(GGML_METAL            OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN           OFF CACHE BOOL "" FORCE)
set(GGML_SYCL             OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE           OFF CACHE BOOL "" FORCE)
# CPU inference is the supported path on Android; GPU backends stay opt-in.
set(GGML_CPU              ON  CACHE BOOL "" FORCE)

if (ANDROID AND ANDROID_ABI STREQUAL "x86_64")
    # Android x86_64 guarantees SSE4.2, not AVX/AVX2/FMA. A universal emulator
    # APK must not crash with SIGILL on older hosts.
    foreach(feature AVX AVX2 AVX512 FMA F16C BMI2)
        set(GGML_${feature} OFF CACHE BOOL "" FORCE)
    endforeach()
endif()

add_subdirectory("${EMBER_LLAMA_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama.cpp" EXCLUDE_FROM_ALL)

# Generate capability evidence from the source actually compiled, never a
# manually maintained model whitelist. Fail closed if upstream layout changes.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
file(GLOB EMBER_CAP_MODEL_SOURCES CONFIGURE_DEPENDS "${EMBER_LLAMA_CPP_DIR}/src/models/*.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/generate-runtime-capabilities.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp.version"
    "${EMBER_LLAMA_CPP_DIR}/src/llama-arch.cpp" "${EMBER_LLAMA_CPP_DIR}/src/llama-model.cpp"
    "${EMBER_LLAMA_CPP_DIR}/src/llama-hparams.h" "${EMBER_LLAMA_CPP_DIR}/ggml/include/ggml.h"
    "${EMBER_LLAMA_CPP_DIR}/gguf-py/gguf/constants.py" ${EMBER_CAP_MODEL_SOURCES})
execute_process(COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/generate-runtime-capabilities.py"
    --source "${EMBER_LLAMA_CPP_DIR}"
    --header "${CMAKE_CURRENT_BINARY_DIR}/generated/runtime_capabilities.h"
    RESULT_VARIABLE EMBER_CAP_RESULT)
if(NOT EMBER_CAP_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not generate runtime architecture capabilities")
endif()

add_library(ember_llama SHARED
    src/ember_llama.cpp
)

target_include_directories(ember_llama
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${EMBER_LLAMA_CPP_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_link_libraries(ember_llama PRIVATE llama)

set_target_properties(ember_llama PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if (NOT MSVC)
    target_compile_options(ember_llama PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# Only Ember's C ABI / JNI are public. Hide static upstream archive symbols;
# permits ELF dead-section elimination without changing upstream model graphs.
if (ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(ember_llama PRIVATE "-Wl,--exclude-libs,ALL" "-Wl,--gc-sections")
endif()

if (ANDROID)
    target_sources(ember_llama PRIVATE src/ember_llama_jni.cpp)
    target_compile_definitions(ember_llama PRIVATE __ANDROID__=1)
    find_library(EMBER_ANDROID_LOG log)
    if (EMBER_ANDROID_LOG)
        target_link_libraries(ember_llama PRIVATE ${EMBER_ANDROID_LOG})
    endif()
endif()

if (EMBER_BUILD_LLAMA_TESTS AND NOT ANDROID)
    enable_testing()
    add_executable(make_quantized_fixture ${CMAKE_CURRENT_SOURCE_DIR}/../tests/make_quantized_fixture.cpp)
    target_link_libraries(make_quantized_fixture PRIVATE llama)
    add_executable(test_llama_engine ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_llama_engine.cpp)
    target_link_libraries(test_llama_engine PRIVATE ember_llama)
    # Release builds define NDEBUG, which would turn every assertion in the test
    # into a no-op; the checks must stay live.
    if (NOT MSVC)
        target_compile_options(test_llama_engine PRIVATE -UNDEBUG)
    endif()
    add_test(NAME ember_llama_engine_tests COMMAND test_llama_engine)
endif()
