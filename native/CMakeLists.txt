cmake_minimum_required(VERSION 3.22.1)
project(ember_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(EMBER_BUILD_TESTS "Build standalone C++ native tests" ON)

add_library(ember_infer SHARED
    src/bpe.cpp
    src/model.cpp
    src/engine.cpp
    src/jni_bridge.cpp
)

target_include_directories(ember_infer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (MSVC)
    target_compile_options(ember_infer PRIVATE /O2 /W4)
else()
    target_compile_options(ember_infer PRIVATE -O3 -Wall -Wextra -fno-fast-math)
endif()

if (ANDROID)
    find_library(android_log_lib log)
    target_link_libraries(ember_infer PRIVATE ${android_log_lib})
else()
    find_package(Threads REQUIRED)
    target_link_libraries(ember_infer PRIVATE Threads::Threads)
endif()

if (EMBER_BUILD_TESTS AND NOT ANDROID)
    enable_testing()
    add_executable(test_native tests/test_native.cpp)
    target_link_libraries(test_native PRIVATE ember_infer Threads::Threads)
    add_test(NAME ember_native_tests COMMAND test_native)
endif()
