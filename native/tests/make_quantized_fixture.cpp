// Host-test ONLY. Explicit fixture conversion with upstream kernels, never an
// import code path and never linked into the APK. Input must be an F32 fixture.
#include "llama.h"
#include <cstdlib>
#include <cstdio>
int main(int argc,char** argv) {
    if(argc!=4) {std::fprintf(stderr,"usage: make_quantized_fixture input output llama_ftype\n");return 2;}
    llama_backend_init();
    auto p=llama_model_quantize_default_params();
    p.nthread=2;p.ftype=static_cast<llama_ftype>(std::atoi(argv[3]));
    p.allow_requantize=false;p.pure=true;p.max_buf_size=16*1024*1024;
    const auto result=llama_model_quantize(argv[1],argv[2],&p);
    llama_backend_free();
    return result;
}
