"""Actual upstream quantization + actual native decode, not header-only claims.

These are explicit host-test conversions. Import never converts weights.
Importance-matrix-only IQ encoders are not certified by these fixtures.
"""
import re,subprocess
from pathlib import Path
import pytest
from ember.gguf_export import export_gguf,read_gguf_header,_runtime_manifest
from ember.model import ModelConfig,Llama
from ember.tokenizer import train_bpe
from ember.inference import LlamaCppEngine,EngineConfig,GenParams

@pytest.fixture(scope='module')
def quant_source(tmp_path_factory):
    path=tmp_path_factory.mktemp('quant-types')/'f32.gguf'
    tok=train_bpe(['Hello world. Quantization kernel integration fixture. '*10],300)
    cfg=ModelConfig(n_layer=1,n_head=4,n_kv_head=2,n_embd=256,block_size=128,vocab_size=tok.vocab_size)
    export_gguf(path,cfg,Llama(cfg,seed=17).params,tok)
    return path

@pytest.mark.parametrize('mode,tensor_type',[
    ('MOSTLY_F16','F16'),('MOSTLY_BF16','BF16'),('MOSTLY_Q4_0','Q4_0'),('MOSTLY_Q4_1','Q4_1'),
    ('MOSTLY_Q5_0','Q5_0'),('MOSTLY_Q5_1','Q5_1'),('MOSTLY_Q8_0','Q8_0'),
    ('MOSTLY_Q2_K','Q2_K'),('MOSTLY_Q3_K_S','Q3_K'),('MOSTLY_Q4_K_S','Q4_K'),
    ('MOSTLY_Q5_K_S','Q5_K'),('MOSTLY_Q6_K','Q6_K'),('MOSTLY_IQ4_NL','IQ4_NL'),('MOSTLY_IQ4_XS','IQ4_XS'),
    ('MOSTLY_TQ1_0','TQ1_0'),('MOSTLY_TQ2_0','TQ2_0'),
    ('MOSTLY_Q1_0','Q1_0'),('MOSTLY_Q2_0','Q2_0')])
def test_upstream_quantization_kernel_and_generation(quant_source,tmp_path,mode,tensor_type):
    header=Path('third_party/llama.cpp/include/llama.h').read_text()
    ftype=int(re.search(r'LLAMA_FTYPE_'+mode+r'\s*=\s*(\d+)',header).group(1))
    path=tmp_path/'quantized.gguf'
    result=subprocess.run(['native/build-llama/make_quantized_fixture',str(quant_source),str(path),str(ftype)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr[-4000:]
    h=read_gguf_header(path);layouts=_runtime_manifest()['tensorTypes']
    assert tensor_type in {layouts[str(t.dtype)]['name'] for t in h.tensors}
    e=LlamaCppEngine()
    try:
        e.load_model(path,EngineConfig(n_ctx=128,n_batch=32,n_ubatch=16,n_threads=2))
        out=e.generate('Hello',GenParams(max_tokens=4,temperature=0,stop_on_eos=False))
        assert out.new_tokens==4
    finally:e.close()


def test_nvfp4_encoder_limitation_is_not_mistaken_for_inference_support(quant_source,tmp_path):
    # The enum/layout exists, but this pinned CPU quantizer rejects its output
    # file type. This does NOT establish whether supplied NVFP4 weights can run.
    result=subprocess.run(['native/build-llama/make_quantized_fixture',str(quant_source),str(tmp_path/'nvfp4'), '39'],capture_output=True,text=True)
    assert result.returncode!=0 and 'invalid output file type' in result.stderr
