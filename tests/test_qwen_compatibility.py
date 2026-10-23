"""Small valid dense Qwen graph fixtures; NOT downloaded/pretrained Qwen models.

Qwen2.5 shares qwen2 GGUF architecture. Qwen3 adds per-head Q/K RMS gains.
Tokenizer metadata is deliberately distinct from Ember's training tokenizer.
"""
import numpy as np
import pytest
from ember.gguf_export import GgufWriter,ember_to_llama_tensors,validate_gguf_file,_VT_STRING,_VT_I32
from ember.gpt2_tokenizer import gguf_vocab_entries
from ember.tokenizer import train_bpe
from ember.model import ModelConfig,Llama
from ember.inference import LlamaCppEngine,EngineConfig,GenParams


def qwen_fixture(path,arch='qwen2',name='Qwen2 test graph',quantized=False,head_dim=32):
    tok=train_bpe(['Hello world. A small deterministic fixture. '*20],vocab_size=300)
    tokens,types,merges=gguf_vocab_entries(tok.vocab,tok.merges,special_ids=(256,257,258,259,260))
    start=len(tokens);end=start+1
    tokens += ['<|im_start|>','<|im_end|>'];types += [3,3]
    cfg=ModelConfig(n_layer=1,n_head=2,n_kv_head=1,n_embd=64,block_size=2048,vocab_size=len(tokens))
    model=Llama(cfg,seed=42)
    tensors=ember_to_llama_tensors(cfg,model.params)
    if arch=='qwen3':
        tensors['blk.0.attn_q_norm.weight']=np.ones(head_dim,np.float32)
        tensors['blk.0.attn_k_norm.weight']=np.ones(head_dim,np.float32)
        if head_dim!=32:
            rng=np.random.default_rng(7)
            for suffix,shape in [('q',(2*head_dim,64)),('k',(head_dim,64)),('v',(head_dim,64)),('output',(64,2*head_dim))]:
                tensors['blk.0.attn_'+suffix+'.weight']=(rng.standard_normal(shape)*.01).astype(np.float32)
    else:
        tensors['blk.0.attn_q.bias']=np.zeros(64,np.float32)
        tensors['blk.0.attn_k.bias']=np.zeros(32,np.float32)
        tensors['blk.0.attn_v.bias']=np.zeros(32,np.float32)
    w=GgufWriter(arch)
    w.add_string('general.architecture',arch);w.add_string('general.name',name)
    for suffix,value in [('context_length',2048),('embedding_length',64),('block_count',1),
        ('feed_forward_length',256),('attention.head_count',2),('attention.head_count_kv',1),('rope.dimension_count',head_dim)]:
        w.add_u32(arch+'.'+suffix,value)
    w.add_u32(arch+'.attention.key_length',head_dim);w.add_u32(arch+'.attention.value_length',head_dim)
    w.add_f32(arch+'.attention.layer_norm_rms_epsilon',1e-6)
    w.add_f32(arch+'.rope.freq_base',1000000)
    w.add_string('tokenizer.ggml.model','gpt2');w.add_string('tokenizer.ggml.pre','qwen2')
    w.add_array('tokenizer.ggml.tokens',tokens,_VT_STRING);w.add_array('tokenizer.ggml.token_type',types,_VT_I32)
    w.add_array('tokenizer.ggml.merges',merges,_VT_STRING)
    w.add_u32('tokenizer.ggml.eos_token_id',end)
    w.add_bool('tokenizer.ggml.add_bos_token',False)
    w.add_string('tokenizer.chat_template',"{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
    for k,v in tensors.items():w.add_tensor(k,v,quantize=quantized and v.ndim==2)
    path.write_bytes(w.build())
    return start,end


@pytest.mark.parametrize('quantized',[False,True])
@pytest.mark.parametrize('arch,name',[('qwen2','Qwen2'),('qwen2','Qwen2.5'),('qwen3','Qwen3')])
def test_dense_qwen_load_template_context_stream_cancel_and_recovery(tmp_path,arch,name,quantized):
    path=tmp_path/(name+'.gguf');start,end=qwen_fixture(path,arch,name,quantized)
    info=validate_gguf_file(path)
    assert info['arch']==arch and info['contextLength']==2048 and info['tokenizerPre']=='qwen2'
    e=LlamaCppEngine()
    try:
        loaded=e.load_model(path,EngineConfig(n_ctx=2048,n_batch=64,n_ubatch=16,n_threads=2))
        assert loaded.n_ctx>=2048 and loaded.eos_token_id==end
        assert e.tokenize('<|im_end|>',parse_special=True)==[end]
        chunks=[]
        def callback(piece,tid,n):chunks.append(piece);return n<3
        result=e.generate('Hello world',GenParams(max_tokens=10,temperature=0,stop_on_eos=False),callback)
        assert result.new_tokens==3 and result.stopped_by=='cancelled'
        rendered='<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n'
        assert result.prompt_tokens==len(e.tokenize(rendered,parse_special=True))
        assert e.generate('Hello',GenParams(max_tokens=4,temperature=0,stop_on_eos=False)).new_tokens==4
        e.unload();assert not e.is_loaded()
    finally:e.close()


@pytest.mark.parametrize('quantized',[False,True])
def test_qwen3_independent_head_dimensions(tmp_path,quantized):
    path=tmp_path/'independent.gguf';qwen_fixture(path,'qwen3',quantized=quantized,head_dim=64)
    assert validate_gguf_file(path)['arch']=='qwen3'
    e=LlamaCppEngine()
    try:
        e.load_model(path,EngineConfig(n_ctx=2048,n_batch=64,n_ubatch=16,n_threads=2))
        assert e.generate('Hello',GenParams(max_tokens=3,temperature=0,stop_on_eos=False)).new_tokens==3
    finally:e.close()
