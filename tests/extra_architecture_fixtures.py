"""Complete random graph fixtures following pinned per-model C++ tensor inventories.

No pretrained or architecture-family certification is implied. In particular the
DeepSeek case exercises its leading-dense path, not MLA/DeepSeek2/3.
"""
import numpy as np
from ember.gguf_export import GgufWriter,_VT_STRING,_VT_I32
from ember.gpt2_tokenizer import gguf_vocab_entries
from ember.tokenizer import train_bpe


def family_fixture(path,family,quantized=False):
    arch='llama' if family=='mixtral-style' else family
    tok=train_bpe(['Hello graph fixture. Not pretrained model weights. '*8],300)
    tokens,types,merges=gguf_vocab_entries(tok.vocab,tok.merges,special_ids=(256,257,258,259,260))
    w=GgufWriter(arch);w.add_string('general.architecture',arch);w.add_string('general.name','Synthetic '+family)
    for key,v in [('context_length',256),('block_count',1),('embedding_length',64),('feed_forward_length',128),
        ('attention.head_count',2),('attention.head_count_kv',1),('rope.dimension_count',32)]:w.add_u32(arch+'.'+key,v)
    w.add_f32(arch+'.attention.layer_norm_rms_epsilon',1e-5)
    if arch in ('phi2','command-r'):w.add_f32(arch+'.attention.layer_norm_epsilon',1e-5)
    if arch=='deepseek':
        w.add_u32(arch+'.leading_dense_block_count',1)
        w.add_u32(arch+'.expert_feed_forward_length',128);w.add_u32(arch+'.expert_shared_count',1)
    if family=='mixtral-style':w.add_u32('llama.expert_count',4);w.add_u32('llama.expert_used_count',2)
    w.add_string('tokenizer.ggml.model','gpt2');w.add_string('tokenizer.ggml.pre','qwen2')
    w.add_array('tokenizer.ggml.tokens',tokens,_VT_STRING);w.add_array('tokenizer.ggml.token_type',types,_VT_I32)
    w.add_array('tokenizer.ggml.merges',merges,_VT_STRING);w.add_u32('tokenizer.ggml.eos_token_id',258)
    rng=np.random.default_rng(987)
    def tensor(name,shape,kind='random'):
        a=np.ones(shape,np.float32) if kind=='norm' else np.zeros(shape,np.float32) if kind=='bias' else (rng.standard_normal(shape)*.02).astype(np.float32)
        w.add_tensor(name,a,quantize=quantized and a.ndim>=2)
    tensor('token_embd.weight',(len(tokens),64));tensor('output_norm.weight',(64,),'norm')
    tensor('blk.0.attn_norm.weight',(64,),'norm')
    for name,shape in [('attn_q.weight',(64,64)),('attn_k.weight',(32,64)),('attn_v.weight',(32,64)),('attn_output.weight',(64,64))]:tensor('blk.0.'+name,shape)
    if arch=='phi2':
        for name,shape in [('output_norm.bias',(64,)),('output.bias',(len(tokens),)),('blk.0.attn_norm.bias',(64,)),
            ('blk.0.attn_output.bias',(64,)),('blk.0.ffn_up.bias',(128,)),('blk.0.ffn_down.bias',(64,))]:tensor(name,shape,'bias')
        tensor('output.weight',(len(tokens),64))
    elif arch!='command-r':tensor('blk.0.ffn_norm.weight',(64,),'norm')
    if family=='mixtral-style':
        tensor('blk.0.ffn_gate_inp.weight',(4,64))
        for name,shape in [('ffn_gate_exps.weight',(4,128,64)),('ffn_up_exps.weight',(4,128,64)),('ffn_down_exps.weight',(4,64,128))]:tensor('blk.0.'+name,shape)
    else:
        if arch!='phi2':tensor('blk.0.ffn_gate.weight',(128,64))
        tensor('blk.0.ffn_up.weight',(128,64));tensor('blk.0.ffn_down.weight',(64,128))
    path.write_bytes(w.build())
