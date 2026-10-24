"""Complete tiny native graph fixtures, random weights, NOT pretrained models.

Tensor names/shapes follow pinned src/models/{mamba,qwen3moe}.cpp. No missing
layers, metadata rewriting at import, graph substitutions, or inference mocks.
"""
import numpy as np
from ember.gguf_export import GgufWriter,_VT_STRING,_VT_I32
from ember.gpt2_tokenizer import gguf_vocab_entries
from ember.tokenizer import train_bpe


def advanced_fixture(path,arch,quantized=False):
    tok=train_bpe(['Hello world. Architecture integration fixture only. '*10],300)
    tokens,types,merges=gguf_vocab_entries(tok.vocab,tok.merges,special_ids=(256,257,258,259,260))
    w=GgufWriter(arch)
    w.add_string('general.architecture',arch)
    w.add_string('general.name','Synthetic '+arch+' graph, NOT pretrained')
    dim=32 if arch=='mamba' else 64
    for suffix,value in [('context_length',256),('embedding_length',dim),('block_count',1)]:
        w.add_u32(arch+'.'+suffix,value)
    w.add_f32(arch+'.attention.layer_norm_rms_epsilon',1e-5)
    w.add_string('tokenizer.ggml.model','gpt2');w.add_string('tokenizer.ggml.pre','qwen2')
    w.add_array('tokenizer.ggml.tokens',tokens,_VT_STRING)
    w.add_array('tokenizer.ggml.token_type',types,_VT_I32)
    w.add_array('tokenizer.ggml.merges',merges,_VT_STRING)
    w.add_u32('tokenizer.ggml.bos_token_id',257);w.add_u32('tokenizer.ggml.eos_token_id',258)
    rng=np.random.default_rng(123)
    def tensor(name,shape,ones=False,negative=False,quant=True):
        a=np.ones(shape,np.float32) if ones else (rng.standard_normal(shape)*.02).astype(np.float32)
        if negative:a=-np.ones(shape,np.float32)
        w.add_tensor(name,a,quantize=quantized and quant and a.ndim>=2)
    tensor('token_embd.weight',(len(tokens),dim));tensor('output_norm.weight',(dim,),ones=True)
    tensor('blk.0.attn_norm.weight',(dim,),ones=True)
    if arch=='mamba':
        for key,value in [('ssm.conv_kernel',4),('ssm.inner_size',64),('ssm.state_size',16),('ssm.time_step_rank',8)]:
            w.add_u32(arch+'.'+key,value)
        for name,shape in [('ssm_in.weight',(128,32)),('ssm_conv1d.weight',(64,4)),
            ('ssm_conv1d.bias',(64,)),('ssm_x.weight',(40,64)),('ssm_dt.weight',(64,8)),
            ('ssm_dt.bias',(64,)),('ssm_out.weight',(32,64))]:tensor('blk.0.'+name,shape)
        tensor('blk.0.ssm_a',(64,16),negative=True,quant=False)
        tensor('blk.0.ssm_d',(64,),ones=True)
    elif arch=='qwen3moe':
        for key,value in [('attention.head_count',2),('attention.head_count_kv',1),
            ('rope.dimension_count',32),('feed_forward_length',128),('expert_count',2),
            ('expert_used_count',1),('expert_feed_forward_length',128)]:w.add_u32(arch+'.'+key,value)
        for name,shape in [('attn_q.weight',(64,64)),('attn_k.weight',(32,64)),
            ('attn_v.weight',(32,64)),('attn_output.weight',(64,64)),('ffn_gate_inp.weight',(2,64)),
            ('ffn_gate_exps.weight',(2,128,64)),('ffn_up_exps.weight',(2,128,64)),
            ('ffn_down_exps.weight',(2,64,128))]:tensor('blk.0.'+name,shape)
        for name,shape in [('attn_q_norm.weight',(32,)),('attn_k_norm.weight',(32,)),('ffn_norm.weight',(64,))]:
            tensor('blk.0.'+name,shape,ones=True)
    else:raise ValueError('No fixture implementation for '+arch)
    path.write_bytes(w.build())
