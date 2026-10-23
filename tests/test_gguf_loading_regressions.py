"""Real native load regressions; fixtures are generated locally, no downloads."""
import os
import struct
import pytest
from ember.gguf_export import export_gguf, read_gguf_header, validate_gguf_file
from ember.inference import LlamaCppEngine, EngineConfig, GenParams, InferenceError
from ember.model import ModelConfig, Llama
from ember.tokenizer import train_bpe


def fixture(path):
    tok=train_bpe(['A small real GGUF fixture for loading and tokenization. '*10],300)
    cfg=ModelConfig(n_layer=1,n_head=2,n_embd=32,block_size=64,vocab_size=tok.vocab_size)
    export_gguf(path,cfg,Llama(cfg,seed=4).params,tok)
    return path


def test_metadata_beyond_old_64_mib_prefix_loads(tmp_path):
    source=fixture(tmp_path/'small.gguf')
    original=source.read_bytes()
    # Insert a valid, ignored metadata string. Its serialized size is aligned so
    # tensor offsets and the pre-existing data padding remain correct.
    size=65*1024*1024
    key=b'fixture.padding'
    prefix=struct.pack('<Q',len(key))+key+struct.pack('<I',8)
    length=size-len(prefix)-8
    p=tmp_path/'large-header.gguf'
    header=bytearray(original[:24])
    struct.pack_into('<Q',header,16,struct.unpack_from('<Q',header,16)[0]+1)
    with p.open('wb') as f:
        f.write(header);f.write(prefix);f.write(struct.pack('<Q',length))
        block=b'x'*(1024*1024)
        while length:
            part=min(len(block),length);f.write(block[:part]);length-=part
        f.write(original[24:])
    assert validate_gguf_file(p)['arch']=='llama'
    e=LlamaCppEngine()
    try:
        assert e.load_model(p,EngineConfig(n_ctx=64)).loaded
        assert e.generate('A small',GenParams(max_tokens=2,temperature=0,stop_on_eos=False)).new_tokens==2
    finally:e.close()


@pytest.mark.skipif(not os.path.isdir('/proc/self/fd'),reason='Linux/Android descriptor path')
def test_seekable_saf_descriptor_equivalent_load_and_reload(tmp_path):
    p=fixture(tmp_path/'model.gguf');e=LlamaCppEngine()
    try:
        with p.open('rb') as f:
            path=f'/proc/self/fd/{f.fileno()}'
            # Selection/metadata reads must not disturb subsequent native opens.
            f.seek(123)
            assert validate_gguf_file(path)['arch']=='llama'
            for _ in range(3):
                assert e.load_model(path,EngineConfig(n_ctx=64)).loaded
                assert e.generate('A small',GenParams(max_tokens=2,temperature=0,stop_on_eos=False)).new_tokens==2
                e.unload();assert not e.is_loaded()
        with pytest.raises(InferenceError):e.load_model(path)
        assert e.load_model(p,EngineConfig(n_ctx=64)).loaded
    finally:e.close()


def test_upstream_tensor_failure_survives_error_boundary(tmp_path):
    p=fixture(tmp_path/'corrupt.gguf')
    header=read_gguf_header(p)
    with p.open('r+b') as f:
        f.seek(header.data_start+header.tensors[0].offset)
        f.write(struct.pack('<f',float('nan')))
    e=LlamaCppEngine()
    try:
        with pytest.raises(InferenceError,match='Upstream:') as exc:
            e.load_model(p,EngineConfig(n_ctx=64,check_tensors=True))
        assert 'tensor' in str(exc.value).lower()
        assert not e.is_loaded()
        valid=fixture(tmp_path/'valid.gguf')
        assert e.load_model(valid,EngineConfig(n_ctx=64)).loaded
    finally:e.close()
