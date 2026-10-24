"""Architecture registry, structural preflight and actual graph/registration gates."""
import json
import struct
from pathlib import Path
from types import SimpleNamespace
import pytest
from ember.backend import EmberBackend
from ember.gguf_export import validate_gguf_file,read_gguf_header,GgufExportError,_dtype_nbytes
from ember.inference import LlamaCppEngine,EngineConfig,GenParams,InferenceError,get_engine
from test_qwen_compatibility import qwen_fixture
from architecture_fixtures import advanced_fixture


def test_capabilities_are_from_compiled_pin_and_all_upstream_factories():
    e=LlamaCppEngine()
    try:
        caps=e.capabilities()
        assert caps==json.loads(Path('ember/runtime_capabilities.json').read_text())
        rows={r['architecture']:r for r in caps['architectures']}
        assert len(rows)>100
        for arch in ['qwen35','qwen35moe','mamba','mamba2','rwkv6','deepseek2','qwen3moe','gpt2','t5','pangu-embedded']:
            assert rows[arch]['nativeLoaderImplemented'] and rows[arch]['emberTextCandidate']
        assert rows['qwen35']['modelKind']=='hybrid'
        assert rows['qwen3moe']['moeGraphAvailable']
        assert not rows['bert']['emberTextCandidate']
        assert not rows['gemma4-assistant']['emberTextCandidate']
        for dtype,layout in caps['tensorTypes'].items():
            assert _dtype_nbytes(int(dtype),(layout['block'],))==layout['bytes']
        assert all(not r['textInferenceVerified'] for r in rows.values())
    finally:e.close()


@pytest.mark.parametrize('arch',['mamba','qwen3moe'])
@pytest.mark.parametrize('quantized',[False,True])
def test_real_recurrent_and_moe_graph_load_generate_cancel_switch(tmp_path,arch,quantized):
    path=tmp_path/(arch+'.gguf');advanced_fixture(path,arch,quantized)
    info=validate_gguf_file(path)
    assert not info['textInferenceVerified'] and not info['nativeValidated']
    assert info['estimatedKvBytesPerToken'] is None
    e=LlamaCppEngine()
    try:
        model=e.load_model(path,EngineConfig(n_ctx=256,n_batch=32,n_ubatch=16,n_threads=2))
        assert model.recurrent==(arch=='mamba')
        assert e.generate('Hello',GenParams(max_tokens=5,temperature=0,stop_on_eos=False)).new_tokens==5
        result=e.generate('Hello',GenParams(max_tokens=8,temperature=0,stop_on_eos=False),lambda text,tid,n:n<2)
        assert result.new_tokens==2 and result.stopped_by=='cancelled'
        e.unload();assert not e.is_loaded()
        assert e.load_model(path,EngineConfig(n_ctx=256,n_batch=32,n_ubatch=16,n_threads=2)).loaded
        assert e.generate('Hello',GenParams(max_tokens=3,temperature=0,stop_on_eos=False)).new_tokens==3
    finally:e.close()


def test_qwen35_is_recognized_but_incomplete_graph_is_not_registered(tmp_path):
    # Intentionally WRONG graph: proving the old gate is gone is not proof of
    # Qwen3.5 inference. Actual upstream must reject missing SSM/rope tensors.
    path=tmp_path/'incomplete.gguf';qwen_fixture(path,'qwen35')
    info=validate_gguf_file(path)
    assert info['arch']=='qwen35' and not info['textInferenceVerified']
    backend=EmberBackend(tmp_path/'app')
    try:
        with pytest.raises(InferenceError,match='Upstream:') as failure:
            backend.dispatch('model.importGguf',{'path':str(path)})
        assert 'qwen35' in str(failure.value) and 'supported dense' not in str(failure.value)
        assert not list(backend.trainer.library_dir.glob('*.gguf'))
        assert backend.trainer._imported_gguf is None and not get_engine().is_loaded()
    finally:backend.close()


def test_first_token_failure_and_cancel_do_not_publish_or_change_selection(tmp_path,monkeypatch):
    path=tmp_path/'good.gguf';qwen_fixture(path)
    b=EmberBackend(tmp_path/'app');engine=get_engine()
    try:
        good=b.dispatch('model.importGguf',{'path':str(path),'name':'existing'})
        assert good['textInferenceVerified'] and good['nativeValidated']
        selected=b.trainer._imported_gguf
        original=engine.generate
        monkeypatch.setattr(engine,'generate',lambda *a,**k:SimpleNamespace(new_tokens=0))
        with pytest.raises(InferenceError,match='first-token'):
            b.dispatch('model.importGguf',{'path':str(path),'name':'unready'})
        assert b.trainer._imported_gguf==selected
        assert len(list(b.trainer.library_dir.glob('*.gguf')))==1
        assert not engine.is_loaded()
        def cancelled(*args,**kwargs):
            b.trainer._model_import_cancel.set()
            return SimpleNamespace(new_tokens=1)
        monkeypatch.setattr(engine,'generate',cancelled)
        with pytest.raises(InterruptedError):b.dispatch('model.importGguf',{'path':str(path),'name':'cancelled'})
        assert len(list(b.trainer.library_dir.glob('*.gguf')))==1 and b.trainer._imported_gguf==selected
        monkeypatch.setattr(engine,'generate',original)
        good=b.dispatch('model.importGguf',{'path':str(path),'name':'verified','nCtx':512})
        assert good['validationStage']=='first-token-verified'
        assert not engine.is_loaded()  # validation handle reused then released before JNI load
    finally:b.close()


def test_unknown_architecture_and_unknown_quantization_are_actionable(tmp_path):
    path=tmp_path/'unknown.gguf';qwen_fixture(path,'not-a-runtime-architecture')
    with pytest.raises(GgufExportError,match='architecture-recognition'):validate_gguf_file(path)
    path=tmp_path/'quant.gguf';qwen_fixture(path)
    data=bytearray(path.read_bytes());tensor=read_gguf_header(path).tensors[0]
    name=tensor.name.encode();start=data.index(struct.pack('<Q',len(name))+name,24)
    pos=start+8+len(name)
    ndim=struct.unpack_from('<I',data,pos)[0]
    struct.pack_into('<I',data,pos+4+8*ndim,999)
    path.write_bytes(data)
    with pytest.raises(GgufExportError,match='type|dtype|quant'):validate_gguf_file(path)


@pytest.mark.parametrize('suffix,value',[('block_count',0),('block_count',513),('expert_count',99999),('expert_count',4)])
def test_invalid_common_hparams_fail_before_upstream_assertions(tmp_path,suffix,value):
    path=tmp_path/'bad.gguf';qwen_fixture(path)
    # A rebuilt metadata-only fixture, never a graph substituted for support.
    from ember.gguf_export import GgufWriter,_VT_STRING,_VT_I32
    w=GgufWriter('qwen35');w.add_string('general.architecture','qwen35')
    for key,v in [('block_count',1),('embedding_length',64),('context_length',256)]:
        w.add_u32('qwen35.'+key,value if key==suffix else v)
    if suffix=='expert_count':w.add_u32('qwen35.expert_count',value)
    import numpy as np
    w.add_tensor('token_embd.weight',np.zeros((4,64),np.float32))
    path.write_bytes(w.build())
    with pytest.raises(GgufExportError,match='metadata-structure'):validate_gguf_file(path)


def test_changed_session_configuration_is_not_ignored_for_same_model(tmp_path):
    path=tmp_path/'graph.gguf';qwen_fixture(path)
    b=EmberBackend(tmp_path/'app')
    try:
        b.dispatch('model.importGguf',{'path':str(path)})
        b.trainer.ensure_engine_model(EngineConfig(n_ctx=256))
        assert get_engine().model_info().n_ctx==256
        b.trainer.ensure_engine_model(EngineConfig(n_ctx=512))
        assert get_engine().model_info().n_ctx==512
    finally:b.close()


def test_native_validation_covers_the_published_private_inode(tmp_path,monkeypatch):
    path=tmp_path/'source.gguf';qwen_fixture(path)
    b=EmberBackend(tmp_path/'app');e=get_engine();load=e.load_model;seen=[]
    def record(candidate,*args,**kwargs):
        p=Path(candidate)
        assert p.suffix=='.partial' and p.parent==b.trainer.library_dir
        seen.append(p.stat().st_ino)
        return load(candidate,*args,**kwargs)
    try:
        monkeypatch.setattr(e,'load_model',record)
        result=b.dispatch('model.importGguf',{'path':str(path)})
        assert seen==[Path(result['path']).stat().st_ino]
        assert result['textInferenceVerified'] and not list(b.trainer.library_dir.glob('*.partial'))
    finally:b.close()


def test_actual_native_template_failure_cannot_be_registered(tmp_path):
    path=tmp_path/'unsupported-template.gguf';qwen_fixture(path)
    template=read_gguf_header(path).metadata['tokenizer.chat_template'].encode()
    path.write_bytes(path.read_bytes().replace(template,b'?'*len(template),1))
    b=EmberBackend(tmp_path/'app')
    try:
        with pytest.raises(InferenceError,match='first-token.*template'):
            b.dispatch('model.importGguf',{'path':str(path)})
        assert not list(b.trainer.library_dir.iterdir())
        assert b.trainer._imported_gguf is None and not get_engine().is_loaded()
    finally:b.close()
