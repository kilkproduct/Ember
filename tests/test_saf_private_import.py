"""Host tests of the registry boundary, not proof of Android SAF permissions."""
import builtins
import hashlib
from pathlib import Path
import pytest
from ember.backend import EmberBackend
from ember.gguf_export import export_gguf
from ember.model import ModelConfig, Llama
from ember.tokenizer import train_bpe
from ember.inference import InferenceError, get_engine, EngineConfig, GenParams


def fixture(path):
    tok=train_bpe(['Private import regression fixture content. '*10],300)
    cfg=ModelConfig(n_layer=1,n_head=2,n_embd=32,vocab_size=tok.vocab_size)
    export_gguf(path,cfg,Llama(cfg,seed=9).params,tok)
    return path


@pytest.mark.parametrize('source',['/proc/self/fd/148','content://provider/model'])
def test_uri_and_descriptor_never_reopened_by_importer(tmp_path,monkeypatch,source):
    backend=EmberBackend(tmp_path/'app'/'ember_v3')
    original_open=builtins.open
    original_exists=Path.exists
    def checked_open(path,*args,**kwargs):
        assert str(path)!=source,'Importer tried to reopen provider path'
        return original_open(path,*args,**kwargs)
    def checked_exists(path):
        assert str(path)!=source,'Importer tried to stat provider path'
        return original_exists(path)
    try:
        monkeypatch.setattr(builtins,'open',checked_open)
        monkeypatch.setattr(Path,'exists',checked_exists)
        with pytest.raises(InferenceError,match='stream-copied'):
            backend.dispatch('model.importGguf',{'path':source})
        with pytest.raises(InferenceError,match='private model staging'):
            backend.dispatch('model.registerExternal',{'path':source,'uri':'content://provider/model'})
        assert backend.trainer._imported_gguf is None
    finally:backend.close()


def test_private_staging_publishes_only_valid_models_and_keeps_source(tmp_path):
    app=tmp_path/'app';staging=app/'model_staging';staging.mkdir(parents=True)
    backend=EmberBackend(app/'ember_v3')
    source=fixture(tmp_path/'source.gguf')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        backend.corpus.add('Existing corpus content must survive a model import.')
        results=[]
        for token in ('a'*32,'b'*32):
            staged=staging/(token+'.gguf');staged.write_bytes(source.read_bytes())
            result=backend.dispatch('model.importGguf',{'path':str(staged),'modelStagingId':token,'name':'same.gguf'})
            assert Path(result['path']).parent==backend.trainer.library_dir
            assert staged.stat().st_ino==Path(result['path']).stat().st_ino
            staged.unlink()  # Android's finally cleanup; published library inode survives.
            assert Path(result['path']).is_file()
            results.append(result)
        assert results[0]['path']!=results[1]['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        assert backend.corpus.count()==1
        engine=get_engine()
        assert engine.load_model(results[-1]['path'],EngineConfig(n_ctx=64)).loaded
        assert engine.generate('Private',GenParams(max_tokens=2,temperature=0,stop_on_eos=False)).new_tokens==2
        engine.unload()
        selected=backend.trainer._imported_gguf
        bad=staging/('c'*32+'.gguf');bad.write_bytes(source.read_bytes()[:100])
        with pytest.raises(ValueError):
            backend.dispatch('model.importGguf',{'path':str(bad),'modelStagingId':'c'*32,'name':'bad'})
        assert backend.trainer._imported_gguf==selected
        assert not (backend.trainer.library_dir/'bad.gguf').exists()
        with pytest.raises(ValueError,match='identity/path'):
            backend.dispatch('model.importGguf',{'path':str(source),'modelStagingId':'a'*32})
    finally:backend.close()


def test_legacy_registration_and_selected_uri_are_preserved_without_reopen(tmp_path):
    b=EmberBackend(tmp_path/'app')
    uri='content://provider/previous-model'
    try:
        b.corpus.set_kv('inference:selected_external',uri)
        b.corpus.set_kv('inference:selected_model','/proc/self/fd/148')
        row=b.dispatch('model.restoreExternalRegistration',{'uri':uri,'name':'Existing model'})
        assert row['requiresPrivateImport'] and row['selected'] and row['valid'] is False
        assert b.corpus.get_kv('inference:selected_external')==uri
        assert b.corpus.get_kv('inference:selected_model')=='/proc/self/fd/148'
        assert b.dispatch('model.list')['models'][-1]['name']=='Existing model'
        with pytest.raises(InferenceError,match='preserved'):b.trainer.require_selected_source_available()
        assert b.dispatch('gen.prepare',{'prompt':'hello'})['modelError']['errorName']=='model-source-unavailable'
    finally:b.close()
