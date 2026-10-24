import pytest
from extra_architecture_fixtures import family_fixture
from ember.inference import LlamaCppEngine,EngineConfig,GenParams

@pytest.mark.parametrize('family',['gemma','phi2','deepseek','command-r','mixtral-style'])
@pytest.mark.parametrize('quantized',[False,True])
def test_additional_upstream_graphs(tmp_path,family,quantized):
    path=tmp_path/'model.gguf';family_fixture(path,family,quantized)
    e=LlamaCppEngine()
    try:
        e.load_model(path,EngineConfig(n_ctx=128,n_batch=32,n_ubatch=16,n_threads=2))
        assert e.detokenize(e.tokenize('Hello'))=='Hello'
        out=e.generate('Hello',GenParams(max_tokens=4,temperature=0,stop_on_eos=False))
        assert out.new_tokens==4
        e.unload();assert not e.is_loaded()
    finally:e.close()
