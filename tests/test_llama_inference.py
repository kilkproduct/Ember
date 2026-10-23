"""Actual upstream llama.cpp execution, not file-existence or compile-only tests."""
import ctypes
import socket
import numpy as np
import pytest
from ember.gguf_export import export_gguf
from ember.inference import LlamaCppEngine, EngineConfig, GenParams, InferenceError
from ember.model import GPT, ModelConfig
from ember.tokenizer import train_bpe

TEXTS = ["It's a test—1234 🔥  done. Café العربية 灯台の記録\n\nhello world! " * 8,
         " don't won't we're I've\t mixed \v whitespace\u00a0 end   " * 8]

@pytest.fixture
def fixture(tmp_path):
    tok = train_bpe(TEXTS, vocab_size=450)
    model = GPT(ModelConfig(n_layer=2, n_head=2, n_embd=64, block_size=64, vocab_size=tok.vocab_size), seed=3)
    path = tmp_path / 'fixture.gguf'
    export_gguf(path, model.cfg, model.params, tok)
    engine = LlamaCppEngine()
    assert engine.available(), engine.load_error
    engine.load_model(path, EngineConfig(n_ctx=64, n_batch=64, n_threads=2))
    yield engine, tok, model, path
    engine.close()

def test_upstream_tokenizer_exact_parity(fixture):
    engine, tok, _, _ = fixture
    cases = TEXTS + ['', ' ', '   ', '\n\n', '\tmixed \v whitespace', 'a\u00a0b', "don't won't we're", 
        'Café e\u0301 العربية 灯台の記録 русский 🔥🚀', '0123 Ⅷ ² ½', 'a_b 100% (((...)))\r\n']
    for text in cases:
        ids = engine.tokenize(text)
        assert ids == tok.encode(text), repr(text)
        assert engine.detokenize(ids) == text

def test_upstream_forward_logits_and_greedy(fixture):
    engine, tok, model, _ = fixture
    for text in ['hi', 'Café 灯台 🔥', 'hello world!']:
        ids = [257] + tok.encode(text)
        expected = model.forward_logits(np.array([ids], dtype=np.int32))[0, -1]
        # Upstream ggml-cpu/vec.h defines GGML_GELU_FP16: even the F32
        # kernel uses its fp16 LUT, unlike NumPy's analytic tanh. Measured max
        # error is 1.26e-4; use an explicit 2e-4 bound, not text plausibility.
        np.testing.assert_allclose(engine.forward_logits(ids), expected, atol=2e-4, rtol=2e-4)
    a = engine.generate('hello', GenParams(max_tokens=12, temperature=0, repeat_penalty=1, seed=7))
    b = engine.generate('hello', GenParams(max_tokens=12, temperature=0, repeat_penalty=1, seed=7))
    assert a.token_ids == b.token_ids and a.text == b.text
    assert a.new_tokens == 12

def test_local_offline_stream_cancel_and_recovery(fixture, monkeypatch):
    engine, _, _, _ = fixture
    def forbidden(*a, **kw): raise AssertionError('inference attempted network I/O')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    pieces = []
    def stop(piece, tid, n):
        pieces.append(piece)
        return n < 6
    out = engine.generate('hello', GenParams(max_tokens=100, temperature=0, stop_on_eos=False), stop)
    assert out.stopped_by == 'cancelled' and out.new_tokens == 6
    assert ''.join(pieces) == out.text
    good = engine.generate('hello', GenParams(max_tokens=8, temperature=0, stop_on_eos=False))
    assert good.new_tokens == 8

def test_q8_0_native_load_and_generate(fixture, tmp_path):
    engine, tok, model, _ = fixture
    path = tmp_path / 'q8.gguf'
    export_gguf(path, model.cfg, model.params, tok, quantization='q8_0')
    engine.unload()
    engine.load_model(path, EngineConfig(n_ctx=64, n_batch=64, n_threads=2))
    assert np.isfinite(engine.forward_logits([257])).all()
    out = engine.generate('hello', GenParams(max_tokens=12, temperature=0, stop_on_eos=False))
    assert out.new_tokens == 12

def test_missing_corrupt_and_legacy_diagnostics(tmp_path):
    engine = LlamaCppEngine()
    assert engine.available(), engine.load_error
    try:
        with pytest.raises(InferenceError, match='not found'):
            engine.load_model(tmp_path/'missing.gguf')
        path = tmp_path/'legacy.emb'; path.write_bytes(b'EMB3'+b'x'*2048)
        with pytest.raises(InferenceError, match='GGUF'):
            engine.load_model(path)
        assert not engine.is_loaded()
    finally: engine.close()

def test_sampling_validation_and_no_fallback(fixture):
    engine, _, _, _ = fixture
    for params in (GenParams(temperature=float('nan')), GenParams(top_k=-2), GenParams(top_p=1.1),
                   GenParams(repeat_penalty=0), GenParams(max_tokens=-1)):
        with pytest.raises(InferenceError): engine.generate('hello', params)
    assert engine.generate('hello', GenParams(max_tokens=0)).new_tokens == 0
    missing = LlamaCppEngine('/no/such/libember_llama.so')
    assert not missing.available()
    with pytest.raises(InferenceError, match='load'):
        missing.generate('hello')

def test_library_selection_survives_restart_and_samples_refresh(fixture, tmp_path):
    from ember.backend import EmberBackend
    from ember.gguf_export import read_gguf_header
    _, _, _, path = fixture
    root = tmp_path/'data'
    b = EmberBackend(root)
    result = b.dispatch('model.importGguf', {'path':str(path), 'name':'local', 'select':True})
    assert result['selected']
    b.close()
    b = EmberBackend(root)
    try:
        assert b.trainer._imported_gguf.name == 'local.gguf'
        out = b.dispatch('gen.run', {'prompt':'hello','maxNew':4,'temperature':0,'stopOnBoundary':False})
        assert out['backend'] == 'llama.cpp' and out['newTokens'] == 4
        b.dispatch('corpus.addPack', {'pack':'starter'})
        b.dispatch('vessel.cast', {'config':{'nLayer':2,'nHead':2,'nEmbd':32,'blockSize':32,'vocabSize':400},'seed':3})
        b.dispatch('train.sync', {'steps':4,'every':2,'validateEvery':0,'batch':2})
        assert [s['step'] for s in b.trainer.samples] == [4,2]
        assert read_gguf_header(b.trainer.active_gguf_path).metadata['ember.source_step'] == 4
    finally: b.close()

def test_random_seed_is_actual_replayable_seed(fixture):
    engine, _, _, _ = fixture
    first = engine.generate('hello', GenParams(max_tokens=8, temperature=.9, stop_on_eos=False))
    assert first.seed != 0xFFFFFFFF
    again = engine.generate('hello', GenParams(max_tokens=8, temperature=.9, seed=first.seed, stop_on_eos=False))
    assert first.token_ids == again.token_ids

def test_repetition_penalty_uses_upstream_sampler_chain(fixture, tmp_path):
    engine, tok, model, _ = fixture
    # A controlled *valid GPT-2 graph*, not a mocked sampler: zero blocks, final
    # norm bias selects dimension 0, and two output rows compete at logits 2/1.8.
    # First token is A. After repeat_penalty=1.6 its score is 1.25, so B must win.
    for weight in model.params.values(): weight.fill(0)
    model.params['ln_f.bias'][0] = 1
    model.params['wte'][ord('A'), 0] = 2
    model.params['wte'][ord('B'), 0] = 1.8
    path = tmp_path/'penalties.gguf'
    export_gguf(path, model.cfg, model.params, tok)
    engine.load_model(path, EngineConfig(n_ctx=64, n_threads=2))
    plain = engine.generate('', GenParams(max_tokens=6, temperature=0, repeat_penalty=1, seed=1))
    penalized = engine.generate('', GenParams(max_tokens=6, temperature=0, repeat_penalty=1.6, seed=1))
    assert plain.token_ids == [ord('A')]*6
    assert penalized.token_ids[:2] == [ord('A'), ord('B')]
    assert plain.text != penalized.text
