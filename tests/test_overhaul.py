"""Deterministic overhaul regressions; no internet or device needed."""
import bz2,csv,gzip,io,json,lzma,threading,urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from ember.autodiff import Tensor
from ember.corpus import Corpus
from ember.forager import Forager
from ember.ingest import ImportManager
from ember.progress import WorkProgress
from ember.model import ModelConfig,Llama,save_model,load_model,read_checkpoint_metadata
from ember.tokenizer import BPETokenizer,train_bpe
from ember.gguf_export import export_gguf,validate_gguf_file
from ember.inference import LlamaCppEngine,EngineConfig,GenParams,InferenceError


def test_measured_eta_unknown_then_rolling_median_and_cancel():
    now=[0.0];p=WorkProgress(clock=lambda:now[0],window=5)
    assert p.update(0,20)['etaSeconds'] is None
    for i in range(1,5):
        now[0]=i*2
        event=p.update(i,20)
    assert event['unitsPerSecond']==.5 and event['etaSeconds']==32
    assert p.update(4,None)['etaSeconds'] is None
    assert p.update(4,20,state='cancelled')['etaSeconds'] is None
    with pytest.raises(ValueError):p.update(3,20)
    # bounded outlier resistance, not a statistical confidence claim
    now[0]=100;assert p.update(5,20)['etaSeconds']==30
    assert len(p.samples)==5


def test_flattened_weight_gradient_and_broadcast_tape_do_not_alias():
    rng=np.random.default_rng(5)
    x=rng.normal(size=(2,3,4,7)).astype(np.float32)
    w=rng.normal(size=(7,9)).astype(np.float32)
    g=rng.normal(size=(2,3,4,9)).astype(np.float32)
    xt,wt=Tensor(x),Tensor(w);result=xt@wt;result.backward(g)
    np.testing.assert_allclose(wt.grad,np.einsum('abci,abcj->ij',x,g),rtol=3e-6,atol=3e-6)
    np.testing.assert_allclose(xt.grad,g@w.T,rtol=2e-6)
    original=g.copy();wt.add_grad(np.ones_like(w));np.testing.assert_array_equal(g,original)
    shared=Tensor(np.ones((3,),np.float32));(shared*shared+shared).backward(np.ones(3,np.float32))
    np.testing.assert_array_equal(shared.grad,[3,3,3])


@pytest.mark.parametrize('codec,compress',[('',lambda x:x),('.gz',gzip.compress),('.bz2',bz2.compress),('.xz',lzma.compress)])
@pytest.mark.parametrize('fmt',['jsonl','csv','tsv'])
def test_streaming_compressed_formats_pause_restart_and_dedup(tmp_path,codec,compress,fmt):
    rows=[{'title':f'Row {i}','text':f'Unique streaming passage number {i} with content that is genuinely different.'} for i in range(205)]
    if fmt=='jsonl':raw=('\n'.join(json.dumps(r) for r in rows)+'\n').encode()
    else:
        text=io.StringIO();w=csv.DictWriter(text,fieldnames=['title','text'],delimiter='\t' if fmt=='tsv' else ',')
        w.writeheader();w.writerows(rows);raw=text.getvalue().encode()
    path=tmp_path/('rows.'+fmt+codec);path.write_bytes(compress(raw))
    corpus=Corpus(tmp_path/'db');mgr=ImportManager(corpus)
    preview=mgr.stage_path('task',path,name=path.name)
    assert preview['resumable']
    def pause(ev):
        if ev['records']>=100:mgr.pause('task')
    result=mgr.commit('task',on_progress=pause)
    assert result['outcome']=='paused' and corpus.count()==100
    mgr=ImportManager(corpus);events=[];result=mgr.resume('task',on_progress=events.append)
    assert result['outcome']=='completed' and corpus.count()==205
    assert events[-1]['percent']==100
    assert result['byteBasis']==('decompressed' if codec else 'source')
    mgr.stage_path('again',path,name=path.name);mgr.commit('again')
    assert corpus.count()==205


def test_malformed_replayed_prefix_not_double_counted(tmp_path):
    path=tmp_path/'data.jsonl';path.write_text('not-json\n'+'\n'.join(json.dumps({'text':f'Passage {i} is a unique import record.'}) for i in range(201)))
    corpus=Corpus(tmp_path/'db');mgr=ImportManager(corpus);mgr.stage_path('x',path)
    mgr.commit('x',on_progress=lambda e:mgr.pause('x') if e['records']>=100 else None)
    result=ImportManager(corpus).resume('x')
    assert result['malformed']==1 and corpus.count()==201


def test_forager_response_cache_survives_restart_and_cancel_does_not_fetch(tmp_path,monkeypatch):
    calls=[]
    def fetch(url,**kwargs):calls.append(url);kwargs['on_bytes'](37);return {'genuine':'fixture response'}
    monkeypatch.setattr('ember.forager._http_json',fetch)
    c=Corpus(tmp_path/'db');f=Forager(c,lambda *a:None)
    url='https://en.wikipedia.org/w/api.php?fixture=cache'
    assert f._request(url)==f._request(url)
    assert len(calls)==1 and f.metrics['cached']==1 and f.metrics['bytesDownloaded']==37
    f2=Forager(c,lambda *a:None);assert f2._request(url)=={'genuine':'fixture response'}
    assert len(calls)==1
    f2._stop.set()
    with pytest.raises(InterruptedError):f2._request(url)
    assert f2.metrics['activeRequests']==0


def test_forager_retries_429_and_releases_host_slot(tmp_path,monkeypatch):
    calls=[]
    def fetch(url,**kwargs):
        calls.append(url)
        if len(calls)<3:raise urllib.error.HTTPError(url,429,'slow',{'Retry-After':'0'},None)
        return {'ok':True}
    monkeypatch.setattr('ember.forager._http_json',fetch)
    f=Forager(Corpus(tmp_path/'db'),lambda *a:None)
    f._host_next.clear()
    assert f._request('https://example.org/test')=={'ok':True}
    assert len(calls)==3 and f.metrics['activeRequests']==0 and f.metrics['failed']==0


def test_forager_parallel_writes_respect_budget_and_preserve_interrupted_topic(tmp_path,monkeypatch):
    f=Forager(Corpus(tmp_path/'db'),lambda *a:None);f._budget=7
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i:f._accept(str(i),f'Unique passage {i} '+('genuine content '*20),'fixture'),range(30)))
    assert f.documents_accepted==7 and f.corpus.count()==7
    f.documents_accepted=0
    monkeypatch.setattr(f,'_fetch_wikipedia',lambda topic,queue:f._stop.set())
    f._worker({'seeds':'unfinished topic','hn':False,'gutenberg':False,'openlibrary':False,'budget':20})
    assert f.saved_state['pendingTopics'][0]=='unfinished topic'
    assert 'unfinished topic' not in f.saved_state['seenTopics']


def test_requested_50k_small_corpus_stays_corpus_limited():
    tok=train_bpe(['hello hello'],vocab_size=50000)
    assert 261<tok.vocab_size<50000
    with pytest.raises(ValueError,match='50,000'):train_bpe(['hello'],50001)
    with pytest.raises(ValueError,match='50,000'):ModelConfig(vocab_size=50001).validate()


def test_actual_50000_vocab_checkpoint_loss_export_and_native_tokenizer(tmp_path):
    # A constructed valid byte-pair merge table, NOT a claim that a tiny corpus
    # learned 49,739 merges. There are no filler/unused fake vocabulary entries.
    vocab=[chr(i) for i in range(256)]+['<pad>','<bos>','<eos>','<doc>','<unk>'];merges=[]
    for a in range(256):
        for b in range(256):
            if len(vocab)==50000:break
            merges.append([chr(a),chr(b)]);vocab.append(chr(a)+chr(b))
        if len(vocab)==50000:break
    tok=BPETokenizer(vocab,merges)
    cfg=ModelConfig(n_layer=1,n_head=2,n_kv_head=1,n_embd=32,block_size=16,vocab_size=50000)
    model=Llama(cfg,seed=42);x=np.array([[0,49999,123,257]],np.int32)
    _,loss=model.forward(x,targets=np.roll(x,-1,axis=1));grads=model.backward()
    assert np.isfinite(loss) and all(np.isfinite(g).all() for g in grads.values())
    ckpt=tmp_path/'large.emb';save_model(ckpt,cfg,model.params,meta={'tokenizer':tok.to_dict(),'step':1})
    assert read_checkpoint_metadata(ckpt)['step']==1
    reloaded=load_model(ckpt);assert reloaded[0].vocab_size==50000
    path=tmp_path/'large.gguf';export_gguf(path,cfg,model.params,tok)
    assert validate_gguf_file(path)['vocabSize']==50000
    e=LlamaCppEngine()
    try:
        e.load_model(path,EngineConfig(n_ctx=16,n_batch=64,n_ubatch=16,n_threads=2))
        text='Hello café Unicode 灯台'
        assert e.tokenize(text)==tok.encode(text)
        assert e.generate(text,GenParams(max_tokens=2,temperature=0,stop_on_eos=False)).new_tokens==2
    finally:e.close()


def test_external_descriptor_selection_and_restart_does_not_reuse_stale_fd(tmp_path):
    from ember.backend import EmberBackend
    path=tmp_path/'external.gguf';tok=train_bpe(['A real fixture passage. '*5],300)
    cfg=ModelConfig(n_layer=1,n_head=2,n_embd=32,vocab_size=tok.vocab_size)
    export_gguf(path,cfg,Llama(cfg,seed=4).params,tok)
    backend=EmberBackend(tmp_path/'app')
    try:
        with path.open('rb') as f:
            fd=f'/proc/self/fd/{f.fileno()}'
            backend.dispatch('model.registerExternal',{'path':fd,'uri':'content://fixture/1','name':'fixture'})
            result=backend.dispatch('model.select',{'path':fd})
            assert result['selected']==fd
            assert backend.trainer.inference_model_path.as_posix()==fd
            assert backend.dispatch('model.list')['models'][-1]['storage']=='external'
    finally:backend.close()
    backend=EmberBackend(tmp_path/'app')
    try:assert backend.trainer._imported_gguf is None
    finally:backend.close()


def test_stream_rejects_unbounded_single_line(tmp_path):
    path=tmp_path/'oversized.jsonl'
    with path.open('w') as f:
        f.write('{"text":"')
        for _ in range(9*1024):f.write('x'*1024)
        f.write('"}\n')
    mgr=ImportManager(Corpus(tmp_path/'db'));mgr.stage_path('oversized',path)
    with pytest.raises(ValueError,match='safety limit'):mgr.commit('oversized')


def test_native_invalid_resources_and_clamped_context_warning(tmp_path):
    tok=train_bpe(['A real fixture passage. '*5],300)
    cfg=ModelConfig(n_layer=1,n_head=2,n_embd=32,block_size=32,vocab_size=tok.vocab_size)
    path=tmp_path/'resources.gguf';export_gguf(path,cfg,Llama(cfg,seed=4).params,tok)
    e=LlamaCppEngine()
    try:
        with pytest.raises(InferenceError,match='microbatch'):e.load_model(path,EngineConfig(n_batch=16,n_ubatch=64))
        with pytest.raises(InferenceError):e.load_model(path,EngineConfig(n_ctx=9000))
        e.load_model(path,EngineConfig(n_ctx=2048,n_batch=64,n_ubatch=16,n_threads=2))
        result=e.generate('A real',GenParams(max_tokens=1,temperature=0))
        assert 'clamped' in result.warning.lower()
    finally:e.close()


def test_reused_cross_entropy_buffer_matches_float64_oracle_without_mutating_logits():
    from ember.model import _cross_entropy_backward
    rng=np.random.default_rng(13);x=rng.standard_normal((2,3,50000),dtype=np.float32)
    before=x.copy();targets=np.array([[49999,1,-100],[0,234,45000]])
    loss,grad=_cross_entropy_backward(x,targets)
    flat=x.reshape(-1,50000).astype(np.float64);labels=targets.ravel();valid=labels>=0
    shifted=flat-flat.max(axis=1,keepdims=True);prob=np.exp(shifted);prob/=prob.sum(axis=1,keepdims=True)
    expected_loss=-np.log(prob[np.flatnonzero(valid),labels[valid]]).mean()
    expected=prob.copy();expected[np.flatnonzero(valid),labels[valid]]-=1;expected[~valid]=0;expected/=valid.sum()
    assert abs(loss-expected_loss)<2e-6
    np.testing.assert_allclose(grad,expected,rtol=2e-6,atol=1e-8)
    np.testing.assert_array_equal(x,before)
