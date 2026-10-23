"""Independent mathematical and integration checks for EMB4 Llama training."""
import os
import numpy as np
import pytest
from ember.model import Llama, ModelConfig, save_model, load_model
from ember.autodiff import Tensor, rmsnorm, rope, silu, repeat_kv
from ember.optim import AdamW


def config(**kw):
    return ModelConfig(**dict(dict(n_layer=1,n_head=4,n_kv_head=2,n_embd=32,
                                  block_size=16,vocab_size=270,intermediate_size=48),**kw))


@pytest.mark.parametrize('kw', [dict(n_kv_head=3),dict(n_embd=36),dict(rope_theta=0),
    dict(rope_scaling='llama3'),dict(dropout=.1),dict(bias=True),dict(training_dtype='float16'),
    dict(norm_eps=0),dict(intermediate_size=-1)])
def test_reject_unsupported_configurations(kw):
    with pytest.raises(ValueError): config(**kw).validate()


def test_rmsnorm_and_swiglu_match_definitions():
    x=Tensor(np.array([[[1.,2.,-3.,4.]]],np.float32))
    gain=Tensor(np.array([1.,2.,3.,4.],np.float32))
    y=rmsnorm(x,gain,1e-5)
    expected=x.data/np.sqrt(np.mean(x.data**2,axis=-1,keepdims=True)+1e-5)*gain.data
    np.testing.assert_allclose(y.data,expected,rtol=2e-6)
    y.backward(np.ones_like(y.data))
    assert np.isfinite(x.grad).all() and np.isfinite(gain.grad).all()
    np.testing.assert_allclose(silu(x).data,x.data/(1+np.exp(-x.data)),rtol=1e-6)


def test_rope_matches_meta_complex_adjacent_pair_definition_and_backward():
    rng=np.random.default_rng(14)
    x=Tensor(rng.normal(size=(2,4,7,8)).astype(np.float32))
    y=rope(x,500000.)
    complex_x=x.data[...,::2]+1j*x.data[...,1::2]
    angles=np.outer(np.arange(7),1/(500000.**(np.arange(0,8,2)/8)))
    expected=complex_x*np.exp(1j*angles)
    np.testing.assert_allclose(y.data[...,::2],expected.real,atol=3e-6)
    np.testing.assert_allclose(y.data[...,1::2],expected.imag,atol=3e-6)
    np.testing.assert_allclose(y.data[...,0,:],x.data[...,0,:],atol=1e-6)
    np.testing.assert_allclose(np.sum(y.data**2,axis=-1),np.sum(x.data**2,axis=-1),rtol=2e-6)
    y.backward(y.data)
    np.testing.assert_allclose(x.grad,x.data,atol=3e-6)


def test_gqa_repeats_contiguous_head_groups_and_reduces_gradients():
    x=Tensor(np.arange(2*3*4).reshape(1,2,3,4))
    y=repeat_kv(x,3)
    np.testing.assert_array_equal(y.data[:,0],x.data[:,0])
    np.testing.assert_array_equal(y.data[:,2],x.data[:,0])
    np.testing.assert_array_equal(y.data[:,3],x.data[:,1])
    y.backward(np.ones_like(y.data))
    np.testing.assert_array_equal(x.grad,np.full_like(x.data,3))


@pytest.mark.parametrize('tied',[True,False])
@pytest.mark.parametrize('kv',[1,2,4])
def test_causality_dimensions_and_all_parameter_gradients(tied,kv):
    m=Llama(config(n_kv_head=kv,tie_embeddings=tied),seed=21)
    x=np.array([[1,2,3,4,5,6]],np.int32)
    a=m.forward(x)[0]; changed=x.copy();changed[:,3:]=19
    b=m.forward(changed)[0]
    assert a.shape==(1,6,270)
    np.testing.assert_array_equal(a[:,:3],b[:,:3])
    _,loss=m.forward(x,targets=np.array([[2,3,4,5,6,7]],np.int32))
    assert np.isfinite(loss)
    grads=m.backward()
    for name in m.params:
        assert grads[name].shape==m.params[name].shape
        assert np.isfinite(grads[name]).all(),name
        assert np.any(grads[name]!=0),name


def test_tiny_batch_overfits_and_atomic_checkpoint_survives_failure(tmp_path,monkeypatch):
    m=Llama(config(),seed=7); opt=AdamW(m.params,lr=.01)
    x=np.array([[1,2,3,4,1,2,3,4]],np.int32)
    y=np.array([[2,3,4,1,2,3,4,1]],np.int32)
    initial=m.forward(x,targets=y)[1]
    for _ in range(90):
        _,loss=m.forward(x,targets=y)
        opt.step(m.backward())
    logits,loss=m.forward(x,targets=y)
    assert loss<initial*.1 and (logits.argmax(-1)==y).mean()>.95
    path=tmp_path/'known-good.emb'
    save_model(path,m.cfg,m.params,{'step':90})
    before=path.read_bytes()
    real_replace=os.replace
    def fail(src,dst):
        if str(dst)==str(path): raise OSError('simulated interrupted publication')
        return real_replace(src,dst)
    monkeypatch.setattr(os,'replace',fail)
    with pytest.raises(OSError):save_model(path,m.cfg,m.params,{'step':91})
    assert path.read_bytes()==before
    cfg,params,meta=load_model(path)
    assert meta['step']==90 and cfg.to_dict()==m.cfg.to_dict()
    for key in params:np.testing.assert_array_equal(params[key],m.params[key])


def test_legacy_gpt_checkpoint_is_rejected_without_modification(tmp_path):
    path=tmp_path/'old.emb'; data=b'EMB3'+b'\0'*100;path.write_bytes(data)
    with pytest.raises(ValueError,match='magic'):load_model(path)
    assert path.read_bytes()==data
