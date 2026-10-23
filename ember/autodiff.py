"""Small float32 reverse-mode tape for the on-device NumPy trainer.

No inference implementation: production decoding exclusively uses llama.cpp.
Operations are intentionally limited to the supported Llama training graph.
"""
import numpy as np


def reduce_like(g, shape):
    while g.ndim > len(shape): g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1: g = g.sum(axis=axis, keepdims=True)
    return g


class Tensor:
    def __init__(self, data, parents=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.parents = parents
        self.grad = None

    def add_grad(self, g):
        g = reduce_like(g,self.data.shape).astype(np.float32,copy=False)
        if self.grad is None:self.grad=g.copy()
        else:self.grad+=g

    def __add__(self, other):
        other = as_tensor(other)
        return Tensor(self.data + other.data, ((self, lambda g:g), (other, lambda g:g)))
    __radd__ = __add__

    def __mul__(self, other):
        other = as_tensor(other)
        return Tensor(self.data * other.data, ((self, lambda g:g*other.data), (other, lambda g:g*self.data)))
    __rmul__ = __mul__

    def __matmul__(self, other):
        other = as_tensor(other)
        def weight_grad(g):
            if other.data.ndim==2 and self.data.ndim>=2:
                return self.data.reshape(-1,self.data.shape[-1]).T @ g.reshape(-1,g.shape[-1])
            return self.data.swapaxes(-1,-2) @ g
        return Tensor(self.data @ other.data, ((self, lambda g:g @ other.data.swapaxes(-1,-2)),(other,weight_grad)))

    def reshape(self, *shape):
        return Tensor(self.data.reshape(*shape), ((self, lambda g:g.reshape(self.data.shape)),))

    def transpose(self, *axes):
        return Tensor(self.data.transpose(axes), ((self, lambda g:g.transpose(np.argsort(axes))),))

    def backward(self, grad):
        order, seen = [], set()
        def visit(t):
            if id(t) in seen: return
            seen.add(id(t))
            for p,_ in t.parents: visit(p)
            order.append(t)
        visit(self)
        self.grad = grad
        for t in reversed(order):
            for p, fn in t.parents: p.add_grad(fn(t.grad))


def as_tensor(x): return x if isinstance(x, Tensor) else Tensor(x)


def rmsnorm(x, gain, eps):
    r = (np.mean(x.data*x.data, axis=-1, keepdims=True)+eps)**-0.5
    norm = Tensor(x.data*r, ((x, lambda g:r*(g-x.data*r*r*np.mean(g*x.data, axis=-1, keepdims=True))),))
    return norm*gain


def silu(x):
    s = 1/(1+np.exp(-np.clip(x.data,-80,80)))
    return Tensor(x.data*s, ((x, lambda g:g*(s+x.data*s*(1-s))),))


def softmax(x):
    e = np.exp(x.data-x.data.max(axis=-1,keepdims=True))
    y = e/e.sum(axis=-1,keepdims=True)
    return Tensor(y, ((x, lambda g:y*(g-(g*y).sum(axis=-1,keepdims=True))),))


def rope(x, theta):
    # Meta complex-number and ggml LLAMA_ROPE_TYPE_NORM: adjacent pairs.
    d = x.data.shape[-1]; t = x.data.shape[-2]
    angle = np.arange(t,dtype=np.float32)[:,None] / theta**(np.arange(0,d,2,dtype=np.float32)/d)
    c,s = np.cos(angle),np.sin(angle)
    def rotate(a, sign):
        u,v = a[...,0::2],a[...,1::2]
        out=np.empty_like(a)
        out[...,0::2]=u*c-sign*v*s
        out[...,1::2]=sign*u*s+v*c
        return out
    return Tensor(rotate(x.data,1), ((x,lambda g:rotate(g,-1)),))


def repeat_kv(x, groups):
    b,h,t,d=x.data.shape
    return Tensor(np.repeat(x.data,groups,axis=1), ((x,lambda g:g.reshape(b,h,groups,t,d).sum(axis=2)),))


def embedding(w, ids):
    def back(g):
        out=np.zeros_like(w.data); np.add.at(out,ids,g); return out
    return Tensor(w.data[ids], ((w,back),))
