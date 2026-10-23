#!/usr/bin/env python3
"""Isolated 50k-output CE benchmark, same inputs; baseline via git show, no checkout."""
import argparse,ast,hashlib,json,os,resource,subprocess,time,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--baseline-ref');a=p.parse_args()
if a.baseline_ref:
    source=subprocess.check_output(['git','show',f'{a.baseline_ref}:ember/model.py'],text=True)
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_cross_entropy_backward')
    namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<baseline-ce>','exec'),namespace)
    loss_fn=namespace['_cross_entropy_backward'];ref=a.baseline_ref
else:
    from ember.model import _cross_entropy_backward as loss_fn
    source=Path('ember/model.py').read_text();ref=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
rng=np.random.default_rng(42);logits=rng.standard_normal((2,128,50000),dtype=np.float32);targets=rng.integers(0,50000,(2,128))
times=[];sums=[]
for i in range(5):
    t=time.perf_counter();loss,gradient=loss_fn(logits,targets);times.append(time.perf_counter()-t)
    sums.append(float(np.abs(gradient).sum(dtype=np.float64)));del gradient
print(json.dumps({'ref':ref,'sourceSha256':hashlib.sha256(source.encode()).hexdigest(),'numpy':np.__version__,
    'openblasThreads':os.getenv('OPENBLAS_NUM_THREADS'),'shape':[2,128,50000],'loss':loss,'absGradientSum':sums,
    'medianSeconds':statistics.median(times),'samplesSeconds':times,'peakRssKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
