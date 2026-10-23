#!/usr/bin/env python3
"""Fixed host microbenchmarks, not Android-device performance claims."""
import sys,os,time,json,platform,subprocess,tempfile,statistics,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from ember.model import Llama,ModelConfig,save_model,load_model
from ember.optim import AdamW
from ember.tokenizer import train_bpe
from ember.gguf_export import export_gguf,validate_gguf_file

def measure(fn,n=5):
    times=[]
    for _ in range(n):
        start=time.perf_counter();fn();times.append(time.perf_counter()-start)
    return {'median_seconds':statistics.median(times),'samples_seconds':times}

def main():
    result={'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'machine':platform.machine(),
      'platform':platform.platform(),'cpu':platform.processor(),'cpus':os.cpu_count(),'openblas_threads':os.getenv('OPENBLAS_NUM_THREADS')}}
    texts=[f'Document {i}: The tide rises and falls. Unicode café 灯台 🔥. '*20 for i in range(200)]
    result['tokenizer']=measure(lambda:train_bpe(texts,vocab_size=500),3)
    tok=train_bpe(texts,vocab_size=1500)
    m=Llama(ModelConfig(n_layer=2,n_head=4,n_kv_head=2,n_embd=128,block_size=32,vocab_size=tok.vocab_size),seed=42)
    opt=AdamW(m.params,lr=.001)
    rng=np.random.default_rng(42);x=rng.integers(0,tok.vocab_size,(2,32),dtype=np.int32);y=np.roll(x,-1,axis=1)
    def step():m.forward(x,targets=y);opt.step(m.backward())
    step();result['training_step']=measure(step,10)
    with tempfile.TemporaryDirectory() as td:
        checkpoint=Path(td)/'m.emb';gguf=Path(td)/'m.gguf'
        result['checkpoint_save']=measure(lambda:save_model(checkpoint,m.cfg,m.params))
        result['checkpoint_load']=measure(lambda:load_model(checkpoint))
        export_gguf(gguf,m.cfg,m.params,tok)
        result['gguf_inspect']=measure(lambda:validate_gguf_file(gguf),10)
        result['gguf_bytes']=gguf.stat().st_size
        from ember.inference import LlamaCppEngine,EngineConfig,GenParams
        e=LlamaCppEngine()
        if e.available():
            result['model_load']=measure(lambda:e.load_model(gguf,EngineConfig(n_ctx=32,n_threads=2)),3)
            result['generation_32_tokens']=measure(lambda:e.generate('The tide',GenParams(max_tokens=32,temperature=0,stop_on_eos=False)),5)
            e.close()
    result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
