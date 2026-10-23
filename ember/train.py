"""Reproducible local training CLI, using the same pipeline as the Android app.

Example: python -m ember.train --data ./data --preset phone --steps 200 \
    --max-train-tokens 100000 --tokenizer-chars 200000 --export model.gguf
Existing corpus data is never cleared. Optional JSONL import is streamed by line.
"""
import argparse
import json
from pathlib import Path
from .backend import EmberBackend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--jsonl',type=Path,help='Optional documents with text/title/source fields')
    p.add_argument('--preset',default='phone',choices=['phone','tiny','small','medium','tablet','desktop'])
    p.add_argument('--config',type=Path,help='JSON architecture/preparation overrides (Android vessel keys)')
    p.add_argument('--resume',help='Existing named checkpoint; do not recast')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gradient-accumulation',type=int,default=1)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--resume-schedule',action='store_true',help='Continue checkpoint scheduler and batch RNG instead of a fresh run schedule')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-train-tokens',type=int,default=3000000)
    p.add_argument('--max-val-tokens',type=int,default=250000)
    p.add_argument('--tokenizer-chars',type=int,default=2000000)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--quantization',choices=['f32','q8_0'],default='f32')
    a=p.parse_args()
    b=EmberBackend(a.data)
    try:
        if a.jsonl:
            with a.jsonl.open(encoding='utf-8') as f:
                for line_no,line in enumerate(f,1):
                    if not line.strip():continue
                    doc=json.loads(line)
                    if not isinstance(doc.get('text'),str):
                        raise ValueError(f'JSONL line {line_no}: text must be a string')
                    b.dispatch('corpus.addText',doc)
        if a.resume:b.trainer.load_checkpoint(a.resume)
        else:
            cfg={'preset':a.preset,'maxTrainTokens':a.max_train_tokens,
                 'maxValTokens':a.max_val_tokens,'tokenizerChars':a.tokenizer_chars}
            if a.config:cfg.update(json.loads(a.config.read_text()))
            b.dispatch('vessel.cast',{'config':cfg,'seed':a.seed})
        print(json.dumps({'preparation':b.trainer.status()},ensure_ascii=False),flush=True)
        run=b.dispatch('train.sync',{'steps':a.steps,'batch':a.batch,'lr':a.lr,'every':0,'validateEvery':50,'gradientAccumulation':a.gradient_accumulation,'weightDecay':a.weight_decay,'resumeSchedule':a.resume_schedule})
        b.trainer.save_checkpoint('cli-latest')
        from .gguf_export import export_gguf
        report=export_gguf(a.export,b.trainer.cfg,b.trainer.model.params,b.trainer.tokenizer,
                           step=b.trainer.step,quantization=a.quantization)
        print(json.dumps({'run':run,'export':report.to_dict()},ensure_ascii=False))
        if run['outcome']=='failed':raise RuntimeError('training failed; inspect run diagnostics')
    except KeyboardInterrupt:
        b.trainer.stop()
        if b.trainer.ready:b.trainer.save_checkpoint('cli-interrupted')
        raise
    finally:b.close()


if __name__=='__main__': main()
