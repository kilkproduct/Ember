#!/usr/bin/env python3
"""Execute native graph checks and write a per-architecture evidence matrix.

Host fixtures are complete synthetic graphs, not pretrained or physical-device
proof. Optional --model requires explicit provenance; weights never enter Git.
"""
import argparse,copy,hashlib,json,platform,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'scripts')]
from ember.inference import LlamaCppEngine,EngineConfig,GenParams
from ember.gguf_export import read_gguf_header
from test_qwen_compatibility import qwen_fixture
from architecture_fixtures import advanced_fixture
from extra_architecture_fixtures import family_fixture
from make_test_gguf import build_fixture


def checksum(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path);p.add_argument('--source-url');p.add_argument('--license')
    p.add_argument('--context',type=int,default=512)
    a=p.parse_args()
    if a.model and not (a.source_url and a.license):p.error('--model requires --source-url and --license (model weight license, not engine license)')
    e=LlamaCppEngine();report=copy.deepcopy(e.capabilities())
    report.update({'scope':'Host CPU graph execution; synthetic fixtures do NOT establish genuine/pretrained family or Android device readiness',
        'executedAtUtc':datetime.now(timezone.utc).isoformat(),'host':{'system':platform.system(),'machine':platform.machine(),'python':platform.python_version()},
        'android':{'arm64-v8a':'See CI build/package evidence; no physical-device inference','x86_64':'See CI build/package evidence; SAF instrumentation not run (universal-only APK constraint)'},
        'originalRequestedFile':{'name':'1a91b7eee80749ae824f504efc8155f2.gguf','available':False,'sha256':None},
        'genuineQwen35':{'status':'not-tested','weightsObtained':False,'source':'https://github.com/QwenLM/Qwen3.8/blob/2ea10dc725823bf7c3e21ce8557cbe15245132ae/README.md',
            'investigation':'QwenLM/Qwen3.5 redirects to Qwen3.8; no GitHub release assets. Official weights linked on Hugging Face/ModelScope, outside sandbox permitted hosts. No model checksum or weight license asserted.',
            'sha256':None,'license':None},'fixtures':[]})
    rows={r['architecture']:r for r in report['architectures']}
    for row in rows.values():row['verificationScope']='Not executed; factory/registry evidence only'
    def verify(path,scope,source,license_name,context):
        model=e.load_model(path,EngineConfig(n_ctx=context,n_batch=64,n_ubatch=16,n_threads=2))
        text='Hello world'
        assert e.detokenize(e.tokenize(text))==text,'tokenizer round-trip failed'
        first=e.generate(text,GenParams(max_tokens=1,temperature=0,stop_on_eos=False))
        continued=e.generate(text,GenParams(max_tokens=4,temperature=0,stop_on_eos=False))
        assert first.new_tokens==1 and continued.new_tokens==4,'native generation incomplete'
        types=sorted({report['tensorTypes'][str(t.dtype)]['name'] for t in read_gguf_header(path).tensors})
        result={'architecture':model.arch,'scope':scope,'source':source,'license':license_name,'sha256':checksum(path),'bytes':path.stat().st_size,
            'tensorTypes':types,'contextRequested':context,'contextEffective':model.n_ctx,
            'nativeLoad':True,'contextAllocated':True,'tokenizerRoundTrip':True,'firstToken':True,'continuedTokens':continued.new_tokens,
            'hostTokensPerSecond':continued.tokens_per_sec,'physicalDeviceTested':False}
        report['fixtures'].append(result)
        row=rows[model.arch];row['textInferenceVerified']=True;row['verificationScope']=scope
        row['testedQuantizations']=sorted(set(row['testedQuantizations']+types))
        e.unload();assert not e.is_loaded()
    try:
        with tempfile.TemporaryDirectory(prefix='ember-architecture-') as td:
            for arch in ['llama','qwen2','qwen3','mamba','qwen3moe','gemma','phi2','command-r','deepseek','mixtral-style']:
                for quantized in [False,True]:
                    path=Path(td)/(arch+('-q8' if quantized else '-f32')+'.gguf')
                    if arch=='llama':build_fixture(path,quantization='q8_0' if quantized else 'f32')
                    elif arch in ['qwen2','qwen3']:qwen_fixture(path,arch,quantized=quantized)
                    elif arch in ['mamba','qwen3moe']:advanced_fixture(path,arch,quantized)
                    else:family_fixture(path,arch,quantized)
                    origin='scripts/make_test_gguf.py' if arch=='llama' else 'tests/test_qwen_compatibility.py' if arch in ['qwen2','qwen3'] else 'tests/architecture_fixtures.py' if arch in ['mamba','qwen3moe'] else 'tests/extra_architecture_fixtures.py'
                    verify(path,'complete synthetic graph; NOT pretrained',origin+' (SHA256 '+checksum(ROOT/origin)+')','repository fixture code; not a downloaded weight license',256)
            if a.model:verify(a.model,'user-supplied actual weights',a.source_url,a.license,a.context)
        report['success']=True
    finally:e.close()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(f'PASS: {len(report["fixtures"])} native graph executions; {len(rows)} architecture evidence rows; no genuine Qwen3.5/device claim')

if __name__=='__main__':main()
