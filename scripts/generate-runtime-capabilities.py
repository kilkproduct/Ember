#!/usr/bin/env python3
"""Derive capability evidence from the EXACT pinned upstream source (stdlib only).

No architecture claims are inferred from branding. Registry + factory mapping
establish loader candidates, NOT successful inference. Ember policy exclusions
are explicit generation-interface limitations, never missing upstream graphs.
"""
import argparse, ast, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

def generate(source):
    pin=json.loads((ROOT/'third_party/llama.cpp.version').read_text())
    stamp=source/'.ember-fetch-stamp'
    if not stamp.exists() or stamp.read_text().strip()!=pin['tag']+':'+pin['sha256']:
        raise ValueError('Runtime source must be fetched and checksum-verified against third_party/llama.cpp.version')
    arch_source=(source/'src/llama-arch.cpp').read_text()
    model_source=(source/'src/llama-model.cpp').read_text()
    names=re.findall(r'\{\s*(LLM_ARCH_\w+),\s*"([^"]+)"\s*\}',arch_source.split('LLM_KV_NAMES')[0])
    factory=dict(re.findall(r'case (LLM_ARCH_\w+):\s*return new (\w+)\(params\);',model_source))
    if len(names)<50 or len(factory)<50:raise ValueError('Upstream registry/factory format changed; review capability generator')
    def category(function):
        body=arch_source.split('bool '+function+'(',1)[1].split('\n}',1)[0]
        return set(re.findall(r'case (LLM_ARCH_\w+):',body))
    recurrent=category('llm_arch_is_recurrent');hybrid=category('llm_arch_is_hybrid');diffusion=category('llm_arch_is_diffusion')
    moe_impls=set()
    for graph in (source/'src/models').glob('*.cpp'):
        text=graph.read_text()
        if 'build_moe_ffn(' in text:
            moe_impls.update(re.findall(r'(llama_model_\w+)::',text))
    limits={}
    for filename,macro,key in [('src/llama-hparams.h','LLAMA_MAX_LAYERS','maxLayers'),
        ('src/llama-hparams.h','LLAMA_MAX_EXPERTS','maxExperts'),('ggml/include/ggml.h','GGML_MAX_DIMS','tensorDimensions')]:
        limits[key]=int(re.search(r'#define\s+'+macro+r'\s+(\d+)',(source/filename).read_text()).group(1))
    rows=[]
    for enum,name in names:
        impl=factory.get(enum)
        reason=None
        if not impl:reason='No model factory implementation in this runtime'
        elif enum in diffusion:reason='Diffusion/denoising generation is not implemented by Ember'
        elif 'bert' in name or name in ('gemma-embedding','gemma-embedding2','llama-embed','t5encoder'):reason='Encoder/embedding-only model; Ember exposes text generation, not embedding retrieval'
        elif name in ('clip','wavtokenizer-dec'):reason='Auxiliary vision/audio model; Ember has no multimodal asset pipeline'
        elif name in ('eagle3','dflash','gemma4-assistant'):reason='Speculative draft/target coupling is not implemented by Ember'
        rows.append({'architecture':name,'upstreamRegistry':True,'nativeLoaderImplemented':bool(impl),
                     'upstreamFactory':impl,'moeGraphAvailable':impl in moe_impls,'emberTextCandidate':reason is None,'limitation':reason,
                     'modelKind':'diffusion' if enum in diffusion else 'hybrid' if enum in hybrid else 'recurrent' if enum in recurrent else 'encoder-decoder' if name=='t5' else 'upstream-defined',
                     'textInferenceVerified':False,'testedQuantizations':[],
                     'multimodal':{'image':False,'audio':False,'video':False,'note':'Text candidate only; projectors/auxiliary modality pipelines are not provided'},
                     'androidAbis':['arm64-v8a','x86_64'],'abiBuildEvidence':'See CI for this commit; registry membership is not a device test'})
    constants=ast.parse((source/'gguf-py/gguf/constants.py').read_text())
    enums={}
    for node in constants.body:
        if isinstance(node,ast.ClassDef) and node.name=='GGMLQuantizationType':
            enums={n.targets[0].id:ast.literal_eval(n.value) for n in node.body if isinstance(n,ast.Assign)}
    def value(node):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.Name) and node.id=='QK_K':return 256
        if isinstance(node,ast.Tuple):return [value(x) for x in node.elts]
        if isinstance(node,ast.BinOp):
            a,b=value(node.left),value(node.right)
            if isinstance(node.op,ast.Add):return a+b
            if isinstance(node.op,ast.Mult):return a*b
            if isinstance(node.op,ast.FloorDiv):return a//b
        raise ValueError(ast.dump(node))
    sizes={}
    for node in constants.body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id=='GGML_QUANT_SIZES':
            for k,v in zip(node.value.keys,node.value.values):
                sizes[str(enums[k.attr])]={'name':k.attr,'block':value(v)[0],'bytes':value(v)[1]}
    if not sizes:raise ValueError('Missing upstream quantization layout table')
    return {'schemaVersion':1,'backend':'llama.cpp','tag':pin['tag'],'commit':pin['commit'],
            'cpuOnly':True,'training':'Ember EMB4 training/export remains independent (Llama graph only)',
            'limits':limits,'architectures':rows,'tensorTypes':sizes,
            'quantizationPolicy':'Recognized tensor layout is not verified CPU kernel/model support; native loading and generation remain authoritative',
            'tokenizerPolicy':'Upstream loads unchanged tokenizer/vocabulary/control IDs; actual prompt tokenization is verified per import'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'third_party/llama.cpp');p.add_argument('--json',type=Path);p.add_argument('--header',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    data=generate(a.source);text=json.dumps(data,indent=2,sort_keys=True)+'\n'
    if a.json:
        if a.check:
            if a.json.read_text()!=text:raise SystemExit('Generated runtime capability manifest is stale')
        else:a.json.write_text(text)
    if a.header:
        a.header.parent.mkdir(parents=True,exist_ok=True)
        rows=data['architectures']
        interfaces='\n'.join('{'+json.dumps(r['architecture'])+','+('true' if r['emberTextCandidate'] else 'false')+','+json.dumps(r['limitation'] or '')+'},' for r in rows)
        a.header.write_text('#pragma once\nstatic const char kEmberRuntimeCapabilities[] = R"EMBERCAP('+json.dumps(data,separators=(',',':'))+')EMBERCAP";\n'
            +'struct EmberArchInterface {const char* name;bool text;const char* reason;};\nstatic const EmberArchInterface kEmberArchInterfaces[] = {\n'+interfaces+'\n};\n')
