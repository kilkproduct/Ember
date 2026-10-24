"""Independent binary fixtures, original upstream v2 writer, and native regression."""
import hashlib,importlib.util,struct,tracemalloc
from pathlib import Path
import numpy as np
import pytest
import gguf
from ember.gguf_export import read_gguf_header,validate_gguf_file,inspect_gguf_file,GgufExportError,export_gguf
from ember.inference import LlamaCppEngine,EngineConfig,GenParams
from ember.model import ModelConfig,Llama
from ember.tokenizer import train_bpe


def wire_fixture(version=3,order='<',alignment=32,relative=0,metadata=None,dims=(2,),dtype=0):
    def pack(fmt,v):return struct.pack(order+fmt,v)
    def string(s):
        b=s.encode();return pack('Q',len(b))+b
    scalar={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'B',10:'Q',11:'q',12:'d'}
    def value(kind,v):
        if kind in scalar:return pack(scalar[kind],v)
        if kind==8:return string(v)
        subtype,items=v
        return pack('I',subtype)+pack('Q',len(items))+b''.join(value(subtype,x) for x in items)
    md=[('general.architecture',8,'fixture'),('general.alignment',4,alignment)]+(metadata or [])
    body=b'GGUF'+pack('I',version)+pack('Q',1)+pack('Q',len(md))
    for key,kind,v in md:body+=string(key)+pack('I',kind)+value(kind,v)
    body+=string('weight')+pack('I',len(dims))+b''.join(pack('Q',d) for d in dims)+pack('I',dtype)+pack('Q',relative)
    start=(len(body)+alignment-1)//alignment*alignment
    return body+b'\0'*(start-len(body)),start


@pytest.mark.parametrize('version,order',[(2,'<'),(3,'<'),(3,'>')])
def test_versions_endian_all_scalar_types_and_homogeneous_arrays(tmp_path,version,order):
    samples=[(0,255),(1,-4),(2,65535),(3,-32000),(4,2**32-1),(5,-2**31),(6,1.25),
        (7,1),(8,'灯台 café'),(10,2**63+7),(11,-2**60),(12,1.125)]
    md=[('test.v'+str(kind),kind,v) for kind,v in samples]
    md+=[('test.array'+str(kind),9,(kind,[v,v])) for kind,v in samples]
    md+=[('test.empty',9,(4,[]))]
    head,start=wire_fixture(version,order,128,metadata=md)
    path=tmp_path/'format.gguf';path.write_bytes(head+struct.pack(order+'ff',1.,2.))
    h=read_gguf_header(path);ref=gguf.GGUFReader(path)
    assert h.version==version and h.byte_order==('little' if order=='<' else 'big')
    assert h.data_start==ref.data_offset==start and h.alignment==128
    np.testing.assert_equal(ref.tensors[0].data,[1,2])
    for kind,v in samples:
        assert h.metadata['test.v'+str(kind)]==v
        assert h.metadata['test.array'+str(kind)]==[v,v]
    assert h.metadata['test.empty']==[]
    if order=='>':assert inspect_gguf_file(path)['compatibilityState']=='unsupported-byte-order'


def original_v2_model(path):
    original=Path(__file__).parent/'reference/gguf_v2_writer.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='0f2d600e97b42a913520bc37576a42e9f5c5b5c0c9301f32e7453c0f61bccbd5'
    spec=importlib.util.spec_from_file_location('original_upstream_v2',original)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.GGUF_VERSION==2
    tok=train_bpe(['A complete genuine v2 container fixture. Hello world. '*10],300)
    cfg=ModelConfig(n_layer=1,n_head=2,n_embd=32,block_size=64,vocab_size=tok.vocab_size)
    src=path.with_suffix('.source');export_gguf(src,cfg,Llama(cfg,seed=52).params,tok)
    h=read_gguf_header(src);writer=module.GGUFWriter(path,'llama')
    try:
        for key,v in h.metadata.items():
            if key=='general.architecture':continue
            method=writer.add_bool if isinstance(v,bool) else writer.add_array if isinstance(v,list) else writer.add_string if isinstance(v,str) else writer.add_float32 if isinstance(v,float) else writer.add_int32 if v<0 else writer.add_uint32
            method(key,v)
        for t in h.tensors:
            arr=np.fromfile(src,dtype='<f4',count=np.prod(t.dims),offset=h.data_start+t.offset).reshape(tuple(reversed(t.dims)))
            writer.add_tensor(t.name,arr)
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file()
    finally:writer.close();src.unlink()
    return path


def test_original_v2_writer_independent_reader_and_actual_native_inference(tmp_path):
    path=original_v2_model(tmp_path/'original-v2.gguf')
    assert path.read_bytes()[:8]==b'GGUF\x02\0\0\0'
    ref=gguf.GGUFReader(path)
    assert int(ref.fields['GGUF.version'].parts[0][0])==2
    h=read_gguf_header(path)
    assert h.version==2 and len(h.tensors)==len(ref.tensors)
    assert validate_gguf_file(path)['ggufVersion']==2
    for t,r in zip(h.tensors,ref.tensors):
        assert t.name==r.name and h.data_start+t.offset==r.data_offset
    e=LlamaCppEngine()
    try:
        e.load_model(path,EngineConfig(n_ctx=64))
        assert e.generate('Hello',GenParams(max_tokens=4,temperature=0,stop_on_eos=False)).new_tokens==4
    finally:e.close()
    from ember.backend import EmberBackend
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    app=EmberBackend(tmp_path/'app')
    try:
        imported=app.dispatch('model.importGguf',{'path':str(path),'nCtx':64})
        assert imported['ggufVersion']==2 and imported['textInferenceVerified']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        assert hashlib.sha256(Path(imported['path']).read_bytes()).hexdigest()==digest
    finally:app.close()


@pytest.mark.parametrize('cut',[0,3,7,12,23,26,40,80,110,125])
def test_truncated_prefixes_fail_cleanly(tmp_path,cut):
    data,_=wire_fixture();p=tmp_path/'cut';p.write_bytes((data+b'\0'*8)[:cut])
    with pytest.raises((GgufExportError,ValueError)):read_gguf_header(p)


@pytest.mark.parametrize('mutate',[
    lambda b: b.__setitem__(slice(0,4),b'NOPE'),
    lambda b: struct.pack_into('<I',b,4,999),
    lambda b: struct.pack_into('<Q',b,8,2**64-1),
    lambda b: struct.pack_into('<Q',b,16,2**64-1),
    lambda b: struct.pack_into('<Q',b,24,2**64-1),
    lambda b: struct.pack_into('<I',b,24+8+len('general.architecture'),999),
])
def test_invalid_magic_future_version_and_allocation_attacks(tmp_path,mutate):
    data,_=wire_fixture();b=bytearray(data+b'\0'*8);mutate(b)
    p=tmp_path/'bad';p.write_bytes(b)
    with pytest.raises(GgufExportError):read_gguf_header(p)


@pytest.mark.parametrize('dims,dtype,offset,alignment',[((),0,0,32),((0,),0,0,32),((2**63,),0,0,32),((2,),999,0,32),((2,),0,1,32),((2,),0,2**64-1,32),((2,),0,0,3),((2,),2,0,32)])
def test_invalid_tensor_geometry_type_offset_and_alignment(tmp_path,dims,dtype,offset,alignment):
    data,_=wire_fixture(dims=dims,dtype=dtype,relative=offset,alignment=alignment)
    p=tmp_path/'bad';p.write_bytes(data+b'\0'*8)
    with pytest.raises(GgufExportError):read_gguf_header(p)


def test_sparse_offset_above_four_gib_has_no_whole_file_heap_allocation(tmp_path):
    data,start=wire_fixture(relative=2**32)
    p=tmp_path/'sparse'
    with p.open('wb') as f:f.write(data);f.seek(start+2**32);f.write(b'\0'*8)
    tracemalloc.start()
    try:
        h=read_gguf_header(p);peak=tracemalloc.get_traced_memory()[1]
        assert h.tensors[0].offset==2**32 and peak<8*1024*1024
    finally:tracemalloc.stop()


def test_shard_arrangement_reported_before_registration(tmp_path):
    data,_=wire_fixture(metadata=[('split.count',2,2),('split.no',2,0)])
    p=tmp_path/'part-00001-of-00002.gguf';p.write_bytes(data+b'\0'*8)
    report=inspect_gguf_file(p)
    assert report['valid'] and report['compatibilityState']=='missing-shards' and report['splitCount']==2


def test_nested_arrays_and_array_length_bombs(tmp_path):
    data,_=wire_fixture(metadata=[('test.nested',9,(9,[(4,[1,2]),(4,[])]))])
    p=tmp_path/'nested';p.write_bytes(data+b'\0'*8)
    assert read_gguf_header(p).metadata['test.nested']==[[1,2],[]]
    b=bytearray(data+b'\0'*8)
    pos=b.index(b'test.nested')+len('test.nested')+8
    struct.pack_into('<Q',b,pos,2**64-1);p.write_bytes(b)
    with pytest.raises(GgufExportError,match='array'):read_gguf_header(p)


@pytest.mark.parametrize('alignment',[32,128,8192])
def test_writer_alignment_count_empty_arrays_and_exact_payload(tmp_path,alignment):
    from ember.gguf_export import GgufWriter
    w=GgufWriter('fixture',alignment)
    w.add_string('general.architecture','fixture');w.add_array('test.empty',[],4)
    array=np.arange(32,dtype=np.float32).reshape(2,16)
    w.add_tensor('weight',array)
    first=w.build();assert w.build()==first
    path=tmp_path/'writer';path.write_bytes(first)
    h=read_gguf_header(path);ref=gguf.GGUFReader(path)
    assert h.alignment==ref.alignment==alignment
    assert h.metadata['test.empty']==[]
    assert path.read_bytes()[h.data_start:h.data_start+array.nbytes]==array.astype('<f4').tobytes()
    np.testing.assert_array_equal(ref.tensors[0].data,array)


def test_writer_conflicting_alignment_is_not_silently_emitted():
    from ember.gguf_export import GgufWriter
    w=GgufWriter(alignment=128);w.add_u32('general.alignment',32)
    with pytest.raises(GgufExportError,match='alignment'):w.build()

@pytest.mark.parametrize('kind,v',[(0,255),(1,-1),(2,60000),(3,-30000),(4,2**32-1),(5,-2**31),
    (6,1.25),(7,True),(8,'灯台'),(10,2**63+1),(11,-2**60),(12,1.125)])
def test_writer_all_scalar_and_array_widths_independently(tmp_path,kind,v):
    from ember.gguf_export import GgufWriter
    w=GgufWriter('fixture');w.add_string('general.architecture','fixture')
    w._add('test.scalar',kind,v);w.add_array('test.values',[v,v],kind)
    w.add_tensor('weight',np.zeros(2,np.float32))
    path=tmp_path/'typed';path.write_bytes(w.build())
    ours=read_gguf_header(path);ref=gguf.GGUFReader(path)
    assert ours.metadata['test.scalar']==v and ours.metadata['test.values']==[v,v]
    assert int(ref.fields['test.scalar'].types[0])==kind
    assert int(ref.fields['test.values'].types[1])==kind


def test_unknown_layout_inspection_is_not_misreported_as_invalid_file(tmp_path):
    data,_=wire_fixture(dtype=999);path=tmp_path/'unknown';path.write_bytes(data+b'\0'*8)
    report=inspect_gguf_file(path)
    assert report['compatibilityState']=='unsupported-tensor-type' and report['valid'] is None
