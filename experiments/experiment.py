# Template: initialize with init_replay.py before executing.
# Recorded mathematics; public wrapper has not been rerun on NPU.
import os,sys,time,json,argparse,math,collections,hashlib
from pathlib import Path
import numpy as np
import torch,torch_npu
sys.path.insert(0,'/workspace/ascend_bge_online_lab')
from bge_lab.core import Engine,load_config,read_jsonl,save_json
P=Path('/workspace/fia_bf16_retest_20261115')
LAB=Path('/workspace/ascend_bge_online_lab')

def cg():
    for path in ['/sys/fs/cgroup/cpu/cpu.stat','/sys/fs/cgroup/cpu.stat']:
        if Path(path).exists(): return {k:int(v) for k,v in (x.split() for x in Path(path).read_text().splitlines())}
    return {}
def affinity():
    out=[]
    for p in Path('/proc/self/task').glob('*/status'):
        lines=p.read_text().splitlines()
        out.append({k:v.strip() for k,v in (l.split(':',1) for l in lines) if k in ['Name','Pid','Cpus_allowed_list','Mems_allowed_list']})
    return out

class OptimizedFIA(torch.nn.Module):
    def __init__(self,orig,shared,layout):
        super().__init__(); self.query=orig.query;self.key=orig.key;self.value=orig.value
        self.heads=orig.num_attention_heads;self.dim=orig.attention_head_size
        self.shared=shared;self.layout=layout
    def forward(self,hidden_states,attention_mask=None,**kwargs):
        b,s,h=hidden_states.shape
        q=self.query(hidden_states);k=self.key(hidden_states);v=self.value(hidden_states)
        if self.layout=='BNSD':
            q=q.reshape(b,s,self.heads,self.dim).transpose(1,2).contiguous()
            k=k.reshape(b,s,self.heads,self.dim).transpose(1,2).contiguous()
            v=v.reshape(b,s,self.heads,self.dim).transpose(1,2).contiguous()
        if self.shared['mask'] is None and attention_mask is not None:
            self.shared['mask']=(attention_mask<0).expand(b,1,s,s).contiguous()
        r=torch_npu.npu_fused_infer_attention_score(q,k,v,num_heads=self.heads,num_key_value_heads=self.heads,input_layout=self.layout,scale=1/math.sqrt(self.dim),atten_mask=self.shared['mask'],sparse_mode=0,pre_tokens=2147483647,next_tokens=2147483647,softmax_lse_flag=False)[0]
        if self.layout=='BNSD':r=r.transpose(1,2).contiguous().reshape(b,s,h)
        return r,None

def patch(engine,layout):
    shared={'mask':None}
    def reset(module,args,kwargs): shared['mask']=None
    encoder=engine.module.encoder
    encoder.register_forward_pre_hook(reset,with_kwargs=True)
    for layer in encoder.encoder.layer:
        layer.attention.self=OptimizedFIA(layer.attention.self,shared,layout)
    engine.module.eval();engine.replaced_layers=24

def stats(x):
    return {'p50_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95)),'mean_ms':float(np.mean(x)),'n':len(x)}
def precision():
    return {'matmul_allow_hf32':torch_npu.npu.matmul.allow_hf32,'conv_allow_hf32':torch_npu.npu.conv.allow_hf32,'jit_compile_false':torch_npu.npu.is_jit_compile_false(),'float32_matmul_precision':torch.get_float32_matmul_precision(),'autocast_npu':torch.is_autocast_enabled('npu')}

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant', required=True, choices=['fp32','bf16','fia','fia_bsh','fia_mask','graph_fp32','graph_bf16']);ap.add_argument('--out',required=True);ap.add_argument('--profile',action='store_true');ap.add_argument('--reference',action='store_true');ap.add_argument('--hf32',choices=['on','off']);ap.add_argument('--steps',type=int,default=64);ap.add_argument('--e2e-steps',type=int,default=32);ap.add_argument('--length',type=int,default=256);a=ap.parse_args()
    out=P/'runs'/a.out;out.mkdir(parents=True,exist_ok=False)
    variant=a.variant; is_graph=variant.startswith('graph_'); base=variant.removeprefix('graph_')
    cfg=load_config(LAB/'configs'/('npu_fp32.json' if base=='fp32' else ('fia.json' if base=='fia' else 'bf16.json')))
    cfg['max_length']=a.length
    if is_graph:cfg['engine']='torchair'
    if a.hf32:torch_npu.npu.matmul.allow_hf32=(a.hf32=='on')
    engine=Engine(cfg,LAB/'models/bge-large-zh-v1.5')
    if base in ['fia_bsh','fia_mask']:patch(engine,'BSH' if base=='fia_bsh' else 'BNSD')
    rows=read_jsonl(LAB/'data/perf_requests.jsonl')[:128]
    texts=[[r['text'] for r in rows[i:i+8]] for i in range(0,len(rows),8)]
    hosts=[engine.tokenize(t)[0] for t in texts]
    inputs=[engine.to_device(h) for h in hosts]
    engine.sync()
    info={'variant':variant,'args':vars(a),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'pid':os.getpid(),'metadata':engine.metadata(),'precision':precision(),'torch_npu_commit':getattr(torch_npu.version,'git_version',None),'env':{k:os.environ.get(k) for k in ['CPU_AFFINITY_CONF','TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING','OMP_NUM_THREADS','MKL_NUM_THREADS']},'data_sha256':hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'first_128_ids':[r['id'] for r in rows]}
    observed=[]
    def observe(mod,args,output):
        observed.append({'module':type(mod).__name__,'input_dtype':str(args[0].dtype) if args else None,'weight_dtype':str(mod.weight.dtype) if hasattr(mod,'weight') else None,'output_dtype':str(output.dtype) if hasattr(output,'dtype') else str(type(output))})
    layer=engine.module.encoder.encoder.layer[0]
    hooks=[]
    if not is_graph:
        for m in [layer.attention.self.query,layer.attention.output.dense,layer.attention.output.LayerNorm,layer.intermediate.dense,layer.output.dense]:hooks.append(m.register_forward_hook(observe))
        engine.run(**inputs[0]);engine.sync()
        for h in hooks:h.remove()
    info['observed_dtypes']=observed
    t0=time.perf_counter()
    for i in range(12):engine.run(**inputs[i%len(inputs)])
    engine.sync();info['warmup_seconds']=time.perf_counter()-t0
    info['threads_after_warmup']=affinity()
    info['parameter_bytes']=sum(p.numel()*p.element_size() for p in engine.module.parameters())
    info['parameter_dtypes']=dict(collections.Counter(str(p.dtype) for p in engine.module.parameters()))
    output=np.concatenate([engine.run(**x).cpu().numpy() for x in inputs])
    refpath=P/f'reference_s{a.length}.npy'
    if a.reference:
        if refpath.exists():raise FileExistsError(refpath)
        np.save(refpath,output)
    if refpath.exists():
        ref=np.load(refpath);cos=(ref*output).sum(1)/(np.linalg.norm(ref,axis=1)*np.linalg.norm(output,axis=1));err=np.abs(ref-output)
        info['quality']={'min_cosine':float(cos.min()),'mean_cosine':float(cos.mean()),'max_abs_error':float(err.max()),'finite':bool(np.isfinite(output).all()),'samples':len(output),'scope':'embedding agreement on first 128 performance texts; not retrieval acceptance'}
    np.save(out/'embeddings.npy',output)
    if a.profile:
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],schedule=torch_npu.profiler.schedule(wait=1,warmup=1,active=3,repeat=1),on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'trace')),record_shapes=True,profile_memory=False,with_stack=False) as prof:
            for i in range(5):
                with torch.profiler.record_function('RETEST.model'):
                    engine.run(**inputs[i]);engine.sync()
                prof.step()
        info['profile_only']=True
    else:
        torch.npu.reset_peak_memory_stats();before=cg();wall=time.perf_counter();cpu=time.process_time();timing=[];enqueue=[];e2e=[]
        graph_before=engine.compiled_graphs
        for i in range(a.steps):
            engine.sync();t=time.perf_counter();r=engine.run(**inputs[i%len(inputs)]);enq=time.perf_counter();engine.sync();end=time.perf_counter();timing.append((end-t)*1000);enqueue.append((enq-t)*1000)
        for i in range(a.e2e_steps):
            engine.sync();t=time.perf_counter();host,n=engine.tokenize(texts[i%len(texts)]);inp=engine.to_device(host);r=engine.run(**inp).cpu().numpy();engine.sync();e2e.append((time.perf_counter()-t)*1000)
        after=cg();info.update({'model':stats(timing),'enqueue_wall':stats(enqueue),'e2e':stats(e2e),'raw_model_ms':timing,'raw_enqueue_ms':enqueue,'raw_e2e_ms':e2e,'cgroup_delta':{k:after[k]-before[k] for k in before},'cpu_process_seconds':time.process_time()-cpu,'measurement_wall_seconds':time.perf_counter()-wall,'peak_allocated_bytes':torch.npu.max_memory_allocated(),'peak_reserved_bytes':torch.npu.max_memory_reserved(),'graph_count':engine.compiled_graphs,'graph_count_delta':engine.compiled_graphs-graph_before})
    save_json(out/'result.json',info)
    print('RESULT',json.dumps({k:info[k] for k in ['variant','model','e2e','quality','cgroup_delta'] if k in info},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
