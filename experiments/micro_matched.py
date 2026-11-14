# Template: initialize with init_replay.py before executing.
# Recorded mathematics; public wrapper has not been rerun on NPU.
import time,json,math,os
from pathlib import Path
import torch,torch_npu,numpy as np
from transformers.models.bert.modeling_bert import eager_attention_forward
P=Path('/workspace/fia_bf16_retest_20261115')
torch.npu.set_device(0);torch.set_num_threads(4);torch.manual_seed(20261115)
@torch.inference_mode()
def run():
    results=[]; dummy=torch.nn.Module().eval()
    for s in [64,256,512]:
        b,n,d=8,16,64; scale=d**-0.5
        q,k,v=[torch.randn(b,s,n*d,device='npu',dtype=torch.bfloat16)*0.2 for _ in range(3)]
        qs,ks,vs=[x.reshape(b,s,n,d).transpose(1,2).contiguous() for x in [q,k,v]]
        mask=torch.zeros(b,1,s,s,dtype=torch.bool,device='npu');mask[:,:,:,-s//4:]=True
        additive=torch.zeros(b,1,s,s,device='npu',dtype=torch.bfloat16).masked_fill(mask,torch.finfo(torch.bfloat16).min)
        def eager_core():
            weights=torch.matmul(qs,ks.transpose(2,3))*scale+additive
            weights=torch.nn.functional.softmax(weights,dim=-1)
            weights=torch.nn.functional.dropout(weights,p=0.0,training=False)
            return torch.matmul(weights,vs)
        def fia_core():
            return torch_npu.npu_fused_infer_attention_score(qs,ks,vs,num_heads=n,num_key_value_heads=n,input_layout='BNSD',scale=scale,atten_mask=mask,sparse_mode=0,pre_tokens=2147483647,next_tokens=2147483647,softmax_lse_flag=False)[0]
        def eager_layout():
            qq,kk,vv=[x.reshape(b,s,n,d).transpose(1,2) for x in [q,k,v]]
            out,_=eager_attention_forward(dummy,qq,kk,vv,attention_mask=additive,scaling=scale,dropout=0.0)
            return out.reshape(b,s,n*d)
        def fia_bsh():
            return torch_npu.npu_fused_infer_attention_score(q,k,v,num_heads=n,num_key_value_heads=n,input_layout='BSH',scale=scale,atten_mask=mask,sparse_mode=0,pre_tokens=2147483647,next_tokens=2147483647,softmax_lse_flag=False)[0]
        funcs={'eager_core':eager_core,'fia_core':fia_core,'eager_from_bsh':eager_layout,'fia_bsh':fia_bsh}
        arrays={k:fn().float().cpu().numpy() for k,fn in funcs.items()}
        checks={}
        for refname,testname in [('eager_core','fia_core'),('eager_from_bsh','fia_bsh')]:
            ref=arrays[refname];test=arrays[testname]
            checks[testname]={'cosine':float((ref*test).sum()/(np.linalg.norm(ref)*np.linalg.norm(test))), 'max_abs_error':float(np.abs(ref-test).max())}
        checks['core_matches_installed_bert']=bool(np.array_equal(arrays['eager_core'].transpose(0,2,1,3).reshape(b,s,n*d),arrays['eager_from_bsh']))
        for r,order in enumerate([list(funcs),list(reversed(funcs)),list(funcs)],1):
            for name in order:
                fn=funcs[name]
                for _ in range(20):fn()
                torch.npu.synchronize();values=[]
                for _ in range(100):
                    t=time.perf_counter();fn();torch.npu.synchronize();values.append((time.perf_counter()-t)*1000)
                row={'s':s,'b':b,'heads':n,'head_dim':d,'dtype':'bfloat16','variant':name,'round':r,'p50_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),'raw_ms':values,'checks':checks}
                results.append(row)
                print('MATCHED_MICRO',s,name,r,row['p50_ms'],flush=True)
    payload={'method':'20 warmup, 100 synchronize-per-call, 3 interleaved rounds; BF16 softmax; precreated masks; no QKV projections','env':{k:os.getenv(k) for k in ['CPU_AFFINITY_CONF','TASK_QUEUE_ENABLE']},'cpu_affinity':sorted(os.sched_getaffinity(0)),'results':results}
    (P/'micro_matched.json').write_text(json.dumps(payload,indent=2))
run()
