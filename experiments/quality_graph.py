# Template: initialize with init_replay.py before executing.
# Recorded mathematics; public wrapper has not been rerun on NPU.
import json,torch,torch_npu
from experiment import P,LAB
from bge_lab.core import Engine,load_config
from bge_lab.quality import evaluate
out=P/'quality';out.mkdir(exist_ok=True)
for variant in ['graph_fp32','graph_bf16']:
    with torch.inference_mode():
        cfg=load_config(LAB/'configs'/('npu_fp32.json' if variant=='graph_fp32' else 'bf16.json'));cfg['engine']='torchair'
        engine=Engine(cfg,LAB/'models/bge-large-zh-v1.5')
        result=evaluate(engine,LAB/'data',out/variant,reference=LAB/'runs/ref_cpu_fp32',gate_path=LAB/'configs/quality_gate.json')
        print('QUALITY',variant,result['status'],result['retrieval'],'compiled_graphs',engine.compiled_graphs,flush=True)
        del engine;torch.npu.empty_cache()
