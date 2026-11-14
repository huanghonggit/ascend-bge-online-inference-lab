# Template: initialize with init_replay.py before executing.
# Recorded mathematics; public wrapper has not been rerun on NPU.
import json,inspect,os
from pathlib import Path
import numpy as np
import torch,torch_npu
from experiment import P,LAB,patch
from bge_lab.core import Engine,load_config
from bge_lab.quality import evaluate
from transformers.models.bert.modeling_bert import eager_attention_forward
out=P/'quality';out.mkdir(exist_ok=True)
probe={'cube_math_type':str(torch_npu.npu.matmul.cube_math_type),'matmul_allow_hf32':torch_npu.npu.matmul.allow_hf32,'conv_allow_hf32':torch_npu.npu.conv.allow_hf32,'reference':str(LAB/'runs/ref_cpu_fp32')}
probe['original_and_optimized_fia_embeddings_equal']=bool(np.array_equal(np.load(P/'runs/baseline_A_fia_r01/embeddings.npy'),np.load(P/'runs/baseline_A_fia_bsh_r01/embeddings.npy')))
(P/'extra_probe.json').write_text(json.dumps(probe,indent=2));print('PRECISION',json.dumps(probe),flush=True)
(P/'bert_eager_source.py').write_text(inspect.getsource(eager_attention_forward))
for variant in ['bf16','fia_bsh']:
 with torch.inference_mode():
  engine=Engine(load_config(LAB/'configs/bf16.json'),LAB/'models/bge-large-zh-v1.5')
  if variant=='fia_bsh':patch(engine,'BSH')
  result=evaluate(engine,LAB/'data',out/variant,reference=LAB/'runs/ref_cpu_fp32',gate_path=LAB/'configs/quality_gate.json')
  print('QUALITY',variant,result['status'],result['retrieval'],flush=True)
  del engine
  torch.npu.empty_cache()
