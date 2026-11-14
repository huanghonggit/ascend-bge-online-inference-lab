# Template: initialize with init_replay.py before executing.
# Recorded mathematics; public wrapper has not been rerun on NPU.
import os,sys,time,json,subprocess
from pathlib import Path
P=Path('/workspace/fia_bf16_retest_20261115')
phase=sys.argv[1]
if phase=='baseline':
    jobs=[('A',v,r) for r,order in enumerate([['fp32','bf16','fia','fia_bsh'],['bf16','fia_bsh','fp32','fia'],['fia','fp32','fia_bsh','bf16']],1) for v in order]
elif phase=='screen':
    jobs=[(g,v,1) for g in ['B','C','D','F','T'] for v in ['bf16','fia_bsh']]
else:jobs=json.loads((P/(phase+'.jobs.json')).read_text())
groups={'A':('0','1',None),'B':('1,npu_affine:1','1',None),'C':('2,npu_affine:1','1',None),'D':('0','2',None),'E':('1,npu_affine:1','2',None),'F':('2,npu_affine:1','2',None),'T':('0','2','48-63')}
for g,v,r in jobs:
    name=f'{phase}_{g}_{v}_r{r:02}'
    env=os.environ.copy();env.pop('ASCEND_LAUNCH_BLOCKING',None);env['CPU_AFFINITY_CONF'],env['TASK_QUEUE_ENABLE'],cpus=groups[g]
    cmd=[sys.executable,str(P/'experiment.py'),'--variant',v,'--out',name]
    if phase=='baseline' and r==1 and v=='fp32':cmd+=['--reference']
    if cpus:cmd=['taskset','-c',cpus]+cmd
    if phase=='profile':cmd+=['--profile']
    print('START',name,flush=True)
    t=time.time()
    with (P/(name+'.log')).open('w') as log:
        result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    print('END',name,'rc',result.returncode,'seconds',round(time.time()-t,1),flush=True)
    if result.returncode:
        print((P/(name+'.log')).read_text()[-5000:],flush=True);sys.exit(result.returncode)
    data=json.loads((P/'runs'/name/'result.json').read_text())
    print(json.dumps({k:data[k] for k in ['variant','model','e2e','quality','cgroup_delta'] if k in data}),flush=True)
