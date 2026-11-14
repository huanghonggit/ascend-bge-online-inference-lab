"""Run one recorded stage in an initialized replay root, with overwrite guards."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ('baseline', 'screen', 'confirm', 'profile', 'quality_eager', 'quality_graph', 'micro', 'micro_accuracy')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=STAGES)
    args = parser.parse_args()
    if not (ROOT / 'replay_manifest.json').is_file():
        parser.error('Use init_replay.py first; do not run this wrapper in archived evidence')
    if args.stage != 'baseline' and not (ROOT / 'stage_baseline.done.json').exists():
        parser.error('Complete baseline first so embedding references and FIA comparisons exist')
    if args.stage.startswith('quality_') and not (Path('/workspace/ascend_bge_online_lab') / 'runs/ref_cpu_fp32/quality.json').is_file():
        parser.error('Create the frozen CPU FP32 quality reference first')
    started = ROOT / ('stage_' + args.stage + '.started.json')
    with started.open('x', encoding='utf-8') as stream:
        json.dump({'stage': args.stage, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, stream)
    env = os.environ.copy()
    env.pop('ASCEND_LAUNCH_BLOCKING', None)
    env.update(CPU_AFFINITY_CONF='0', TASK_QUEUE_ENABLE='1', PYTHONUNBUFFERED='1')
    if args.stage in ('baseline', 'screen', 'confirm', 'profile'):
        cmd = [sys.executable, str(ROOT / 'runner.py'), args.stage]
    else:
        script = {'quality_eager': 'quality_probe.py', 'quality_graph': 'quality_graph.py',
                  'micro': 'micro_matched.py', 'micro_accuracy': 'micro_accuracy.py'}[args.stage]
        cmd = [sys.executable, str(ROOT / script)]
        if args.stage != 'quality_graph':
            env['TASK_QUEUE_ENABLE'] = '2'
            cmd = ['taskset', '-c', '48-63'] + cmd
    log_path = ROOT / ('stage_' + args.stage + '.log')
    print('RUN', json.dumps(cmd), '\nLOG', log_path, flush=True)
    with log_path.open('x', encoding='utf-8') as log:
        completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   cwd='/workspace/ascend_bge_online_lab')
    if completed.returncode:
        print(log_path.read_text(encoding='utf-8', errors='replace')[-8000:])
        raise SystemExit(completed.returncode)
    if args.stage in ('quality_eager', 'quality_graph'):
        variants = ('bf16', 'fia_bsh') if args.stage == 'quality_eager' else ('graph_fp32', 'graph_bf16')
        for variant in variants:
            quality = json.loads((ROOT / 'quality' / variant / 'quality.json').read_text(encoding='utf-8'))
            if quality['status'] != 'passed':
                raise RuntimeError('Quality gate failed: ' + variant)
    (ROOT / ('stage_' + args.stage + '.done.json')).write_text(
        json.dumps({'stage': args.stage, 'returncode': 0,
                    'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, indent=2), encoding='utf-8')
    print('DONE', args.stage, flush=True)

if __name__ == '__main__':
    main()
