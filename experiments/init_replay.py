"""Materialize the recorded experiment scripts in a new, isolated directory."""
import argparse
import ast
import json
import re
from pathlib import Path

ORIGINAL = '/workspace/fia_bf16_retest_20261115'
ORIGINAL_LAB = '/workspace/ascend_bge_online_lab'
FILES = ('experiment.py', 'runner.py', 'confirm.jobs.json', 'profile.jobs.json',
         'quality_probe.py', 'quality_graph.py', 'micro_matched.py',
         'micro_accuracy.py', 'replay_stage.py', 'summarize_results.py', 'summarize_profiles.py')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='New absolute output directory')
    parser.add_argument('--lab', required=True, type=Path, help='Absolute repository directory in the container')
    parser.add_argument('--cpus', required=True, help='Verified NPU-local CPU list for taskset, for example 48-63')
    args = parser.parse_args()
    if not args.root.is_absolute() or not args.lab.is_absolute():
        parser.error('--root and --lab must be absolute')
    target, lab = args.root.resolve(), args.lab.resolve()
    if not re.fullmatch(r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*', args.cpus):
        parser.error('--cpus must be a taskset CPU list')
    for part in args.cpus.split(','):
        bounds = [int(x) for x in part.split('-')]
        if len(bounds) == 2 and bounds[0] > bounds[1]:
            parser.error('CPU range start must not exceed end')
    if not (lab / 'bge_lab/core.py').is_file():
        parser.error('--lab must contain bge_lab/core.py')
    if str(target).replace('\\', '/') == ORIGINAL:
        parser.error('Refusing the archived experiment directory')
    if target.exists():
        parser.error('--root already exists; choose a new directory')
    source = Path(__file__).resolve().parent
    prepared = {}
    for name in FILES:
        text = (source / name).read_text(encoding='utf-8')
        text = text.replace(repr(ORIGINAL), repr(str(target)))
        text = text.replace(repr(ORIGINAL_LAB), repr(str(lab)))
        text = text.replace(repr('48-63'), repr(args.cpus))
        if name.endswith('.py'):
            ast.parse(text, filename=name)
        else:
            json.loads(text)
        prepared[name] = text
    target.mkdir(parents=True, exist_ok=False)
    for name, text in prepared.items():
        (target / name).write_text(text, encoding='utf-8', newline='\n')
    manifest = {'source': str(source), 'root': str(target), 'lab': str(lab),
                'cpus': args.cpus, 'files': list(FILES),
                'note': 'Initialization only. No NPU experiment was run.'}
    (target / 'replay_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
