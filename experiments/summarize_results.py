"""Summarize raw process P50s without mixing screen and confirmation runs."""
import argparse, csv, json, statistics
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
groups = defaultdict(list)
for p in sorted((args.root / 'runs').glob('*/result.json')):
    data = json.loads(p.read_text(encoding='utf-8'))
    if data.get('profile_only'):
        continue
    phase, group, rest = p.parent.name.split('_', 2)
    variant, round_id = rest.rsplit('_r', 1)
    groups[(phase, group, variant)].append(data)
rows = []
for (phase, group, variant), items in sorted(groups.items()):
    model = [x['model']['p50_ms'] for x in items]
    e2e = [x['e2e']['p50_ms'] for x in items]
    rows.append({'phase': phase, 'group': group, 'variant': variant, 'processes': len(items),
                 'model_p50_median_ms': statistics.median(model), 'model_min_ms': min(model), 'model_max_ms': max(model),
                 'e2e_p50_median_ms': statistics.median(e2e),
                 'all_steady_recompile_zero': all(x.get('graph_count_delta', 0) == 0 for x in items)})
if not rows:
    parser.error('No timing results found')
writer = csv.DictWriter(__import__('sys').stdout, fieldnames=list(rows[0]))
writer.writeheader()
writer.writerows(rows)
