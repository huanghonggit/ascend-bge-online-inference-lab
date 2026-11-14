"""Audit checked-in evidence with Python's standard library; no NPU or model download."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/2026-11-15'

def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-8), (a, b)

def pct(values, q):
    values = sorted(values)
    index = (len(values)-1) * q
    low = int(index)
    high = min(low+1, len(values)-1)
    return values[low] + (values[high]-values[low]) * (index-low)

def main():
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = load(DATA / 'SHA256SUMS.json')
    assert set(manifest) == {str(p.relative_to(DATA)).replace('\\','/') for p in DATA.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.json'}
    for name, expected in manifest.items():
        assert digest(DATA / name) == expected, name
    dataset = load(ROOT / 'data/manifest.json')
    for name, expected in dataset['sha256'].items():
        assert digest(ROOT / 'data' / name) == expected, name
    groups = defaultdict(list)
    raw = {}
    profiles = []
    for p in sorted((DATA / 'runs').glob('*/result.json')):
        r = load(p)
        if r.get('profile_only'):
            profiles.append(r)
            continue
        raw[p.parent.name] = r
        phase, group, rest = p.parent.name.split('_', 2)
        variant, round_id = rest.rsplit('_r', 1)
        assert r['variant'] == variant
        groups[(phase, group, variant)].append(r)
        for values_key, stat_key, n in [('raw_model_ms','model',64),('raw_enqueue_ms','enqueue_wall',64),('raw_e2e_ms','e2e',32)]:
            values = r[values_key]
            assert len(values) == n and all(math.isfinite(v) and v > 0 for v in values)
            close(statistics.median(values), r[stat_key]['p50_ms'])
            close(pct(values, .95), r[stat_key]['p95_ms'])
        assert r.get('graph_count_delta', 0) == 0
        for key in ('nr_throttled','throttled_time','throttled_usec'):
            assert r['cgroup_delta'].get(key, 0) == 0
    assert len(raw) == 43 and len(profiles) == 4 and len(groups) == 21
    assert len({r['data_sha256'] for r in raw.values()}) == 1
    for r in load(DATA / 'group_summary.json'):
        key = (r['phase'],r['group'],r['variant'])
        items = groups[key]
        assert len(items) == r['processes'] == (1 if r['phase']=='screen' else 3)
        for field, prefix in [('model','model'),('e2e','e2e'),('enqueue_wall','enqueue')]:
            v = [i[field]['p50_ms'] for i in items]
            for suffix, fn in [('median',statistics.median),('min',min),('max',max)]:
                close(fn(v),r[prefix+'_p50_ms_'+suffix])
    with (DATA / 'process_summary.csv').open(encoding='utf-8-sig',newline='') as stream:
        process_rows = list(csv.DictReader(stream))
    assert len(process_rows) == 43
    for r in process_rows:
        p = raw[r['run']]
        for field in ('model','e2e'):
            close(r[field+'_p50_ms'],p[field]['p50_ms'])
            close(r[field+'_p95_ms'],p[field]['p95_ms'])
    micro = load(DATA / 'micro_matched.json')['results']
    assert len(micro) == 36
    mg = defaultdict(list)
    for r in micro:
        assert len(r['raw_ms']) == 100
        close(statistics.median(r['raw_ms']),r['p50_ms'])
        mg[(r['s'],r['variant'])].append(r['p50_ms'])
    with (DATA / 'micro_summary.csv').open(encoding='utf-8-sig',newline='') as stream:
        for r in csv.DictReader(stream):
            v = mg[(int(r['S']),r['variant'])]
            assert len(v) == int(r['rounds']) == 3
            close(statistics.median(v),r['p50_median_ms'])
            close(min(v),r['p50_min_ms']);close(max(v),r['p50_max_ms'])
    for r in load(DATA / 'micro_accuracy_fp64.json')['results']:
        assert r['checks']['core_matches_installed_bert']
        for variant in ('fia_core','fia_bsh'):
            assert .99999 <= r['checks'][variant]['cosine'] <= 1
            assert r['checks'][variant]['max_abs_error'] < .002
    ps = load(DATA / 'profiles.json')
    for name, expected in ps.items():
        folder = DATA / 'profiler' / name
        with (folder / 'kernel_details.csv').open(encoding='utf-8-sig',newline='') as f:
            kernels = list(csv.DictReader(f))
        close(len(kernels)/3, expected['kernels_per_step'])
        with (folder / 'step_trace_time.csv').open(encoding='utf-8-sig',newline='') as f:
            steps = list(csv.DictReader(f))
        close(statistics.mean(float(r['Computing']) for r in steps)/1000,expected['computing_ms'])
        for s in expected['spans']:
            intervals = sorted((float(r['Start Time(us)']),float(r['Start Time(us)'])+float(r['Duration(us)'])) for r in kernels if r['Step Id'] == s['step'])
            start, end = intervals[0]
            union = 0
            for a,b in intervals[1:]:
                if a > end:
                    union += end-start
                    start,end = a,b
                else:
                    end = max(end,b)
            union += end-start
            span = max(b for a,b in intervals)-intervals[0][0]
            close((span-union)/1000,s['gap_ms'])
            close(union/1000,s['union_ms'])
    quality = load(DATA / 'quality.json')
    assert set(quality) == {'bf16','fia_bsh','graph_fp32','graph_bf16'}
    for variant,r in quality.items():
        assert r['status'] == 'passed' and r['comparison']['passed'], variant
        close(r['retrieval']['recall5'],1.0)
        close(r['retrieval']['mrr10'],.9453125)
    print(json.dumps({'passed':True,'timing_processes':len(raw),'profile_processes':len(profiles),
        'groups':len(groups),'model_samples':sum(len(r['raw_model_ms']) for r in raw.values()),
        'e2e_samples':sum(len(r['raw_e2e_ms']) for r in raw.values()),'micro_samples':3600,
        'quality_variants':len(quality),'data_files_hashed':len(manifest)},indent=2))

if __name__ == '__main__':
    main()
