"""Regenerate the published figures from the checked-in measurements; no NPU."""
import csv
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/2026-11-15'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
ROWS = json.loads((DATA / 'group_summary.json').read_text(encoding='utf-8'))
BG, INK, MUTED = '#f5f7fa', '#172b44', '#536477'
BLUE, TEAL, RED, GRAY = '#3877bb', '#168b87', '#cb6b58', '#8796a8'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.titlesize': 13, 'axes.labelcolor': MUTED, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.spines.left': False,
    'axes.edgecolor': '#cad4df', 'svg.fonttype': 'path', 'svg.hashsalt': 'bge-lab',
    'figure.facecolor': BG, 'axes.facecolor': BG})

def row(phase, group, variant):
    return next(r for r in ROWS if (r['phase'], r['group'], r['variant']) == (phase, group, variant))

def save(fig, name, title, subtitle, footer):
    fig.suptitle(title, x=.06, y=.97, ha='left', fontsize=19, fontweight='bold')
    fig.text(.06, .89, subtitle, fontsize=10.5, color=MUTED)
    fig.text(.06, .035, footer, fontsize=9, color=MUTED)
    fig.subplots_adjust(left=.07, right=.97, top=.78, bottom=.2, wspace=.3)
    for ext in ('png', 'svg'):
        fig.savefig(OUT / (name + '.' + ext), dpi=180, facecolor=BG,
                    metadata={'Date': None} if ext == 'svg' else {})
        if ext == 'svg':
            svg_path = OUT / (name + '.svg')
            svg_text = svg_path.read_text(encoding='utf-8')
            svg_path.write_text('\n'.join(line.rstrip() for line in svg_text.splitlines()) + '\n',
                                encoding='utf-8', newline='\n')
    plt.close(fig)

def bars(ax, rows, labels, metric, colors, limit=None):
    vals = np.array([r[metric + '_p50_ms_median'] for r in rows])
    low = np.array([r[metric + '_p50_ms_min'] for r in rows])
    high = np.array([r[metric + '_p50_ms_max'] for r in rows])
    bars = ax.bar(np.arange(len(rows)), vals, color=colors, width=.58, zorder=3,
                  yerr=np.array([vals-low, high-vals]), capsize=4,
                  error_kw={'ecolor': INK, 'linewidth': 1})
    top = limit or max(high) * 1.25
    for b, value, hi in zip(bars, vals, high):
        ax.text(b.get_x()+b.get_width()/2, hi+top*.025, f'{value:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(np.arange(len(rows)), labels)
    ax.tick_params(axis='both', length=0)
    ax.set_ylim(0, top)
    ax.grid(axis='y', color='#dde4ec', zorder=0)
    ax.set_ylabel('Latency (ms) | lower is better')

r = [row('baseline','A',v) for v in ['fp32','bf16','fia','fia_bsh']]
fig, axs = plt.subplots(1,2,figsize=(13,5.5))
for ax, metric, title in zip(axs, ['model','e2e'], ['Model only', 'In-process end to end']):
    bars(ax, r, ['FP32\neager','BF16\neager','FIA\nBNSD','FIA BSH\nmask reuse'], metric, [GRAY,BLUE,RED,TEAL])
    ax.set_title(title, loc='left', pad=15)
save(fig,'baseline','A direct FIA replacement was not enough',
     'Ascend910B3 | BGE-large-zh-v1.5 | B8 / S256 | Group A: queue=1, affinity=0, no taskset',
     '3 processes / variant; bar = median of process P50s; whisker = min-max. E2E excludes HTTP and queueing.')

fig, axs = plt.subplots(1,2,figsize=(13,5.5))
for ax, metric, title in zip(axs,['model','e2e'],['Model only','In-process end to end']):
    for j,(v,label,c) in enumerate([('bf16','BF16 eager',BLUE),('fia_bsh','FIA BSH + mask reuse',TEAL)]):
        rr=[row('baseline','A',v),row('confirm','D',v),row('confirm','T',v)]
        vals=np.array([r[metric+'_p50_ms_median'] for r in rr])
        lo=np.array([r[metric+'_p50_ms_min'] for r in rr]);hi=np.array([r[metric+'_p50_ms_max'] for r in rr])
        xx=np.arange(3)+(j-.5)*.35
        ax.bar(xx,vals,width=.32,color=c,label=label,zorder=3,yerr=[vals-lo,hi-vals],capsize=3,error_kw={'ecolor':INK,'linewidth':1})
        for x,y,h in zip(xx,vals,hi):ax.text(x,h+1,f'{y:.2f}',ha='center',fontsize=9)
    ax.set_xticks(range(3),['A\nqueue=1','D\nqueue=2','T\nqueue=2 + taskset'])
    ax.set_ylabel('Latency (ms) | lower is better');ax.set_ylim(0,52 if metric=='e2e' else 37)
    ax.grid(axis='y',color='#dde4ec',zorder=0);ax.tick_params(length=0);ax.set_title(title,loc='left',pad=15)
axs[0].legend(frameon=False,loc='upper right',fontsize=9)
save(fig,'queue_affinity','Queueing and CPU affinity changed the outcome',
     'B8 / S256 | T used CPU48-63 on the measured host; choose CPUs from your own NPU topology',
     'A and D/T were measured in separate stages. These are combined observations, not an additive causal decomposition.')

fig, axs=plt.subplots(1,2,figsize=(11,5.5))
rr=[row('confirm','A',v) for v in ['graph_fp32','graph_bf16']]
for ax,metric,title in zip(axs,['model','e2e'],['Model only','In-process end to end']):
    bars(ax,rr,['TorchAir FP32','TorchAir BF16'],metric,[GRAY,TEAL]);ax.set_title(title,loc='left',pad=15)
speed=rr[0]['model_p50_ms_median']/rr[1]['model_p50_ms_median']
save(fig,'torchair_dtype',f'BF16 delivered {speed:.2f}x model speedup in graph mode',
     'Same TorchAir execution mode | Group A | B8 / S256 | 3 interleaved processes / dtype',
     'First compilation is in warmup. Each process compiled one graph; no steady-state recompilation. No FIA BSH combination.')

micro=list(csv.DictReader((DATA/'micro_summary.csv').open(encoding='utf-8-sig',newline='')))
fig,axs=plt.subplots(1,2,figsize=(13,5.5))
for ax,variants,title in zip(axs,[['eager_core','fia_core'],['eager_from_bsh','fia_bsh']],['Contiguous BNSD core','Attention path starting from BSH']):
    for j,(v,c,label) in enumerate(zip(variants,[BLUE,TEAL],['Eager reference','FIA'])):
        rr=[next(r for r in micro if int(r['S'])==s and r['variant']==v) for s in [64,256,512]]
        y=np.array([float(r['p50_median_ms']) for r in rr]);lo=np.array([float(r['p50_min_ms']) for r in rr]);hi=np.array([float(r['p50_max_ms']) for r in rr])
        x=np.arange(3)+(j-.5)*.35
        ax.bar(x,y,width=.32,color=c,label=label,zorder=3,yerr=[y-lo,hi-y],capsize=3,error_kw={'ecolor':INK,'linewidth':1})
        for xx,yy,hh in zip(x,y,hi):ax.text(xx,hh+.018,f'{yy:.3f}',ha='center',fontsize=9)
    ax.set_xticks(range(3),['S64','S256','S512']);ax.set_ylim(0,.76)
    ax.set_title(title,loc='left',pad=15);ax.set_ylabel('Latency (ms) | lower is better');ax.grid(axis='y',color='#dde4ec',zorder=0);ax.tick_params(length=0)
axs[0].legend(frameon=False)
save(fig,'attention_micro','FIA gains depend on shape and measurement boundary',
     'BF16 | B8 / N16 / D64 | 20 warmups + 100 synchronized calls x 3 rounds | queue=2 + taskset',
     'Masks precreated; QKV projections excluded. At S256 the BNSD FIA core was slower, while the BSH path was faster.')

profiles=json.loads((DATA/'profiles.json').read_text(encoding='utf-8'))
p=[profiles['profile_A_'+v+'_r01'] for v in ['fp32','bf16','fia','fia_bsh']]
fig,axs=plt.subplots(1,3,figsize=(14,5.5))
for ax,vals,title,unit in zip(axs,[[r['computing_ms'] for r in p],[r['kernels_per_step'] for r in p],[statistics.mean(s['gap_ms'] for s in r['spans']) for r in p]],['Device Computing','Kernel count','Gap within kernel span'],['ms / step','kernels / step','ms / step']):
    ax.bar(range(4),vals,color=[GRAY,BLUE,RED,TEAL],width=.62,zorder=3)
    for x,y in enumerate(vals):ax.text(x,y+max(vals)*.04,f'{y:.2f}' if unit=='ms / step' else f'{y:.0f}',ha='center',fontsize=9)
    ax.set_xticks(range(4),['FP32','BF16','FIA\nBNSD','FIA\nBSH']);ax.set_ylim(0,max(vals)*1.24);ax.set_ylabel(unit);ax.set_title(title,loc='left',pad=15)
    ax.grid(axis='y',color='#dde4ec',zorder=0);ax.tick_params(length=0)
save(fig,'profiler','Faster device compute did not guarantee faster eager execution',
     'Group A | Separate profiler runs, 3 active steps each | BF16 Computing decreased while kernel count and gaps grew',
     'Gap is span minus union of kernel intervals; not pure CPU time or whole-device idle rate. Do not subtract from non-profile wall time.')

fig,ax=plt.subplots(figsize=(13,6));ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
def box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.018',facecolor=color,edgecolor='none'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color='white',linespacing=1.5)
def arrow(x1,y1,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':MUTED,'lw':1.5})
ax.text(.01,.90,'Original FIA adapter  /  repeated at every one of 24 layers',fontweight='bold',fontsize=13)
for x,w,t,c in [(.01,.15,'Q / K / V\nprojection',BLUE),(.22,.21,'BSH -> BNSD\ncontiguous',RED),(.49,.23,'mask materialization\n+ FIA BNSD',RED),(.78,.20,'BNSD -> BSH\ncontiguous',RED)]:box(x,.58,w,.19,t,c)
for a,b in [(.16,.22),(.43,.49),(.72,.78)]:arrow(a,.675,b,.675)
ax.text(.01,.44,'Revised FIA adapter  /  serial eager only',fontweight='bold',fontsize=13)
for x,w,t,c in [(.01,.15,'Q / K / V\nprojection',BLUE),(.27,.31,'FIA directly in BSH\nshared mask for 24 layers',TEAL),(.72,.26,'BSH output\nno layout conversion',TEAL)]:box(x,.13,w,.19,t,c)
arrow(.16,.225,.27,.225);arrow(.58,.225,.72,.225)
ax.text(.28,.02,'Reset mask at each encoder forward; shared state is not a concurrency-safe design.',fontsize=10,color=MUTED)
save(fig,'fia_path','Optimize the whole attention call path',
     'Layout and repeated mask work can offset the savings of a fused attention kernel',
     'Schematic only. The measured BSH FIA kernel itself was not faster than BNSD; the end-to-end attention path improved.')
print('Generated 6 PNG + 6 SVG figures in', OUT)
