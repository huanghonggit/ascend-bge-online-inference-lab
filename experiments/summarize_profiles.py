import csv,json,collections,statistics
from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--runs',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists(): parser.error('Output already exists; choose a new file')
summaries={}
for root in args.runs.glob('profile*'):
 d=next(root.rglob('kernel_details.csv')).parent
 rows=list(csv.DictReader((d/'kernel_details.csv').open(encoding='utf-8-sig')))
 groups=collections.defaultdict(list)
 for r in rows: groups[r['Name']].append(float(r['Duration(us)']))
 top=sorted([{'name':k,'count_per_step':len(v)/3,'ms_per_step':sum(v)/3000} for k,v in groups.items()],key=lambda x:-x['ms_per_step'])
 step=list(csv.DictReader((d/'step_trace_time.csv').open(encoding='utf-8-sig')))
 events=json.loads((d/'trace_view.json').read_text(encoding='utf-8'))
 host=collections.Counter(e['name'] for e in events if e.get('cat')=='cpu_op' and e.get('ph')=='X')
 selected={k:v/3 for k,v in host.items() if any(w in k for w in ['contiguous','clone','copy','lt.','aten::lt','softmax','fused_infer','transpose','reshape','to','cast'])}
 spans=[]
 for s in sorted(set(r['Step Id'] for r in rows)):
  ints=sorted((float(r['Start Time(us)']),float(r['Start Time(us)'])+float(r['Duration(us)'])) for r in rows if r['Step Id']==s)
  total=0;start,end=ints[0]
  for a,b in ints[1:]:
   if a>end:total+=end-start;start,end=a,b
   else:end=max(end,b)
  total+=end-start
  spans.append({'step':s,'span_ms':(max(b for a,b in ints)-ints[0][0])/1000,'union_ms':total/1000,'gap_ms':(max(b for a,b in ints)-ints[0][0]-total)/1000})
 out={'kernels_per_step':len(rows)/3,'computing_ms':statistics.mean(float(x['Computing']) for x in step)/1000,'free_ms':statistics.mean(float(x['Free']) for x in step)/1000,'spans':spans,'kernels':top,'host_selected_counts_per_step':selected}
 summaries[root.name]=out
if not summaries: parser.error('No profile runs found')
with args.out.open('x',encoding='utf-8') as stream: json.dump(summaries,stream,indent=2)
print('WROTE',args.out,'profiles',len(summaries))
