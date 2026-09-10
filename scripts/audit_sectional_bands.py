"""One predeclared validation-only broad-band regional-prior check."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import os
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_bps_surface_queries import read_canonical_ply,sha256,memory_snapshot
from scripts.run_sectional_prior_diagnostic import _sample_surface_candidates,stable_seed
from scripts.audit_sectional_region_coverage import diverse
from manip.geometry.sectional_prior import build_contact_section_frame


def band_orders(points,source,frame,extent):
    delta=points-source
    travel=delta@np.asarray(frame.inward)/extent
    horizontal=np.abs((points-np.asarray(frame.horizontal_origin))@np.asarray(frame.up))/extent
    vertical=np.abs((points-np.asarray(frame.centre))@np.asarray(frame.vertical_normal))/extent
    eligible=np.flatnonzero(np.linalg.norm(delta,axis=1)>=.08*extent)
    distances={'horizontal_band':horizontal,'vertical_band':vertical,'union_band':np.minimum(horizontal,vertical)}
    # Only widening changes: preserve original inward penalty/reward coefficients.
    return {name:sorted(eligible.tolist(),key=lambda i:(max(float(d[i])-.08,0)+.5*max(-float(travel[i]),0)-.05*float(travel[i]),i)) for name,d in distances.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root_folder',type=Path,required=True)
    p.add_argument('--baseline_results',type=Path,required=True)
    p.add_argument('--diverse_results',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    a=p.parse_args();root=a.data_root_folder.resolve();out=a.output_dir.resolve()
    if out==root or root in out.parents:raise ValueError('Output inside data root')
    if out.exists():raise FileExistsError(out)
    base=json.loads((a.baseline_results/'summary.json').read_text());assert base['config']['eval_split']=='validation'
    raw=list(csv.DictReader((a.baseline_results/'samples.csv').open(encoding='utf-8')))
    prior=list(csv.DictReader((a.diverse_results/'samples.csv').open(encoding='utf-8')))
    assert len(raw)==len(prior)
    names=sorted({r['object_name'] for r in raw})
    paths=[a.baseline_results/'summary.json',a.baseline_results/'samples.csv',a.diverse_results/'samples.csv',Path(__file__)]
    paths += [Path(__file__).with_name(n) for n in ('audit_bps_surface_queries.py','run_sectional_prior_diagnostic.py','audit_sectional_region_coverage.py')]
    paths += [Path(__file__).resolve().parents[1]/'manip/geometry/sectional_prior.py']
    paths += [root/'rest_object_geo'/f'{n}.ply' for n in names]
    hashes={str(p):sha256(p) for p in paths};out.mkdir(parents=True,exist_ok=False)
    protocol=dict(primary='union_band',secondary=['horizontal_band','vertical_band'],
        half_width_extent_ratio=.08,separation_extent_ratio=.08,seed=1,candidate_count=512,
        gate='Primary top5 gain >= .05 over diverse antipode and paired sequence CI lower > 0; not below previous diverse chord_normal',
        scope='EXPLORATORY_VALIDATION_ONLY; no parameter search or per-object winner selection',input_hashes=hashes)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    start=time.perf_counter();surfaces={};rows=[]
    for name in names:
        v,f=read_canonical_ply(root/'rest_object_geo'/f'{name}.ply');v=v.astype(np.float32).astype(np.float64)
        points,_=_sample_surface_candidates(np,SimpleNamespace(vertices=v,faces=f),512,stable_seed(1,name))
        surfaces[name]=(points,(v.min(0)+v.max(0))/2,float(np.ptp(v,axis=0).max()))
    for r,b in zip(raw,prior):
        assert all(r[k]==b[k] for k in ('sequence_name','absolute_frame','source_hand','object_name'))
        points,center,extent=surfaces[r['object_name']]
        source=np.array([float(r['source_'+k]) for k in 'xyz']);target=np.array([float(r['target_'+k]) for k in 'xyz'])
        assert np.linalg.norm(points-source,axis=1).min()<1e-10
        frame=build_contact_section_frame(source,center,[float(r['up_'+k]) for k in 'xyz'],extent)
        result={k:r[k] for k in ('sequence_name','absolute_frame','source_hand','object_name')}
        result['antipode_top5']=float(b['center_antipode_diverse_top5_hit']);result['previous_best_top5']=float(b['section_chord_normal_diverse_top5_hit'])
        for name,order in band_orders(points,source,frame,extent).items():
            chosen=diverse(order,points,.08*extent);assert len(chosen)==10
            for k in (1,5,10):result[f'{name}_top{k}']=float(np.linalg.norm(points[chosen[:k]]-target,axis=1).min()<=.08*extent)
        rows.append(result)
    groups={s:[r for r in rows if r['sequence_name']==s] for s in sorted({r['sequence_name'] for r in rows})}
    fields=[k for k in rows[0] if k.endswith(('top1','top5','top10'))]
    per_sequence={k:np.array([np.mean([r[k] for r in rr]) for rr in groups.values()]) for k in fields}
    scores={k:float(v.mean()) for k,v in per_sequence.items()}
    difference=per_sequence['union_band_top5']-per_sequence['antipode_top5']
    rng=np.random.default_rng(1);ix=rng.integers(0,len(groups),size=(10000,len(groups)))
    ci=np.quantile(difference[ix].mean(1),[.025,.975])
    passed=bool(difference.mean()>=.05 and ci[0]>0 and scores['union_band_top5']>=scores['previous_best_top5'])
    assert hashes=={str(p):sha256(p) for p in paths}
    report=dict(gate='PASS_VALIDATION_ONLY' if passed else 'NO_GO_CURRENT_PRIMARY',scores=scores,
        primary_gain=float(difference.mean()),paired_sequence_95=ci.tolist(),samples=len(rows),sequences=len(groups),
        input_hashes_unchanged=True,elapsed_seconds=time.perf_counter()-start,memory=memory_snapshot())
    with (out/'samples.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
