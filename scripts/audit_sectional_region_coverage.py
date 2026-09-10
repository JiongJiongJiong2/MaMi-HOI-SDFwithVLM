"""Frozen-row validation ablation: diverse regions, no learned/contact optimizer."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_bps_surface_queries import read_canonical_ply,sha256,memory_snapshot
from scripts.run_sectional_prior_diagnostic import _sample_surface_candidates,stable_seed
from manip.geometry.sectional_prior import build_contact_section_frame,rank_surface_candidates,METHODS


def diverse(order,points,radius,count=10):
    chosen=[]
    for i in order:
        if not chosen or np.all(np.linalg.norm(points[chosen]-points[i],axis=1)>=radius):
            chosen.append(i)
            if len(chosen)==count:break
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root_folder',type=Path,required=True)
    p.add_argument('--baseline_results',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();data=a.data_root_folder.resolve()
    if out==data or data in out.parents:raise ValueError('Output inside data')
    if out.exists():raise FileExistsError(out)
    baseline=json.loads((a.baseline_results/'summary.json').read_text())
    assert baseline['config']['eval_split']=='validation'
    cfg=baseline['config'];assert cfg['seed']==1 and cfg['candidate_count']==512 and cfg['hit_radius_ratio']==.08
    raw=list(csv.DictReader((a.baseline_results/'samples.csv').open(encoding='utf-8')))
    objects=sorted({r['object_name'] for r in raw})
    files=[a.baseline_results/'summary.json',a.baseline_results/'samples.csv',Path(__file__),
           Path(__file__).with_name('run_sectional_prior_diagnostic.py'),
           Path(__file__).with_name('audit_bps_surface_queries.py'),
           Path(__file__).resolve().parents[1]/'manip/geometry/sectional_prior.py']
    files += [data/'rest_object_geo'/f'{n}.ply' for n in objects]
    before={str(p):sha256(p) for p in files};out.mkdir(parents=True,exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(dict(status='EXPLORATORY_VALIDATION_ONLY',
        change='Greedy separation >= existing hit radius; identical treatment of all four rankings',
        candidate_count=512,seed=1,separation_extent_ratio=.08,top_k=[1,5,10],
        no_parameter_search=True,input_hashes=before),indent=2))
    start=time.perf_counter();surfaces={};rows=[];max_replay_error=0.
    for name in objects:
        mesh_path=data/'rest_object_geo'/f'{name}.ply'
        v,f=read_canonical_ply(mesh_path)
        # Baseline trimesh honors PLY `property float` as float32, then stores
        # vertices as float64. Reproduce that declared precision, not a new mesh.
        with mesh_path.open(encoding='ascii') as handle:
            header=[]
            for line in handle:
                header.append(line.strip())
                if line.strip()=='end_header':break
        assert 'property float x' in header or 'property float32 x' in header
        v=v.astype(np.float32).astype(np.float64)
        pts,normals=_sample_surface_candidates(np,SimpleNamespace(vertices=v,faces=f),512,stable_seed(1,name))
        surfaces[name]=(pts,normals,(v.min(0)+v.max(0))/2,float(np.ptp(v,axis=0).max()))
    for idx,r in enumerate(raw):
        points,normals,center,extent=surfaces[r['object_name']]
        source=np.array([float(r['source_'+k]) for k in 'xyz']);target=np.array([float(r['target_'+k]) for k in 'xyz'])
        up=np.array([float(r['up_'+k]) for k in 'xyz']);si=int(np.linalg.norm(points-source,axis=1).argmin())
        assert np.linalg.norm(points[si]-source)<1e-10
        frame=build_contact_section_frame(source,center,up,extent)
        rankings=rank_surface_candidates(points,normals,si,frame,
            seed=stable_seed(1,r['sequence_name'],int(r['absolute_frame']),r['source_hand']),
            exclusion_ratio=cfg['exclusion_ratio'],normal_weight=cfg['normal_weight'])
        result={k:r[k] for k in ('sequence_name','absolute_frame','source_hand','object_name')}
        for method,order in rankings.items():
            chosen=diverse(order,points,.08*extent)
            result[method+'_diverse_count']=len(chosen)
            for k in (1,5,10):
                original=float(np.linalg.norm(points[order[:k]]-target,axis=1).min()/extent)
                max_replay_error=max(max_replay_error,abs(original-float(r[f'{method}_top{k}_error_norm'])))
                assert abs(original-float(r[f'{method}_top{k}_error_norm']))<1e-9
                error=float(np.linalg.norm(points[chosen[:k]]-target,axis=1).min()/extent)
                result[f'{method}_diverse_top{k}_error_norm']=error
                result[f'{method}_diverse_top{k}_hit']=float(error<=.08)
                result[f'{method}_original_top{k}_hit']=float(r[f'{method}_top{k}_hit'])
        rows.append(result)
        if (idx+1)%500==0:print('Processed',idx+1,flush=True)
    sequences=sorted({r['sequence_name'] for r in rows})
    def macro(rr,key):
        seq=sorted({r['sequence_name'] for r in rr})
        return float(np.mean([np.mean([r[key] for r in rr if r['sequence_name']==s]) for s in seq]))
    summary={m:{f'{variant}_top{k}':macro(rows,f'{m}_{variant}_top{k}_hit') for variant in ('original','diverse') for k in (1,5,10)} for m in METHODS}
    by_object={n:{m:macro([r for r in rows if r['object_name']==n],f'{m}_diverse_top5_hit') for m in METHODS} for n in objects}
    groups={s:[r for r in rows if r['sequence_name']==s] for s in sequences}
    rng=np.random.default_rng(1);indices=rng.integers(0,len(sequences),size=(10000,len(sequences)))
    intervals={}
    for variant in ('original','diverse'):
        for method in ('section_chord','section_chord_normal'):
            diff=np.array([np.mean([r[f'{method}_{variant}_top5_hit']-r[f'center_antipode_{variant}_top5_hit'] for r in groups[s]]) for s in sequences])
            intervals[f'{method}_{variant}_minus_antipode']={'mean':float(diff.mean()),'sequence_bootstrap_95':np.quantile(diff[indices].mean(1),[.025,.975]).tolist()}
    assert before=={str(p):sha256(p) for p in files}
    report=dict(status='EXPLORATORY_VALIDATION_NOT_TEST',sample_count=len(rows),sequence_count=len(sequences),
        methods=summary,by_object_diverse_top5=by_object,paired_bootstrap=intervals,max_replay_error=max_replay_error,
        min_diverse_count=min(r[m+'_diverse_count'] for r in rows for m in METHODS),
        elapsed_seconds=time.perf_counter()-start,memory=memory_snapshot(),input_hashes_unchanged=True)
    with (out/'samples.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
