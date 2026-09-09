"""CPU local rim connectivity and exterior-path diagnostic. No solid labels."""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_bps_surface_queries import read_canonical_ply, nearest_triangles, sha256, memory_snapshot
from scripts.material_oracle_probe import ray_hits


def topology(v,f):
    vertices,vi=np.unique(v,axis=0,return_inverse=True)
    mapped=vi[f]
    _,first,inverse=np.unique(np.sort(mapped,axis=1),axis=0,return_index=True,return_inverse=True)
    faces=mapped[first]
    edges=defaultdict(list)
    for i,t in enumerate(faces):
        for a,b in ((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):edges[tuple(sorted((a,b)))].append(i)
    return vertices,faces,first,inverse,edges


def face_path(edges,start,end,allowed,manifold_only):
    neighbors=defaultdict(set)
    for ids in edges.values():
        if manifold_only and len(ids)!=2:continue
        kept=[i for i in ids if i in allowed]
        for a in kept:neighbors[a].update(b for b in kept if b!=a)
    parents={start:None};todo=deque([start])
    while todo:
        a=todo.popleft()
        if a==end:
            path=[]
            while a is not None:path.append(a);a=parents[a]
            return path[::-1]
        for b in sorted(neighbors[a]):
            if b not in parents:parents[b]=a;todo.append(b)
    return None


def certify_segment(a,b,triangles,epsilon,max_queries=2048):
    """Numerical clearance bound using 1-Lipschitz exact triangle UDF.

    On each interval nearest endpoint is at most length/2 away. Thus
    min(endpoint distances)-length/2 is a lower bound for the entire interval.
    Bound is relative to available triangles, not unknown physical surfaces.
    """
    a,b=np.asarray(a,float),np.asarray(b,float)
    distances,_=nearest_triangles(np.array([a,b]),triangles,query_batch=8)
    intervals=[(a,b,float(distances[0]),float(distances[1]))]
    queries=2;lower=[];minimum=float(min(distances))
    while intervals:
        pending=[]
        for x,y,dx,dy in intervals:
            bound=min(dx,dy)-np.linalg.norm(y-x)/2
            if bound>epsilon:lower.append(float(bound))
            elif min(dx,dy)<=epsilon:
                return dict(status='NO_CLEARANCE_CERTIFICATE',queries=queries,min_sample_distance_m=minimum)
            else:pending.append((x,y,dx,dy))
        if not pending:break
        if queries+len(pending)>max_queries:
            return dict(status='QUERY_BUDGET_NO_CERTIFICATE',queries=queries,min_sample_distance_m=minimum)
        mids=np.array([(x+y)/2 for x,y,_,_ in pending])
        dm,_=nearest_triangles(mids,triangles,query_batch=8)
        queries+=len(dm);minimum=min(minimum,float(dm.min()));intervals=[]
        for (x,y,dx,dy),mid,d in zip(pending,mids,dm):
            intervals.extend([(x,mid,dx,float(d)),(mid,y,float(d),dy)])
    return dict(status='CLEAR_TO_AVAILABLE_TRIANGLES',queries=queries,
                certified_lower_bound_m=min(lower),min_sample_distance_m=minimum,
                interval_count=len(lower),epsilon_m=epsilon)


def audit_pair(row,v,f,cv,cf,first,inverse,edges):
    tri=v[f];extent=float(np.ptp(v,axis=0).max());lo,hi=v.min(0),v.max(0)
    a,b=row['hits'];q=np.array([(a['coordinate_m']+b['coordinate_m'])/2,row['scan_y_m'],row['z_m']])
    start,end=int(inverse[a['source_face_ids'][0]]),int(inverse[b['source_face_ids'][0]])
    radius=np.array([.035,.035,.05]);rlo,rhi=q-radius,q+radius
    candidate=cv[cf]
    allowed=set(np.flatnonzero(np.all(candidate.max(1)>=rlo,axis=1)&np.all(candidate.min(1)<=rhi,axis=1)).tolist())
    paths={}
    for label,manifold in (('manifold_edges_only',True),('all_shared_edges',False)):
        path=face_path(edges,start,end,allowed,manifold)
        if path is None:paths[label]=dict(status='NO_LOCAL_PATH')
        else:
            verts=candidate[path]
            paths[label]=dict(status='CONNECTED',face_hops=len(path)-1,
                              original_face_ids=first[path].tolist(),
                              path_vertex_bounds_m=[verts.min((0,1)).tolist(),verts.max((0,1)).tolist()],
                              face_centroids_m=verts.mean(1).tolist())
    directions=[np.array(d,float) for d in ((0,0,-1),(0,0,1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0))]
    rays=[]
    for d in directions:
        h,_=ray_hits(q,d,tri,extent)
        rays.append(dict(direction=d.tolist(),hit_count=len(h),first_hit_m=float(h[0]) if len(h) else None))
    path_info=None;perturb=[]
    clear=next((r for r in rays if r['hit_count']==0),None)
    if clear:
        direction=np.array(clear['direction']);axis=int(np.flatnonzero(direction)[0])
        target=q.copy();target[axis]=(hi[axis]+.1*extent) if direction[axis]>0 else (lo[axis]-.1*extent)
        path_info=dict(start_m=q.tolist(),end_m=target.tolist(),
                       end_outside_aabb=bool(np.any(target<lo)|np.any(target>hi)),
                       **certify_segment(q,target,tri,extent*1e-7))
        # Perturbations test ray intersection only, not the full clearance bound.
        for delta in np.vstack((np.eye(3),-np.eye(3)))*.00025:
            h,_=ray_hits(q+delta,direction,tri,extent)
            perturb.append(dict(delta_m=delta.tolist(),hit_count=len(h)))
    return dict(tag=row['tag'],gap_midpoint_m=q.tolist(),source_face_groups=[a['source_face_ids'],b['source_face_ids']],
                roi_bounds_m=[rlo.tolist(),rhi.tolist()],local_face_count=len(allowed),
                local_paths=paths,axis_rays=rays,exterior_segment=path_info,
                perturbation_ray_tests=perturb,semantic_label='unknown')


def plot(rows,v,f,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.audit_material_sections import section
    from matplotlib.collections import LineCollection
    fig,axs=plt.subplots(1,2,figsize=(12,7))
    r=rows[1];q=np.array(r['gap_midpoint_m']);extent=np.ptp(v,axis=0).max()
    s,_,_=section(v[f],1,q[1],extent*1e-10)
    for ax in axs:
        ax.add_collection(LineCollection(s*1000,colors='#64748b',linewidths=.8))
        for row in rows:
            p=row['exterior_segment']
            if p:
                points=np.array([p['start_m'],p['end_m']])[:,[0,2]]*1000
                ax.plot(points[:,0],points[:,1],'-o',linewidth=1,label=row['tag'])
        ax.set_xlabel('x (mm)');ax.set_ylabel('z (mm)');ax.set_aspect('equal');ax.grid(alpha=.2)
    axs[0].set_xlim(-240,260);axs[0].set_ylim(-220,180);axs[0].set_title('Paths toward exterior; full section')
    axs[1].set_xlim(165,225);axs[1].set_ylim(60,160);axs[1].set_title('Rim zoom: lines do not assign material')
    axs[1].legend();fig.tight_layout();fig.savefig(out/'rim_escape_paths.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root_folder',type=Path,required=True)
    p.add_argument('--section_trace',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    args=p.parse_args();root=args.data_root_folder.resolve();out=args.output_dir.resolve()
    if out==root or root in out.parents:raise ValueError('Output inside data root')
    if out.exists():raise FileExistsError(out)
    mesh=root/'rest_object_geo/plasticbox.ply'
    if sha256(mesh)!='e0a9c29124314f02d6b189b36716c5a01aeeb973b4001e8f28ee76f3382a1a76':raise ValueError('Unexpected mesh')
    files=[mesh,args.section_trace,Path(__file__)]+[Path(__file__).with_name(n) for n in ('audit_bps_surface_queries.py','material_oracle_probe.py','audit_material_sections.py')]
    before={str(p):sha256(p) for p in files};out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    v,f=read_canonical_ply(mesh);cv,cf,first,inverse,edges=topology(v,f)
    rows=[]
    for row in json.loads(args.section_trace.read_text(encoding='utf-8')):
        result=audit_pair(row,v,f,cv,cf,first,inverse,edges);rows.append(result)
        print(result['tag'],{k:p['status'] for k,p in result['local_paths'].items()},result['exterior_segment'],flush=True)
    plot(rows,v,f,out)
    assert before=={str(p):sha256(p) for p in files},'Input changed'
    report=dict(rows=rows,topology=dict(raw_faces=len(f),unique_geometric_faces=len(cf),
                 boundary_edges=sum(len(e)==1 for e in edges.values()),nonmanifold_edges=sum(len(e)>2 for e in edges.values())),
                 input_hashes=before,input_hashes_unchanged=True,elapsed_seconds=time.perf_counter()-start,
                 memory=memory_snapshot(),status='CPU_LOCAL_GEOMETRY_ONLY_U1_BLOCKED',
                 limitations='Exterior connectivity is relative to available triangles, not proof of physical material thickness. No unknown probe relabelled. Local face paths do not certify a closed solid.')
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
