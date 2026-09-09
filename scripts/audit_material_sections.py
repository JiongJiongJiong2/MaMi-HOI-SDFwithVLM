"""CPU triangle-section topology audit; never infers material from parity alone."""
from __future__ import annotations
import argparse
from collections import defaultdict
import itertools
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_bps_surface_queries import read_canonical_ply, sha256, memory_snapshot


def section(tri, axis, value, eps):
    """Keep face provenance; flag coplanar faces rather than inventing boundaries."""
    signed = tri[:, :, axis] - value
    candidates = np.flatnonzero((signed.min(1) <= eps) & (signed.max(1) >= -eps))
    segments, ids, coplanar = [], [], []
    for i in candidates:
        t, d = tri[i], signed[i]
        if np.max(np.abs(d)) <= eps:
            coplanar.append(int(i))
            continue
        hits = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if abs(d[a]) <= eps:
                hits.append(t[a])
            if (d[a] < -eps and d[b] > eps) or (d[b] < -eps and d[a] > eps):
                hits.append(t[a] + (t[b] - t[a]) * d[a] / (d[a] - d[b]))
        unique = []
        for h in hits:
            if not any(np.linalg.norm(h-u) <= eps for u in unique):
                unique.append(h)
        if len(unique) == 2:
            segments.append(unique)
            ids.append(int(i))
    keep = [k for k in range(3) if k != axis]
    return np.asarray(segments).reshape(-1, 2, 3)[:, :, keep], ids, coplanar


def graph(segments, face_ids, tol):
    """Tolerance endpoint weld only; no gap filling or overlapping-edge splitting."""
    nodes, buckets, edges = [], defaultdict(list), defaultdict(list)
    def node(p):
        key = tuple(np.floor(p/tol).astype(np.int64))
        for delta in itertools.product((-1, 0, 1), repeat=2):
            for idx in buckets[tuple(a+b for a,b in zip(key, delta))]:
                if np.linalg.norm(nodes[idx]-p) <= tol:
                    return idx
        idx = len(nodes)
        nodes.append(p)
        buckets[key].append(idx)
        return idx
    collapsed = 0
    for s, fid in zip(segments, face_ids):
        a,b = node(s[0]), node(s[1])
        if a == b:
            collapsed += 1
        else:
            edges[tuple(sorted((a,b)))].append(fid)
    adjacency = defaultdict(set)
    for a,b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    components, unseen = [], set(adjacency)
    nodes = np.asarray(nodes).reshape(-1, 2)
    loops = []
    while unseen:
        start = min(unseen)
        todo, members = [start], set()
        while todo:
            a = todo.pop()
            if a in members:
                continue
            members.add(a)
            todo.extend(adjacency[a]-members)
        unseen -= members
        degrees = [len(adjacency[a]) for a in members]
        cycle = len(members) >= 3 and all(d == 2 for d in degrees)
        if cycle:
            order, previous, current = [start], None, start
            while True:
                nxt = min(adjacency[current] - ({previous} if previous is not None else set()))
                if nxt == start:
                    break
                order.append(nxt)
                previous, current = current, nxt
            polygon = nodes[order]
            area = abs(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)-polygon[:,1]*np.roll(polygon[:,0],-1)))/2
            loops.append(dict(node_ids=order, area_m2=float(area)))
        components.append(dict(nodes=len(members), endpoints=degrees.count(1),
                               branch_nodes=sum(d>2 for d in degrees), cycle=cycle))
    edge_list = [dict(nodes=list(k), source_face_ids=v) for k,v in edges.items()]
    return dict(nodes=nodes.tolist(), edges=edge_list, components=components, loops=loops,
                raw_segments=len(segments), unique_edges=len(edges),
                duplicate_segments=sum(len(v)-1 for v in edges.values()), collapsed_segments=collapsed,
                endpoints=sum(len(v)==1 for v in adjacency.values()),
                branch_nodes=sum(len(v)>2 for v in adjacency.values()))


def proper_crossings(g, tol):
    """Count transverse crossings between nonincident edges; tangencies not certified."""
    nodes = np.asarray(g['nodes']).reshape(-1,2)
    edges = np.array([e['nodes'] for e in g['edges']], dtype=int).reshape(-1,2)
    def cross(a,b): return a[...,0]*b[...,1]-a[...,1]*b[...,0]
    count = 0
    for i,(a,b) in enumerate(edges):
        rest = edges[i+1:]
        rest = rest[~np.any((rest == a)|(rest == b),axis=1)]
        if not len(rest): continue
        p,q = nodes[a],nodes[b]
        u,v = nodes[rest[:,0]],nodes[rest[:,1]]
        margin1 = tol*np.linalg.norm(q-p)
        margin2 = tol*np.linalg.norm(v-u,axis=1)
        s1,s2 = cross(q-p,u-p),cross(q-p,v-p)
        t1,t2 = cross(v-u,p-u),cross(v-u,q-u)
        count += int(np.sum((((s1>margin1)&(s2 < -margin1))|((s2>margin1)&(s1 < -margin1))) &
                            (((t1>margin2)&(t2 < -margin2))|((t2>margin2)&(t1 < -margin2)))))
    return count


def cycle_core(g, tol):
    """Remove graph leaves only, without joining endpoints or changing surfaces.

    This recovers a ring with dangling spurs, not a repaired material boundary.
    Keep every original face ID for surviving edges.
    """
    adjacency=defaultdict(set)
    for e in g['edges']:
        a,b=e['nodes'];adjacency[a].add(b);adjacency[b].add(a)
    todo=[a for a,n in adjacency.items() if len(n)<2]
    while todo:
        a=todo.pop()
        if len(adjacency[a])!=1:continue
        b=next(iter(adjacency[a]));adjacency[a].clear();adjacency[b].remove(a)
        if len(adjacency[b])==1:todo.append(b)
    nodes=np.asarray(g['nodes']).reshape(-1,2)
    segments,ids=[],[]
    for e in g['edges']:
        a,b=e['nodes']
        if b in adjacency[a]:
            for fid in e['source_face_ids']:
                segments.append(nodes[[a,b]]);ids.append(fid)
    core=graph(np.array(segments).reshape(-1,2,2),ids,tol)
    core['removed_unique_edges']=g['unique_edges']-core['unique_edges']
    core['transverse_crossings']=proper_crossings(core,tol)
    active={i for e in core['edges'] for i in e['nodes']}
    core['cycle_rank']=core['unique_edges']-len(active)+len(core['components'])
    # Containment is diagnostic only, and withheld if any transverse crossing.
    polygons=[np.asarray(core['nodes'])[l['node_ids']] for l in core['loops']]
    def inside(p,poly):
        a=poly;b=np.roll(poly,-1,axis=0)
        mask=(a[:,1]>p[1])!=(b[:,1]>p[1])
        a,b=a[mask],b[mask]
        x=a[:,0]+(p[1]-a[:,1])*(b[:,0]-a[:,0])/(b[:,1]-a[:,1])
        return bool(np.count_nonzero(x>p[0])%2)
    core['loop_containment']=None if core['transverse_crossings'] else [
        [j for j,q in enumerate(polygons) if i!=j and inside(p[0],q)] for i,p in enumerate(polygons)]
    return core


def scanlines(g, bounds, tol):
    nodes=np.asarray(g['nodes']).reshape(-1,2)
    result=[]
    for travel in (0,1):
        other=1-travel
        for fraction in (.25,.5,.75):
            value=bounds[0,other]+fraction*(bounds[1,other]-bounds[0,other])
            hits=[]
            for edge in g['edges']:
                a,b=nodes[edge['nodes']]
                # Half-open convention counts a shared crossing vertex once.
                if (a[other] <= value < b[other]) or (b[other] <= value < a[other]):
                    coord=a[travel]+(b[travel]-a[travel])*(value-a[other])/(b[other]-a[other])
                    hits.append((float(coord),edge['source_face_ids']))
            hits.sort(key=lambda h:h[0])
            groups=[]
            for coord,ids in hits:
                if groups and abs(coord-groups[-1]['coordinate_m'])<=tol:
                    groups[-1]['source_face_ids'].extend(ids)
                else: groups.append(dict(coordinate_m=coord,source_face_ids=list(ids)))
            coords=[h['coordinate_m'] for h in groups]
            result.append(dict(travel_axis_in_section=travel, fraction=fraction,
                               fixed_coordinate_m=float(value),hits=groups,
                               consecutive_gaps_m=np.diff(coords).tolist()))
    return result


def analyze(v,f,axis,value):
    extent=float(np.ptp(v,axis=0).max())
    s,ids,coplanar=section(v[f],axis,value,extent*1e-10)
    g=graph(s,ids,extent*1e-8)
    sensitive=graph(s,ids,extent*1e-6)
    keep=[k for k in range(3) if k!=axis]
    scans=scanlines(g,np.array([v.min(0),v.max(0)])[:,keep],extent*1e-8)
    g.update(axis=axis,value_m=float(value),coplanar_face_ids=coplanar,
             transverse_crossings=proper_crossings(g,extent*1e-8),scanlines=scans,
             tolerance_sensitivity={k:[g[k],sensitive[k]] for k in ('unique_edges','endpoints','branch_nodes')},
             loops_at_two_tolerances=[len(g['loops']),len(sensitive['loops'])])
    # Audit exact geometric duplicate triangles and their RAW orientation.
    normals=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
    opposed=0
    for e in g['edges']:
        ids=e['source_face_ids']
        n=normals[ids]
        if len(n)>1 and np.any(n@n[0] < 0): opposed+=1
    g['duplicate_edges_with_opposed_raw_normals']=opposed
    g['leaf_pruned_core']=cycle_core(g,extent*1e-8)
    return g


def render(rows,name,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    selected=[r for r in rows if r['tag'] in ('x_center','y_center','z_0.500','z_0.895','z_0.900','z_0.905')]
    fig,axs=plt.subplots(2,3,figsize=(15,10))
    for ax,r in zip(axs.flat,selected):
        n=np.asarray(r['nodes']).reshape(-1,2)
        e=np.array([x['nodes'] for x in r['edges']],int).reshape(-1,2)
        if len(e): ax.add_collection(LineCollection(n[e],colors='#475569',linewidths=.6))
        for loop in r['loops']:
            p=n[loop['node_ids']+[loop['node_ids'][0]]]
            ax.plot(p[:,0],p[:,1],lw=1.1)
        ax.autoscale();ax.set_aspect('equal');ax.grid(alpha=.2)
        axes=[a for i,a in enumerate('xyz') if i!=r['axis']]
        ax.set_xlabel(axes[0]+' (m)');ax.set_ylabel(axes[1]+' (m)')
        ax.set_title(f"{r['tag']} | simple rings={len(r['loops'])}, ends={r['endpoints']}\nbranches={r['branch_nodes']}, core rings={len(r['leaf_pruned_core']['loops'])}")
    fig.suptitle(name+' | unique section edges; colored = degree-2 cycle (not material truth)')
    fig.tight_layout();fig.savefig(out/f'{name}_quantitative_sections.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root_folder',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    a=p.parse_args();root=a.data_root_folder.resolve();out=a.output_dir.resolve()
    if out==root or root in out.parents: raise ValueError('Output inside data root')
    if out.exists(): raise FileExistsError(out)
    paths=[root/'rest_object_geo'/f'{n}.ply' for n in ('plasticbox','trashcan','smalltable')]
    paths += [Path(__file__),Path(__file__).with_name('audit_bps_surface_queries.py')]
    before={str(p):sha256(p) for p in paths};out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();summary={}
    for name in ('plasticbox','trashcan','smalltable'):
        v,f=read_canonical_ply(root/'rest_object_geo'/f'{name}.ply')
        lo,hi=v.min(0),v.max(0)
        planes=[('x_center',0,(lo[0]+hi[0])/2),('y_center',1,(lo[1]+hi[1])/2)]
        planes += [(f'z_{t:.3f}',2,lo[2]+t*(hi[2]-lo[2])) for t in (.15,.5,.85,.895,.9,.905,.95,.99)]
        rows=[]
        for tag,axis,value in planes:
            r=analyze(v,f,axis,value);r['tag']=tag;rows.append(r)
        (out/f'{name}.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        render(rows,name,out)
        summary[name]=[{k:r[k] for k in ('tag','value_m','raw_segments','unique_edges','duplicate_segments',
                           'duplicate_edges_with_opposed_raw_normals','endpoints','branch_nodes',
                           'transverse_crossings','loops_at_two_tolerances','tolerance_sensitivity')} for r in rows]
        print(name,[(r['tag'],len(r['loops']),r['endpoints'],r['branch_nodes']) for r in rows],flush=True)
    assert before=={str(p):sha256(p) for p in paths},'Input changed'
    report=dict(objects=summary,input_hashes=before,input_hashes_unchanged=True,
                elapsed_seconds=time.perf_counter()-start,memory=memory_snapshot(),
                limitations='No material assignment; coplanar faces reported; no gap fill, overlap splitting or complete tangency test. Cycle count is not a solid certificate.')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
