"""Independent bounded CPU material/free/unknown probe. Never reads old SDF.

Review geometry first with --review_only, then supply an audited --probe_manifest.
Only new diagnostic directories are written. NumPy geometry; matplotlib for
review figures. No Torch, dataset constructor, trainer or GPU is invoked.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import sys
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_bps_surface_queries import read_canonical_ply, nearest_triangles, sha256, memory_snapshot

OBJECTS = ('plasticbox', 'trashcan', 'smalltable')
RAYS = np.array([[1,.137,.263],[-.219,1,.317],[.193,-.271,1],
                 [-1,.371,-.113],[.419,-1,.229],[-.307,.173,-1],[1,1.173,.719]], dtype=float)
RAYS /= np.linalg.norm(RAYS, axis=1)[:,None]


def edges_and_components(faces):
    edges = defaultdict(list)
    for i, f in enumerate(faces):
        for a, b in zip(f, np.roll(f,-1)):
            edges[min(a,b),max(a,b)].append((i, 1 if a < b else -1))
    parent = np.arange(len(faces))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for incident in edges.values():
        first = find(incident[0][0])
        for i,_ in incident[1:]:
            parent[find(i)] = first
    groups = defaultdict(list)
    for i in range(len(faces)):
        groups[find(i)].append(i)
    return edges, sorted(groups.values(), key=lambda x: (-len(x),x[0]))


def topology(vertices, faces):
    edges, components = edges_and_components(faces)
    component_id = np.empty(len(faces),dtype=int)
    for k,ids in enumerate(components): component_id[ids] = k
    counts = [dict(boundary_edges=0,nonmanifold_edges=0,inconsistent_edges=0) for _ in components]
    for incident in edges.values():
        row = counts[component_id[incident[0][0]]]
        if len(incident)==1: row['boundary_edges'] += 1
        if len(incident)>2: row['nonmanifold_edges'] += 1
        if len(incident)==2 and incident[0][1]==incident[1][1]: row['inconsistent_edges'] += 1
    details=[]
    for ids, c in zip(components,counts):
        tri=vertices[faces[ids]]
        lo,hi=tri.min((0,1)),tri.max((0,1))
        t=tri-(lo+hi)/2
        vol=np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6
        details.append(dict(face_count=len(ids),bounds=[lo.tolist(),hi.tolist()],signed_volume_m3=float(vol),**c))
    duplicate_count=len(faces)-len(np.unique(np.sort(faces,axis=1),axis=0))
    return dict(vertex_count=len(vertices),face_count=len(faces),duplicate_unordered_faces=duplicate_count,
                boundary_edges=sum(c['boundary_edges'] for c in counts),
                nonmanifold_edges=sum(c['nonmanifold_edges'] for c in counts),components=details),components


def conservative_candidate(vertices,faces):
    """Exact-position weld + duplicate/zero-area removal + coherent orientation.

    No vertex displacement, hole fill, thickening, decimation or mesh export.
    Component outward orientation is a candidate solid-union assumption, never
    an independent material label. Nested shells require separate semantics.
    """
    verts,inverse=np.unique(vertices,axis=0,return_inverse=True)
    f=inverse[faces]
    tri=verts[f]
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    f=f[area2>0]
    _,ids=np.unique(np.sort(f,axis=1),axis=0,return_index=True)
    f=f[np.sort(ids)].copy()
    edges,components=edges_and_components(f)
    neighbors=defaultdict(list)
    for inc in edges.values():
        if len(inc)==2:
            (a,sa),(b,sb)=inc
            neighbors[a].append((b,-sa*sb)); neighbors[b].append((a,-sa*sb))
    orientation=np.zeros(len(f),dtype=int)
    conflicts=0
    for start in range(len(f)):
        if orientation[start]:continue
        orientation[start]=1;queue=deque([start])
        while queue:
            a=queue.popleft()
            for b,relative in neighbors[a]:
                expected=orientation[a]*relative
                if not orientation[b]:orientation[b]=expected;queue.append(b)
                elif orientation[b]!=expected:conflicts+=1
    flip=orientation<0
    f[flip]=f[flip][:,[0,2,1]]
    info,components=topology(verts,f)
    outward=[]
    for k,(ids,c) in enumerate(zip(components,info['components'])):
        if not any(c[x] for x in ('boundary_edges','nonmanifold_edges','inconsistent_edges')):
            if c['signed_volume_m3']<0:f[ids]=f[ids][:,[0,2,1]]
            outward.append(k)
    info,components=topology(verts,f)
    info.update(operation='exact weld, zero-area removal, unordered face dedup, manifold-edge orientation',
                moved_vertices=0,filled_holes=0,orientation_conflicts=conflicts//2,
                closed_outward_components=outward,self_intersections='NOT_TESTED',
                physical_solid_certified=False)
    return verts,f,info,components


def winding(points,triangles,query_batch=8,triangle_chunk=1024):
    """Direct generalized winding number via signed triangle solid angles."""
    points=np.asarray(points,dtype=float);out=np.zeros(len(points))
    for start in range(0,len(points),query_batch):
        p=points[start:start+query_batch];total=np.zeros(len(p))
        for off in range(0,len(triangles),triangle_chunk):
            t=triangles[off:off+triangle_chunk]
            a,b,c=(t[None,:,i,:]-p[:,None,:] for i in range(3))
            la,lb,lc=(np.linalg.norm(x,axis=-1) for x in (a,b,c))
            numerator=np.einsum('qti,qti->qt',a,np.cross(b,c))
            denominator=la*lb*lc+np.einsum('qti,qti->qt',a,b)*lc+np.einsum('qti,qti->qt',b,c)*la+np.einsum('qti,qti->qt',c,a)*lb
            total+=(2*np.arctan2(numerator,denominator)).sum(1)
        out[start:start+len(p)]=total/(4*np.pi)
    return out


def ray_hits(point,direction,triangles,extent,triangle_chunk=1024,positive=True):
    """Independent Moller-Trumbore intersections; retain face-hit and unique counts.

    Coincident hits (including opposite duplicate faces and shared edges) are
    clustered spatially. This fixes duplicate counting, not missing surfaces.
    """
    direction=np.asarray(direction,dtype=float);direction=direction/np.linalg.norm(direction)
    hits=[]
    for off in range(0,len(triangles),triangle_chunk):
        t=triangles[off:off+triangle_chunk]
        edge1,edge2=t[:,1]-t[:,0],t[:,2]-t[:,0]
        h=np.cross(direction,edge2);det=np.einsum('ij,ij->i',edge1,h)
        valid=abs(det)>1e-12*np.linalg.norm(edge1,axis=1)*np.linalg.norm(edge2,axis=1)
        inv=np.divide(1.,det,out=np.zeros_like(det),where=valid)
        delta=np.asarray(point)-t[:,0]
        u=np.einsum('ij,ij->i',delta,h)*inv
        q=np.cross(delta,edge1)
        v=np.einsum('j,ij->i',direction,q)*inv
        distance=np.einsum('ij,ij->i',edge2,q)*inv
        valid&=(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)
        if positive:valid&=distance>1e-8*extent
        hits.extend(distance[valid].tolist())
    hits=np.sort(hits)
    if not len(hits):return np.array([]),0
    unique=hits[np.r_[True,np.diff(hits)>1e-7*extent]]
    return unique,len(hits)


def parity(points,triangles,extent,directions=RAYS):
    unique=np.empty((len(points),len(directions)),dtype=int)
    naive=np.empty_like(unique)
    for i,p in enumerate(points):
        for j,d in enumerate(directions):
            hits,count=ray_hits(p,d,triangles,extent)
            unique[i,j]=len(hits)%2;naive[i,j]=count%2
    return unique,naive


def cross_section(triangles,axis,value,extent):
    """Triangle-plane segment soup for visual review, not inside/outside labels."""
    segments=[]
    for tri in triangles:
        hits=[]
        for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            da,db=a[axis]-value,b[axis]-value
            if da*db<0:hits.append(a+(b-a)*da/(da-db))
            elif abs(da)<1e-10*extent:hits.append(a)
        if len(hits)>=2:
            hs=np.unique(np.round(np.asarray(hits),12),axis=0)
            if len(hs)==2 and np.linalg.norm(hs[0]-hs[1])>1e-10*extent:segments.append(hs)
    return np.asarray(segments).reshape(-1,2,3)


def render_review(vertices,faces,name,out,probes=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    lo,hi=vertices.min(0),vertices.max(0);center=(lo+hi)/2;extent=max(hi-lo)
    tri=vertices[faces]
    planes=[(0,center[0]),(1,center[1]),(2,lo[2]+.15*(hi[2]-lo[2])),
            (2,lo[2]+.5*(hi[2]-lo[2])),(2,lo[2]+.9*(hi[2]-lo[2])),(2,lo[2]+.99*(hi[2]-lo[2]))]
    fig,axs=plt.subplots(2,3,figsize=(15,10))
    sections=[]
    for ax,(axis,value) in zip(axs.flat,planes):
        seg=cross_section(tri,axis,value,extent);keep=[i for i in range(3) if i!=axis]
        ax.add_collection(LineCollection(seg[:,:,keep],linewidths=.45,color='#334155'))
        ax.set_xlim(lo[keep[0]]-.06*extent,hi[keep[0]]+.06*extent)
        ax.set_ylim(lo[keep[1]]-.06*extent,hi[keep[1]]+.06*extent)
        ax.set_aspect('equal');ax.set_xlabel('xyz'[keep[0]]+' (m)');ax.set_ylabel('xyz'[keep[1]]+' (m)')
        ax.set_title(f'{"xyz"[axis]}={value:.5f} m; {len(seg)} segments')
        ax.grid(alpha=.2)
        if probes:
            for row in probes:
                p=np.array(row['point'])
                # Projections are explicitly labelled, not asserted on section.
                color={'material':'#dc2626','free':'#16a34a','unknown':'#d97706'}[row['label']]
                ax.plot(p[keep[0]],p[keep[1]],'.',color=color,markersize=4)
        sections.append(dict(axis=axis,value=float(value),segment_count=len(seg)))
    fig.suptitle(name+' raw mesh sections'+(' | projected probes: red=material, green=free, orange=unknown' if probes else ''))
    fig.tight_layout();fig.savefig(out/(name+'_sections.png'),dpi=150);plt.close(fig)
    return sections


def review_object(root,name,out):
    path=root/'rest_object_geo'/(name+'.ply');v,f=read_canonical_ply(path)
    raw,_=topology(v,f);cv,cf,candidate,groups=conservative_candidate(v,f)
    lo,hi=v.min(0),v.max(0);center=(lo+hi)/2;extent=max(hi-lo)
    lines=[]
    for axis in range(3):
        positions=[center.copy()]
        if axis in (0,1):
            positions=[]
            for frac in (.03,.15,.5,.9,.99):
                p=center.copy();p[2]=lo[2]+frac*(hi[2]-lo[2]);positions.append(p)
        for p in positions:
            direction=np.eye(3)[axis]
            hits,_=ray_hits(p,direction,v[f],extent,positive=False)
            lines.append(dict(axis=axis,origin=p.tolist(),intersection_coordinates=(p[axis]+hits).tolist()))
    sections=render_review(v,f,name,out)
    return dict(object=name,mesh_sha256=sha256(path),bounds=[lo.tolist(),hi.tolist()],raw=raw,
                candidate=candidate,line_intersections=lines,sections=sections)


def analytic_fixture(cup=False):
    """Boundary of an axis-aligned cell union; cup has an OPEN cavity."""
    axes=[np.array([-1.,-.8,.8,1.]),np.array([-1.,-.8,.8,1.]),np.array([0.,.2,2.])] if cup else [np.array([-1.,1.])]*3
    shape=tuple(len(a)-1 for a in axes)
    occupied=set(idx for idx in np.ndindex(shape) if not cup or idx[2]==0 or idx[0]!=1 or idx[1]!=1)
    verts=[];lookup={};faces=[]
    def vertex(point):
        key=tuple(point)
        if key not in lookup:lookup[key]=len(verts);verts.append(key)
        return lookup[key]
    for idx in sorted(occupied):
        for axis in range(3):
            other=[a for a in range(3) if a!=axis]
            for side in (-1,1):
                neighbor=list(idx);neighbor[axis]+=side
                if tuple(neighbor) in occupied:continue
                corners=[]
                for u,v in ((0,0),(1,0),(1,1),(0,1)):
                    p=[axes[a][idx[a]] for a in range(3)]
                    p[axis]=axes[axis][idx[axis]+int(side>0)]
                    p[other[0]]=axes[other[0]][idx[other[0]]+u]
                    p[other[1]]=axes[other[1]][idx[other[1]]+v]
                    corners.append(np.array(p))
                if np.cross(corners[1]-corners[0],corners[2]-corners[0])[axis]*side<0:corners=corners[::-1]
                ids=[vertex(p) for p in corners]
                faces.extend([[ids[0],ids[1],ids[2]],[ids[0],ids[2],ids[3]]])
    return np.asarray(verts),np.asarray(faces,dtype=int)


def analytic_truth(points,cup=False):
    """Independent solid definition and rectangle distances, not triangle code."""
    p=np.asarray(points,dtype=float)
    if not cup:
        inside=(abs(p)<1).all(1)
        d=np.where(inside,(1-abs(p)).min(1),np.linalg.norm(np.maximum(abs(p)-1,0),axis=1))
        return inside.astype(int),d
    inside=(abs(p[:,:2])<1).all(1)&(p[:,2]>0)&(p[:,2]<2)&((abs(p[:,0])>.8)|(abs(p[:,1])>.8)|(p[:,2]<.2))
    rectangles=[]
    for axis in (0,1):
        for side in (-1,1):
            lo=np.array([-1.,-1.,0.]);hi=np.array([1.,1.,2.]);lo[axis]=hi[axis]=side;rectangles.append((lo,hi))
            lo=np.array([-.8,-.8,.2]);hi=np.array([.8,.8,2.]);lo[axis]=hi[axis]=side*.8;rectangles.append((lo,hi))
    rectangles.extend([(np.array([-1,-1,0.]),np.array([1,1,0.])),(np.array([-.8,-.8,.2]),np.array([.8,.8,.2]))])
    for axis in (0,1):
        for side in (-1,1):
            lo=np.array([-1.,-1.,2.]);hi=np.array([1.,1.,2.])
            lo[axis],hi[axis]=(-1.,-.8) if side<0 else (.8,1.)
            rectangles.append((lo,hi))
    d=np.min([np.linalg.norm(p-np.clip(p,lo,hi),axis=1) for lo,hi in rectangles],axis=0)
    return inside.astype(int),d


def run_fixtures():
    results={}
    for name,cup in (('closed_box',False),('open_thick_cup',True)):
        v,f=analytic_fixture(cup);tri=v[f]
        points=np.random.default_rng(17).uniform([-1.4,-1.4,-.4],[1.4,1.4,2.4],(160,3))
        if cup:points=np.vstack([points,[[.9,0,1],[0,0,.1],[0,0,1],[0,0,2.2],[1.2,0,1],[.79,0,1],[.81,0,1]]])
        labels,expected=analytic_truth(points,cup)
        actual,cp=nearest_triangles(points,tri,8,1024)
        w=winding(points,tri);r,_=parity(points,tri,2.)
        assert np.all((w>.5)==labels) and np.all(r==labels[:,None]),name
        assert np.max(abs(actual-expected))<1e-10,name
        exit_errors=[]
        for p,q,inside,d in zip(points,cp,labels,actual):
            if inside and d>1e-8:
                hits,_=ray_hits(p,q-p,tri,2.)
                assert len(hits)>0
                exit_errors.append(abs(hits[0]-d))
        assert max(exit_errors,default=0)<1e-9
        results[name]=dict(count=len(points),material_count=int(labels.sum()),occupancy_accuracy=1.,
                           distance_max_error_m=float(max(abs(actual-expected))),exit_max_error_m=float(max(exit_errors)),
                           no_cap_at_cavity=(bool(analytic_truth([[0,0,2.]],True)[1][0]>.5) if cup else None))
    return results


def classification_metrics(predictions,rows):
    labels=np.array([r['label'] for r in rows]);pred=np.asarray(predictions)
    def accuracy(label,expected):
        mask=labels==label
        return dict(count=int(mask.sum()),correct=int((pred[mask]==expected).sum()),
                    accuracy=float((pred[mask]==expected).mean()) if mask.any() else None,
                    unknown_predictions=int((pred[mask]==-1).sum()))
    unknown=labels=='unknown'
    return dict(material=accuracy('material',1),free=accuracy('free',0),
                unknown_label_count=int(unknown.sum()),unknown_label_fraction=float(unknown.mean()),
                predicted_unknown_fraction=float((pred==-1).mean()),
                unknown_label_abstention_fraction=float((pred[unknown]==-1).mean()) if unknown.any() else None,
                misclassified_ids=[r['id'] for r,v in zip(rows,pred) if r['label']!='unknown' and v!=-1 and v!=int(r['label']=='material')],
                abstained_known_ids=[r['id'] for r,v in zip(rows,pred) if r['label']!='unknown' and v==-1])


def evaluate_variant(points,triangles,extent,rows,perturb_scale):
    offsets=np.vstack([np.zeros(3),np.eye(3)*perturb_scale,-np.eye(3)*perturb_scale])
    expanded=(points[:,None,:]+offsets[None,:,:]).reshape(-1,3)
    w=winding(expanded,triangles).reshape(len(points),7)
    rays,naive=parity(points,triangles,extent)
    # Ray perturbations also tested, independently of winding.
    perturbed_rays,_=parity(points+np.array([1,-.73,.39])*perturb_scale,triangles,extent)
    d,cp=nearest_triangles(points,triangles,8,1024)
    thresholds=(.25,.5,.75)
    votes=np.stack([w>t for t in thresholds],axis=-1)
    stable_w=(votes==votes[:,0:1,1:2]).all((1,2))
    stable_r=(rays==rays[:,0:1]).all(1)&(perturbed_rays==rays).all(1)
    consensus=np.where(stable_w&stable_r&((w[:,0]>.5)==rays[:,0])&(d>perturb_scale*2),rays[:,0],-1)
    summary=dict(winding={str(t):classification_metrics(w[:,0]>t,rows) for t in thresholds},
                 parity_majority=classification_metrics((rays.mean(1)>.5).astype(int),rows),
                 agreement_filter_NOT_calibrated_confidence=classification_metrics(consensus,rows),
                 ray_direction_sensitive_fraction=float((np.ptp(rays,axis=1)>0).mean()),
                 ray_position_sensitive_fraction=float((perturbed_rays!=rays).any(1).mean()),
                 winding_threshold_sensitive_fraction=float(((w[:,0]>.25)!=(w[:,0]>.75)).mean()),
                 winding_position_sensitive_fraction=float(((w>.5)!=(w[:,0:1]>.5)).any(1).mean()),
                 duplicate_hit_parity_sensitive_fraction=float((rays!=naive).any(1).mean()),
                 global_reverse_winding_label_change_fraction=float(((w[:,0]>.5)!=(-w[:,0]>.5)).mean()),
                 occupancy_true_labels_generated_here=False)
    for label in ('material','free','unknown'):
        mask=np.array([r['label']==label for r in rows])
        summary['stability_'+label]=dict(count=int(mask.sum()),
            ray_direction_sensitive_count=int((np.ptp(rays[mask],axis=1)>0).sum()),
            ray_position_sensitive_count=int((perturbed_rays[mask]!=rays[mask]).any(1).sum()),
            winding_threshold_sensitive_count=int(((w[mask,0]>.25)!=(w[mask,0]>.75)).sum()),
            winding_position_sensitive_count=int(((w[mask]>.5)!=(w[mask,0:1]>.5)).any(1).sum()))
    detail=[]
    for i,row in enumerate(rows):
        detail.append(dict(id=row['id'],point=points[i].tolist(),label=row['label'],region=row['region'],
                           source=row['source'],winding=float(w[i,0]),winding_perturbation_range=[float(w[i].min()),float(w[i].max())],
                           ray_votes=rays[i].tolist(),naive_face_hit_parity=naive[i].tolist(),
                           perturbed_ray_votes=perturbed_rays[i].tolist(),consensus=int(consensus[i]),
                           unsigned_distance_m=float(d[i]),closest=cp[i].tolist()))
    return summary,detail,d,cp


def geometric_label_checks(points,rows,triangles,extent):
    """Record local evidence checks; NEVER infer labels from occupancy results."""
    records=[]
    for p,row in zip(points,rows):
        record=dict(id=row['id'],label=row['label'],region=row['region'],source=row['source'])
        if row.get('checks')=='open_vertical_corridor':
            hits,_=ray_hits(p,[0,0,1],triangles,extent)
            record['upward_surface_hit_count']=len(hits)
            if len(hits):raise ValueError('Reviewed cavity corridor obstructed: '+row['id'])
        if row.get('checks')=='paired_axis_brackets':
            brackets=[]
            for axis in row.get('bracket_axes',[0,1,2]):
                hits,_=ray_hits(p,np.eye(3)[axis],triangles,extent,positive=False)
                neg,pos=hits[hits<0],hits[hits>0]
                brackets.append(dict(axis=axis,negative_nearest=float(neg[-1]) if len(neg) else None,
                                     positive_nearest=float(pos[0]) if len(pos) else None))
                if not len(neg) or not len(pos):raise ValueError('Material bracket absent: '+row['id'])
            record['brackets']=brackets
        records.append(record)
    return records


def run_object(root,name,manifest,out):
    start=time.perf_counter();path=root/'rest_object_geo'/(name+'.ply')
    assert sha256(path)==manifest['mesh_sha256'],'Probe manifest mesh SHA mismatch'
    v,f=read_canonical_ply(path);raw,raw_components=topology(v,f)
    cv,cf,candidate,components=conservative_candidate(v,f)
    rows=manifest['probes'];points=np.asarray([r['point'] for r in rows]);extent=float(np.ptp(v,axis=0).max())
    if not rows or len(rows)>128 or points.shape!=(len(rows),3) or not np.isfinite(points).all():raise ValueError('Invalid or oversized probes')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate probe IDs')
    for row in rows:
        if row['label'] not in ('material','free','unknown') or not row.get('source'):raise ValueError('Each label needs evidence')
    tri,ctri=v[f],cv[cf]
    evidence=geometric_label_checks(points,rows,tri,extent)
    perturb_scale=extent*1e-4
    variants={};details={};distances={};closest={}
    for variant,t in (('raw',tri),('candidate',ctri)):
        print('[CPU probe]',name,variant,len(rows),'points',flush=True)
        variants[variant],details[variant],distances[variant],closest[variant]=evaluate_variant(points,t,extent,rows,perturb_scale)
    # Component interpretations are measured, not silently chosen as truth.
    component_w=np.array([winding(points,ctri[ids]) for ids in components])
    largest_labels=component_w[0]>.5
    union_labels=(component_w>.5).any(0)
    component_metrics=dict(largest_only=classification_metrics(largest_labels,rows),
                           outward_component_union=classification_metrics(union_labels,rows),
                           largest_vs_union_changed_ids=[r['id'] for r,a,b in zip(rows,largest_labels,union_labels) if a!=b],
                           per_component_winding=component_w.tolist())
    component_metrics['raw_vs_candidate_winding_changed_ids']=[r['id'] for r,a,b in zip(rows,details['raw'],details['candidate']) if (a['winding']>.5)!=(b['winding']>.5)]
    # Short lines are label-free spatial diagnostics; point labels remain separate.
    line_results=[]
    for line in manifest.get('lines',[]):
        a,b=np.asarray(line['start']),np.asarray(line['end']);q=np.linspace(a,b,17)
        line_d,line_cp=nearest_triangles(q,ctri,8,1024)
        line_w=winding(q,ctri)
        line_results.append(dict(**line,points=q.tolist(),candidate_winding=line_w.tolist(),
                                 candidate_udf_m=line_d.tolist(),labels='UNKNOWN unless separately reviewed probe',
                                 candidate_signed_jump_ratio_max=float(np.max(abs(np.diff(np.where(line_w>.5,-line_d,line_d))))/(np.linalg.norm(b-a)/16))))
    exit_rows=[]
    for i,row in enumerate(rows):
        if row['label']!='material':continue
        cp=closest['candidate'][i];d=distances['candidate'][i]
        if d<=1e-8*extent:continue
        direction=(cp-points[i])/d
        hits,_=ray_hits(points[i],direction,ctri,extent)
        pair=np.array([cp-direction*perturb_scale,cp+direction*perturb_scale])
        side_w=winding(pair,ctri)
        exit_rows.append(dict(id=row['id'],closest_boundary_distance_m=float(d),
                              first_ray_hit_m=float(hits[0]) if len(hits) else None,
                              ray_vs_closest_abs_error_m=float(abs(hits[0]-d)) if len(hits) else None,
                              boundary_minus_plus_winding=side_w.tolist(),
                              inside_to_free_transition=bool(side_w[0]>.5 and side_w[1]<.5),
                              scope='local reviewed material; not global certified solid'))
    # Geometric preservation probes use face interiors, not BPS or old SDF.
    rng=np.random.default_rng(23);ids=rng.choice(len(tri),min(64,len(tri)),replace=False)
    surface=tri[ids].mean(1);sd,_=nearest_triangles(surface,ctri,8,1024)
    region_metrics={region:{variant:{'winding_0.5':classification_metrics(
        np.array([r['winding']>.5 for r in details[variant]])[np.array([r['region']==region for r in rows])],
        [r for r in rows if r['region']==region])} for variant in details} for region in sorted({r['region'] for r in rows})}
    render_review(v,f,name,out,rows)
    with (out/(name+'_detail.json')).open('x') as handle:
        json.dump(dict(evidence=evidence,variants=details,lines=line_results,exit_distances=exit_rows,
                       component_handling=component_metrics),handle,indent=2,allow_nan=False)
    finite_exit=[r['ray_vs_closest_abs_error_m'] for r in exit_rows if r['ray_vs_closest_abs_error_m'] is not None]
    has_material=any(r['label']=='material' for r in rows)
    wrong=variants['candidate']['winding']['0.5']['material']['correct']+variants['candidate']['winding']['0.5']['free']['correct']
    known=sum(r['label']!='unknown' for r in rows)
    parity_correct=sum(variants['candidate']['parity_majority'][k]['correct'] for k in ('material','free'))
    local_pass=(has_material and wrong==known and parity_correct==known and not candidate['boundary_edges'] and not candidate['nonmanifold_edges'] and all(r['inside_to_free_transition'] for r in exit_rows))
    decision='A_LOCAL_GO_GLOBAL_GATE_BLOCKED' if local_pass else 'A_NO_GO_FOR_TESTED_LOW_COST_PROCESSING'
    assert sha256(path)==manifest['mesh_sha256'],'Input changed'
    return dict(object=name,probe_count=len(rows),mesh_sha256=manifest['mesh_sha256'],raw_topology=raw,
                candidate_topology=candidate,variants=variants,regions=region_metrics,
                semantic_unknown_fraction=sum(r['label']=='unknown' for r in rows)/len(rows),
                component_handling=component_metrics,perturbation_m=perturb_scale,
                surface_distance=dict(raw_vs_candidate_query_max_abs_error_m=float(max(abs(distances['raw']-distances['candidate']))),
                                      raw_face_interior_to_candidate_max_m=float(max(sd)),
                                      truth='exact triangle union; preservation only, not physical scan accuracy'),
                exit_distance=dict(material_count=len(exit_rows),max_ray_consistency_error_m=max(finite_exit) if finite_exit else None,
                                   local_exit_transitions_pass=all(r['inside_to_free_transition'] for r in exit_rows) if exit_rows else None,
                                   unavailable_reason=None if exit_rows else 'No independently supported material-interior labels; no fabricated exit depth'),
                decision=decision,semantic_review=manifest['review'],elapsed_seconds=time.perf_counter()-start,
                future_repair_scope='No automatic claim that all possible low-cost repairs were exhausted')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root_folder',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--objects',nargs='+',default=list(OBJECTS),choices=OBJECTS)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--review_only',action='store_true')
    mode.add_argument('--probe_manifest')
    args=parser.parse_args()
    root,out=Path(args.data_root_folder).resolve(),Path(args.output_dir).resolve()
    if out.exists() or out==root or root in out.parents:parser.error('Require a NEW output directory outside data root')
    if len(set(args.objects))!=len(args.objects):parser.error('Duplicate objects')
    out.mkdir(parents=True,exist_ok=False)
    if args.review_only:
        objects=[]
        for name in args.objects:
            print('[CPU review]',name,flush=True);objects.append(review_object(root,name,out))
        with (out/'geometry_review.json').open('x') as f:json.dump(objects,f,indent=2,allow_nan=False)
        print('REVIEW_COMPLETE; no semantic labels inferred from occupancy',flush=True)
    else:
        manifest_path=Path(args.probe_manifest)
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('version')!=1:raise ValueError('Unsupported probe manifest')
        started=time.perf_counter()
        fixtures=run_fixtures()
        print('[CPU] analytic box/cup PASS',flush=True)
        objects=[run_object(root,name,manifest['objects'][name],out) for name in args.objects]
        memory=memory_snapshot()
        repo=Path(__file__).resolve().parents[1]
        revision=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip()
        summary=dict(status='CPU_MATERIAL_PROBE_COMPLETE_U1_BLOCKED',objects=objects,fixtures=fixtures,
                     script_sha256=sha256(__file__),triangle_primitive_sha256=sha256(Path(__file__).with_name('audit_bps_surface_queries.py')),
                     manifest_sha256=sha256(manifest_path),manifest=manifest,environment=dict(python=sys.version,numpy=np.__version__,device='CPU'),
                     base_git_commit=revision,
                     elapsed_seconds=time.perf_counter()-started,memory=memory,
                     limitations=['No old SDF/cache/BPS/trainer/evaluator was read or changed',
                                  'Real semantic labels are geometry-reviewed local evidence, not measured physical thickness',
                                  'No universal solid certificate or reliable self-intersection test',
                                  'Algorithm agreement and threshold stability are not semantic confidence',
                                  'No whole-object A GO from a sparse probe set; unknown regions remain blocked'])
        with (out/'summary.json').open('x') as handle:json.dump(summary,handle,indent=2,allow_nan=False)
        print(json.dumps(dict(status=summary['status'],output=str(out),decisions={o['object']:o['decision'] for o in objects})),flush=True)


if __name__=='__main__':main()
