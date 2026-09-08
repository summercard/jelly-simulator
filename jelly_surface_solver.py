"""Shape-independent compliant surface solver for Blender's evaluated meshes.

A surface XPBD approximation: edge lengths, volume, pinned vertices, contact.
Kinematic collider meshes include evaluated armatures and shape keys. Active soft
bodies exchange contact corrections. This is not a volumetric FEM solver.
"""
import math
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree
_COLLIDER_HISTORY={}
_COLLIDER_CACHE={}


def volume_only(x,tri):
    a,b,c=x[tri[:,0]],x[tri[:,1]],x[tri[:,2]]
    return float(np.sum(a*np.cross(b,c))/6)


def volume_gradient(x,tri):
    a,b,c=x[tri[:,0]],x[tri[:,1]],x[tri[:,2]]
    g=np.zeros_like(x)
    np.add.at(g,tri[:,0],np.cross(b,c)/6)
    np.add.at(g,tri[:,1],np.cross(c,a)/6)
    np.add.at(g,tri[:,2],np.cross(a,b)/6)
    return float(np.sum(a*np.cross(b,c))/6),g


def init(st):
    body,rest=next((o,r) for o,r in st['entries'] if o.get('jelly_body'))
    import bpy
    mesh=body.data;temporary=None;evaluated=None;original_rest=rest
    if len(rest)>1200:
        # Solve a compact cage and interpolate displacement to the render mesh.
        copied=mesh.copy();copied.vertices.foreach_set('co',rest.ravel());copied.update()
        temporary=bpy.data.objects.new('_jelly_solver_cage',copied)
        st['scene'].collection.objects.link(temporary)
        modifier=temporary.modifiers.new('计算网格','DECIMATE');modifier.ratio=900/len(rest)
        evaluated=temporary.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=evaluated.to_mesh()
        values=np.empty(len(mesh.vertices)*3,dtype=np.float32);mesh.vertices.foreach_get('co',values);rest=values.reshape(-1,3)
    mesh.calc_loop_triangles()
    tri=np.array([t.vertices[:] for t in mesh.loop_triangles],dtype=np.int32)
    edges=np.array([tuple(e.vertices) for e in mesh.edges],dtype=np.int32)
    if temporary:
        evaluated.to_mesh_clear();data=temporary.data;bpy.data.objects.remove(temporary,do_unlink=True);bpy.data.meshes.remove(data)
    pins=rest[:,2]<1e-5 if not st['anchor'].get('jelly_custom_soft') else np.zeros(len(rest),dtype=bool)
    tree=KDTree(len(rest))
    for i,p in enumerate(rest):tree.insert(Vector(p),i)
    tree.balance()
    followers={}
    for obj,r in st['entries']:
        if obj==body and rest is original_rest:continue
        ids=[];weights=[]
        for p in r:
            hits=tree.find_n(Vector(p),12);ids.append([h[1] for h in hits])
            dist=np.array([h[2] for h in hits]);support=max(float(dist[-1])*1.001,1e-5)
            q=np.clip(dist/support,0,1);w=(1-q)**4*(4*q+1);weights.append(w/max(w.sum(),1e-12))
        followers[obj.as_pointer()]=(np.array(ids,dtype=np.int32),np.array(weights))
    render_edges=np.array([tuple(e.vertices) for e in body.data.edges],dtype=np.int32)
    st['surface_solver']={'degree':np.bincount(edges.ravel(),minlength=len(rest)).clip(1),'render_degree':np.bincount(render_edges.ravel(),minlength=len(original_rest)).clip(1),'render_edges':render_edges,'body':body,'rest':rest.copy(),'tri':tri,'edges':edges,'pins':pins,'followers':followers,'x':None,'velocity':np.zeros_like(rest),'contact_count':0}
    return st['surface_solver']


def transform(points,matrix):
    m=np.array(matrix);return points@m[:3,:3].T+m[:3,3]


class Collider:
    def __init__(self,obj,depsgraph):
        evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh()
        try:
            points=np.empty(len(mesh.vertices)*3,dtype=np.float32);mesh.vertices.foreach_get('co',points)
            self.vertices=transform(points.reshape(-1,3),evaluated.matrix_world)
            self.faces=[tuple(p.vertices) for p in mesh.polygons]
        finally:evaluated.to_mesh_clear()
        self.target=self.vertices.copy()
        key=obj.as_pointer();previous=_COLLIDER_HISTORY.get(key)
        self.previous=previous if previous is not None and previous.shape==self.vertices.shape else self.target
        _COLLIDER_HISTORY[key]=self.target.copy()
        cached=_COLLIDER_CACHE.get(key)
        if cached is not None and cached.faces==self.faces and np.array_equal(cached.vertices,self.target):
            self.tree=cached.tree;self.low=cached.low;self.high=cached.high
        else:self.rebuild(self.target)
        _COLLIDER_CACHE[key]=self

    def rebuild(self,vertices):
        if hasattr(self,'tree') and np.array_equal(self.vertices,vertices):return
        self.vertices=vertices
        self.tree=BVHTree.FromPolygons(vertices.tolist(),self.faces)
        self.low=vertices.min(axis=0);self.high=vertices.max(axis=0)

    def contacts(self,x,margin=.003,inward=None):
        ids=np.flatnonzero(np.all((x>=self.low-margin)&(x<=self.high+margin),axis=1))
        result=[]
        for i in ids:
            near,normal,face,distance=self.tree.find_nearest(Vector(x[i]))
            if near is not None:
                signed=float((Vector(x[i])-near).dot(normal))
                if signed<margin:
                    if inward is not None and signed<0:
                        hit,n,_,_=self.tree.ray_cast(Vector(x[i]),Vector(inward[i]))
                        if hit is not None:near,normal=hit,n
                    result.append((i,np.array(near)+np.array(normal)*margin))
        return result


def edge_solve(d,base,h,k):
    x=d['x'];weights=(~d['pins']).astype(float);edges=d['edges']
    a,b=edges[:,0],edges[:,1];delta=x[a]-x[b];length=np.linalg.norm(delta,axis=1)
    rest=d['edge_lengths']
    alpha=(.00004*30/max(k,1))/(h*h)
    correction=delta/np.maximum(length[:,None],1e-8)*((length-rest)/(weights[a]+weights[b]+alpha))[:,None]
    degree=d['degree']
    for axis in range(3):
        summed=np.bincount(a,weights=-correction[:,axis],minlength=len(x))+np.bincount(b,weights=correction[:,axis],minlength=len(x))
        x[:,axis]+=.9*weights*summed/degree
    # Bending regularisation acts on displacement, retaining the rest silhouette.
    offset=x-base
    for axis in range(3):
        neighbours=np.bincount(a,weights=offset[b,axis],minlength=len(x))+np.bincount(b,weights=offset[a,axis],minlength=len(x))
        x[:,axis]+=.42*weights*(neighbours/degree-offset[:,axis])


def volume_solve(d,target):
    volume,g=volume_gradient(d['x'],d['tri'])
    influence=d.get('influence')
    if influence is not None:g*=influence[:,None]
    g[d['pins']]=0
    denominator=float(np.sum(g*g))
    if denominator>1e-10:
        step=np.clip((target-volume)/denominator,-.2,.2)
        d['x']+=g*step*.8


def shared_contact_plane(first,second):
    """Stable manifold for two coplanar, fixed-cut jelly supports.

    Their cap disks cannot cross; a shared separator avoids oscillating nearest
    triangle normals and guarantees separation of the rendered surfaces too.
    Other orientations/custom soft meshes use the general mesh constraints.
    """
    ra=first.get('root');rb=second.get('root')
    if ra is None or rb is None or ra.get('jelly_custom_soft') or rb.get('jelly_custom_soft'):return None
    na=ra.matrix_world.to_3x3().normalized()@Vector((0,0,1));nb=rb.matrix_world.to_3x3().normalized()@Vector((0,0,1))
    delta=rb.matrix_world.translation-ra.matrix_world.translation
    if abs(na.dot(nb))<.995 or abs(delta.dot(na))>.025 or delta.length<1.995:return None
    direction=np.array(delta.normalized());middle=np.array((ra.matrix_world.translation+rb.matrix_world.translation)*.5)
    count=0
    for d,sign in [(first,1.),(second,-1.)]:
        distance=(d['x']-middle)@direction*sign
        ids=np.flatnonzero((distance>-.0005)&(~d['pins']))
        d['x'][ids]-=(distance[ids]+.0005)[:,None]*direction[None,:]*sign
        count+=len(ids)
    return count


def soft_contacts(first,second):
    """Symmetric correction: distribute opposing impulse to hit face vertices."""
    plane_count=shared_contact_plane(first,second)
    if plane_count is not None:return plane_count
    x,y=first['x'],second['x'];tri=second['tri']
    if np.any(x.max(axis=0)<y.min(axis=0)-.002) or np.any(y.max(axis=0)<x.min(axis=0)-.002):return 0
    tree=BVHTree.FromPolygons(y.tolist(),tri.tolist(),all_triangles=True)
    low=y.min(axis=0)-.002;high=y.max(axis=0)+.002
    ids=np.flatnonzero(np.all((x>=low)&(x<=high),axis=1))
    count=0
    for i in ids:
        near,n,face,distance=tree.find_nearest(Vector(x[i]))
        if near is None:continue
        signed=(Vector(x[i])-near).dot(n)
        if signed>=.002:continue
        vertices=tri[face];wa=0. if first['pins'][i] else 1.
        wb=(~second['pins'][vertices]).astype(float)/3
        denom=wa+np.sum(wb)/3
        if denom<1e-8:continue
        correction=np.array(n)*min(.08,.002-signed)/denom
        x[i]+=wa*correction
        y[vertices]-=wb[:,None]*correction
        count+=1
    return count


def step(scene,states,dt,api):
    import bpy
    depsgraph=bpy.context.evaluated_depsgraph_get()
    objects=[o for o in scene.objects if o.type=='MESH' and (o.get('jelly_collider') or o.get('jelly_finger')) and not o.get('jelly_deform') and not o.hide_viewport]
    colliders=[Collider(o,depsgraph) for o in objects]
    data=[]
    for st in states:
        d=st.get('surface_solver') or init(st)
        root=st['anchor'];custom=root.get('jelly_custom_soft',False)
        rest=d['rest'];base_local=rest.copy() if custom else api.deform_positions(rest,st['q'],st['r'])
        base=transform(base_local,root.matrix_world)
        if d['x'] is None:d['x']=base.copy()
        elif 'root_matrix' in d:
            movement=root.matrix_world@d['root_matrix'].inverted()
            d['x']=transform(d['x'],movement)
            d['velocity']=d['velocity']@np.array(movement.to_3x3()).T
        d['root_matrix']=root.matrix_world.copy()
        d['base']=base;d['base_local']=base_local;d['root']=root;d['state']=st
        d['contact_count']=0;d['soft_contact_count']=0;data.append(d)
        edges=d['edges'];d['edge_lengths']=np.linalg.norm(base[edges[:,0]]-base[edges[:,1]],axis=1)
        target,normals=volume_gradient(base,d['tri']);d['target_volume']=target
        d['inward']=-normals/np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-8)
    # Real elapsed time is capped; two substeps maintain responsive contact.
    motion=max([float(np.max(np.linalg.norm(c.target-c.previous,axis=1))) for c in colliders]+[0.])
    steps=max(2,min(24,max(math.ceil(dt*120),math.ceil(motion/.06))));h=min(dt,.05)/steps
    for substep in range(steps):
        for c in colliders:
            if motion>1e-6:c.rebuild(c.previous+(c.target-c.previous)*((substep+1)/steps))
        for d in data:
            x=d['x'];old=x.copy();d['old']=old
            stiffness=scene.jelly_k*2
            d['velocity']+=h*(stiffness*(d['base']-x)-scene.jelly_damp*d['velocity'])
            x+=h*d['velocity'];x[d['pins']]=d['base'][d['pins']]
        for iteration in range(scene.jelly_solver_iterations):
            for d in data:
                edge_solve(d,d['base'],h,scene.jelly_k)
                # Preserve physical volume; optional extra bulge scales contact volume loss.
                volume_solve(d,d['target_volume']+(scene.jelly_bulge-1)*d.get('contact_loss',0))
                contact_ids=[]
                for collider in colliders:
                    for i,p in collider.contacts(d['x'],inward=d['inward']):
                        if not d['pins'][i]:d['x'][i]=p;d['contact_count']+=1;contact_ids.append(i)
                if contact_ids:
                    samples=d['x'][np.unique(contact_ids)[::max(1,len(contact_ids)//24)]]
                    distance2=np.min(np.sum((d['x'][:,None,:]-samples[None,:,:])**2,axis=2),axis=1)
                    d['influence']=.3+.7*np.exp(-distance2/(.3*scene.jelly_bulge_spread)**2)
                    current=volume_only(d['x'],d['tri'])
                    d['contact_loss']=max(0,d['target_volume']-current)
                else:d['influence']=None;d['contact_loss']=0
                d['x'][d['pins']]=d['base'][d['pins']]
            for i,d in enumerate(data):
                for other in data[i+1:]:
                    count=soft_contacts(d,other)+soft_contacts(other,d)
                    d['contact_count']+=count;other['contact_count']+=count
                    d['soft_contact_count']+=count;other['soft_contact_count']+=count
        for d in data:
            d['velocity']=(d['x']-d['old'])/h
            speed=np.linalg.norm(d['velocity'],axis=1)
            d['velocity']*=np.minimum(1,8/np.maximum(speed,1e-8))[:,None]
    for d in data:
        st=d['state'];local=transform(d['x'],d['root'].matrix_world.inverted())
        offset=local-d['base_local'];body=d['body']
        for obj,rest in st['entries']:
            mapping=d['followers'].get(obj.as_pointer())
            displacement=offset if mapping is None else np.sum(offset[mapping[0]]*mapping[1][:,:,None],axis=1)
            fixed=rest[:,2]<1e-5 if not d['root'].get('jelly_custom_soft') else np.zeros(len(rest),dtype=bool)
            if obj==body:
                # Smooth displacement, not positions: preserve the original surface
                # and avoid adding faces. Alternate with contact rather than one
                # destructive final projection of independently mapped vertices.
                edges=d['render_edges'];a,b=edges[:,0],edges[:,1]
                degree=d['render_degree']
                for _ in range(5):
                    for axis in range(3):
                        neighbours=np.bincount(a,weights=displacement[b,axis],minlength=len(rest))+np.bincount(b,weights=displacement[a,axis],minlength=len(rest))
                        displacement[:,axis]+=.45*(neighbours/degree-displacement[:,axis])
                    displacement[fixed]=0
                base_local=rest if d['root'].get('jelly_custom_soft') else api.deform_positions(rest,st['q'],st['r'])
                points=transform(base_local+displacement,d['root'].matrix_world)
                # A final render-mesh contact pass removes interpolation penetration.
                normals=transform(rest,d['root'].matrix_world)-np.array(d['root'].matrix_world.translation)
                inward=-normals/np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-8)
                for _ in range(3):
                    for collider in colliders:
                        for i,p in collider.contacts(points,inward=inward):
                            if not fixed[i]:points[i]=p
                    # A short tangential fairing pass removes alternating facet
                    # projections without pulling the fixed cap out of plane.
                    local_points=transform(points,d['root'].matrix_world.inverted())
                    delta=local_points-base_local
                    for axis in range(3):
                        neighbours=np.bincount(a,weights=delta[b,axis],minlength=len(rest))+np.bincount(b,weights=delta[a,axis],minlength=len(rest))
                        delta[:,axis]+=.15*(neighbours/degree-delta[:,axis])
                    delta[fixed]=0
                    points=transform(base_local+delta,d['root'].matrix_world)
                for collider in colliders:
                    for i,p in collider.contacts(points,inward=inward):
                        if not fixed[i]:points[i]=p
                displacement=transform(points,d['root'].matrix_world.inverted())-base_local
            if not d['root'].get('jelly_custom_soft'):
                collar=np.clip(rest[:,2]/.10,0,1)
                displacement*= (collar*collar*(3-2*collar))[:,None]
            displacement[fixed]=0
            st['dent'][obj.as_pointer()]=[displacement.copy(),np.zeros_like(displacement)]
        st['dent_active']=np.max(np.abs(offset))>2e-5
        st['max_dent']=float(np.max(np.linalg.norm(offset,axis=1)))
        api.apply_state(st,True)
        if d['root'].get('jelly_custom_soft'):
            rest=next(r for o,r in st['entries'] if o==body)
            body.data.vertices.foreach_set('co',(rest+st['dent'][body.as_pointer()][0]).ravel());body.data.update()

    rendered=[]
    for d in data:
        st=d['state'];body=d['body'];rest=next(r for o,r in st['entries'] if o==body)
        values=np.empty(len(rest)*3,dtype=np.float32);body.data.vertices.foreach_get('co',values)
        local=values.reshape(-1,3);world=transform(local,d['root'].matrix_world)
        body.data.calc_loop_triangles()
        tri=np.array([tuple(t.vertices) for t in body.data.loop_triangles],dtype=np.int32)
        rendered.append({'x':world,'tri':tri,'pins':rest[:,2]<1e-5 if not d['root'].get('jelly_custom_soft') else np.zeros(len(rest),dtype=bool),'body':body,'root':d['root'],'state':st,'local':local.copy()})
    for _ in range(3):
        for i,first in enumerate(rendered):
            for second in rendered[i+1:]:
                soft_contacts(first,second);soft_contacts(second,first)
    for r in rendered:
        local=transform(r['x'],r['root'].matrix_world.inverted());delta=local-r['local']
        if np.max(np.abs(delta))>1e-7:
            r['body'].data.vertices.foreach_set('co',local.ravel());r['body'].data.update()
            r['state']['dent'][r['body'].as_pointer()][0]+=delta
