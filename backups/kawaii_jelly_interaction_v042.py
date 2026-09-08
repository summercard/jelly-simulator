bl_info={'name':'Q弹果冻 · 重力与固定切面','author':'Codex','version':(4,2,0),'blender':(4,2,0),'location':'3D视图 > N侧栏 > Q弹果冻','description':'固定切面、重力下垂、抓拉快甩、摇晃与硬软调节','category':'3D View'}
import bpy,math,time
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils
from bpy.props import FloatProperty,BoolProperty,EnumProperty,IntProperty
from bpy.app.handlers import persistent
_STATE=None
_STATES={}
_MODAL=None
_UPDATING=False
_FINGER_CONTEXT={}

def anchors(scene):
    return sorted([o for o in scene.objects if o.get('jelly_anchor')],key=lambda o:o.get('jelly_instance_id',1))

def anchor(scene):
    items=anchors(scene)
    return next((o for o in items if o.get('jelly_instance_id',1)==scene.jelly_active),items[0] if items else None)

def finger(scene):
    return next((o for o in scene.objects if o.get('jelly_finger')),None)

def finger_handle(scene,create=False):
    handle=next((o for o in scene.objects if o.get('jelly_handle')),None)
    if handle is None and create and finger(scene):
        handle=bpy.data.objects.new('球体控制柄_XYZ',None)
        scene.collection.objects.link(handle)
        handle['jelly_handle']=True
        handle.empty_display_type='PLAIN_AXES';handle.empty_display_size=.55
        handle.show_in_front=True;handle.location=finger(scene).location
        handle['操作']='移动此控制柄：G 后按 X/Y/Z，或拖动三轴箭头。实体球受碰撞限制。'
    return handle

def sync_handle(scene,position):
    handle=finger_handle(scene,True)
    if handle:handle.location=position

def finger_size_changed(self,context):
    ball=finger(self)
    if ball:
        radius=float(ball.get('jelly_radius',.23))
        ball.scale=(self.jelly_ball_radius/radius,)*3
    parameter_changed(self,context)


def setup_plates(scene):
    for root in anchors(scene):
        for obj in list(root.children):
            if obj.type=='MESH' and not obj.get('jelly_deform') and ('托盘' in obj.name or obj.get('jelly_plate')):
                world=obj.matrix_world.copy();obj.parent=None;obj.matrix_world=world
                obj['jelly_plate']=True;obj['jelly_plate_root']=root.name;obj['jelly_collider']=False
                obj.lock_location=(False,)*3;obj.lock_rotation=(False,)*3
                obj['jelly_plate_last']=list(np.array(world).ravel())
                obj['操作']='移动此托盘，会带着整个果冻移动；托盘不参与碰撞'

def sync_plates(scene):
    bpy.context.view_layer.update()
    for obj in scene.objects:
        if obj.get('jelly_plate'):
            root=scene.objects.get(obj.get('jelly_plate_root',''))
            if root is None:continue
            previous=np.array(obj.get('jelly_plate_last',list(np.array(obj.matrix_world).ravel()))).reshape(4,4)
            plate=np.array(obj.matrix_world);base=np.array(root.matrix_world)
            if np.max(np.abs(plate-previous))<1e-6 and np.max(np.abs(base-previous))>1e-6:
                obj.matrix_world=root.matrix_world.copy()
            else:
                # Two rigid fixed cut disks cannot occupy the same plane/area.
                # Their jelly boundary (not the larger plate rim) limits dragging.
                candidate=obj.matrix_world.copy();normal=candidate.to_3x3().normalized()@Vector((0,0,1))
                if not root.get('jelly_custom_soft'):
                    for other in anchors(scene):
                        if other==root or other.get('jelly_custom_soft'):continue
                        other_normal=other.matrix_world.to_3x3().normalized()@Vector((0,0,1))
                        delta=candidate.translation-other.matrix_world.translation
                        lateral=delta-normal*delta.dot(normal)
                        if abs(normal.dot(other_normal))>.995 and abs(delta.dot(normal))<.025 and lateral.length<2.002:
                            old=Vector(previous[:3,3])-other.matrix_world.translation
                            old-=normal*old.dot(normal)
                            direction=old.normalized() if old.dot(lateral)<0 or lateral.length<1e-6 else lateral.normalized()
                            if direction.length<1e-6:direction=Vector((1,0,0))
                            candidate.translation=other.matrix_world.translation+normal*delta.dot(normal)+direction*2.002
                    obj.matrix_world=candidate
                root.matrix_world=obj.matrix_world.copy()
            obj['jelly_plate_last']=list(np.array(obj.matrix_world).ravel())
    bpy.context.view_layer.update()

class JELLY_OT_select_plate(bpy.types.Operator):
    bl_idname='jelly.select_plate';bl_label='移动当前盘子和果冻'
    def execute(self,context):
        if _MODAL:_MODAL.finish(context)
        setup_plates(context.scene)
        obj=next((o for o in context.scene.objects if o.get('jelly_plate_root')==anchor(context.scene).name),None)
        if obj is None:return {'CANCELLED'}
        for selected in context.selected_objects:selected.select_set(False)
        obj.select_set(True);context.view_layer.objects.active=obj
        if context.area and context.area.type=='VIEW_3D':
            context.space_data.show_gizmo=True;context.space_data.show_gizmo_object_translate=True
            bpy.ops.wm.tool_set_by_id(name='builtin.move')
        wake(context.scene);return {'FINISHED'}

def squeeze_changed(self,context):
    if _UPDATING:return
    for obj in self.objects:
        side=obj.get('jelly_squeeze_side')
        if side is not None:
            obj.location.x=float(obj['jelly_squeeze_home'])-float(side)*self.jelly_squeeze
    wake(self)


def rotation(scene):
    a=anchor(scene)
    return a.matrix_world.to_3x3().normalized() if a else None

def local_vector(scene,vec):
    r=rotation(scene)
    return np.array(r.transposed()@Vector(vec) if r else vec,dtype=float)

def deform_positions(rest,q,ripple=0.0):
    t=np.clip(rest[:,2],0,1);w=t*t*(3-2*t)
    stretch=np.maximum(.43,1.0+q[2]*w)
    radial=1.0/np.sqrt(stretch)
    radial+=ripple*w*np.sin(t*math.pi*2)*.10
    dst=rest.copy()
    dst[:,0]=rest[:,0]*radial+q[0]*w
    dst[:,1]=rest[:,1]*radial+q[1]*w
    dst[:,2]=rest[:,2]+q[2]*w+ripple*w*np.sin(t*math.pi)*.05
    # Bend cross sections towards the centre-line slope: sag is curved, not a rigid tilt.
    bend=.22*w
    dz=-bend*(q[0]*rest[:,0]+q[1]*rest[:,1])
    dst[:,2]+=dz
    dst[t<1e-7]=rest[t<1e-7]
    return dst

def new_state(scene,root=None):
    root=root or anchor(scene)
    entries=[]
    for obj in scene.objects:
        if obj.type!='MESH' or not obj.get('jelly_deform') or (root and obj.parent!=root):continue
        attr=obj.data.attributes.get('jelly_rest')
        if not attr:continue
        values=np.empty(len(obj.data.vertices)*3,dtype=np.float32);attr.data.foreach_get('vector',values)
        entries.append((obj,values.reshape((-1,3))))
    return {'scene':scene,'anchor':root,'dent':{},'dent_active':False,'max_dent':0.0,'entries':entries,'q':np.zeros(3),'v':np.zeros(3),'r':0.0,'rv':0.0,'target':None,'last':time.perf_counter(),'alive':True,'shake_until':0.0,'shake_phase':0.0,'demo':False,'next_poke':0.0,'last_draw':None,'sleeping':False,'peak_drag':0.0,'release_count':0}

def state(scene=None,root=None):
    global _STATE
    scene=scene or bpy.context.scene;root=root or anchor(scene);key=root.as_pointer() if root else 0
    if key not in _STATES or _STATES[key]['scene']!=scene:_STATES[key]=new_state(scene,root)
    _STATE=_STATES[key]
    return _STATE

def all_states(scene):
    result=[]
    for a in anchors(scene):
        key=a.as_pointer()
        if key not in _STATES:_STATES[key]=new_state(scene,a)
        result.append(_STATES[key])
    if not result:result=[state(scene)]
    return result

def gravity_force(scene):
    if not scene.jelly_gravity or not scene.use_gravity:return np.zeros(3)
    return local_vector(scene,scene.gravity)*np.array((1.25,1.25,.85))*scene.jelly_g

def spring_coefficients(scene):
    # A fixed cap is more compliant in bending than normal compression.
    tension=.70 if gravity_force(scene)[2]>0 else 1.0
    return np.array((.60,.60,tension))*scene.jelly_k

def limit_pose(q,v=None):
    old=q.copy();q[:2]=np.clip(q[:2],-1.15,1.15);q[2]=np.clip(q[2],-.47,1.05)
    if v is not None:
        for i in range(3):
            if old[i]!=q[i] and v[i]*(old[i]-q[i])>0:v[i]=0.0
    return q

def equilibrium(scene):
    return limit_pose(gravity_force(scene)/spring_coefficients(scene))

def integrate(s,dt,k=None,damping=None,force=None):
    scene=s['scene'];coeff=spring_coefficients(scene) if k is None else np.asarray(k,dtype=float)
    if coeff.ndim==0:coeff=np.repeat(coeff,3)
    damping=scene.jelly_damp if damping is None else damping
    force=gravity_force(scene) if force is None else np.asarray(force,dtype=float)
    count=max(1,math.ceil(dt*180));h=dt/count
    for _ in range(count):
        accel=force-coeff*s['q']-damping*s['v']
        if s['target'] is not None:accel=200*(s['target']-s['q'])-21*s['v']
        s['v']+=accel*h;s['q']+=s['v']*h;limit_pose(s['q'],s['v'])
        s['rv']+=(-85*s['r']-3.8*s['rv'])*h;s['r']+=s['rv']*h

def apply_state(s,force=False):
    signature=tuple(s['q'])+(s['r'],)
    if not force and not s['dent_active'] and s['last_draw'] and max(abs(a-b) for a,b in zip(signature,s['last_draw']))<.000003:return
    for obj,rest in s['entries']:
        try:
            co=deform_positions(rest,s['q'],s['r']);dent=s['dent'].get(obj.as_pointer())
            if dent is not None:co+=dent[0]
            obj.data.vertices.foreach_set('co',co.ravel());obj.data.update()
        except ReferenceError:continue
    s['last_draw']=signature
    for win in bpy.context.window_manager.windows:
        for a in win.screen.areas:
            if a.type=='VIEW_3D':a.tag_redraw()

def finger_context(scene):
    key=scene.as_pointer();ball=finger(scene)
    if key not in _FINGER_CONTEXT:
        _FINGER_CONTEXT[key]={'desired':ball.location.copy(),'last':ball.location.copy(),'candidate':ball.location.copy(),'active':scene.jelly_active,'auto':False,'blocked_count':0}
    return _FINGER_CONTEXT[key]

def surface_tree(st,dented=False):
    body,rest=next((o,r) for o,r in st['entries'] if o.get('jelly_body'))
    if 'surface_faces' not in st:st['surface_faces']=[tuple(p.vertices) for p in body.data.polygons]
    signature=tuple(st['q'])+(st['r'],)
    if not dented and st.get('base_tree_signature')==signature:return st['base_tree']
    co=deform_positions(rest,st['q'],st['r'])
    if dented and body.as_pointer() in st['dent']:co+=st['dent'][body.as_pointer()][0]
    tree=BVHTree.FromPolygons(co.tolist(),st['surface_faces'],all_triangles=False)
    if not dented:st['base_tree_signature']=signature;st['base_tree']=tree
    return tree

def position_finger(scene):
    ball=finger(scene);root=anchor(scene)
    if not ball or not root:return
    ctx=finger_context(scene)
    tool=active_tool(scene)
    if tool and tool.get('jelly_last_location') is not None and (tool.location-Vector(tool['jelly_last_location'])).length>.0001:
        scene.jelly_finger_auto=False;ctx['desired']=tool.location+tool_offset(scene,tool);sync_handle(scene,ctx['desired'])
    handle=finger_handle(scene)
    if scene.jelly_finger_auto and ctx['auto'] and handle is not None and (handle.location-ctx['desired']).length>.0001:
        scene.jelly_finger_auto=False
    if scene.jelly_finger_auto:
        st=state(scene,root);base=np.array([[.35,0,.93675]],dtype=np.float32)
        contact=Vector(deform_positions(base,st['q'],st['r'])[0]);normal=Vector((.35,0,.93675)).normalized();radius=float(ball.get('jelly_radius',.23))*max(ball.scale)
        outside=root.matrix_world@(contact+normal*(radius+.12))
        if not ctx['auto'] or ctx['active']!=scene.jelly_active:
            ball.location=outside;ctx['last']=outside.copy()
        depth=scene.jelly_press_depth if scene.jelly_press else -.12
        ctx['desired']=root.matrix_world@(contact+normal*(radius-depth))
        sync_handle(scene,ctx['desired'])
    else:
        handle=finger_handle(scene,True)
        if handle is not None and (handle.location-ctx['desired']).length>.00001:
            ctx['desired']=handle.location.copy()
        elif (ball.location-ctx['last']).length>.0001:
            ctx['desired']=ball.location.copy()
            sync_handle(scene,ctx['desired'])
    ctx['auto']=scene.jelly_finger_auto;ctx['active']=scene.jelly_active

def prepare_finger_contacts(scene,states):
    ball=finger(scene)
    if not ball:return
    ctx=finger_context(scene);wanted=ctx['desired'];start=ctx['last'];radius=ball.get('jelly_radius',.23)*max(ball.scale)
    for st in states:st['pressure_center']=None;st['pressure_force']=np.zeros(3)
    trees=[(st,surface_tree(st),st['anchor'].matrix_world.inverted()) for st in states]
    delta=wanted-start;steps=min(400,max(1,math.ceil(delta.length/(radius*.28))))
    collision=None
    for i in range(steps+1):
        point=start+delta*(i/steps)
        for st,tree,inv in trees:
            local=inv@point;near,n,face,dist=tree.find_nearest(local)
            if near is not None and (dist<radius-.0001 or (local-near).dot(n)<0):
                end_local=inv@wanted
                if (end_local-near).dot(n)>=radius and (inv.to_3x3()@delta).dot(n)>0:continue
                collision=(st,tree,inv,near,n);break
        if collision:break
    ctx['candidate']=wanted.copy()
    if collision:
        st,tree,inv,near,n=collision;local_wanted=inv@wanted
        end,en,ei,ed=tree.find_nearest(local_wanted)
        if end is not None and en.dot(n)>.35 and (end-near).length<radius*3:near,n=end,en
        requested=max(0,radius-(local_wanted-near).dot(n))
        max_indent=min(.65,radius*(.35+.80*(90-scene.jelly_k)/78)*scene.jelly_compression)
        depth=min(requested,max_indent)
        effective=near+n*(radius-depth)
        if not st['anchor'].get('jelly_custom_soft'):
            # Protect the disk and a small collar. Collision projects the tool,
            # never the cap; radius test also covers side-on approaches.
            radial=math.hypot(effective.x,effective.y)
            if radial<1+radius and effective.z<radius+.015:
                effective.z=radius+.015
                depth=max(0,radius-(effective-near).dot(n))
        st['pressure_center']=np.array(effective)
        st['pressure_normal']=np.array(n)
        st['pressure_depth']=depth
        st['pressure_force']=-np.array(n)*depth*min(15,scene.jelly_k*.32)
        ctx['candidate']=st['anchor'].matrix_world@effective
        ctx['blocked_count']+=1
        ctx['last_contact_depth']=depth
    return ctx

def resolve_solid_finger(scene,states):
    ball=finger(scene)
    if not ball:return
    ctx=finger_context(scene);point=ctx['candidate'].copy();radius=ball.get('jelly_radius',.23)*max(ball.scale)
    trees=[(st,surface_tree(st,True),st['anchor'].matrix_world.inverted()) for st in states if st.get('pressure_center') is None]
    for _ in range(32):
        changed=False
        for st,tree,inv in trees:
            local=inv@point;near,n,idx,dist=tree.find_nearest(local)
            if near is None:continue
            signed=(local-near).dot(n)
            if signed<0 or dist<radius+.001:
                direction=n if signed<=0 or dist<1e-7 else (local-near).normalized()
                point=st['anchor'].matrix_world@(near+direction*(radius+.0015));changed=True
        if not changed:break
    # Concave triangle corners may alternate nearest faces. Retreat along the
    # contact entry direction if nearest-point iterations have not separated them.
    for _ in range(80):
        blocked=False
        for st,tree,inv in trees:
            local=inv@point;near,n,idx,dist=tree.find_nearest(local)
            if near is not None and (dist<radius+.001 or (local-near).dot(n)<0):
                direction=Vector(st.get('pressure_normal',n))
                point+=st['anchor'].matrix_world.to_3x3()@direction*(radius*.025)
                blocked=True
        if not blocked:break
    ball.location=point;ctx['last']=point.copy()

def move_finger(scene,position):
    ctx=finger_context(scene);ctx['desired']=Vector(position);scene.jelly_finger_auto=False
    sync_handle(scene,position)
    if scene.jelly_surface_mode:
        prepare_finger_contacts(scene,all_states(scene));finger(scene).location=ctx['candidate'];ctx['last']=ctx['candidate'].copy();sync_active_tool(scene);wake(scene);return
    states=all_states(scene);prepare_finger_contacts(scene,states)
    for st in states:update_dents(st,1/60);apply_state(st,True)
    resolve_solid_finger(scene,states)


def mesh_triangles(st,body):
    if 'volume_triangles' not in st:
        body.data.calc_loop_triangles()
        st['volume_triangles']=np.array([t.vertices[:] for t in body.data.loop_triangles],dtype=np.int32)
    return st['volume_triangles']

def mesh_volume(co,tri):
    a,b,c=co[tri[:,0]],co[tri[:,1]],co[tri[:,2]]
    return float(np.sum(a*np.cross(b,c))/6)

def contact_fields(rest,co,centre,radius,scene,normal,depth):
    """Monotone spherical contact patch and a tangent-continuous shoulder.

    The patch parameter follows the collider surface; compensation only changes
    the free shoulder, so it cannot fold the contact patch back on itself.
    """
    diff=co-centre;axial=diff@normal
    tangent=diff-axial[:,None]*normal
    lateral=np.linalg.norm(tangent,axis=1)
    direction=tangent/np.maximum(lateral[:,None],1e-8)
    shell=radius+.0025
    patch=radius*math.sqrt(max(.02,1-max(0,1-depth/radius)**2))
    angle=min(1.48,math.acos(np.clip(1-depth/radius,-1,1))+.12*min(1,depth/radius))
    theta=angle*np.clip(lateral/patch,0,1)
    mapped_radius=shell*np.sin(theta)
    mapped_axial=-shell*np.cos(theta)
    width=radius*(.6+.35*scene.jelly_bulge_spread)
    u=np.clip((lateral-patch)/width,0,1)
    # Cubic Hermite shoulder starts tangent to the spherical contact patch.
    h00=2*u**3-3*u*u+1;h10=u**3-2*u*u+u
    h01=-2*u**3+3*u*u;h11=u**3-u*u
    end_radius=patch+width
    radial_outer=h00*(shell*math.sin(angle))+h10*(width*.65*math.cos(angle))+h01*end_radius+h11*width
    axial_outer=h00*(-shell*math.cos(angle))+h10*(width*.65*math.sin(angle))+h01*axial
    inner=lateral<=patch
    mapped_radius=np.where(inner,mapped_radius,radial_outer)
    mapped_axial=np.where(inner,mapped_axial,axial_outer)
    mapped=centre+direction*mapped_radius[:,None]+normal[None,:]*mapped_axial[:,None]
    front=np.clip((axial+radius*1.4)/(radius*.4),0,1)
    t=np.clip(rest[:,2]/.18,0,1);pin=t*t*(3-2*t)
    weight=pin*front*(lateral<patch+width)
    target=(mapped-co)*weight[:,None]
    # The volume shoulder meets the contact at zero value AND zero slope.
    ring=16*u*u*(1-u)*(1-u)*weight
    field=normal[None,:]*ring[:,None]
    return target,field


class SphereContactSurface:
    """Surface adapter: signed distance is positive outside the collider."""
    def __init__(self,centre,radius):
        self.centre=np.asarray(centre);self.radius=radius

    def query(self,points):
        diff=points-self.centre;length=np.linalg.norm(diff,axis=1)
        normals=diff/np.maximum(length[:,None],1e-8)
        normals[length<1e-8]=(0,0,1)
        nearest=self.centre+normals*self.radius
        return nearest,normals,length-self.radius

class MeshContactSurface:
    """Adapter for a closed, outward-wound mesh in the same coordinate space.

    Future non-spherical tools supply evaluated collider vertices and faces.
    Sharp/concave meshes additionally require continuous collision detection;
    this adapter alone does not enable them as interactive tools in the UI.
    """
    def __init__(self,vertices,faces):
        self.tree=BVHTree.FromPolygons(vertices,faces,all_triangles=False)

    def query(self,points):
        nearest=np.empty_like(points);normals=np.empty_like(points);signed=np.empty(len(points))
        for i,p in enumerate(points):
            hit,normal,_,distance=self.tree.find_nearest(Vector(p))
            if hit is None:raise ValueError('Collider surface has no faces')
            nearest[i]=hit
            delta=p-nearest[i]
            sign=1. if np.dot(delta,normal)>=0 else -1.
            signed[i]=distance*sign
            normals[i]=delta/max(distance,1e-8)*sign if distance>1e-6 else normal
        return nearest,normals,signed

def project_contact_surface(points,surface,clearance=.0025):
    """Shared unilateral constraint, independent of collider shape."""
    nearest,normals,signed=surface.query(points)
    return points+normals*np.maximum(clearance-signed,0)[:,None]

def sphere_surface(co,centre,radius):
    return project_contact_surface(co,SphereContactSurface(centre,radius))


def update_dents(st,dt):
    scene=st['scene'];ball=finger(scene)
    if not st['anchor'] or not st['entries']:return
    centre=st.get('pressure_center')
    radius=ball.get('jelly_radius',.23)*max(ball.scale) if ball else .23
    fields=[];amplitude=0.;body_info=None
    for obj,rest in st['entries']:
        co=deform_positions(rest,st['q'],st['r'])
        if centre is None:target=np.zeros_like(rest);field=np.zeros_like(rest)
        else:target,field=contact_fields(rest,co,centre,radius,scene,st['pressure_normal'],st['pressure_depth'])
        fields.append((obj,rest,co,target,field))
        if obj.get('jelly_body'):body_info=(obj,co,target,field)
    if centre is not None and body_info:
        obj,co,target,field=body_info;tri=mesh_triangles(st,obj)
        base=mesh_volume(co,tri);pressed=mesh_volume(co+target,tri)
        loss=max(0,abs(base)-abs(pressed))
        desired=pressed+math.copysign(loss*scene.jelly_bulge,base)
        # Solve one shared bulge amplitude from the actual closed body volume.
        # 1.0 compensates the lost volume; values above 1 exaggerate the bulge.
        for _ in range(8):
            current=sphere_surface(co+target+field*amplitude,centre,radius)
            volume=mesh_volume(current,tri)
            if abs(desired-volume)<max(abs(base)*1e-5,1e-8):break
            derivative=(mesh_volume(sphere_surface(co+target+field*(amplitude+.001),centre,radius),tri)-volume)/.001
            if abs(derivative)<1e-8:break
            amplitude=float(np.clip(amplitude+(desired-volume)/derivative,0,.45))
        st['displaced_volume']=loss;st['bulge_amplitude']=amplitude
    else:st['displaced_volume']=0.;st['bulge_amplitude']=0.
    active=False;peak=0.
    for obj,rest,co,target,field in fields:
        key=obj.as_pointer()
        if key not in st['dent']:st['dent'][key]=[np.zeros_like(rest),np.zeros_like(rest)]
        offset,vel=st['dent'][key];target=target+field*amplitude
        if centre is not None:target=sphere_surface(co+target,centre,radius)-co
        stiffness=65+scene.jelly_k*3.;damping=math.sqrt(stiffness)*1.05
        count=max(1,math.ceil(dt*120));h=dt/count
        for _ in range(count):
            vel[:]=(vel+h*stiffness*(target-offset))/(1+h*damping+h*h*stiffness)
            offset+=h*vel
        if centre is not None:
            # Radial contact allows the raised rim to wrap the side of the
            # sphere rather than cutting it back down to the rear hemisphere.
            contact=np.linalg.norm(co+target-centre,axis=1)<=radius+.003
            offset[contact]=target[contact];vel[contact]=0
            offset[:]=sphere_surface(co+offset,centre,radius)-co
        pins=rest[:,2]<1e-7;offset[pins]=0;vel[pins]=0
        magnitude=float(np.max(np.linalg.norm(offset,axis=1)));peak=max(peak,magnitude)
        if magnitude>.00002 or np.max(np.abs(vel))>.00003:active=True
        else:offset[:]=0;vel[:]=0
    st['dent_active']=active;st['max_dent']=peak


def tick():
    global _STATE
    if not _STATE or not _STATE['alive']:return None
    try:
        scene=_STATE['scene'];sync_plates(scene);now=time.perf_counter()
        if not scene.jelly_running:return .06
        states=all_states(scene);position_finger(scene);idle=True
        for st in states:
            dt=min(.08,max(.001,now-st['last']));st['step_dt']=dt;st['last']=now
            external=gravity_force(scene)+st.get('pressure_force',np.zeros(3))
            current=st['anchor']==anchor(scene);shaking=(scene.jelly_continuous_shake and current) or now<st['shake_until']
            if shaking:
                st['shake_phase']+=dt;env=1 if scene.jelly_continuous_shake and current else min(1,max(0,(st['shake_until']-now)/.4))
                external+=local_vector(scene,(18*math.sin(st['shake_phase']*8.7),7*math.cos(st['shake_phase']*7.1),0))*scene.jelly_shake_strength*env
            integrate(st,dt,force=external)
        if scene.jelly_surface_mode:
            import jelly_surface_solver,sys
            ball=finger(scene)
            if ball:
                # Same depth/stiffness limit in native and general contact modes.
                prepare_finger_contacts(scene,states)
                ctx=finger_context(scene);ball.location=ctx['candidate'];ctx['last']=ball.location.copy()
                sync_active_tool(scene)
            bpy.context.view_layer.update()
            jelly_surface_solver.step(scene,states,min(.05,max(st['step_dt'] for st in states)),sys.modules[__name__])
            return 1/60
        prepare_finger_contacts(scene,states)
        for st in states:
            was_dent=st['dent_active'];update_dents(st,st['step_dt']);apply_state(st,was_dent)
            st['sleeping']=np.linalg.norm(st['v'])<.0002 and np.linalg.norm(st['q']-equilibrium(scene))<.0001 and abs(st['r'])+abs(st['rv'])<.0002 and st['target'] is None and not st['dent_active']
            idle=idle and st['sleeping']
        resolve_solid_finger(scene,states)
        return .05 if idle and _MODAL is None else 1/60
    except (ReferenceError,RuntimeError):_STATE=None;return None

def wake(scene=None):
    s=state(scene);s['alive']=True;s['last']=time.perf_counter();s['sleeping']=False
    if not bpy.app.timers.is_registered(tick):bpy.app.timers.register(tick,first_interval=.01)
    return s

def settle(scene=None):
    s=state(scene);s.pop('surface_solver',None); s['q'][:]=equilibrium(s['scene']);s['v'][:]=0;s['r']=s['rv']=0;s['target']=None;s['demo']=False;s['shake_until']=0
    s['scene'].jelly_continuous_shake=False;apply_state(s,True);return s

def surface_mode_changed(self,context):
    for st in _STATES.values():st.pop('surface_solver',None)
    parameter_changed(self,context)

def press_depth_changed(self,context):
    if _UPDATING:return
    self.jelly_finger_auto=True;self.jelly_press=True
    wake(self)

def parameter_changed(self,context):
    if not _UPDATING and any(o.get('jelly_body') for o in self.objects):wake(self)

def finger_press_changed(self,context):
    if _UPDATING:return
    self.jelly_finger_auto=True
    wake(self)

def active_changed(self,context):
    if _UPDATING:return
    count=max(1,len(anchors(self)))
    if self.jelly_active>count:self.jelly_active=count
    wake(self)

def fit_front_camera(scene):
    roots=anchors(scene);cam=scene.camera
    if not roots or not cam:return
    mid=sum(a.location.x for a in roots)/len(roots)
    cam.location=(mid,-9,2.10);cam.rotation_euler=(Vector((mid,0,2.10))-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.ortho_scale=4.3+2.3*(len(roots)-1)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                rv=area.spaces.active.region_3d;rv.view_rotation=cam.rotation_euler.to_quaternion();rv.view_location=Vector((mid+cam.data.ortho_scale*.16,0,2.1));rv.view_distance=4.2+2.4*(len(roots)-1);rv.view_perspective='ORTHO'
                area.tag_redraw()

def orientation_changed(self,context):
    if _UPDATING:return
    for a in anchors(self):a.rotation_euler=(math.radians(self.jelly_tilt),0,0)
    if context and context.view_layer:context.view_layer.update()
    s=wake(self);s['target']=None
    if _MODAL:_MODAL.dragging=False
    fit_front_camera(self)

def reset(scene=None):
    return settle(scene)

class JELLY_OT_grab(bpy.types.Operator):
    bl_idname='jelly.grab';bl_label='开始抓拉互动';bl_description='左键慢拉、快甩或轻戳；松手按当前重力回摆；Esc退出'
    def invoke(self,context,event):
        global _MODAL
        if _MODAL is not None:self.report({'INFO'},'抓拉已开启，直接左键拖动果冻');return {'CANCELLED'}
        if context.window is None or context.area is None or context.area.type!='VIEW_3D':return {'CANCELLED'}
        if not context.scene.jelly_running:context.scene.jelly_running=True
        self.area=context.area;self.region=next(r for r in self.area.regions if r.type=='WINDOW');self.rv3d=context.space_data.region_3d
        self.dragging=False;self.finger_drag=False;self.closed=False;self.s=wake(context.scene);_MODAL=self
        context.window_manager.modal_handler_add(self);self.area.header_text_set('Q弹果冻 | 左键抓拉 / 快甩 / 轻戳 | 松手重力回弹 | 右键 / Esc退出');context.window.cursor_modal_set('HAND')
        return {'RUNNING_MODAL'}
    def ray(self,event):
        p=(event.mouse_x-self.region.x,event.mouse_y-self.region.y)
        d=view3d_utils.region_2d_to_vector_3d(self.region,self.rv3d,p)
        a=anchor(self.s['scene']);centre=a.matrix_world@Vector((0,0,.5)) if a else Vector((0,0,.5))
        point=view3d_utils.region_2d_to_location_3d(self.region,self.rv3d,p,centre)
        return point-d*15,d
    def finish(self,context):
        global _MODAL
        self.s['target']=None;self.dragging=False;self.closed=True
        if _MODAL is self:_MODAL=None
        try:self.area.header_text_set(None);context.window.cursor_modal_restore()
        except (ReferenceError,AttributeError):pass
        return {'FINISHED'}
    def modal(self,context,event):
        if self.closed:return {'FINISHED'}
        if not self.s['alive'] or (event.type in {'ESC','RIGHTMOUSE'} and event.value=='PRESS'):return self.finish(context)
        if not context.scene.jelly_running:return {'PASS_THROUGH'}
        if event.type=='LEFTMOUSE' and event.value=='PRESS':
            if not(self.region.x<=event.mouse_x<=self.region.x+self.region.width and self.region.y<=event.mouse_y<=self.region.y+self.region.height):return {'PASS_THROUGH'}
            origin,direction=self.ray(event)
            hit,loc,n,idx,obj,mat=context.scene.ray_cast(context.evaluated_depsgraph_get(),origin,direction)
            if hit and obj.get('jelly_finger'):
                self.dragging=True;self.finger_drag=True;self.ball=obj.original if hasattr(obj,'original') else obj
                context.scene.jelly_finger_auto=False;self.grab_world=loc.copy();self.normal=self.rv3d.view_rotation@Vector((0,0,1));self.finger_start=self.ball.location.copy();self.finger_depth=0.0
                return {'RUNNING_MODAL'}
            if not hit or not obj.get('jelly_deform'):return {'PASS_THROUGH'}
            self.finger_drag=False
            parent=obj.parent
            if parent and parent.get('jelly_anchor'):
                parent=parent.original if hasattr(parent,'original') else parent
                context.scene.jelly_active=parent.get('jelly_instance_id',1);self.s=state(context.scene,parent)
            a=anchor(context.scene);local=a.matrix_world.inverted()@loc if a else loc
            # Use the hit polygon's undeformed height: sag can put visible vertices below the cut plane in world space.
            original=obj.original if hasattr(obj,'original') else obj
            attr=original.data.attributes.get('jelly_rest')
            if attr and 0<=idx<len(original.data.polygons):t=sum(attr.data[i].vector.z for i in original.data.polygons[idx].vertices)/len(original.data.polygons[idx].vertices)
            else:t=float(local.z)
            if t<.045:return {'RUNNING_MODAL'}
            self.dragging=True;self.grab_world=loc.copy();self.normal=self.rv3d.view_rotation@Vector((0,0,1));self.startq=self.s['q'].copy()
            t=max(.045,min(1,float(t)));self.weight=max(.22,t*t*(3-2*t))
            self.s['target']=self.s['q'].copy();self.s['rv']+=.3
            self.last_target=self.s['q'].copy();self.last_motion=time.perf_counter();self.mouse_velocity=np.zeros(3);self.moved=0.0
            self.s['grab_count']=self.s.get('grab_count',0)+1
            return {'RUNNING_MODAL'}
        if self.dragging and self.finger_drag and event.type in {'WHEELUPMOUSE','WHEELDOWNMOUSE'}:
            shift=-.045 if event.type=='WHEELUPMOUSE' else .045
            self.finger_depth+=shift;move_finger(context.scene,finger_context(context.scene)['desired']+self.normal*shift)
            return {'RUNNING_MODAL'}
        if event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'} and self.dragging:
            o,d=self.ray(event);denom=d.dot(self.normal)
            if abs(denom)>1e-6:
                pos=o+d*((self.grab_world-o).dot(self.normal)/denom)
                if self.finger_drag:
                    move_finger(context.scene,self.finger_start+(pos-self.grab_world)+self.normal*self.finger_depth)
                    return {'RUNNING_MODAL'}
                delta=local_vector(context.scene,pos-self.grab_world)/self.weight
                target=limit_pose(self.startq+delta);now=time.perf_counter();dt=max(.012,now-self.last_motion)
                velocity=(target-self.last_target)/dt;speed=np.linalg.norm(velocity)
                if speed>5:velocity*=5/speed
                self.mouse_velocity=.4*self.mouse_velocity+.6*velocity;self.last_target=target.copy();self.last_motion=now;self.moved=max(self.moved,float(np.linalg.norm(delta)))
                self.s['target']=target;self.s['q']+=.5*(target-self.s['q']);apply_state(self.s)
                self.s['last_drag_target']=target.tolist();self.s['peak_drag']=max(self.s['peak_drag'],float(np.linalg.norm(self.s['q']-equilibrium(context.scene))))
            return {'RUNNING_MODAL'}
        if event.type=='LEFTMOUSE' and event.value=='RELEASE' and self.dragging:
            if self.finger_drag:
                self.dragging=False;self.finger_drag=False
                return {'RUNNING_MODAL'}
            if self.s['target'] is not None:self.s['q']+=.55*(self.s['target']-self.s['q'])
            recent=max(0,1-(time.perf_counter()-self.last_motion)/.16)
            self.s['v']=.35*self.s['v']+.65*self.mouse_velocity*recent
            if self.moved<.025:self.s['v']+=local_vector(context.scene,-self.normal)*1.25
            self.s['last_release_q']=self.s['q'].tolist();self.s['last_release_velocity']=self.s['v'].tolist();self.s['release_count']+=1
            self.s['target']=None;self.s['rv']+=1.0;self.dragging=False;apply_state(self.s)
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}

class JELLY_OT_poke(bpy.types.Operator):
    bl_idname='jelly.poke';bl_label='戳一下'
    def execute(self,context):s=wake(context.scene);s['v']+=np.array((1.4,-.25,-1.8));s['rv']+=1.8;return {'FINISHED'}
class JELLY_OT_squash(bpy.types.Operator):
    bl_idname='jelly.squash';bl_label='压扁一下'
    def execute(self,context):s=wake(context.scene);s['v']+=np.array((0,0,-2.5));s['rv']+=2;return {'FINISHED'}
class JELLY_OT_shake(bpy.types.Operator):
    bl_idname='jelly.shake';bl_label='来回摇晃3秒';bl_description='交替施力，基座位置与朝向不动；结束后保留自然余晃'
    def execute(self,context):s=wake(context.scene);s['shake_until']=time.perf_counter()+3;s['shake_phase']=0;return {'FINISHED'}
class JELLY_OT_settle(bpy.types.Operator):
    bl_idname='jelly.settle';bl_label='停止晃动并归稳'
    def execute(self,context):settle(context.scene);return {'FINISHED'}
class JELLY_OT_pose(bpy.types.Operator):
    bl_idname='jelly.pose';bl_label='设置固定切面朝向'
    pose:EnumProperty(items=[('UP','平放',''),('SIDE','竖放',''),('DOWN','倒挂','')],default='UP')
    def execute(self,context):
        context.scene.jelly_tilt={'UP':0.0,'SIDE':90.0,'DOWN':180.0}[self.pose];wake(context.scene);return {'FINISHED'}
class JELLY_OT_preset(bpy.types.Operator):
    bl_idname='jelly.preset';bl_label='选择硬软预设'
    preset:EnumProperty(items=[('FIRM','偏硬',''),('BOUNCY','Q弹',''),('SOFT','软糯','')],default='BOUNCY')
    def execute(self,context):
        s=context.scene
        s.jelly_k,s.jelly_damp={'FIRM':(75,4.0),'BOUNCY':(30,2.2),'SOFT':(16,1.7)}[self.preset]
        wake(s);return {'FINISHED'}
class JELLY_OT_finger_home(bpy.types.Operator):
    bl_idname='jelly.finger_home';bl_label='球形手指放到当前果冻前'
    def execute(self,context):context.scene.jelly_finger_auto=True;context.scene.jelly_press=False;position_finger(context.scene);wake(context.scene);return {'FINISHED'}
class JELLY_OT_control_ball(bpy.types.Operator):
    bl_idname='jelly.control_ball';bl_label='三轴操控球体';bl_description='选择独立控制柄，用移动箭头、G X/Y/Z 或坐标输入施压'
    def execute(self,context):
        if _MODAL:_MODAL.finish(context)
        scene=context.scene;ball=finger(scene)
        if not ball:return {'CANCELLED'}
        scene.jelly_finger_auto=False;scene.jelly_running=True
        ctx=finger_context(scene);ctx['desired']=ball.location.copy();sync_handle(scene,ball.location)
        for obj in context.view_layer.objects:
            if obj.select_get():obj.select_set(False)
        handle=finger_handle(scene,True);handle.hide_set(False);handle.select_set(True)
        context.view_layer.objects.active=handle
        scene.transform_orientation_slots[0].type='GLOBAL'
        if context.area and context.area.type=='VIEW_3D':
            context.space_data.show_gizmo=True;context.space_data.show_gizmo_object_translate=True
            context.space_data.overlay.show_overlays=True
            bpy.ops.wm.tool_set_by_id(name='builtin.move')
        wake(scene);return {'FINISHED'}

class JELLY_OT_ball_step(bpy.types.Operator):
    bl_idname='jelly.ball_step';bl_label='球体轴向微调'
    axis:IntProperty(default=0,min=0,max=2)
    direction:FloatProperty(default=1.)
    def execute(self,context):
        scene=context.scene
        if not finger(scene):return {'CANCELLED'}
        position=finger_context(scene)['desired'].copy()
        position[self.axis]+=self.direction*scene.jelly_move_step
        move_finger(scene,position);wake(scene);return {'FINISHED'}


class JELLY_OT_clone(bpy.types.Operator):
    bl_idname='jelly.clone';bl_label='复制一个到旁边';bl_description='复制网格与固定基座；每个果冻独立响应抓拉、挤压与冲量'
    def execute(self,context):
        scene=context.scene;source=anchor(scene)
        if not source:return {'CANCELLED'}
        if len(anchors(scene))>=6:self.report({'INFO'},'当前最多保留6个，避免实时模拟变慢');return {'CANCELLED'}
        from mathutils import Matrix
        n=max(a.get('jelly_instance_id',1) for a in anchors(scene))+1
        coll=bpy.data.collections.new('复制果冻_%02d'%n);scene.collection.children.link(coll)
        new=source.copy();new.name='固定切面基座_%02d'%n;new['jelly_instance_id']=n;coll.objects.link(new);new.location=(3.0*(n-1),0,2.2)
        for obj in list(scene.objects):
            if obj.parent!=source:continue
            clone=obj.copy();clone.data=obj.data.copy();clone.name=obj.name+'_副本%02d'%n;coll.objects.link(clone);clone.parent=new;clone.matrix_parent_inverse=Matrix.Identity(4)
            if clone.get('jelly_deform'):
                attr=clone.data.attributes.get('jelly_rest');arr=np.empty(len(clone.data.vertices)*3,dtype=np.float32);attr.data.foreach_get('vector',arr);clone.data.vertices.foreach_set('co',arr);clone.data.update()
        plate=next((o for o in scene.objects if o.get('jelly_plate_root')==source.name),None)
        if plate:
            copy=plate.copy();copy.data=plate.data.copy();scene.collection.objects.link(copy)
            copy.name=plate.name+'_副本%02d'%n;copy['jelly_plate_root']=new.name
            copy.matrix_world=new.matrix_world.copy();copy['jelly_plate_last']=list(np.array(copy.matrix_world).ravel())
        scene.jelly_active=n;bpy.context.view_layer.update();settle(scene);wake(scene);position_finger(scene)
        cam=scene.camera
        if cam:
            fit_front_camera(scene)
        return {'FINISHED'}

def active_tool(scene):
    return next((o for o in scene.objects if o.get('jelly_tool_active')),None)

def tool_offset(scene,tool):
    thickness={'HAND':.14,'BOX':.25,'OVAL':.3}.get(tool.get('jelly_tool_kind'),scene.jelly_ball_radius)
    return tool.matrix_world.to_3x3().normalized()@Vector((0,0,scene.jelly_ball_radius-thickness))

def sync_active_tool(scene):
    tool=active_tool(scene);ball=finger(scene)
    if tool and ball:
        tool.location=ball.location-tool_offset(scene,tool)
        tool['jelly_last_location']=list(tool.location)

def activate_tool(scene,tool):
    ball=finger(scene)
    for root in scene.objects:
        if root.get('jelly_tool_kind'):
            enabled=root==tool;root['jelly_tool_active']=enabled
            for obj in [root]+list(root.children):
                obj.hide_viewport=not enabled;obj.hide_render=not enabled
                if obj.type=='MESH':obj['jelly_collider']=enabled
    ball.hide_viewport=tool is not None;ball.hide_render=tool is not None
    sync_active_tool(scene)
    for obj in bpy.context.selected_objects:obj.select_set(False)
    selected=tool or ball
    selected.hide_set(False);selected.select_set(True);bpy.context.view_layer.objects.active=selected
    scene.jelly_finger_auto=True;scene.jelly_press=False
    for st in all_states(scene):st.pop('surface_solver',None)
    import jelly_surface_solver
    jelly_surface_solver._COLLIDER_HISTORY.clear()


class JELLY_OT_demo_shape(bpy.types.Operator):
    bl_idname='jelly.demo_shape';bl_label='一键替换交互形状'
    shape:EnumProperty(items=[('SPHERE','球体',''),('BOX','圆角方块',''),('OVAL','椭球',''),('HAND','骨骼手掌示例','')],default='BOX')
    def execute(self,context):
        from mathutils import Matrix
        scene=context.scene;a=anchor(scene)
        if _MODAL:_MODAL.finish(context)
        if self.shape=='SPHERE':
            activate_tool(scene,None);wake(scene);return {'FINISHED'}
        existing=next((o for o in scene.objects if o.get('jelly_tool_kind')==self.shape),None)
        if existing:
            activate_tool(scene,existing);wake(scene);return {'FINISHED'}
        location=a.matrix_world@Vector((0,0,1.35))
        if self.shape=='HAND':
            bpy.ops.object.armature_add(location=location)
            rig=context.object;rig.name='交互手掌_骨骼示例'
            rig.matrix_world=a.matrix_world@Matrix.Translation((0,0,1.35))
            bpy.ops.object.mode_set(mode='EDIT');rig.data.edit_bones.remove(rig.data.edit_bones[0])
            palm=rig.data.edit_bones.new('掌心');palm.head=(0,-.3,0);palm.tail=(0,.15,0)
            for i in range(5):
                x=(i-2)*.22
                b=rig.data.edit_bones.new('手指%d'%i);b.head=(x,.1,0);b.tail=(x,.65,0);b.parent=palm
            bpy.ops.object.mode_set(mode='OBJECT')
            pieces=[('掌心',(0,-.08,0),(.56,.33,.14))]+[('手指%d'%i,((i-2)*.22,.46,0),(.105,.38,.11)) for i in range(5)]
            for bone,centre,scale in pieces:
                bpy.ops.mesh.primitive_uv_sphere_add(segments=20,ring_count=12)
                obj=context.object;obj.name='手掌碰撞_'+bone
                for v in obj.data.vertices:
                    v.co=Vector((v.co.x*scale[0]+centre[0],v.co.y*scale[1]+centre[1],v.co.z*scale[2]+centre[2]))
                obj.parent=rig;obj.matrix_basis=Matrix.Identity(4)
                obj['jelly_collider']=True
                group=obj.vertex_groups.new(name=bone);group.add(list(range(len(obj.data.vertices))),1.,'REPLACE')
                modifier=obj.modifiers.new('骨骼表面变形','ARMATURE');modifier.object=rig
                for face in obj.data.polygons:face.use_smooth=True
            rig['jelly_hand_demo']=True
            context.view_layer.objects.active=rig
            for obj in context.selected_objects:obj.select_set(False)
            rig.select_set(True)
        else:
            if self.shape=='BOX':
                bpy.ops.mesh.primitive_cube_add(size=1,location=location)
                obj=context.object;obj.scale=(.8,.65,.5)
                bevel=obj.modifiers.new('圆角','BEVEL');bevel.width=.12;bevel.segments=4
            else:
                bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=20,location=location)
                obj=context.object;obj.scale=(.55,.25,.3)
            obj.rotation_euler=a.rotation_euler;obj.name='交互_'+self.shape;obj['jelly_collider']=True
        root=rig if self.shape=='HAND' else obj
        root['jelly_tool_kind']=self.shape
        activate_tool(scene,root)
        scene.jelly_surface_mode=True;wake(scene);return {'FINISHED'}

def hand_curl_changed(self,context):
    for rig in self.objects:
        if rig.type=='ARMATURE' and rig.get('jelly_hand_demo'):
            for bone in rig.pose.bones:
                if bone.name.startswith('手指'):
                    bone.rotation_mode='XYZ';bone.rotation_euler.x=-math.radians(self.jelly_hand_curl)
    parameter_changed(self,context)


class JELLY_OT_collider(bpy.types.Operator):
    bl_idname='jelly.collider';bl_label='选中网格设为交互物体'
    remove:BoolProperty(default=False)
    def execute(self,context):
        meshes=[o for o in context.selected_objects if o.type=='MESH' and not o.get('jelly_deform')]
        if not meshes:self.report({'WARNING'},'请选择封闭的网格物体');return {'CANCELLED'}
        for obj in meshes:obj['jelly_collider']=not self.remove
        if not self.remove:context.scene.jelly_surface_mode=True
        wake(context.scene);return {'FINISHED'}

class JELLY_OT_soft_mesh(bpy.types.Operator):
    bl_idname='jelly.soft_mesh';bl_label='选中网格设为软物体'
    def execute(self,context):
        from mathutils import Matrix
        obj=context.object
        if not obj or obj.type!='MESH' or obj.get('jelly_deform'):
            self.report({'WARNING'},'请选择独立的封闭网格');return {'CANCELLED'}
        if len(anchors(context.scene))>=6:
            self.report({'WARNING'},'最多6个软物体');return {'CANCELLED'}
        if obj.modifiers or obj.parent:
            self.report({'WARNING'},'软物体请先应用修改器并解除父级；骨骼模型请设为交互物体');return {'CANCELLED'}
        scene=context.scene;n=max([a.get('jelly_instance_id',1) for a in anchors(scene)]+[0])+1
        root=bpy.data.objects.new('软物体基座_%02d'%n,None);scene.collection.objects.link(root)
        root.matrix_world=obj.matrix_world.copy();root['jelly_anchor']=True;root['jelly_custom_soft']=True;root['jelly_instance_id']=n
        obj.data=obj.data.copy();obj.parent=root;obj.matrix_parent_inverse=Matrix.Identity(4);obj.matrix_basis=Matrix.Identity(4)
        obj['jelly_body']=True;obj['jelly_deform']=True;obj['jelly_collider']=False
        attr=obj.data.attributes.get('jelly_rest') or obj.data.attributes.new('jelly_rest','FLOAT_VECTOR','POINT')
        values=np.empty(len(obj.data.vertices)*3,dtype=np.float32);obj.data.vertices.foreach_get('co',values);attr.data.foreach_set('vector',values)
        scene.jelly_surface_mode=True;wake(scene);return {'FINISHED'}


class JELLY_PT_panel(bpy.types.Panel):
    bl_label='Q弹果冻 · 重力实验台';bl_idname='JELLY_PT_panel';bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='Q弹果冻'
    @classmethod
    def poll(cls,context):return any(o.get('jelly_body') for o in context.scene.objects)
    def draw(self,context):
        l=self.layout;s=context.scene
        l.prop(s,'jelly_active');l.operator('jelly.clone',icon='DUPLICATE')
        l.operator('jelly.select_plate',icon='TRANSFORM_MOVE')
        l.prop(s,'jelly_squeeze',slider=True)
        row=l.row();row.scale_y=1.3;row.operator('jelly.grab',icon='HAND')
        l.label(text='左键抓拉 / 快甩，松手回弹');l.label(text='右键或 Esc 退出抓拉')
        box=l.box();box.label(text='1. 固定切面朝向',icon='LOCKED');r=box.row(align=True)
        for key,label in [('UP','平放'),('SIDE','竖放'),('DOWN','倒挂')]:r.operator('jelly.pose',text=label).pose=key
        box.prop(s,'jelly_tilt',slider=True);box.label(text='只在你调整朝向时转动切面')
        box=l.box();box.label(text='2. 重力与硬软');box.prop(s,'jelly_gravity');box.prop(s,'jelly_g',slider=True)
        r=box.row(align=True)
        for key,label in [('FIRM','偏硬'),('BOUNCY','Q弹'),('SOFT','软糯')]:r.operator('jelly.preset',text=label).preset=key
        box.prop(s,'jelly_k',slider=True);box.prop(s,'jelly_damp',slider=True);box.label(text='硬度越小，下垂越大')
        box=l.box();box.label(text='通用表面交互')
        box.prop(s,'jelly_surface_mode')
        row=box.row(align=True)
        for shape,label in [('SPHERE','球体'),('BOX','方块'),('OVAL','椭球'),('HAND','骨骼手')]:
            row.operator('jelly.demo_shape',text=label).shape=shape
        box.prop(s,'jelly_hand_curl',slider=True)
        box.operator('jelly.collider')
        op=box.operator('jelly.collider',text='移除选中物体的交互');op.remove=True
        box.operator('jelly.soft_mesh')
        box.prop(s,'jelly_solver_iterations')
        box.label(text='支持骨骼 / 形态键变形后的封闭表面')
        box.label(text='软物体之间双向挤压；抓拉与回弹保留')
        box=l.box();box.label(text='3. 球体三轴与体积挤压')
        box.operator('jelly.control_ball',icon='EMPTY_AXIS')
        handle=finger_handle(s)
        if handle:
            coords=box.column();coords.enabled=not s.jelly_finger_auto
            coords.prop(handle,'location',text='目标坐标')
        box.label(text='控制柄：G 后按 X / Y / Z，或拖箭头')
        box.prop(s,'jelly_move_step')
        row=box.row(align=True)
        for axis,label in enumerate(('X','Y','Z')):
            for sign in (-1,1):
                op=row.operator('jelly.ball_step',text=label+('−' if sign<0 else '+'));op.axis=axis;op.direction=sign
        box.prop(s,'jelly_ball_radius',slider=True)
        box.operator('jelly.finger_home');box.prop(s,'jelly_press',toggle=True)
        box.prop(s,'jelly_press_depth',slider=True)
        box.prop(s,'jelly_compression',slider=True)
        box.prop(s,'jelly_bulge',slider=True);box.prop(s,'jelly_bulge_spread',slider=True)
        box.label(text='鼓出 1 = 补偿压掉体积；>1 加强鼓包')
        box.label(text='替换后共用三轴控制柄与压入参数')
        box=l.box();box.label(text='4. 摇晃与回弹');r=box.row(align=True);r.operator('jelly.poke');r.operator('jelly.squash')
        box.operator('jelly.shake',icon='FORCE_HARMONIC');box.prop(s,'jelly_continuous_shake');box.prop(s,'jelly_shake_strength',slider=True)
        box.operator('jelly.settle',icon='LOOP_BACK');box.prop(s,'jelly_running',toggle=True)
        l.label(text='切面不会因重力或摇晃掉落',icon='PINNED')
        l.label(text='参数作用于全部；抓拉与冲量分别响应')
        l.label(text='详细操作：同目录《使用说明》')

@persistent
def before_load(_):
    import sys
    if 'jelly_surface_solver' in sys.modules:sys.modules['jelly_surface_solver']._COLLIDER_HISTORY.clear()
    global _STATE,_MODAL
    for st in _STATES.values():st['alive']=False
    _STATES.clear()
    if _STATE:_STATE['alive']=False
    if bpy.app.timers.is_registered(tick):bpy.app.timers.unregister(tick)
    _STATE=None;_MODAL=None;_FINGER_CONTEXT.clear()
@persistent
def after_load(_):
    if any(o.get('jelly_body') for o in bpy.context.scene.objects):
        scene=bpy.context.scene
        setup_plates(scene)
        finger_handle(scene,True);finger_size_changed(scene,bpy.context)
        for st in all_states(scene):st['q'][:]=equilibrium(scene);apply_state(st,True)
        wake(scene)

_CLASSES=(JELLY_OT_select_plate,JELLY_OT_grab,JELLY_OT_poke,JELLY_OT_squash,JELLY_OT_shake,JELLY_OT_settle,JELLY_OT_pose,JELLY_OT_preset,JELLY_OT_finger_home,JELLY_OT_control_ball,JELLY_OT_ball_step,JELLY_OT_clone,JELLY_OT_demo_shape,JELLY_OT_collider,JELLY_OT_soft_mesh,JELLY_PT_panel)
_PROPS=('jelly_k','jelly_damp','jelly_g','jelly_gravity','jelly_tilt','jelly_shake_strength','jelly_continuous_shake','jelly_running','jelly_active','jelly_finger_auto','jelly_press','jelly_press_depth','jelly_ball_radius','jelly_compression','jelly_bulge','jelly_bulge_spread','jelly_move_step','jelly_surface_mode','jelly_solver_iterations','jelly_hand_curl','jelly_squeeze')
def register():
    for cls in _CLASSES:bpy.utils.register_class(cls)
    bpy.types.Scene.jelly_squeeze=FloatProperty(name='两侧方块向中间挤压',default=0,min=0,max=.7,update=squeeze_changed)
    bpy.types.Scene.jelly_hand_curl=FloatProperty(name='示例手指弯曲',default=0,min=0,max=80,update=hand_curl_changed)
    bpy.types.Scene.jelly_surface_mode=BoolProperty(name='通用软体表面求解',default=False,update=surface_mode_changed)
    bpy.types.Scene.jelly_solver_iterations=IntProperty(name='接触迭代次数',default=4,min=2,max=12)
    bpy.types.Scene.jelly_ball_radius=FloatProperty(name='球体半径',default=.38,min=.12,max=.65,update=finger_size_changed)
    bpy.types.Scene.jelly_compression=FloatProperty(name='可压入程度',default=1.25,min=.2,max=2.,update=parameter_changed)
    bpy.types.Scene.jelly_bulge=FloatProperty(name='体积鼓出程度',default=1.,min=0,max=2.,update=parameter_changed)
    bpy.types.Scene.jelly_bulge_spread=FloatProperty(name='鼓出扩散范围',default=1.1,min=.5,max=2.5,update=parameter_changed)
    bpy.types.Scene.jelly_move_step=FloatProperty(name='轴向步长',default=.05,min=.005,max=.5,precision=3)
    bpy.types.Scene.jelly_active=IntProperty(name='当前果冻编号',default=1,min=1,max=6,update=active_changed)
    bpy.types.Scene.jelly_finger_auto=BoolProperty(name='手指跟随当前果冻',default=True)
    bpy.types.Scene.jelly_press=BoolProperty(name='按下球体 / 松开',default=False,update=finger_press_changed)
    bpy.types.Scene.jelly_press_depth=FloatProperty(name='挤压深度',default=.36,min=0,max=.65,update=press_depth_changed)
    bpy.types.Scene.jelly_k=FloatProperty(name='硬度（越小越软）',default=30,min=12,max=90,update=parameter_changed)
    bpy.types.Scene.jelly_damp=FloatProperty(name='阻尼（越小余晃越久）',default=2.2,min=.6,max=10,update=parameter_changed)
    bpy.types.Scene.jelly_g=FloatProperty(name='重力倍率',default=1,min=0,max=2,update=parameter_changed)
    bpy.types.Scene.jelly_gravity=BoolProperty(name='启用重力下垂',default=True,update=parameter_changed)
    bpy.types.Scene.jelly_tilt=FloatProperty(name='切面倾角 °',default=0,min=0,max=180,update=orientation_changed)
    bpy.types.Scene.jelly_shake_strength=FloatProperty(name='摇晃力度',default=1,min=.1,max=2.5,update=parameter_changed)
    bpy.types.Scene.jelly_continuous_shake=BoolProperty(name='持续来回摇晃',default=False,update=parameter_changed)
    bpy.types.Scene.jelly_running=BoolProperty(name='实时模拟：开 / 暂停',default=True,update=parameter_changed)
    if before_load not in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.append(before_load)
    if after_load not in bpy.app.handlers.load_post:bpy.app.handlers.load_post.append(after_load)
def unregister():
    global _STATE
    if _STATE:_STATE['alive']=False
    if bpy.app.timers.is_registered(tick):bpy.app.timers.unregister(tick)
    if before_load in bpy.app.handlers.load_pre:bpy.app.handlers.load_pre.remove(before_load)
    if after_load in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(after_load)
    for cls in reversed(_CLASSES):bpy.utils.unregister_class(cls)
    for p in _PROPS:
        if hasattr(bpy.types.Scene,p):delattr(bpy.types.Scene,p)
    _STATE=None
if __name__=='__main__':register()
