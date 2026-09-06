from pathlib import Path
p=Path('/Users/summercards/ShellStorm2/output/kawaii_jelly_v003/kawaii_jelly_interaction.py');s=p.read_text()
s=s.replace('from mathutils import Vector','from mathutils import Vector\nfrom mathutils.bvhtree import BVHTree',1)
s=s.replace('_UPDATING=False','_UPDATING=False\n_FINGER_CONTEXT={}',1)
a=s.index('def position_finger(scene):');b=s.index('\ndef update_dents',a)
s=s[:a]+'''def finger_context(scene):
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
    if scene.jelly_finger_auto:
        st=state(scene,root);base=np.array([[-.12,-.67,.732]],dtype=np.float32)
        contact=Vector(deform_positions(base,st['q'],st['r'])[0]);normal=Vector((-.12,-.67,.732)).normalized();radius=float(ball.get('jelly_radius',.23))
        outside=root.matrix_world@(contact+normal*(radius+.12))
        if not ctx['auto'] or ctx['active']!=scene.jelly_active:
            ball.location=outside;ctx['last']=outside.copy()
        depth=scene.jelly_press_depth if scene.jelly_press else -.12
        ctx['desired']=root.matrix_world@(contact+normal*(radius-depth))
    elif (ball.location-ctx['last']).length>.0001:
        # Native G movement is treated as a requested position, never as permission to tunnel.
        ctx['desired']=ball.location.copy()
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
            if near is not None and (dist<radius-.0001 or (local-near).dot(n)<0):collision=(st,tree,inv,near,n);break
        if collision:break
    ctx['candidate']=wanted.copy()
    if collision:
        st,tree,inv,near,n=collision;local_wanted=inv@wanted
        end,en,ei,ed=tree.find_nearest(local_wanted)
        if end is not None and en.dot(n)>.35 and (end-near).length<radius*3:near,n=end,en
        requested=max(0,radius-(local_wanted-near).dot(n))
        max_indent=.12+.25*(90-scene.jelly_k)/78
        depth=min(requested,max_indent)
        effective=near+n*(radius-depth)
        st['pressure_center']=np.array(effective)
        st['pressure_force']=-np.array(n)*depth*min(15,scene.jelly_k*.32)
        ctx['candidate']=st['anchor'].matrix_world@effective
        ctx['blocked_count']+=1
        ctx['last_contact_depth']=depth
    return ctx

def resolve_solid_finger(scene,states):
    ball=finger(scene)
    if not ball:return
    ctx=finger_context(scene);point=ctx['candidate'].copy();radius=ball.get('jelly_radius',.23)*max(ball.scale)
    trees=[(st,surface_tree(st,True),st['anchor'].matrix_world.inverted()) for st in states]
    for _ in range(8):
        changed=False
        for st,tree,inv in trees:
            local=inv@point;near,n,idx,dist=tree.find_nearest(local)
            if near is None:continue
            signed=(local-near).dot(n)
            if signed<0 or dist<radius+.001:
                direction=n if signed<=0 or dist<1e-7 else (local-near).normalized()
                point=st['anchor'].matrix_world@(near+direction*(radius+.0015));changed=True
        if not changed:break
    ball.location=point;ctx['last']=point.copy()

def move_finger(scene,position):
    ctx=finger_context(scene);ctx['desired']=Vector(position);scene.jelly_finger_auto=False
    states=all_states(scene);prepare_finger_contacts(scene,states)
    for st in states:update_dents(st,1/60);apply_state(st,True)
    resolve_solid_finger(scene,states)

''' +s[b:]
s=s.replace("    centre=np.array(root.matrix_world.inverted()@ball.location) if ball else np.array((100,100,100))","    centre=st.get('pressure_center')\n    if centre is None:centre=np.array((100,100,100))")
a=s.index('def tick():');b=s.index('\ndef wake(',a)
s=s[:a]+'''def tick():
    global _STATE
    if not _STATE or not _STATE['alive']:return None
    try:
        scene=_STATE['scene'];now=time.perf_counter()
        if not scene.jelly_running:return .06
        states=all_states(scene);position_finger(scene);idle=True
        for st in states:
            dt=min(.08,max(.001,now-st['last']));st['step_dt']=dt;st['last']=now
            external=gravity_force(scene)+st.get('pressure_force',np.zeros(3))
            current=st['anchor']==anchor(scene);shaking=(scene.jelly_continuous_shake and current) or now<st['shake_until']
            if shaking:
                st['shake_phase']+=dt;env=1 if scene.jelly_continuous_shake and current else min(1,max(0,(st['shake_until']-now)/.4))
                external+=local_vector(scene,(18*sin(st['shake_phase']*8.7),7*cos(st['shake_phase']*7.1),0))*scene.jelly_shake_strength*env
            integrate(st,dt,force=external)
        prepare_finger_contacts(scene,states)
        for st in states:
            was_dent=st['dent_active'];update_dents(st,st['step_dt']);apply_state(st,was_dent)
            st['sleeping']=np.linalg.norm(st['v'])<.0002 and np.linalg.norm(st['q']-equilibrium(scene))<.0001 and abs(st['r'])+abs(st['rv'])<.0002 and st['target'] is None and not st['dent_active']
            idle=idle and st['sleeping']
        resolve_solid_finger(scene,states)
        return .05 if idle and _MODAL is None else 1/60
    except (ReferenceError,RuntimeError):_STATE=None;return None
''' +s[b:]
s=s.replace("for a in anchors(self):a.rotation_euler=(0,math.radians(self.jelly_tilt),0)","for a in anchors(self):a.rotation_euler=(math.radians(self.jelly_tilt),0,0)")
s=s.replace("    if _MODAL:_MODAL.dragging=False\n", "    if _MODAL:_MODAL.dragging=False\n    fit_front_camera(self)\n",1)
pos=s.index('def orientation_changed(')
s=s[:pos]+'''def fit_front_camera(scene):
    roots=anchors(scene);cam=scene.camera
    if not roots or not cam:return
    mid=sum(a.location.x for a in roots)/len(roots)
    cam.location=(mid,-9,2.10);cam.rotation_euler=(Vector((mid,0,2.10))-cam.location).to_track_quat('-Z','Y').to_euler()
    cam.data.ortho_scale=4.3+2.3*(len(roots)-1)

'''+s[pos:]
s=s.replace("            self.finger_depth+=shift;self.ball.location+=self.normal*shift", "            self.finger_depth+=shift;move_finger(context.scene,finger_context(context.scene)['desired']+self.normal*shift)")
s=s.replace("                    self.ball.location=self.finger_start+(pos-self.grab_world)+self.normal*self.finger_depth", "                    move_finger(context.scene,self.finger_start+(pos-self.grab_world)+self.normal*self.finger_depth)")
s=s.replace("            centre=Vector((1.5*(n-1),0,2.05));direction=Vector((4.3,-7,2.45));cam.location=centre+direction;cam.rotation_euler=(centre-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=4.3+2.3*(n-1)","            fit_front_camera(scene)")
s=s.replace("互动中拖球，按住时滚轮推入/拉出","拖球或滚轮施压；碰撞阻止穿透")
p.write_text(s);compile(s,str(p),'exec')
print('solid sphere upgraded')
