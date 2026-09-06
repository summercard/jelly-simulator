from pathlib import Path
p=Path('/Users/summercards/ShellStorm2/output/kawaii_jelly_v003/kawaii_jelly_interaction.py')
s=p.read_text().replace("'version':(2,0,0)","'version':(3,0,0)")
s=s.replace('from bpy.props import FloatProperty,BoolProperty,EnumProperty','from bpy.props import FloatProperty,BoolProperty,EnumProperty,IntProperty')
s=s.replace('_STATE=None\n_MODAL=None','_STATE=None\n_STATES={}\n_MODAL=None',1)
a=s.index('def anchor(scene):');b=s.index('\ndef rotation(scene):',a)
s=s[:a]+'''def anchors(scene):
    return sorted([o for o in scene.objects if o.get('jelly_anchor')],key=lambda o:o.get('jelly_instance_id',1))

def anchor(scene):
    items=anchors(scene)
    return next((o for o in items if o.get('jelly_instance_id',1)==scene.jelly_active),items[0] if items else None)

def finger(scene):
    return next((o for o in scene.objects if o.get('jelly_finger')),None)
''' +s[b:]
s=s.replace('def new_state(scene):\n    entries=[]','def new_state(scene,root=None):\n    root=root or anchor(scene)\n    entries=[]')
s=s.replace("        if obj.type!='MESH' or not obj.get('jelly_deform'):continue","        if obj.type!='MESH' or not obj.get('jelly_deform') or (root and obj.parent!=root):continue",1)
s=s.replace("return {'scene':scene,'entries':entries,","return {'scene':scene,'anchor':root,'dent':{},'dent_active':False,'max_dent':0.0,'entries':entries,",1)
a=s.index('def state(scene=None):');b=s.index('\ndef gravity_force',a)
s=s[:a]+'''def state(scene=None,root=None):
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
''' +s[b:]
s=s.replace("    if not force and s['last_draw'] and", "    if not force and not s['dent_active'] and s['last_draw'] and")
s=s.replace("        try:obj.data.vertices.foreach_set('co',deform_positions(rest,s['q'],s['r']).ravel());obj.data.update()", "        try:\n            co=deform_positions(rest,s['q'],s['r']);dent=s['dent'].get(obj.as_pointer())\n            if dent is not None:co+=dent[0]\n            obj.data.vertices.foreach_set('co',co.ravel());obj.data.update()")
a=s.index('def tick():');b=s.index('\ndef wake(',a)
s=s[:a]+'''def position_finger(scene):
    ball=finger(scene);root=anchor(scene)
    if not ball or not root or not scene.jelly_finger_auto:return
    st=state(scene,root)
    base=np.array([[-.12,-.67,.732]],dtype=np.float32)
    contact=deform_positions(base,st['q'],st['r'])[0]
    normal=Vector((-.12,-.67,.732)).normalized()
    radius=float(ball.get('jelly_radius',.23))
    depth=scene.jelly_press_depth if scene.jelly_press else -.10
    ball.location=root.matrix_world@(Vector(contact)+normal*(radius-depth))

def update_dents(st,dt):
    scene=st['scene'];ball=finger(scene);root=st['anchor']
    if not root:return
    centre=np.array(root.matrix_world.inverted()@ball.location) if ball else np.array((100,100,100))
    radius=ball.get('jelly_radius',.23)*max(ball.scale) if ball else .23
    active=False;peak=0.0
    for obj,rest in st['entries']:
        key=obj.as_pointer()
        if key not in st['dent']:st['dent'][key]=[np.zeros_like(rest),np.zeros_like(rest)]
        offset,vel=st['dent'][key]
        co=deform_positions(rest,st['q'],st['r']);diff=co-centre;distance=np.linalg.norm(diff,axis=1)
        penetration=np.maximum(radius-distance,0)
        target=np.zeros_like(rest)
        if np.max(penetration)>.00001:
            direct=diff/np.maximum(distance[:,None],.00001)
            target=direct*penetration[:,None]
            # Soft tissue around the contact follows a broad depression; hard jelly localises it.
            nearest=int(np.argmin(distance));normal=direct[nearest];depth=float(penetration[nearest])
            spread=radius*(.62+22/scene.jelly_k)
            lateral=np.linalg.norm(co-co[nearest],axis=1)
            band=np.maximum(distance-radius,0)
            surround=np.exp(-lateral*lateral/(2*spread*spread))*(1-np.exp(-band*band/(radius*radius*.16)))
            target+=normal[None,:]*(depth*.50*surround[:,None])
        t=np.clip(rest[:,2]/.18,0,1);pin_weight=t*t*(3-2*t)
        target*=pin_weight[:,None]
        stiffness=65+scene.jelly_k*3.0;damping=math.sqrt(stiffness)*1.2
        vel[:]=(vel+dt*stiffness*(target-offset))/(1+dt*damping+dt*dt*stiffness)
        offset+=dt*vel
        # Actual sphere contact cannot be ignored on a fast frame.
        if np.max(penetration)>.00001:offset[:]=.55*offset+.45*target
        pins=rest[:,2]<1e-7;offset[pins]=0;vel[pins]=0
        magnitude=float(np.max(np.linalg.norm(offset,axis=1)));peak=max(peak,magnitude)
        if magnitude>.00002 or np.max(np.abs(vel))>.00003:active=True
        else:offset[:]=0;vel[:]=0
    st['dent_active']=active;st['max_dent']=peak

def tick():
    global _STATE
    if not _STATE or not _STATE['alive']:return None
    try:
        scene=_STATE['scene'];now=time.perf_counter();position_finger(scene)
        idle=True
        for st in all_states(scene):
            dt=min(.10,max(.001,now-st['last']));st['last']=now
            if not scene.jelly_running:continue
            external=gravity_force(scene)
            current=st['anchor']==anchor(scene)
            shaking=(scene.jelly_continuous_shake and current) or now<st['shake_until']
            if shaking:
                st['shake_phase']+=dt;env=1 if scene.jelly_continuous_shake and current else min(1,max(0,(st['shake_until']-now)/.4))
                wave=(18*sin(st['shake_phase']*8.7),7*cos(st['shake_phase']*7.1),0)
                external+=local_vector(scene,wave)*scene.jelly_shake_strength*env
            integrate(st,dt,force=external);was_dent=st['dent_active'];update_dents(st,dt);apply_state(st,was_dent)
            st['sleeping']=np.linalg.norm(st['v'])<.0002 and np.linalg.norm(st['q']-equilibrium(scene))<.0001 and abs(st['r'])+abs(st['rv'])<.0002 and st['target'] is None and not shaking and not st['dent_active']
            idle=idle and st['sleeping']
        return .04 if idle and _MODAL is None else 1/60
    except (ReferenceError,RuntimeError):_STATE=None;return None
''' +s[b:]
s=s.replace("    a=anchor(self)\n    if a:\n        a.rotation_euler=(0,math.radians(self.jelly_tilt),0)\n        if context and context.view_layer:context.view_layer.update()", "    for a in anchors(self):a.rotation_euler=(0,math.radians(self.jelly_tilt),0)\n    if context and context.view_layer:context.view_layer.update()")
s=s.replace("self.dragging=False;self.closed=False;self.s=wake(context.scene)","self.dragging=False;self.finger_drag=False;self.closed=False;self.s=wake(context.scene)")
s=s.replace("            if not hit or not obj.get('jelly_deform'):return {'PASS_THROUGH'}",'''            if hit and obj.get('jelly_finger'):
                self.dragging=True;self.finger_drag=True;self.ball=obj.original if hasattr(obj,'original') else obj
                context.scene.jelly_finger_auto=False;self.grab_world=loc.copy();self.normal=self.rv3d.view_rotation@Vector((0,0,1));self.finger_start=self.ball.location.copy();self.finger_depth=0.0
                return {'RUNNING_MODAL'}
            if not hit or not obj.get('jelly_deform'):return {'PASS_THROUGH'}
            self.finger_drag=False
            parent=obj.parent
            if parent and parent.get('jelly_anchor'):
                parent=parent.original if hasattr(parent,'original') else parent
                context.scene.jelly_active=parent.get('jelly_instance_id',1);self.s=state(context.scene,parent)''')
s=s.replace("        if event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'} and self.dragging:\n",'''        if self.dragging and self.finger_drag and event.type in {'WHEELUPMOUSE','WHEELDOWNMOUSE'}:
            shift=-.045 if event.type=='WHEELUPMOUSE' else .045
            self.finger_depth+=shift;self.ball.location+=self.normal*shift
            return {'RUNNING_MODAL'}
        if event.type in {'MOUSEMOVE','INBETWEEN_MOUSEMOVE'} and self.dragging:
''')
s=s.replace("                delta=local_vector(context.scene,pos-self.grab_world)/self.weight",'''                if self.finger_drag:
                    self.ball.location=self.finger_start+(pos-self.grab_world)+self.normal*self.finger_depth
                    return {'RUNNING_MODAL'}
                delta=local_vector(context.scene,pos-self.grab_world)/self.weight''')
s=s.replace("        if event.type=='LEFTMOUSE' and event.value=='RELEASE' and self.dragging:\n",'''        if event.type=='LEFTMOUSE' and event.value=='RELEASE' and self.dragging:
            if self.finger_drag:
                self.dragging=False;self.finger_drag=False
                return {'RUNNING_MODAL'}
''')
# New controls share gravity/stiffness, while gesture and dent state are per-copy.
insert=s.index('class JELLY_PT_panel')
s=s[:insert]+'''class JELLY_OT_finger_home(bpy.types.Operator):
    bl_idname='jelly.finger_home';bl_label='球形手指放到当前果冻前'
    def execute(self,context):context.scene.jelly_finger_auto=True;context.scene.jelly_press=False;position_finger(context.scene);wake(context.scene);return {'FINISHED'}
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
        scene.jelly_active=n;bpy.context.view_layer.update();settle(scene);wake(scene);position_finger(scene)
        cam=scene.camera
        if cam:
            centre=Vector((1.5*(n-1),0,2.05));direction=Vector((4.3,-7,2.45));cam.location=centre+direction;cam.rotation_euler=(centre-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=4.3+2.3*(n-1)
        return {'FINISHED'}

''' +s[insert:]
s=s.replace("        l=self.layout;s=context.scene", "        l=self.layout;s=context.scene\n        l.prop(s,'jelly_active');l.operator('jelly.clone',icon='DUPLICATE')")
s=s.replace("        box=l.box();box.label(text='3. 摇晃与回弹');", "        box=l.box();box.label(text='3. 球形手指挤压');box.operator('jelly.finger_home');box.prop(s,'jelly_press',toggle=True);box.prop(s,'jelly_press_depth',slider=True);box.label(text='互动中拖球，按住时滚轮推入/拉出')\n        box=l.box();box.label(text='4. 摇晃与回弹');")
s=s.replace("        l.label(text='详细操作：同目录《使用说明》')", "        l.label(text='参数作用于全部；抓拉与冲量分别响应')\n        l.label(text='详细操作：同目录《使用说明》')")
s=s.replace("    global _STATE,_MODAL\n    if _STATE:_STATE['alive']=False", "    global _STATE,_MODAL\n    for st in _STATES.values():st['alive']=False\n    _STATES.clear()\n    if _STATE:_STATE['alive']=False")
s=s.replace("if any(o.get('jelly_body') for o in bpy.context.scene.objects):settle(bpy.context.scene);wake(bpy.context.scene)", "if any(o.get('jelly_body') for o in bpy.context.scene.objects):\n        scene=bpy.context.scene\n        for st in all_states(scene):st['q'][:]=equilibrium(scene);apply_state(st,True)\n        wake(scene)")
s=s.replace("JELLY_OT_preset,JELLY_PT_panel)","JELLY_OT_preset,JELLY_OT_finger_home,JELLY_OT_clone,JELLY_PT_panel)")
s=s.replace("'jelly_running')","'jelly_running','jelly_active','jelly_finger_auto','jelly_press','jelly_press_depth')",1)
s=s.replace("    for cls in _CLASSES:bpy.utils.register_class(cls)\n", "    for cls in _CLASSES:bpy.utils.register_class(cls)\n    bpy.types.Scene.jelly_active=IntProperty(name='当前果冻编号',default=1,min=1,max=6,update=parameter_changed)\n    bpy.types.Scene.jelly_finger_auto=BoolProperty(name='手指跟随当前果冻',default=True)\n    bpy.types.Scene.jelly_press=BoolProperty(name='按下球体 / 松开',default=False,update=parameter_changed)\n    bpy.types.Scene.jelly_press_depth=FloatProperty(name='挤压深度',default=.24,min=0,max=.42,update=parameter_changed)\n")
p.write_text(s)
compile(s,str(p),'exec')
print('controller expanded',len(s.splitlines()),'lines')
