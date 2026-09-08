import bpy,sys,json,time,math
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kawaii_jelly_interaction as j
import jelly_surface_solver as solver
j.register();bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_重力挤压复制_v003.blend'))
s=bpy.context.scene;j._UPDATING=True;s.jelly_gravity=False;s.jelly_k=30;s.jelly_solver_iterations=3;s.jelly_surface_mode=True
for a in j.anchors(s):a.rotation_euler=(0,0,0)
s.jelly_active=1;bpy.context.view_layer.update();ball=j.finger(s);ball.hide_viewport=True
# Use one low-poly closed sphere as a deformable specimen, translated away from originals.
bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=16,radius=1,location=(8,0,0))
obj=bpy.context.object;obj.name='测试软球';bpy.ops.jelly.soft_mesh();root=obj.parent
st=j.new_state(s,root)
# A rounded rectangular object must produce a rectangular contact, not a spherical dent.
bpy.ops.mesh.primitive_cube_add(size=1,location=(8,0,1.0));tool=bpy.context.object;tool.scale=(.65,.5,.65);tool['jelly_collider']=True
bpy.context.view_layer.update();report={};start=time.perf_counter()
for _ in range(25):solver.step(s,[st],1/60,j)
d=st['surface_solver'];collider=solver.Collider(tool,bpy.context.evaluated_depsgraph_get());contacts=collider.contacts(d['x'],margin=-.001)
report['box']={'max_deformation':st['max_dent'],'penetrating_vertices':len(contacts),'mean_step_ms':(time.perf_counter()-start)*1000/25}
assert st['max_dent']>.1 and not contacts,report
# Shape keys change the evaluated collision surface without replacing the object.
tool.shape_key_add(name='Basis');key=tool.shape_key_add(name='侧向鼓起')
for v in key.data:
    if v.co.x>0:v.co.z-=.45
before=collider.vertices.copy();key.value=1;bpy.context.view_layer.update()
after=solver.Collider(tool,bpy.context.evaluated_depsgraph_get()).vertices
assert np.max(np.abs(after-before))>.1
for _ in range(10):solver.step(s,[st],1/60,j)
assert not solver.Collider(tool,bpy.context.evaluated_depsgraph_get()).contacts(d['x'],margin=-.003)
report['shape_key_surface_updates']=True
# Armature evaluation: every collision rebuild must see the posed geometry.
bpy.ops.object.armature_add(location=tool.location)
rig=bpy.context.object;rig.name='测试碰撞骨骼';bone=rig.data.bones[0].name
vg=tool.vertex_groups.new(name=bone);vg.add(list(range(len(tool.data.vertices))),1,'REPLACE')
modifier=tool.modifiers.new('骨骼驱动碰撞','ARMATURE');modifier.object=rig
bpy.context.view_layer.update();before=solver.Collider(tool,bpy.context.evaluated_depsgraph_get()).vertices
rig.pose.bones[bone].rotation_mode='XYZ';rig.pose.bones[bone].rotation_euler.y=.35
bpy.context.view_layer.update();after=solver.Collider(tool,bpy.context.evaluated_depsgraph_get()).vertices
assert np.max(np.abs(after-before))>.05
for _ in range(10):solver.step(s,[st],1/60,j)
assert not solver.Collider(tool,bpy.context.evaluated_depsgraph_get()).contacts(d['x'],margin=-.003)
report['armature_contact']=True
# Remove tool and observe elastic recovery.
tool['jelly_collider']=False
for _ in range(150):solver.step(s,[st],1/60,j)
report['recovery']=st['max_dent'];assert st['max_dent']<.04,report
# Two freely deformable bodies should both respond.
bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=16,radius=.7,location=(8,0,1.35))
other=bpy.context.object
for v in other.data.vertices:
    v.co.x*=1.15+.12*math.sin(v.co.z*5);v.co.y*=.8
bpy.ops.jelly.soft_mesh();st2=j.new_state(s,other.parent)
for _ in range(30):solver.step(s,[st,st2],1/60,j)
report['soft_soft']={'first':st['max_dent'],'second':st2['max_dent']}
assert st['max_dent']>.03 and st2['max_dent']>.03
assert np.isfinite(st['surface_solver']['x']).all() and np.isfinite(st2['surface_solver']['x']).all()
# Panel must draw without invalid RNA property calls.
class Layout:
    def row(self,**kw):return self
    def box(self,**kw):return self
    def column(self,**kw):return self
    def prop(self,obj,name,**kw):assert hasattr(obj,name),name
    def label(self,**kw):pass
    def operator(self,*args,**kw):return type('Op',(),{})()
from types import SimpleNamespace
j.JELLY_PT_panel.draw(SimpleNamespace(layout=Layout()),bpy.context)
report['panel_draw']=True
report['passed']=True
(ROOT/'通用软体_网格骨骼双向交互测试.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('GENERIC_PASS',report)
