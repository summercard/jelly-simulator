import bpy,sys,json
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kawaii_jelly_interaction as j
import jelly_surface_solver as solver
j.register();bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_双盘双向挤压_v009.blend'))
s=bpy.context.scene;s.jelly_gravity=False
for obj in s.objects:
 if obj.get('jelly_squeeze_side') is not None:obj['jelly_collider']=False
roots=j.anchors(s);states=[j.new_state(s,a) for a in roots];root=roots[0]
plate=next(o for o in s.objects if o.get('jelly_plate_root')==root.name)
solver.step(s,states,1/60,j)
body,rest=next((o,r) for o,r in states[0]['entries'] if o.get('jelly_body'))
def world_mesh():
 values=np.empty(len(rest)*3,dtype=np.float32);body.data.vertices.foreach_get('co',values)
 return solver.transform(values.reshape(-1,3),root.matrix_world)
before=world_mesh();location=plate.location.copy();shift=Vector((-.25,0,0))
plate.location=location+shift;bpy.context.view_layer.update();j.sync_plates(s)
for _ in range(3):j.sync_plates(s)
solver.step(s,states,1/60,j)
error=float(np.max(np.abs(world_mesh()-before-np.array(shift))))
assert error<1e-5,error
assert not plate.get('jelly_collider') and plate.parent is None
# A locked cut cannot be forced through another locked cut, while rims overlap.
plate.location.x=.5;bpy.context.view_layer.update();j.sync_plates(s)
assert (roots[1].location-root.location).length>=2.001
report={'plate_carries_whole_mesh_error':error,'plate_is_not_collider':True,'cap_overlap_guard':True,'hardness':[]}
# Run the same box motion at two hardness values and compare residual response.
for obj in s.objects:
 if obj.get('jelly_squeeze_side') is not None:obj['jelly_collider']=True
for hardness in [16,75]:
 s.jelly_k=hardness;states=[j.new_state(s,a) for a in roots]
 solver._COLLIDER_HISTORY.clear()
 for frame in range(30):
  s.jelly_squeeze=.62*min(1,frame/20);bpy.context.view_layer.update();solver.step(s,states,1/60,j)
 d=states[0]['surface_solver'];extent=float(np.max(np.linalg.norm(d['x']-d['base'],axis=1)))
 s.jelly_squeeze=0;bpy.context.view_layer.update()
 for _ in range(15):solver.step(s,states,1/60,j)
 residual=states[0]['max_dent']
 report['hardness'].append({'hardness':hardness,'pressed':extent,'residual_after_quarter_second':residual})
assert abs(report['hardness'][0]['residual_after_quarter_second']-report['hardness'][1]['residual_after_quarter_second'])>.001
report['passed']=True
(ROOT/'v009_盘子移动与手感验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('PLATE_PASS',report)
