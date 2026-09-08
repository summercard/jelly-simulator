import bpy,sys,time,json
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kawaii_jelly_interaction as j
import jelly_surface_solver as solver
j.register();bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_通用软体交互_v008.blend'))
s=bpy.context.scene;j._UPDATING=True
s.jelly_active=1;s.jelly_gravity=False;s.jelly_k=30;s.jelly_solver_iterations=5;s.jelly_running=True;s.jelly_press=False;s.jelly_bulge_spread=2.5
roots=j.anchors(s)
for i,a in enumerate(roots):a.location=(2.002*i,0,2.2)
j.setup_plates(s);j.sync_plates(s)
ball=j.finger(s);ball.location=(6,0,3);j._FINGER_CONTEXT.clear();j.finger_context(s);j.sync_handle(s,ball.location);s.jelly_finger_auto=False
for obj in s.objects:
 if obj.get('jelly_tool_kind'):
  for child in [obj]+list(obj.children):child.hide_viewport=True;child.hide_render=True;child['jelly_collider']=False
  obj['jelly_tool_active']=False
for side,x in [(-1,-1.3),(1,3.302)]:
 bpy.ops.mesh.primitive_cube_add(size=1,location=(x,-.75,2.1));obj=bpy.context.object
 obj.name='左侧挤压BOX' if side<0 else '右侧挤压BOX';obj.scale=(.32,1.1,2.6)
 obj['jelly_collider']=True;obj['jelly_squeeze_side']=side;obj['jelly_squeeze_home']=x
 bevel=obj.modifiers.new('接触圆角','BEVEL');bevel.width=.06;bevel.segments=3
 mat=bpy.data.materials.new(obj.name+'_材质');mat.diffuse_color=(.12,.35,.52,1);obj.data.materials.append(mat)
bpy.context.view_layer.update()
for st in j.all_states(s):st['q'][:]=0;st['dent'].clear();j.apply_state(st,True)
s.camera.location=(3.8,-8,4.1);target=Vector((0.875,-.2,2.1));s.camera.rotation_euler=(target-s.camera.location).to_track_quat('-Z','Y').to_euler();s.camera.data.ortho_scale=6.1
j.fit_front_camera(s)
for filename in ['kawaii_jelly_interaction.py','jelly_surface_solver.py']:
 text=bpy.data.texts.get(filename) or bpy.data.texts.new(filename);text.clear();text.write((ROOT/filename).read_text())
s['版本说明']='v009：移动托盘带动果冻；托盘不碰撞；双向软体接触和两侧BOX挤压'
j._UPDATING=False;bpy.ops.jelly.select_plate()
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'Q弹果冻_双盘双向挤压_v009.blend'))
# Staged inward motion is the real controller path.
states=[j.new_state(s,a) for a in roots]
start=time.perf_counter()
for frame in range(65):
 s.jelly_squeeze=.65*min(1,frame/40)
 bpy.context.view_layer.update();solver.step(s,states,1/60,j)
report={'step_ms':(time.perf_counter()-start)*1000/65,'bodies':[]}
for st in states:
 d=st['surface_solver'];report['bodies'].append({'dent':st['max_dent'],'contacts':d['contact_count'],'soft_contacts':d['soft_contact_count']})
# Signed closest-surface penetration of the actual render meshes, not just cage.
meshes=[]
for st in states:
 body,rest=next((o,r) for o,r in st['entries'] if o.get('jelly_body'))
 co=np.empty(len(rest)*3,dtype=np.float32);body.data.vertices.foreach_get('co',co);co=solver.transform(co.reshape(-1,3),st['anchor'].matrix_world)
 body.data.calc_loop_triangles();tri=[tuple(t.vertices) for t in body.data.loop_triangles]
 meshes.append((co,tri,rest))
from mathutils.bvhtree import BVHTree
penetration=[]
for i in range(2):
 co,_,rest=meshes[i];other,tri,_=meshes[1-i];tree=BVHTree.FromPolygons(other.tolist(),tri,all_triangles=True)
 worst=0;count=0
 for point in co:
  near,n,_,distance=tree.find_nearest(Vector(point));signed=(Vector(point)-near).dot(n)
  if signed<-.003:worst=max(worst,-signed);count+=1
 penetration.append({'max':worst,'count':count})
report['render_penetration']=penetration
assert all(item['count']==0 for item in penetration),report
assert all(item['soft_contacts']>0 for item in report['bodies']),report
(ROOT/'v009_两侧挤压验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('SQUEEZE_REPORT',report,flush=True)
s.render.engine='BLENDER_WORKBENCH';s.render.resolution_x=1100;s.render.resolution_y=750;s.render.resolution_percentage=100
s.display.shading.light='STUDIO';s.display.shading.color_type='MATERIAL';s.display.shading.show_shadows=False;s.display.shading.show_cavity=True
s.camera.location=(3.8,-8,4.1);target=Vector((0.875,-.2,2.1));s.camera.rotation_euler=(target-s.camera.location).to_track_quat('-Z','Y').to_euler();s.camera.data.ortho_scale=5.5
preview=ROOT/'previews_v009';preview.mkdir(exist_ok=True);s.render.filepath=str(preview/'双BOX挤压双果冻.png');bpy.ops.render.render(write_still=True)
