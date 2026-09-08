import bpy,sys,time,json
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kawaii_jelly_interaction as j
import jelly_surface_solver as solver
j.register();bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_通用软体交互_v007.blend'))
s=bpy.context.scene;j._UPDATING=True;s.jelly_active=1;s.jelly_gravity=False;s.jelly_k=30;s.jelly_compression=1.25;s.jelly_surface_mode=True
states=[j.new_state(s,a) for a in j.anchors(s)];st=states[0];root=st['anchor'];ball=j.finger(s)
body,rest=next((o,r) for o,r in st['entries'] if o.get('jelly_body'));count=len(body.data.vertices)
report={'cases':[],'render_vertices':count}
for normal,depth in [((0,0,1),.3),((.98,0,.2),.55),((.6,0,.8),.6)]:
 st.pop('surface_solver',None);st['dent'].clear()
 normal=Vector(normal).normalized();outside=root.matrix_world@(normal*1.8)
 ball.location=outside;j._FINGER_CONTEXT.clear();ctx=j.finger_context(s)
 ctx['desired']=root.matrix_world@(normal*(1+s.jelly_ball_radius-depth))
 j.prepare_finger_contacts(s,states);ball.location=ctx['candidate'];ctx['last']=ball.location.copy();bpy.context.view_layer.update()
 for _ in range(35):solver.step(s,states,1/60,j)
 values=np.empty(count*3,dtype=np.float32);body.data.vertices.foreach_get('co',values);co=values.reshape(-1,3)
 err=float(np.max(np.abs(co[rest[:,2]<1e-5]-rest[rest[:,2]<1e-5])))
 assert err==0 and len(body.data.vertices)==count and np.isfinite(co).all()
 report['cases'].append({'normal':list(normal),'cut_error':err,'max_displacement':float(np.max(np.linalg.norm(co-rest,axis=1)))})
# Real UI callback and kinematic depth cap, with positions compared in local space.
j._UPDATING=False;s.jelly_press_depth=.1
assert s.jelly_press and s.jelly_finger_auto
j.position_finger(s);j.prepare_finger_contacts(s,states);shallow=j.finger_context(s)['candidate'].copy()
s.jelly_press_depth=.6;j.position_finger(s);j.prepare_finger_contacts(s,states);deep=j.finger_context(s)['candidate'].copy()
assert (deep-shallow).length>.1
s.jelly_compression=.2;j.position_finger(s);j.prepare_finger_contacts(s,states);hardlimit=j.finger_context(s)['candidate'].copy()
s.jelly_compression=1.5;j.position_finger(s);j.prepare_finger_contacts(s,states);softlimit=j.finger_context(s)['candidate'].copy()
assert (hardlimit-softlimit).length>.1
report['depth_position_difference']=(deep-shallow).length;report['compression_position_difference']=(hardlimit-softlimit).length
# Repeated replacement reuses meshes and disables previous shape collision.
bpy.ops.jelly.demo_shape(shape='HAND');hand=j.active_tool(s);handname=hand.name
bpy.ops.jelly.demo_shape(shape='BOX');assert not hand.get('jelly_tool_active') and hand.hide_viewport
bpy.ops.jelly.demo_shape(shape='HAND');assert j.active_tool(s).name==handname
s.jelly_hand_curl=40;assert abs(hand.pose.bones['手指0'].rotation_euler.x)>.6
bpy.ops.jelly.demo_shape(shape='SPHERE');assert j.active_tool(s) is None and not ball.hide_viewport
report['replacement_reuses_hand']=True;report['passed']=True
(ROOT/'v008_切面深度替换验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('V008_PASS',report)
