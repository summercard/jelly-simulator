import bpy,sys,time,json
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kawaii_jelly_interaction as j
import jelly_surface_solver as solver
j.register();bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_重力挤压复制_v003.blend'))
s=bpy.context.scene;j._UPDATING=True;s.jelly_active=1
s.jelly_surface_mode=True;s.jelly_solver_iterations=3;s.jelly_ball_radius=.38;s.jelly_bulge=1;s.jelly_bulge_spread=1.1
s.jelly_press_depth=.3;s.jelly_running=True;s.jelly_continuous_shake=False
j.finger_size_changed(s,bpy.context)
for st in j.all_states(s):st['q'][:]=j.equilibrium(s);j.apply_state(st,True)
j.position_finger(s);ball=j.finger(s);ball.location=j.finger_context(s)['desired'];j.finger_context(s)['last']=ball.location.copy()
bpy.ops.jelly.control_ball()
for screen in bpy.data.screens:
 for area in screen.areas:
  if area.type=='VIEW_3D':
   area.spaces.active.show_gizmo=True;area.spaces.active.show_gizmo_object_translate=True;area.spaces.active.overlay.show_overlays=True;area.spaces.active.show_region_ui=True
   with bpy.context.temp_override(screen=screen,area=area):bpy.ops.wm.tool_set_by_id(name='builtin.move')
for filename in ['kawaii_jelly_interaction.py','jelly_surface_solver.py']:
 text=bpy.data.texts.get(filename) or bpy.data.texts.new(filename);text.clear();text.write((ROOT/filename).read_text())
s['版本说明']='v008：切面显示网格硬锁定、接触位移平滑、压入限制与深度恢复、一键替换骨骼手'
# Preload replaceable assets; switch back to the sphere for the initial scene.
bpy.ops.jelly.demo_shape(shape='HAND');bpy.ops.jelly.demo_shape(shape='BOX');bpy.ops.jelly.demo_shape(shape='SPHERE')
bpy.ops.jelly.control_ball()
j._UPDATING=False
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'Q弹果冻_通用软体交互_v008.blend'))
# QA: actual rounded block pressing the original jelly, no substitute specimen.
j._UPDATING=True;s.jelly_gravity=False;ball.hide_viewport=True;ball.hide_render=True
states=[j.new_state(s,a) for a in j.anchors(s)]
bpy.ops.jelly.demo_shape(shape='BOX');tool=bpy.context.object
root=j.anchor(s);tool.location=root.matrix_world@Vector((.1,0,.95))
ball.hide_viewport=True;ball.hide_render=True
bpy.context.view_layer.update();start=time.perf_counter()
for _ in range(30):solver.step(s,states,1/60,j)
report={'rounded_box_step_ms':(time.perf_counter()-start)*1000/30}
d=states[0]['surface_solver'];report['fixed_cap_max_error']=float(np.max(np.abs(d['x'][d['pins']]-d['base'][d['pins']])))
assert report['fixed_cap_max_error']<1e-6
report['solver_vertices']=len(d['x']);report['render_vertices']=len(d['body'].data.vertices)
print('REAL_SCENE_STEP_MS',report['rounded_box_step_ms'],flush=True)
for obj in s.objects:
 if obj.parent==j.anchors(s)[1]:obj.hide_render=True
s.render.engine='BLENDER_WORKBENCH';s.render.resolution_x=900;s.render.resolution_y=800;s.render.resolution_percentage=100
s.display.shading.light='STUDIO';s.display.shading.color_type='MATERIAL';s.display.shading.show_shadows=False;s.display.shading.show_cavity=True
s.display.shading.cavity_type='BOTH'
s.camera.location=(4,-5,3.4);target=Vector((0,-.3,2.1));s.camera.rotation_euler=(target-s.camera.location).to_track_quat('-Z','Y').to_euler();s.camera.data.type='ORTHO';s.camera.data.ortho_scale=3
preview=ROOT/'previews_v008';preview.mkdir(exist_ok=True)
s.render.filepath=str(preview/'圆角方块_实际果冻接触.png');bpy.ops.render.render(write_still=True)
# Bone-driven demo, saved separately so the main scene starts clean.
tool['jelly_collider']=False;tool.hide_render=True
bpy.ops.jelly.demo_shape(shape='HAND');rig=bpy.context.object
rig.location=root.matrix_world@Vector((0,-.15,1.02));s.jelly_hand_curl=30
j.hand_curl_changed(s,bpy.context);bpy.context.view_layer.update()
start=time.perf_counter()
for _ in range(25):solver.step(s,states,1/60,j)
report['hand_step_ms']=(time.perf_counter()-start)*1000/25
(ROOT/'通用软体_实际场景验证.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
s.render.filepath=str(preview/'骨骼手掌_实际果冻接触.png');bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'Q弹果冻_骨骼手掌演示_v008.blend'))
print('GENERIC_BUILD_PASS')
