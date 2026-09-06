import bpy
from mathutils import Vector
import kawaii_jelly_interaction as j
OUT='/Users/summercards/ShellStorm2/output/kawaii_jelly_v003'
s=bpy.context.scene;s.name='Q弹果冻_重力_球体挤压_复制';j._UPDATING=True
for i,a in enumerate(j.anchors(s)):a['jelly_instance_id']=i+1
s.jelly_active=1;s.jelly_finger_auto=True;s.jelly_press=False;s.jelly_press_depth=.24
ball=j.finger(s)
if ball is None:
 bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=20,radius=.23)
 ball=bpy.context.object;ball.name='球形手指_拖动挤压';ball['jelly_finger']=True;ball['jelly_radius']=.23
 for face in ball.data.polygons:face.use_smooth=True
 m=bpy.data.materials.get('盘沿奶白')
 if m:ball.data.materials.append(m)
 col=bpy.data.collections.new('03_球形手指_交互');s.collection.children.link(col)
 for c in list(ball.users_collection):c.objects.unlink(ball)
 col.objects.link(ball)
j._UPDATING=False
if len(j.anchors(s))==1:bpy.ops.jelly.clone()
s.jelly_active=1;s.jelly_continuous_shake=False;s.jelly_running=True
j.settle(s);j.position_finger(s)
s.render.engine='BLENDER_EEVEE_NEXT'
for ar in bpy.context.screen.areas:
 if ar.type=='VIEW_3D':
  ar.spaces.active.shading.type='SOLID';ar.spaces.active.shading.color_type='MATERIAL';ar.spaces.active.overlay.show_overlays=False;ar.spaces.active.show_region_ui=True
  ar.spaces.active.region_3d.view_perspective='CAMERA';ar.spaces.active.region_3d.view_camera_zoom=6
s['球形手指操作']='侧栏按下球体，或开始抓拉后拖球，按住球时滚轮推进/拉出'
s['复制操作']='复制一个到旁边：独立网格和基座；当前编号选择按钮操作对象，直接抓取自动选择'
bpy.ops.wm.save_as_mainfile(filepath=OUT+'/Q弹果冻_重力挤压复制_v003.blend')
def start():
 if not (hasattr(bpy.types,'blendermcp_server') and bpy.types.blendermcp_server and bpy.types.blendermcp_server.running):bpy.ops.blendermcp.start_server()
 j.wake(s)
 for ar in bpy.context.screen.areas:
  if ar.type=='VIEW_3D':
   for r in ar.regions:
    if r.type=='UI':
     try:r.active_panel_category='Q弹果冻'
     except AttributeError:pass
 return None
bpy.app.timers.register(start,first_interval=1.5)
print('V003 all features ready')
