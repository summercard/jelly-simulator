"""Build the updated portable scene and render contact QA from the real mesh."""
import bpy,sys,math
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import kawaii_jelly_interaction as j
j.register()
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_重力挤压复制_v003.blend'))
# Increase contact sampling; interpolate the rest attribute with the mesh.
for obj in list(bpy.context.scene.objects):
    if obj.get('jelly_body'):
        bpy.context.view_layer.objects.active=obj
        mod=obj.modifiers.new('接触面细分','SUBSURF');mod.subdivision_type='SIMPLE';mod.levels=1
        bpy.ops.object.modifier_apply(modifier=mod.name)
ball=j.finger(bpy.context.scene)
materials=list(ball.data.materials)
bpy.ops.mesh.primitive_uv_sphere_add(segments=96,ring_count=64,radius=ball.get('jelly_radius',.23))
temp=bpy.context.object;ball.data=temp.data.copy();bpy.data.objects.remove(temp,do_unlink=True)
for mat in materials:ball.data.materials.append(mat)
for face in ball.data.polygons:face.use_smooth=True
j._STATES.clear();j._STATE=None
s=bpy.context.scene;j._UPDATING=True
s.jelly_active=1;s.jelly_ball_radius=.38;s.jelly_compression=1.25;s.jelly_bulge=1.25;s.jelly_bulge_spread=1.1;s.jelly_press_depth=.36
s.jelly_k=30;s.jelly_damp=2.2;s.jelly_press=False;s.jelly_finger_auto=True;s.jelly_continuous_shake=False;s.jelly_running=True
j.finger_size_changed(s,bpy.context)
for st in j.all_states(s):st['q'][:]=j.equilibrium(s);j.apply_state(st,True)
j.position_finger(s);j.resolve_solid_finger(s,j.all_states(s))
bpy.ops.jelly.control_ball()
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.show_region_ui=True;area.spaces.active.show_gizmo=True
            area.spaces.active.show_gizmo_object_translate=True
            area.spaces.active.overlay.show_overlays=True
            with bpy.context.temp_override(screen=screen,area=area):
                bpy.ops.wm.tool_set_by_id(name='builtin.move')
                for region in area.regions:
                    if region.type=='UI':
                        try:region.active_panel_category='Q弹果冻'
                        except AttributeError:pass
# Bundle the editable source as a Text datablock for portability and inspection.
text=bpy.data.texts.get('kawaii_jelly_interaction.py') or bpy.data.texts.new('kawaii_jelly_interaction.py')
text.clear();text.write((ROOT/'kawaii_jelly_interaction.py').read_text())
s['版本说明']='v006：独立 XYZ 控制柄、球体大小、压入程度、体积补偿鼓出、扩散范围'
j._UPDATING=False
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'Q弹果冻_球面贴合挤压_v006.blend'))
# QA renders use no gravity so before/after isolate the volume displacement.
j._UPDATING=True;s.jelly_gravity=False;s.jelly_finger_auto=False
states=[j.new_state(s,a) for a in j.anchors(s)];st=states[0];a=st['anchor'];ball=j.finger(s)
for obj in s.objects:
    if obj.parent==j.anchors(s)[1]:obj.hide_render=True
s.render.engine='BLENDER_WORKBENCH';s.render.resolution_x=900;s.render.resolution_y=800;s.render.resolution_percentage=100
s.display.shading.light='STUDIO';s.display.shading.studiolight_rotate_z=.4
s.display.shading.color_type='MATERIAL';s.display.shading.show_shadows=False;s.display.shading.show_cavity=True
s.display.shading.cavity_type='BOTH';s.display.shading.show_specular_highlight=True
s.display.shading.background_type='WORLD';s.world.color=(.12,.14,.18)
s.camera.location=(3,-6,4.4);target=Vector((0,0,2.15));s.camera.rotation_euler=(target-s.camera.location).to_track_quat('-Z','Y').to_euler();s.camera.data.type='ORTHO';s.camera.data.ortho_scale=3.2
normal=Vector((.35,0,.93675)).normalized();contact=normal.copy()
outside=a.matrix_world@(contact+normal*.7)
ball.location=outside;j._FINGER_CONTEXT.clear();ctx=j.finger_context(s)
for t in states:j.apply_state(t,True)
preview=ROOT/'previews_v006';preview.mkdir(exist_ok=True)
s.render.filepath=str(preview/'体积挤压_未按压.png');bpy.ops.render.render(write_still=True)
ctx['desired']=a.matrix_world@(contact+normal*(s.jelly_ball_radius-.36))
for i in range(120):
    j.prepare_finger_contacts(s,states)
    for t in states:j.update_dents(t,1/60)
    j.resolve_solid_finger(s,states)
for t in states:j.apply_state(t,True)
s.render.filepath=str(preview/'体积挤压_按压鼓出.png');bpy.ops.render.render(write_still=True)
ball.hide_render=True
s.render.filepath=str(preview/'体积挤压_凹坑剖看.png');bpy.ops.render.render(write_still=True)
ball.hide_render=False
s.camera.location=(6,-1.5,2.9);target=Vector((0,-.35,2.2));s.camera.rotation_euler=(target-s.camera.location).to_track_quat('-Z','Y').to_euler()
s.camera.data.ortho_scale=2.8
s.render.filepath=str(preview/'接触鼓唇_侧视.png');bpy.ops.render.render(write_still=True)
print('BUILD_COMPLETE')
