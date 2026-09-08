import bpy,sys,importlib.util,json,time
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kawaii_jelly_interaction as j
import jelly_surface_solver as solver
j.register();bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_通用软体交互_v007.blend'))
s=bpy.context.scene;j._UPDATING=True;s.jelly_active=1;s.jelly_gravity=False;s.jelly_k=30;s.jelly_compression=1.25;s.jelly_solver_iterations=3
spec=importlib.util.spec_from_file_location('old_solver',ROOT/'backups/jelly_surface_solver_v040.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
root=j.anchor(s);ball=j.finger(s);normal=Vector((.98,0,.2)).normalized()
report={}
for name,engine in [('v007',old),('v008',solver)]:
 states=[j.new_state(s,a) for a in j.anchors(s)];st=states[0]
 ball.location=root.matrix_world@(normal*1.8);j._FINGER_CONTEXT.clear();ctx=j.finger_context(s)
 ctx['desired']=root.matrix_world@(normal*(1+s.jelly_ball_radius-.55));j.prepare_finger_contacts(s,states)
 ball.location=ctx['candidate'];bpy.context.view_layer.update()
 for _ in range(35):engine.step(s,states,1/60,j)
 body,rest=next((o,r) for o,r in st['entries'] if o.get('jelly_body'));v=np.empty(len(rest)*3,dtype=np.float32);body.data.vertices.foreach_get('co',v);co=v.reshape(-1,3)
 e=np.array([tuple(e.vertices) for e in body.data.edges]);a,b=e[:,0],e[:,1];degree=np.bincount(e.ravel(),minlength=len(rest)).clip(1);offset=co-rest
 lap=np.zeros_like(offset)
 for axis in range(3):lap[:,axis]=(np.bincount(a,weights=offset[b,axis],minlength=len(rest))+np.bincount(b,weights=offset[a,axis],minlength=len(rest)))/degree-offset[:,axis]
 report[name]={'laplacian_rms':float(np.sqrt(np.mean(lap[rest[:,2]>.12]**2))),'cut_error':float(np.max(np.abs(offset[rest[:,2]<1e-5]))),'vertices':len(rest)}
assert report['v008']['cut_error']==0
assert report['v008']['laplacian_rms']<report['v007']['laplacian_rms'],report
report['roughness_reduction']=1-report['v008']['laplacian_rms']/report['v007']['laplacian_rms']
(ROOT/'v008_同面数侧压平滑对照.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
for o in s.objects:
 if o.parent==j.anchors(s)[1]:o.hide_render=True
s.render.engine='BLENDER_WORKBENCH';s.render.resolution_x=900;s.render.resolution_y=800;s.render.resolution_percentage=100
s.display.shading.light='STUDIO';s.display.shading.color_type='MATERIAL';s.display.shading.show_shadows=False;s.display.shading.show_cavity=True
s.camera.location=(5,-5,4.2);target=Vector((0,0,2.2));s.camera.rotation_euler=(target-s.camera.location).to_track_quat('-Z','Y').to_euler();s.camera.data.type='ORTHO';s.camera.data.ortho_scale=3.1
s.render.filepath=str(ROOT/'previews_v008/切面附近侧压.png');bpy.ops.render.render(write_still=True)
print('FAIRING_PASS',report)
