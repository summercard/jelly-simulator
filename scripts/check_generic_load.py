import bpy,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kawaii_jelly_interaction as j
j.register()
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_通用软体交互_v007.blend'))
s=bpy.context.scene
for name in ['kawaii_jelly_interaction.py','jelly_surface_solver.py']:
 assert bpy.data.texts[name].as_string()==(ROOT/name).read_text(),name
assert s.jelly_surface_mode
j._STATE['last']=time.perf_counter()-.016
assert j.tick() is not None
# UI-created hand responds to the actual slider callback.
bpy.ops.jelly.demo_shape(shape='HAND')
s.jelly_hand_curl=45
rig=next(o for o in s.objects if o.get('jelly_hand_demo'))
assert abs(rig.pose.bones['手指0'].rotation_euler.x)>.7
assert j.tick() is not None
print('LOAD_TIMER_HAND_PASS')
