import bpy,sys,time,json,importlib.util,re
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import kawaii_jelly_interaction as j
import jelly_surface_solver as new
spec=importlib.util.spec_from_file_location('old',ROOT/'backups/jelly_surface_solver_v042.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
j.register()
icons=bpy.types.UILayout.bl_rna.functions['operator'].parameters['icon'].enum_items.keys()
for icon in re.findall(r"icon=['\"]([^'\"]+)", (ROOT/'kawaii_jelly_interaction.py').read_text()):assert icon in icons,icon
results={};outputs=[]
for name,solver in [('before',old),('after',new)]:
 bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_双盘双向挤压_v009.blend'))
 s=bpy.context.scene;states=[j.new_state(s,a) for a in j.anchors(s)];times=[];frames=[]
 for frame in range(65):
  s.jelly_squeeze=.65*min(1,frame/40);bpy.context.view_layer.update()
  start=time.perf_counter();solver.step(s,states,1/60,j);elapsed=time.perf_counter()-start
  if frame>0:times.append(elapsed*1000)
  frames.append(np.concatenate([np.array([v.co[:] for v in st['surface_solver']['body'].data.vertices]).ravel() for st in states]))
 results[name]={'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times))}
 outputs.append(np.array(frames))
results['max_vertex_coordinate_difference']=float(np.max(np.abs(outputs[0]-outputs[1])))
assert results['max_vertex_coordinate_difference']<1e-6,results
(ROOT/'v010_性能与效果验证.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(results,flush=True)
# Save the uncompressed starting scene with updated embedded sources.
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_双盘双向挤压_v009.blend'))
for filename in ['kawaii_jelly_interaction.py','jelly_surface_solver.py']:
 text=bpy.data.texts.get(filename) or bpy.data.texts.new(filename);text.clear();text.write((ROOT/filename).read_text())
bpy.context.scene['版本说明']='v010：修复侧栏图标；保持求解效果的重复计算缓存优化'
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'Q弹果冻_控件修复与优化_v010.blend'))
