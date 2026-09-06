import bpy,numpy as np,json
from mathutils import Vector
import kawaii_jelly_interaction as j
s=bpy.context.scene;ball=j.finger(s);oldball=ball.matrix_world.copy();settings={k:getattr(s,k) for k in ['jelly_k','jelly_active','jelly_finger_auto','jelly_press','jelly_g']};j._UPDATING=True
out='/Users/summercards/ShellStorm2/output/kawaii_jelly_v003'
report={}
try:
 s.jelly_active=1;s.jelly_k=30;s.jelly_finger_auto=False;s.jelly_g=1
 roots=j.anchors(s);one=j.new_state(s,roots[0]);two=j.new_state(s,roots[1]);one['q'][:]=j.equilibrium(s);two['q'][:]=j.equilibrium(s)
 a=roots[0];before=[np.array(r.matrix_world) for r in roots]
 local_rest=np.array([[-.12,-.67,.732]],dtype=np.float32);contact=j.deform_positions(local_rest,one['q'])[0];normal=Vector(local_rest[0]).normalized();ball.location=a.matrix_world@(Vector(contact)+normal*(.23-.26));bpy.context.view_layer.update()
 for i in range(180):j.update_dents(one,1/60);j.update_dents(two,1/60)
 body,rest=next((o,r) for o,r in one['entries'] if o.get('jelly_body'));offset=one['dent'][body.as_pointer()][0];lengths=np.linalg.norm(offset,axis=1)
 pins=rest[:,2]<1e-7
 report['pressed_max_dent']=float(np.max(lengths));report['fixed_cut_max_displacement']=float(np.max(np.abs(offset[pins])));report['affected_vertex_fraction']=float(np.mean(lengths>.005));report['other_jelly_max_dent']=two['max_dent']
 report['independent_mesh_data']=not(set(o.data.as_pointer() for o,r in one['entries']) & set(o.data.as_pointer() for o,r in two['entries']))
 report['anchor_world_error']=max(float(np.max(np.abs(np.array(r.matrix_world)-m))) for r,m in zip(roots,before))
 # stiffness changes spatial extent rather than letting a rigid spherical finger pass through the cap.
 report['hardness_contact_width']=[]
 for k in [16,75]:
  s.jelly_k=k;st=j.new_state(s,a);st['q'][:]=one['q']
  for i in range(120):j.update_dents(st,1/60)
  offsets=st['dent'][body.as_pointer()][0]
  report['hardness_contact_width'].append({'hardness':k,'affected_fraction':float(np.mean(np.linalg.norm(offsets,axis=1)>.004))})
 s.jelly_k=30;ball.location+=Vector((0,-10,0));bpy.context.view_layer.update()
 for i in range(300):j.update_dents(one,1/60)
 report['recovery_max_offset_5s']=one['max_dent']
 # A force applied to one instance must not move the independent copy.
 q2=two['q'].copy();one['v'][:]=(0,2.5,0)
 for i in range(120):j.integrate(one,1/60)
 report['copy_unchanged_under_other_impulse']=float(np.max(np.abs(two['q']-q2)))
 report['passed']=report['pressed_max_dent']>.05 and report['fixed_cut_max_displacement']==0 and report['other_jelly_max_dent']<1e-7 and report['independent_mesh_data'] and report['anchor_world_error']==0 and report['recovery_max_offset_5s']<.001 and report['copy_unchanged_under_other_impulse']==0 and report['affected_vertex_fraction']<.5 and report['hardness_contact_width'][0]['affected_fraction']>report['hardness_contact_width'][1]['affected_fraction']
finally:
 for k,v in settings.items():setattr(s,k,v)
 ball.matrix_world=oldball;bpy.context.view_layer.update();j._UPDATING=False
json.dump(report,open(out+'/球体凹陷_恢复_复制独立性测试.json','w'),ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False,indent=2))
