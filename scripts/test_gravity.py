import bpy,json,math,numpy as np
from mathutils import Vector
import kawaii_jelly_interaction as j
OUT='/Users/summercards/ShellStorm2/output/kawaii_jelly_v003'
s=bpy.context.scene;a=j.anchor(s)
original={k:getattr(s,k) for k in ['jelly_k','jelly_damp','jelly_g','jelly_gravity','jelly_tilt','jelly_continuous_shake','jelly_running']};matrix=a.matrix_world.copy();j._UPDATING=True
report={'engine':s.render.engine,'tests':[],'max_fixed_cut_world_error':0,'all_finite':True}
body=next(o for o in s.objects if o.get('jelly_body') and o.parent==a);data=j.new_state(s);rest=next(r for o,r in data['entries'] if o==body);pins=rest[:,2]<1e-7
try:
 s.jelly_gravity=True;s.jelly_g=1
 for tilt in [0,90,180]:
  a.rotation_euler=(math.radians(tilt),0,0);bpy.context.view_layer.update()
  rot=np.array(a.matrix_world.to_3x3());offset=np.array(a.matrix_world.translation)
  fixed=rest[pins]@rot.T+offset;anchor_before=np.array(a.matrix_world)
  for k in [75,30,16]:
   s.jelly_k=k;s.jelly_damp=2.2;st=j.new_state(s);peak=0
   for frame in range(540):
    j.integrate(st,1/60);peak=max(peak,float(np.linalg.norm(st['q'])))
    if frame%30==0:
     deformed=j.deform_positions(rest,st['q'],st['r']);world=deformed[pins]@rot.T+offset
     report['max_fixed_cut_world_error']=max(report['max_fixed_cut_world_error'],float(np.max(np.abs(world-fixed))))
     report['all_finite'] &= bool(np.all(np.isfinite(deformed)))
   q=st['q'].copy();deformed=j.deform_positions(rest,q)
   tip_world_delta=rot@(deformed[-2]-rest[-2])
   report['tests'].append({'tilt_degrees':tilt,'hardness':k,'gravity':1,'equilibrium_q':q.tolist(),'gravity_down_tip_m':float(-tip_world_delta[2]),'equilibrium_error':float(np.linalg.norm(q-j.equilibrium(s))),'anchor_matrix_error':float(np.max(np.abs(np.array(a.matrix_world)-anchor_before))),'minimum_world_z':float(np.min((deformed@rot.T+offset)[:,2])),'peak_deformation':peak})
 # Gravity zero truly eliminates sag at all orientations.
 s.jelly_gravity=False;st=j.new_state(s)
 for _ in range(120):j.integrate(st,1/60)
 report['zero_gravity_displacement']=float(np.linalg.norm(st['q']))
 # Quantitative gravity multiplier ordering, below saturation.
 s.jelly_gravity=True;s.jelly_k=30;a.rotation_euler=(math.pi/2,0,0);bpy.context.view_layer.update()
 report['gravity_multiplier_curve']=[]
 for g in [0,.5,1,1.5]:
  s.jelly_g=g;report['gravity_multiplier_curve'].append({'gravity':g,'sag':float(np.linalg.norm(j.equilibrium(s)))})
 # release after a fast drag is not lost; fixed-step spring changes sign about equilibrium.
 s.jelly_g=1;st=j.new_state(s);eq=j.equilibrium(s);st['q'][:]=eq+np.array((0,.35,.12));st['v'][:]=(0,2,0);ys=[]
 for frame in range(600):j.integrate(st,1/60);ys.append(float(st['q'][1]-eq[1]))
 report['release_rebounds']=min(ys)<-.03 and max(ys)>.1
 report['release_final_error']=float(np.linalg.norm(st['q']-eq))
 # Alternating force shakes jelly without writing the anchor transform.
 st=j.new_state(s);st['q'][:]=j.equilibrium(s);q0=st['q'].copy();maxdev=0;anchor_before=np.array(a.matrix_world)
 for frame in range(480):
  t=frame/60;f=j.gravity_force(s)
  if t<3:f+=j.local_vector(s,(18*math.sin(t*8.7),7*math.cos(t*7.1),0))
  j.integrate(st,1/60,force=f);maxdev=max(maxdev,float(np.linalg.norm(st['q']-q0)))
 report['shake']={'peak_deviation':maxdev,'anchor_error':float(np.max(np.abs(np.array(a.matrix_world)-anchor_before))),'residual_after_5s':float(np.linalg.norm(st['q']-q0))}
 side=[t for t in report['tests'] if t['tilt_degrees']==90]
 report['softer_sags_more']=side[0]['gravity_down_tip_m']<side[1]['gravity_down_tip_m']<side[2]['gravity_down_tip_m']
 report['passed']=report['max_fixed_cut_world_error']<1e-7 and report['all_finite'] and report['zero_gravity_displacement']==0 and report['softer_sags_more'] and report['release_rebounds'] and report['release_final_error']<.003 and report['shake']['peak_deviation']>.1 and report['shake']['anchor_error']==0 and all(t['equilibrium_error']<.005 and t['anchor_matrix_error']==0 for t in report['tests'])
finally:
 for k,v in original.items():setattr(s,k,v)
 a.matrix_world=matrix;bpy.context.view_layer.update();j._UPDATING=False
json.dump(report,open(OUT+'/重力_硬软_固定切面_摇晃测试.json','w'),ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False,indent=2))
