import bpy,sys,json,numpy as np
from mathutils import Vector
j=sys.modules['kawaii_jelly_interaction'];s=bpy.context.scene;ball=j.finger(s)
props={k:getattr(s,k) for k in ['jelly_active','jelly_k','jelly_finger_auto','jelly_press']};oldball=ball.location.copy();oldctx=j._FINGER_CONTEXT.copy();j._FINGER_CONTEXT={};j._UPDATING=True
report={}
try:
 s.jelly_active=1;s.jelly_k=30;s.jelly_finger_auto=False;s.jelly_press=False
 roots=j.anchors(s);states=[j.new_state(s,a) for a in roots];st=states[0];a=roots[0]
 for t in states:t['q'][:]=j.equilibrium(s)
 normal=Vector((0,0,1));base=np.array([[0,0,1]],dtype=np.float32);contact=Vector(j.deform_positions(base,st['q'],0)[0]);outside=a.matrix_world@(contact+normal*.6)
 ball.location=outside;ctx=j.finger_context(s);ctx['desired']=a.matrix_world@(contact-normal*3)
 margins=[];anchors_before=[a.matrix_world.copy() for a in roots]
 for i in range(100):
  j.prepare_finger_contacts(s,states)
  for t in states:j.update_dents(t,1/60)
  j.resolve_solid_finger(s,states)
  local=a.matrix_world.inverted()@ball.location;near,n,idx,dist=j.surface_tree(st,True).find_nearest(local)
  margins.append(dist-.23 if (local-near).dot(n)>=0 else -dist-.23)
 body,rest=next((o,r) for o,r in st['entries'] if o.get('jelly_body'));dent=st['dent'][body.as_pointer()][0]
 report['minimum_surface_clearance']=min(margins);report['dent_peak']=st['max_dent'];report['sphere_stayed_front']=bool((a.matrix_world.inverted()@ball.location).z>0)
 report['fixed_cut_error']=float(np.max(np.abs(dent[rest[:,2]<1e-7])));report['other_jelly_dent']=states[1]['max_dent']
 ctx['desired']=outside
 j.prepare_finger_contacts(s,states);j.resolve_solid_finger(s,states)
 report['withdraw_position_error']=(ball.location-outside).length
 for i in range(240):
  j.prepare_finger_contacts(s,states)
  for t in states:j.update_dents(t,1/60)
  j.resolve_solid_finger(s,states)
 report['release_dent_after_4s']=st['max_dent'];report['anchor_error']=max(max(abs(x-y) for rowa,rowb in zip(a.matrix_world,b) for x,y in zip(rowa,rowb)) for a,b in zip(roots,anchors_before))
 report['same_depth']=max(a.location.y for a in roots)-min(a.location.y for a in roots)
 report['camera_faces_cut']=float((s.camera.rotation_euler.to_matrix()@Vector((0,0,1))).dot(a.matrix_world.to_3x3()@normal))
 report['passed']=bool(min(margins)>-.001 and report['dent_peak']>.03 and report['sphere_stayed_front'] and report['withdraw_position_error']<.001 and report['fixed_cut_error']==0 and report['other_jelly_dent']==0 and report['release_dent_after_4s']<.001 and report['anchor_error']==0 and report['same_depth']==0 and report['camera_faces_cut']>.999)
finally:
 for k,v in props.items():setattr(s,k,v)
 ball.location=oldball;j._FINGER_CONTEXT=oldctx;j._UPDATING=False
path='/Users/summercards/ShellStorm2/output/kawaii_jelly_v003/球体防穿透_正面并排测试.json'
open(path,'w').write(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
