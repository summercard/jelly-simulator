"""Run: Blender --background --factory-startup --python scripts/test_volume_control.py"""
import bpy,sys,json,math,time
import numpy as np
from pathlib import Path
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import kawaii_jelly_interaction as j
j.register()
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'Q弹果冻_球面贴合挤压_v006.blend'))
s=bpy.context.scene;j._UPDATING=True
s.jelly_active=1;s.jelly_gravity=False;s.jelly_k=30;s.jelly_ball_radius=.38
j.finger_size_changed(s,bpy.context)
s.jelly_finger_auto=False;s.jelly_compression=1.25;s.jelly_bulge_spread=1.1
roots=j.anchors(s);ball=j.finger(s);a=roots[0]
report={'cases':[]}

def case(bulge=1.,depth=.36,k=30,normal=(0,0,1),frames=90):
    s.jelly_bulge=bulge;s.jelly_k=k
    states=[j.new_state(s,r) for r in roots];st=states[0]
    normal=Vector(normal).normalized();contact=normal.copy()
    outside=a.matrix_world@(contact+normal*.8)
    ball.location=outside;j._FINGER_CONTEXT.clear()
    ctx=j.finger_context(s);ctx['desired']=a.matrix_world@(contact+normal*(s.jelly_ball_radius-depth))
    minmargin=1.;start=time.perf_counter()
    for i in range(frames):
        j.prepare_finger_contacts(s,states)
        for t in states:j.update_dents(t,1/60)
        j.resolve_solid_finger(s,states)
        local=a.matrix_world.inverted()@ball.location
        near,n,idx,dist=j.surface_tree(st,True).find_nearest(local)
        minmargin=min(minmargin,dist-s.jelly_ball_radius if (local-near).dot(n)>=0 else -dist-s.jelly_ball_radius)
    body,rest=next((o,r) for o,r in st['entries'] if o.get('jelly_body'))
    offset=st['dent'][body.as_pointer()][0];tri=j.mesh_triangles(st,body)
    baseline=abs(j.mesh_volume(rest,tri));volume=abs(j.mesh_volume(rest+offset,tri))
    outward=rest/np.maximum(np.linalg.norm(rest,axis=1)[:,None],1e-6)
    radial=np.sum(offset*outward,axis=1)
    result={'bulge':bulge,'depth':depth,'hardness':k,'normal':list(normal),'max_inward':float(-radial.min()),'max_outward':float(radial.max()),'volume_ratio':volume/baseline,'volume_compensation_amplitude':st['bulge_amplitude'],'clearance':minmargin,'fixed_cap_error':float(np.max(np.abs(offset[rest[:,2]<1e-7]))),'other_dent':states[1]['max_dent'],'mean_step_ms':(time.perf_counter()-start)*1000/frames}
    assert np.isfinite(offset).all() and result['fixed_cap_error']==0 and result['other_dent']==0
    assert minmargin>-.002,result
    print('CASE',json.dumps(result),flush=True)
    report['cases'].append(result)
    return states,result,outside

for b in [0,1,2]:case(bulge=b)
assert report['cases'][0]['max_outward']<report['cases'][1]['max_outward']<report['cases'][2]['max_outward']
assert abs(report['cases'][1]['volume_ratio']-1)<.02
for d in [.1,.25,.5]:case(depth=d)
assert report['cases'][3]['max_inward']<report['cases'][4]['max_inward']<report['cases'][5]['max_inward']
for k in [12,90]:case(k=k,depth=3)
for n in [(1,0,.55),(-1,0,.55),(0,1,.55),(0,-1,.55)]:case(normal=n)
states,_,outside=case();j.finger_context(s)['desired']=outside
for _ in range(300):
    j.prepare_finger_contacts(s,states)
    for st in states:j.update_dents(st,1/60)
    j.resolve_solid_finger(s,states)
report['recovery_after_5s']=states[0]['max_dent'];assert report['recovery_after_5s']<.001
# Absolute coordinate inputs and native target transforms must preserve all three axes.
ball.location=(8,8,8);j._FINGER_CONTEXT.clear();j.finger_context(s);handle=j.finger_handle(s,True)
for axis in range(3):
    target=Vector((8,8,8));target[axis]+=.5;handle.location=target
    j.position_finger(s);j.prepare_finger_contacts(s,states);j.resolve_solid_finger(s,states)
    assert (ball.location-target).length<1e-5
report['xyz_target_follow']=True
for radius in [.2,.6,.38]:
    s.jelly_ball_radius=radius;j.finger_size_changed(s,bpy.context);j.finger_size_changed(s,bpy.context)
    assert abs(ball.get('jelly_radius')*max(ball.scale)-radius)<1e-6
report['radius_repeatable']=True
# Exercise the real timer's shake branch, not a substitute equation.
s.jelly_running=True;s.jelly_continuous_shake=True;j._STATE=j.state(s);j._STATE['last']=time.perf_counter()-.016
assert j.tick() is not None
report['shake_timer']=True
s.jelly_finger_auto=True;s.jelly_press=True
j.position_finger(s)
handle=j.finger_handle(s);target=handle.location.copy();target.x+=.1;handle.location=target
j.position_finger(s)
assert not s.jelly_finger_auto and (j.finger_context(s)['desired']-target).length<1e-5
report['native_handle_takes_over_auto']=True
s.jelly_continuous_shake=False
results=[]
for tilt in [0,90,180]:
    s.jelly_gravity=True;s.jelly_g=1;s.jelly_bulge=1.;s.jelly_k=30
    for root in roots:root.rotation_euler=(math.radians(tilt),0,0)
    bpy.context.view_layer.update()
    states=[j.new_state(s,r) for r in roots]
    for t in states:t['q'][:]=j.equilibrium(s)
    st=states[0];normal=Vector((.35,0,.93675)).normalized()
    contact=Vector(j.deform_positions(np.array([normal],dtype=np.float32),st['q'])[0])
    ball.location=a.matrix_world@(contact+normal*.8);j._FINGER_CONTEXT.clear();ctx=j.finger_context(s)
    ctx['desired']=a.matrix_world@(contact+normal*(s.jelly_ball_radius-.36))
    qother=states[1]['q'].copy();margin=1.
    for i in range(240):
        for t in states:j.integrate(t,1/60,force=j.gravity_force(s)+t.get('pressure_force',np.zeros(3)))
        j.prepare_finger_contacts(s,states)
        for t in states:j.update_dents(t,1/60)
        j.resolve_solid_finger(s,states)
        local=a.matrix_world.inverted()@ball.location;near,n,_,dist=j.surface_tree(st,True).find_nearest(local)
        margin=min(margin,dist-s.jelly_ball_radius if (local-near).dot(n)>=0 else -dist-s.jelly_ball_radius)
    assert margin>-.002 and states[1]['max_dent']==0 and np.linalg.norm(states[1]['q']-qother)<1e-6
    assert all(np.isfinite(t['q']).all() for t in states)
    results.append({'tilt':tilt,'clearance':margin,'dent':st['max_dent'],'finite':True,'other_unchanged':True})
report['dynamic_gravity_contacts']=results
report['passed']=True
(ROOT/'三轴控制_体积鼓出测试.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('VOLUME_TEST',json.dumps(report,ensure_ascii=False))
