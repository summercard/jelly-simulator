"""Validate actual sphere-contact gap and the non-spherical surface adapter."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
exec((ROOT/'scripts/test_volume_control.py').read_text().split('for b in [0,1,2]:')[0])
results=[]
for depth in [.1,.25,.36,.5]:
    states,_,_=case(depth=depth,frames=60)
    st=states[0];tree=j.surface_tree(st,True)
    n=st['pressure_normal'];c=st['pressure_center'];r=s.jelly_ball_radius
    axis=np.array((1.,0,0));axis-=n*np.dot(axis,n);axis/=np.linalg.norm(axis);second=np.cross(n,axis)
    angle=min(1.48,math.acos(np.clip(1-st['pressure_depth']/r,-1,1))+.12*min(1,st['pressure_depth']/r))
    gaps=[]
    for theta in np.linspace(.1,angle*.95,12):
        for phi in np.linspace(0,2*math.pi,48,endpoint=False):
            point=c+r*(-n*math.cos(theta)+(axis*math.cos(phi)+second*math.sin(phi))*math.sin(theta))
            near,normal,_,distance=tree.find_nearest(Vector(point));gaps.append(distance)
    assert max(gaps)<.007,(depth,max(gaps))
    results.append({'depth':depth,'max_surface_gap':max(gaps),'mean_gap':float(np.mean(gaps))})
# Non-spherical closed cube: points inside/on/outside project to its surface.
bpy.ops.mesh.primitive_cube_add(size=2)
obj=bpy.context.object
surface=j.MeshContactSurface([tuple(v.co) for v in obj.data.vertices],[tuple(p.vertices) for p in obj.data.polygons])
points=np.array([[.9,.2,.1],[1.,.1,.2],[1.5,.2,.1],[-.9,.1,.2]])
near,normals,distance=surface.query(points)
projected=j.project_contact_surface(points,surface)
_,_,after=surface.query(projected)
assert np.all(after>=.0024)
assert np.allclose(projected[2],points[2])
assert np.allclose(projected[0],[1.0025,.2,.1])
report={'passed':True,'sphere_contact_gap':results,'mesh_adapter_cube':{'passed':True,'signed_distances':distance.tolist(),'projected':projected.tolist()}}
(ROOT/'球面贴合_通用表面接口测试.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('WRAP_PASS',report)
