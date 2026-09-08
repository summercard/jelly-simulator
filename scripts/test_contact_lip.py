"""Regression: contact rim must rise towards the sphere, around its footprint."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'scripts/test_volume_control.py').read_text()
exec(source.split('for b in [0,1,2]:')[0])
results=[]
for depth in [.15,.36,.5]:
    states,metrics,_=case(depth=depth)
    st=states[0];body,rest=next((o,r) for o,r in st['entries'] if o.get('jelly_body'))
    offset=st['dent'][body.as_pointer()][0]
    normal=st['pressure_normal'];centre=st['pressure_center'];diff=rest-centre
    axial=diff@normal;lateral=np.linalg.norm(diff-axial[:,None]*normal,axis=1)
    rise=offset@normal;index=int(rise.argmax())
    near=(lateral>.38)&(lateral<.76)&(rest[:,2]>.2)
    far=lateral>.9
    result={'depth':depth,'contact_rim_rise':float(rise[near].max()),'peak_distance_from_axis':float(lateral[index]),'far_surface_rise':float(rise[far].max()),'volume_ratio':metrics['volume_ratio']}
    assert result['contact_rim_rise']>.01
    assert .38<result['peak_distance_from_axis']<.76
    assert result['far_surface_rise']<result['contact_rim_rise']*.15
    results.append(result)
assert results[0]['contact_rim_rise']<results[1]['contact_rim_rise']<results[2]['contact_rim_rise']
import json
(ROOT/'接触鼓唇_侧面轮廓测试.json').write_text(json.dumps({'passed':True,'cases':results},ensure_ascii=False,indent=2))
print('LIP_PASS',results)
