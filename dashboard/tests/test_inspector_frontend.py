"""Inspector frontend behavior against contract-shaped synthetic fixtures."""
from test_frontend import node_assert, STATIC
import json
import ast
from pathlib import Path


def test_rejected_and_incomplete_sources_fail_closed_and_table_is_bounded():
    browser_assert(r"""
get('inspector-phase').value='forward';get('inspector-phase').fire('change');
update.phases[0].source={...verified,status:'rejected',snippet:'DO_NOT_RENDER'};
source.snapshot({data:JSON.stringify(base)});
assert.ok(get('inspector-card').textContent.includes('검증 거부'));
assert.ok(!get('inspector-card').textContent.includes('DO_NOT_RENDER'));
update.phases[0].source={...verified,start_line:null,snippet:'NO_INVENTED_LINE'};
source.snapshot({data:JSON.stringify(base)});
assert.ok(!get('inspector-card').textContent.includes('NO_INVENTED_LINE'));
base.profiling.stages.pretrain.bridge.operations[0].source_evidence={...verified,status:'rejected',snippet:'REJECTED_BRIDGE_CODE'};
source.snapshot({data:JSON.stringify(base)});
assert.ok(!get('bridge-content').textContent.includes('REJECTED_BRIDGE_CODE'));
get('inspector-phase').value='';get('inspector-phase').fire('change');
update.kernels=Array.from({length:80},(_,i)=>({...kernel,name:'k'+i,total_ms:80-i}));
source.snapshot({data:JSON.stringify(base)});
const flatten=n=>[n,...n.children.flatMap(flatten)];
const buttons=flatten(get('nsight-content')).filter(n=>n.tagName==='button');
assert.equal(buttons.length,64);
buttons[3].fire('click');assert.ok(get('inspector-card').textContent.includes('k3'));
source.snapshot({data:JSON.stringify(base)});assert.ok(get('inspector-card').textContent.includes('k3'));
""")


def test_rank_control_uses_actual_list_and_human_category():
    browser_assert(r"""
base.profiling.stages.pretrain.nsight.inspector.updates.push({...update,rank:1,update:8,kernels:[{...kernel,name:'rank-one',category:'GEMM'}]});
source.snapshot({data:JSON.stringify(base)});
get('inspector-rank').value='1';get('inspector-rank').fire('change');
assert.equal(get('inspector-update').value,'8');
assert.ok(get('inspector-card').textContent.includes('rank-one'));
assert.ok(get('inspector-card').textContent.includes('행렬 곱'));
assert.ok(!get('inspector-candidates').textContent.includes('selected count 2'));
""")


def test_dense_inspector_layout_has_responsive_local_styles():
    css=(STATIC/'style.css').read_text()
    assert '.inspector-layout' in css
    assert '.source-code' in css
    assert '.inspector-controls' in css
    assert 'Inspector' in (STATIC/'index.html').read_text()


def test_measured_scope_filters_and_never_falls_back_to_startup():
    node_assert("""
const n={kernels:[{name:'startup',count:999}],inspector:{available:true,updates:[{rank:0,update:3,kernels:[{name:'measured',count:7}],phases:[{id:'forward',kernels:[{name:'phase',count:2}]}]},{rank:1,update:4,kernels:[{name:'other'}],phases:[]}]}};
assert.equal(app.inspectorScope(n,{}).kernels[0].name,'measured');
assert.equal(app.inspectorScope(n,{rank:'0',update:'3',phase:'forward'}).kernels[0].count,2);
assert.deepEqual(app.inspectorScope(n,{rank:'0',update:'3',phase:'missing'}).kernels,[]);
assert.equal(app.inspectorScope(n,{view:'capture'}).kernels[0].name,'startup');
assert.equal(app.inspectorScope({kernels:n.kernels},{}).view,'capture');
""")


def browser_assert(assertions):
    # Reuse the baseline's dependency-free DOM harness, not its assertions.
    tree = ast.parse(Path(__file__).with_name('test_frontend.py').read_text())
    harness = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value,str) and 'class Element {' in n.value)
    harness = harness.split('vm.runInNewContext')[0]
    fixture = r"""
const code='<img src=x onerror=alert(1)>\nwith nvphase("forward"):';
const verified={status:'host_scope_verified',kind:'host_scope',path:'saved/native.py',start_line:12,end_line:13,file_sha256:'filehash',snippet_sha256:'snippethash',snippet:code};
const kernel={name:'gemm<hostile>',category:'행렬 곱',meaning:'입력과 가중치를 곱합니다',meaning_evidence:'symbol_reference',count:7,total_ms:3,average_us:428,percent:80,source_attribution:'unavailable',source:null};
const update={rank:0,update:3,name:'measured/update_3',host_ms:20,kernel_count:7,kernels:[kernel],phases:[{id:'forward',name:'phase/forward',host_ms:15,kernel_count:2,source:verified,kernels:[{...kernel,count:2,total_ms:1}]}]};
base.profiling.stages.pretrain.nsight.inspector={available:true,schema_version:1,scope:'measured fixture',callstacks:{python_source_attribution:false,runtime_rows_with_callchain:0},updates:[update,{...update,update:4,kernels:[{...kernel,name:'second',count:9}]}],candidates:[{priority:1,title:'검토 대상',rank:0,update:3,phase:'forward',evidence:'selected count 2',interpretation:'critical path unknown',action:'bounded A/B',confidence:'candidate'}]};
base.profiling.stages.pretrain.nsight.kernels=[{name:'startup FillFunctor',count:999,total_ms:999}];
base.profiling.stages.pretrain.bridge.operations[0].source_evidence={...verified,status:'snapshot_verified',kind:'explicit_bridge_operation'};
vm.runInNewContext(fs.readFileSync(SCRIPT_PATH,'utf8'),{window,console});
(async()=>{await new Promise(resolve=>setImmediate(resolve));
const get=id=>nodes.get(id);
"""
    node_assert((harness+fixture+assertions+"\n})().catch(e=>{console.error(e);process.exitCode=1;});").replace('SCRIPT_PATH',json.dumps(str(STATIC/'app.js'))).replace('HTML_PATH',json.dumps(str(STATIC/'index.html'))))


def test_inspector_controls_source_confidence_selection_and_stage_isolation():
    browser_assert(r"""
assert.equal(get('inspector-view').value,'measured');
assert.ok(get('inspector-card').textContent.includes('입력과 가중치를 곱합니다'));
assert.ok(!get('nsight-content').textContent.includes('startup FillFunctor'));
get('inspector-phase').value='forward';get('inspector-phase').fire('change');
assert.ok(get('inspector-card').textContent.includes('커널 호출 소스 미수집'));
assert.ok(get('inspector-card').textContent.includes('호스트 구간 소스 검증 / 커널 호출 스택 아님'));
assert.ok(get('inspector-card').textContent.includes('12  <img src=x onerror=alert(1)>'));
assert.ok(get('inspector-card').textContent.includes('filehash'));
const row=get('profile-kernels').children.find(c=>c.attrs.role==='button');
row.fire('pointerenter');assert.ok(get('profile-kernels-tooltip').textContent.includes('호출 수 2'));
assert.ok(get('profile-kernels-tooltip').textContent.includes('커널 소스 미확인'));
row.fire('keydown',{key:'Enter'});assert.ok(get('inspector-card').textContent.includes('gemm<hostile>'));
assert.ok(get('inspector-candidates').textContent.includes('bounded A/B'));
assert.ok(get('bridge-content').textContent.includes('교육용 별도 계측'));
assert.ok(get('bridge-content').textContent.includes('12  <img src=x onerror=alert(1)>'));
assert.ok(get('bridge-content').textContent.includes('13  with nvphase("forward"):'));
get('inspector-phase').value='';get('inspector-phase').fire('change');
get('inspector-update').value='4';get('inspector-update').fire('change');
source.snapshot({data:JSON.stringify(base)});
assert.equal(get('inspector-update').value,'4');
assert.ok(get('inspector-card').textContent.includes('second'));
get('inspector-view').value='capture';get('inspector-view').fire('change');
assert.ok(get('nsight-content').textContent.includes('startup FillFunctor'));
get('profile-stage-select').value='mopd';get('profile-stage-select').fire('change');
assert.ok(!get('nsight-content').textContent.includes('second'));
assert.ok(!get('nsight-content').textContent.includes(code));
""")
