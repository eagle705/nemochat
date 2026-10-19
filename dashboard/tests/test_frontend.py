"""Offline frontend contract checks; test fixtures are synthetic, not run data."""
from pathlib import Path
import json
import shutil
import subprocess


STATIC = Path(__file__).resolve().parents[1] / "static"
NODE = shutil.which("node")


def node_assert(code):
    assert NODE, "Node is required for frontend behavior validation"
    assert (STATIC / "app.js").exists(), "Frontend app.js has not been implemented"
    result = subprocess.run(
        [NODE, "-e", "const assert = require('node:assert/strict'); const app = require(" + json.dumps(str(STATIC / "app.js")) + ");\n(() => {\n" + code + "\n})();"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_initial_run_selection_uses_latest_observed_run_and_preserves_manual_choice():
    node_assert("""
const base={schema_version:1,runs:[{id:'old',stage:'pretrain',updated_at:'2026-10-13T00:00:00Z'},{id:'new',stage:'sft',updated_at:'2026-10-13T10:00:00Z'}],stages:[],insights:[],host:{},collection:{},profiling:{}};
const store=app.createStore();store.receive(base);
assert.equal(store.selectedRun().id,'new');assert.equal(store.state.profileStage,'sft');
store.select('old');store.receive({...base,runs:[...base.runs,{id:'newer',stage:'rlvr',updated_at:'2026-10-13T12:00:00Z'}]});
assert.equal(store.selectedRun().id,'old');
""")


def test_contract_validation_and_honest_missing_values():
    node_assert("""
assert.equal(app.validateSnapshot({schema_version: 2, runs: []}), false);
assert.equal(app.validateSnapshot({schema_version: 1, runs: 'wrong'}), false);
assert.equal(app.validateSnapshot({schema_version: 1, runs: [], stages: [], insights: [], host: {}, collection: {}, profiling: {}}), true);
assert.equal(app.validateSnapshot({schema_version: 1, runs: [null], stages: [], insights: [], host: {}, collection: {}, profiling: {}}), false);
assert.equal(app.validateSnapshot({schema_version: 1, runs: [{id:42}], stages: [], insights: [], host: {}, collection: {}, profiling: {}}), false);
assert.equal(app.formatNumber(null), '미수집');
assert.equal(app.formatNumber(0), '0');
assert.equal(app.formatNumber('42'), '미수집');
assert.equal(app.formatDuration(null), '미수집');
assert.equal(app.formatDuration(0), '0초');
assert.equal(app.progressPercent({step: 0, total_steps: 0}), null);
assert.equal(app.progressPercent({step: 5, total_steps: 10}), 50);
""")


def test_selection_pause_resume_and_invalid_snapshot_preserve_last_display():
    node_assert("""
const base = {schema_version: 1, runs: [{id:'a'}, {id:'b'}], stages: [], insights: [], host: {}, collection: {}, profiling: {}};
const store = app.createStore();
assert.equal(store.receive(base), true);
store.select('b');
assert.equal(store.selectedRun().id, 'b');
store.setPaused(true);
store.receive({...base, runs: [{id:'b', name:'updated'}]});
assert.equal(store.selectedRun().name, undefined);
assert.equal(store.state.pending.runs[0].name, 'updated');
store.setPaused(false);
assert.equal(store.selectedRun().name, 'updated');
assert.equal(store.receive({schema_version:9}), false);
assert.equal(store.selectedRun().name, 'updated');
store.receive({...base, runs: [{id:'c'}]});
assert.equal(store.selectedRun().id, 'c');
store.select('<bad>');
assert.equal(store.selectedRun().id, 'c');
store.receive({...base, runs: [], generated_at:'2026-10-13T00:00:10Z'});
assert.equal(store.selectedRun(), null);
assert.equal(store.receive({...base, generated_at:'2026-10-13T00:00:00Z'}), false);
""")


def test_chart_series_preserves_counter_types_gaps_and_zero():
    node_assert("""
const history = [{step:3,loss:0,processed_tokens:300,loss_bearing_tokens:250}, {step:1,loss:2,processed_tokens:100}, {step:2,loss:null,processed_tokens:200}, {step:3,loss:1,processed_tokens:301,loss_bearing_tokens:251}];
assert.deepEqual(app.chartSeries(history,'loss',100), [{step:1,value:2},{step:2,value:null},{step:3,value:1}]);
assert.equal(app.chartSeries(history,'processed_tokens',2)[1].value,301);
assert.equal(app.chartSeries(history,'loss_bearing_tokens',2)[1].value,251);
assert.deepEqual(app.chartSeries([{step:null,loss:1}], 'loss'), []);
assert.equal(app.chartSeries([{step:0,loss:0}], 'loss')[0].value, 0);
""")


def test_transport_snapshot_event_poll_bounds_reconnect_and_cleanup():
    node_assert("""
(async () => {
const base = {schema_version:1,runs:[],stages:[],insights:[],host:{},collection:{interval_seconds:0},profiling:{}};
let source, fetched=0; const timers=new Map(); let next=0; const received=[], statuses=[];
class FakeSource { constructor(url){assert.equal(url,'/api/events'); source=this;} addEventListener(name,fn){assert.equal(name,'snapshot');this.snapshot=fn;} close(){this.closed=true;} }
const transport=app.createTransport({EventSource:FakeSource,fetch:async(url)=>{assert.equal(url,'/api/snapshot');fetched++;return {ok:true,json:async()=>base};},setTimeout:(fn,ms)=>{const id=++next;timers.set(id,{fn,ms});return id;},clearTimeout:id=>timers.delete(id)}, value=>received.push(value), value=>statuses.push(value));
transport.start(); await new Promise(resolve=>setImmediate(resolve));
assert.equal(fetched,1); assert.equal(received.length,1);
source.onopen(); assert.equal(statuses.at(-1),'live');
source.snapshot({data:JSON.stringify(base)}); assert.equal(received.length,2);
source.snapshot({data:'not json'}); assert.equal(statuses.at(-1),'invalid');
source.onerror(); await new Promise(resolve=>setImmediate(resolve));
assert.equal(statuses.includes('reconnecting'),true);
assert.equal(fetched,1); source.onerror(); assert.equal(fetched,1);
assert.equal(timers.size,1); assert.equal([...timers.values()][0].ms,2000);
source.onopen(); assert.equal(timers.size,0);
transport.stop(); assert.equal(source.closed,true);
assert.equal(timers.size,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")


def test_stage_profiles_do_not_infer_attribution_from_global_imports():
    node_assert("""
const globalImport={lens:{available:true},nsight:{available:false},bridge:{available:false},stages:{pretrain:{lens:{available:true},optimizations:[{confidence:'candidate',title:'test hypothesis'}]},sft:{lens:{available:false}}}};
assert.equal(app.stageProfile(globalImport,'pretrain').lens.available,true);
assert.equal(app.stageProfile(globalImport,'sft').lens.available,false);
assert.equal(app.stageProfile(globalImport,'mopd'),null);
assert.equal(app.stageProfile(globalImport,'unscoped').lens.available,true);
const store=app.createStore(); store.selectProfileStage('rlvr');
assert.equal(store.state.profileStage,'rlvr');
store.selectProfileStage('malformed'); assert.equal(store.state.profileStage,'rlvr');
const defaults=app.createStore(); defaults.receive({schema_version:1,runs:[{id:'long-run',stage:'long_context'},{id:'sft-run',stage:'sft'}],stages:[],insights:[],host:{},collection:{},profiling:{}});
assert.equal(defaults.state.profileStage,'long');
defaults.select('sft-run'); assert.equal(defaults.state.profileStage,'sft');
defaults.selectProfileStage('mopd'); defaults.select('long-run'); assert.equal(defaults.state.profileStage,'long');
assert.equal(defaults.state.profilePinned,false);
""")


def test_selected_run_stage_changes_clear_reference_pin():
    node_assert(r"""
const store=app.createStore();
const base={schema_version:1,runs:[{id:'run',stage:'pretrain'}],stages:[],insights:[],host:{},collection:{},profiling:{}};
store.receive(base);store.selectProfileStage('pretrain');
store.receive({...base,runs:[{id:'run',stage:'sft'}]});
assert.equal(store.state.profileStage,'sft');assert.equal(store.state.profilePinned,false);
store.selectProfileStage('pretrain');store.select('run');
assert.equal(store.state.profileStage,'sft','explicit run selection exits reference mode even for same run');
store.selectProfileStage('pretrain');store.receive({...base,runs:[{id:'next',stage:'mid'}]});
assert.equal(store.state.profileStage,'mid');
""")


def test_static_document_has_accessible_local_controls_and_safe_dom():
    html_path = STATIC / 'index.html'
    assert html_path.exists(), 'Dashboard document has not been implemented'
    html = html_path.read_text()
    script = (STATIC / 'app.js').read_text()
    assert 'lang="ko"' in html
    for identifier in ('run-select', 'pause-button', 'chart-range', 'connection-status', 'chart-grid'):
        assert f'id="{identifier}"' in html
    assert 'aria-live="polite"' in html
    assert 'https://' not in html and 'http://' not in html
    assert 'innerHTML' not in script and 'insertAdjacentHTML' not in script
    assert 'textContent' in script


def test_browser_bootstrap_dom_interactions_and_untrusted_text():
    # Minimal browser primitives exercise the actual renderer without installing a DOM package.
    script_path = json.dumps(str(STATIC / 'app.js'))
    html_path = json.dumps(str(STATIC / 'index.html'))
    node_assert(r"""
const fs=require('node:fs'), vm=require('node:vm');
const nodes=new Map();
class Element {
  constructor(tag){this.tagName=tag;this.children=[];this.attrs={};this.listeners={};this.dataset={};this._text='';this.className='';this.hidden=false;this.value='';this.classList={contains:value=>this.className.split(' ').includes(value)};}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(child=>child.textContent).join('');}
  set innerHTML(value){throw Error('Unsafe HTML sink');}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this._text='';this.children=[...children];}
  setAttribute(name,value){this.attrs[name]=String(value);if(name==='id'){this.id=String(value);nodes.set(this.id,this);}if(name==='class')this.className=String(value);}
  removeAttribute(name){delete this.attrs[name];}
  addEventListener(name,fn){this.listeners[name]=fn;}
  fire(name,event={}){this.listeners[name]?.({target:this,preventDefault(){},...event});}
  focus(){document.activeElement=this;}
  getBoundingClientRect(){return {left:0,width:360};}
}
const document={activeElement:null, getElementById:id=>nodes.get(id),createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)};
for(const match of fs.readFileSync(HTML_PATH,'utf8').matchAll(/id="([^"]+)"/g)){const node=new Element('div');node.setAttribute('id',match[1]);}
let source; const timers=new Map(); let next=0;
class Source {constructor(){source=this;}addEventListener(name,fn){this.snapshot=fn;}close(){this.closed=true;}}
const missing={available:false,source:null,warnings:[]};
const run={provenance:{adapter:'canonical',schema_version:1,metrics_scope:'designated writer',counter_scope:'global cumulative',sources:[{path:'metrics<hostile>.jsonl',kind:'canonical metrics',available:false,updated_at:null,truncated:true}]},id:'a',name:'<img src=x onerror=alert(1)>',stage:'pretrain',status:'completed',source:'test-fixture.json',model:{name:'synthetic test only'},tokens:{processed:100,loss_bearing:70,source_read:200,target:null},timing:{elapsed_seconds:0,eta_seconds:null,scope:'test'},metrics:{loss:0,learning_rate:0,grad_norm:null},history:[{step:1,loss:2,processed_tokens:50,loss_bearing_tokens:30},{step:2,loss:null,processed_tokens:75,loss_bearing_tokens:50},{step:3,loss:0,processed_tokens:100,loss_bearing_tokens:70}]};
const base={schema_version:1,generated_at:'2026-10-13T00:00:00Z',collection:{interval_seconds:5,duration_ms:0,warnings:[]},host:{cpu:{},memory:{},gpus:[],warnings:[]},runs:[run,{id:'b',stage:'rlvr',metrics:{loss:null}}],stages:[{id:'pretrain',status:'observed',run_ids:['a']}],insights:[],profiling:{lens:missing,nsight:missing,bridge:missing,stages:{pretrain:{lens:{available:true,source:'fixture',spans:[{name:'<script>untrusted</script>',duration_ms:0,trace_id:'abc',span_id:'def',attributes:{role:'test'}}],warnings:[]},nsight:{available:true,source:'fixture',kernels:[{name:'raw_kernel<test>',count:0,total_ms:0,average_us:0,percent:0},{name:'<img src=x onerror=alert(1)>',count:null,total_ms:null,average_us:null,percent:null}],summary:{scope:'selected fixture process',kernel_count:0,summed_kernel_ms_not_wall:0,selected_ranges:{trace_ranks:[0],updates:[{rank:0,update:3,cpu_nvtx_ms:10,associated_correlated_kernel_count:7,phases:[{name:'forward',cpu_nvtx_ms:8,associated_correlated_kernel_count:5}]}]}},warnings:[]},bridge:{available:true,source:'fixture',operations:[{id:'raw-operation',title:'Actual operation',duration_ms:0,role:'test',explanation:'test only',evidence:'explicit_python_scope'}],warnings:[]},optimizations:[{title:'test candidate',confidence:'candidate',evidence:null,hypothesis:'not measured',next_experiment:'bounded A/B',source:'fixture'}]}}}};
const window={document,EventSource:Source,fetch:async()=>({ok:true,json:async()=>base}),setTimeout:(fn,ms)=>{const id=++next;timers.set(id,{fn,ms});return id;},clearTimeout:id=>timers.delete(id),setInterval:()=>0,clearInterval(){},addEventListener(){}};
vm.runInNewContext(fs.readFileSync(SCRIPT_PATH,'utf8'),{window,console});
(async()=>{
await new Promise(resolve=>setImmediate(resolve));
const get=id=>nodes.get(id);
assert.equal(get('metric-loss').textContent,'0');
assert.equal(get('metric-grad').textContent,'미수집');
assert.equal(get('elapsed').textContent,'0초');
assert.equal(get('tokens-processed').textContent,'100');
assert.equal(get('tokens-loss').textContent,'70');
assert.equal(get('tokens-source').textContent,'200');
assert.equal(get('run-title').textContent,run.name);
assert.ok(get('run-provenance').textContent.includes('canonical'));
assert.ok(get('run-provenance').textContent.includes('metrics<hostile>.jsonl'));
assert.ok(get('run-provenance').textContent.includes('파일 미관측'));
assert.ok(get('run-provenance').textContent.includes('global cumulative'));
assert.ok(get('run-provenance').textContent.includes('일부 읽기'));
assert.equal(get('lens-content').textContent.includes('<script>untrusted</script>'),true);
assert.equal(get('nsight-content').textContent.includes('raw_kernel<test>'),true);
assert.equal(get('nsight-content').textContent.includes('선택 업데이트 · 생산자 분석'),true);
assert.equal(get('nsight-content').textContent.includes('"cpu_nvtx_ms"'),false);
assert.equal(get('nsight-content').textContent.includes('forward'),true);
const kernelChart=get('profile-kernels');
assert.ok(kernelChart, 'Nsight cumulative CUDA bars must render');
const kernelRow=kernelChart.children.find(child=>child.attrs.role==='button');
assert.ok(kernelRow, 'kernel bars support keyboard and tap');
assert.equal(kernelRow.attrs.tabindex,'0');
kernelRow.fire('pointerenter');
const kernelTip=get('profile-kernels-tooltip');
assert.equal(kernelTip.hidden,false);
assert.ok(kernelTip.textContent.includes('raw_kernel<test>'));
assert.ok(kernelTip.textContent.includes('사전학습'));
assert.ok(kernelTip.textContent.includes('매핑 미확인'));
assert.ok(kernelTip.textContent.includes('미분류'));
assert.ok(kernelTip.textContent.includes('0 ms'));
assert.ok(kernelTip.textContent.includes('0 µs'));
assert.ok(kernelTip.textContent.includes('0 %'));
kernelRow.fire('pointerleave');assert.equal(kernelTip.hidden,true);
kernelRow.fire('focus');assert.equal(kernelTip.hidden,false);
kernelRow.fire('keydown',{key:'Escape'});assert.equal(kernelTip.hidden,true);
kernelRow.fire('click');assert.equal(kernelTip.hidden,false);
const nullRow=kernelChart.children.filter(child=>child.attrs.role==='button')[1];
nullRow.fire('pointermove');
assert.ok(kernelTip.textContent.includes('<img src=x onerror=alert(1)>'));
assert.ok(kernelTip.textContent.includes('CUDA 합계 미수집'));
assert.ok(!kernelTip.textContent.includes('NaN'));
assert.ok(kernelTip.textContent.includes('출처 fixture'));
const phaseChart=get('profile-phases');
const phaseRow=phaseChart.children.find(child=>child.attrs.role==='button');
phaseRow.fire('focus');
assert.ok(get('profile-phases-tooltip').textContent.includes('HOST NVTX'));
assert.ok(get('profile-phases-tooltip').textContent.includes('8 ms'));
assert.ok(get('profile-phases-tooltip').textContent.includes('Rank 0 / Update 3'));
assert.ok(get('profile-phases-tooltip').textContent.includes('GPU 커널 시간 아님'));
assert.equal(get('bridge-content').textContent.includes('raw-operation'),true);
for (const id of ['profile-lens','profile-bridge']) {
  assert.ok(get(id), `${id} duration bars render`);
  const row=get(id).children.find(child=>child.attrs.role==='button');
  row.fire('focus');
  assert.ok(get(`${id}-tooltip`).textContent.includes('GPU 커널 시간 아님'));
  assert.ok(get(`${id}-tooltip`).textContent.includes('0 ms'));
}
assert.ok(get('profile-lens-tooltip').textContent.includes('<script>untrusted</script>'));
assert.ok(get('profile-bridge-tooltip').textContent.includes('test only'));
assert.ok(get('profile-bridge-tooltip').textContent.includes('explicit_python_scope'));
assert.ok(get('profile-bridge-tooltip').textContent.includes('소스 식별자 미검증'));
assert.equal(get('optimizations').textContent.includes('후보 가설 · 미측정'),true);
assert.equal(get('optimizations').textContent.includes('정확성 + 비프로파일 전체 업데이트 A/B'),true);
assert.equal(get('profile-stage-cards').children.length,6);
const spanDetails=get('lens-content').children.find(child=>child.tagName==='details'); spanDetails.open=true; spanDetails.fire('toggle');
const loss=get('chart-loss');
assert.equal(loss.children.find(child=>child.tagName==='path').attrs.d.match(/M/g).length,2);
loss.fire('keydown',{key:'Home'});
assert.equal(get('chart-grid').children[0].textContent.includes('Step 1 · Loss 2'),true);
loss.fire('keydown',{key:'ArrowRight'});
assert.equal(get('chart-grid').children[0].textContent.includes('Step 2 · Loss 미수집'),true);
get('pause-button').fire('click');
source.snapshot({data:JSON.stringify({...base,runs:[{...run,metrics:{loss:99}},base.runs[1]]})});
assert.equal(get('metric-loss').textContent,'0');
assert.equal(get('pause-button').attrs['aria-pressed'],'true');
get('pause-button').fire('click');assert.equal(get('metric-loss').textContent,'99');
assert.equal(get('lens-content').children.find(child=>child.tagName==='details').open,true);
get('profile-stage-select').value='mopd';get('profile-stage-select').fire('change');
get('run-select').value='b';get('run-select').fire('change');
assert.equal(get('metric-loss').textContent,'미수집');
assert.equal(get('profile-stage-select').value,'rlvr');
assert.equal(get('lens-content').textContent.includes('캡처가 아직 없습니다'),true);
get('tab-nsight').fire('click');assert.equal(get('panel-nsight').hidden,false);assert.equal(get('panel-lens').hidden,true);
get('tab-nsight').fire('keydown',{key:'ArrowRight'});assert.equal(get('tab-bridge').attrs['aria-selected'],'true');
source.snapshot({data:'{"schema_version":3}'});assert.equal(get('metric-loss').textContent,'미수집');
assert.equal(get('connection-status').textContent.includes('형식 오류'),true);
})().catch(error=>{console.error(error);process.exitCode=1;});
""".replace('SCRIPT_PATH', script_path).replace('HTML_PATH', html_path))


def test_kernel_name_reference_meaning_does_not_claim_source_or_phase():
    node_assert("""
for (const name of ['ncclAllGather_RING_LL', 'ncclAllReduce', 'ncclReduceScatter', 'ncclAlltoAll', 'gemm_kernel', 'FillFunctor<int>', 'layer_norm', 'softmax']) {
 const meaning=app.kernelMeaning(name);
 assert.ok(meaning.includes('명칭 기반 참고 설명 / 소스 미확인'));
 assert.equal(meaning.includes('미분류'),false);
}
assert.ok(app.kernelMeaning('hostile<script>').includes('미분류'));
assert.ok(app.kernelMeaning(null).includes('미분류'));
assert.ok(app.kernelMeaning('FillFunctor<int>').includes('Python 호출을 증명하지 않음'));
""")


def test_profile_default_and_responsive_duration_graph_styles():
    html = (STATIC / 'index.html').read_text()
    css = (STATIC / 'style.css').read_text()
    assert 'id="tab-nsight" role="tab" aria-selected="true"' in html
    assert 'id="panel-nsight" role="tabpanel" aria-labelledby="tab-nsight" hidden' not in html
    assert '.profile-bars' in css and '.profile-tooltip' in css
    assert '.profile-tooltip{position:fixed' in css
    assert 'overflow-wrap:anywhere' in css
    assert '.profile-chart.host .profile-bar' in css
    assert 'min-height:44px' in css


def test_polling_only_upper_bound_http_error_and_stop():
    node_assert("""
(async()=>{
const base={schema_version:1,runs:[],stages:[],insights:[],host:{},collection:{interval_seconds:999},profiling:{}};
const timers=new Map();let next=0;const statuses=[];let ok=true;let received=0;
const transport=app.createTransport({fetch:async()=>({ok,json:async()=>base}),setTimeout:(fn,ms)=>{const id=++next;timers.set(id,{fn,ms});return id;},clearTimeout:id=>timers.delete(id)},()=>received++,status=>statuses.push(status));
transport.start();await new Promise(resolve=>setImmediate(resolve));
assert.equal(received,1);assert.equal([...timers.values()][0].ms,60000);
ok=false;const [id,timer]=[...timers.entries()][0];timers.delete(id);timer.fn();await new Promise(resolve=>setImmediate(resolve));
assert.equal(received,1);assert.equal(statuses.at(-1),'offline');assert.equal(timers.size,1);
transport.stop();assert.equal(timers.size,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
""")
