"""Timeline presentation model: explicit synthetic nanosecond fixtures."""


def test_timeline_renderer_provides_bounded_ordered_table_and_event_details():
    node_assert("const timeline=require("+json.dumps(str(STATIC/'timeline.js'))+");"+r"""
class Node {
 constructor(tag){this.tagName=tag;this.children=[];this.attrs={};this.handlers={};this.value='';this.own='';}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=children;this.own='';}
 setAttribute(k,v){this.attrs[k]=String(v);}
 addEventListener(k,fn){this.handlers[k]=fn;}
 fire(k){this.handlers[k]?.({target:this});}
 get textContent(){return this.own+this.children.map(c=>c.textContent).join(' ');}
 set textContent(v){this.own=String(v);this.children=[];}
}
const doc={createElement:tag=>new Node(tag)}, target=new Node('section');
const fields=['row_id','name_index','gpu_start_ns','gpu_end_ns','device_id','context_id','stream_id','correlation_id','api_start_ns','api_end_ns','thread_index','phase_index','api_name_index'];
const update={timeline:{schema_version:1,available:true,origin:'selected_host_nvtx_start',host_duration_ns:200,span_duration_ns:220,order:'gpu_start_ns,row_id',total_events:120,returned_events:120,truncated:false,names:['exact_raw_symbol<template>'],api_names:['cudaLaunchKernel'],threads:['12345678901234567'],phases:[{id:'forward',start_ns:0,end_ns:200}],fields,events:Array.from({length:120},(_,i)=>[i,0,i,i+10,0,1,7,i,i,i+1,0,0,0])}};
let selected=null;
timeline.render(target,update,{document:doc,onSelectEvent:e=>selected=e});
const flatten=n=>[n,...n.children.flatMap(flatten)];
let all=flatten(target), rows=all.filter(n=>n.tagName==='tbody')[0].children;
assert.equal(rows.length,100);
assert.ok(target.textContent.includes('CPU API enqueue'));
assert.ok(target.textContent.includes('GPU Start order'));
const button=flatten(rows[2]).find(n=>n.tagName==='button');button.fire('click');
assert.equal(selected.id,2);
assert.ok(target.textContent.includes('12345678901234567'));
assert.ok(target.textContent.includes('CUDA runtime correlationId'));
assert.ok(target.textContent.includes('exact_raw_symbol<template>'));
const order=all.find(n=>n.attrs['aria-label']==='Event order');order.value='api';order.fire('change');
const next=flatten(target).find(n=>n.tagName==='button' && n.textContent==='Next 100');next.fire('click');
rows=flatten(target).filter(n=>n.tagName==='tbody')[0].children;assert.equal(rows.length,20);
""")

import json
from test_frontend import node_assert, STATIC


def test_decoder_explains_ring_topology_and_ll_acronym():
    from dashboard.kernel_symbols import decode_symbol
    parts={p['token']:p for p in decode_symbol('ncclDevKernel_AllGather_RING_LL')['parts']}
    assert '논리적 링' in parts['RING']['meaning']
    assert 'Low Latency' in parts['LL']['meaning']


def test_importer_regenerates_symbol_parts_instead_of_trusting_artifact(tmp_path):
    from test_capture_analysis import fixture
    import capture_analysis as analysis
    import inspector
    path,selection=fixture(tmp_path)
    with __import__('sqlite3').connect(path) as db:
        db.execute("UPDATE StringIds SET value=? WHERE id=1",('ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)',))
    document=analysis.analyze_capture(tmp_path,path,selection)
    document['sqlite']=inspector.sqlite_identity(tmp_path,path)
    document['updates'][0]['kernels'][0]['symbol_analysis']={'parts':[{'token':'FORGED_MEANING','sources':['https://evil.example']}], 'warnings':[]}
    (tmp_path/'symbols.json').write_text(json.dumps(document))
    safe=inspector.load_inspector(tmp_path,{'nsight_sqlite':'capture.sqlite','nsight_analysis':'symbols.json'})
    assert safe['available']
    decoded=safe['updates'][0]['kernels'][0]['symbol_analysis']
    assert decoded['family']=='NCCL'
    assert 'FORGED_MEANING' not in json.dumps(safe)
    assert 'https://evil.example' not in json.dumps(safe)
    assert any(p['token']=='4096' and p['confidence']=='documented' for p in decoded['parts'])


def test_inspector_wires_recorded_iteration_timeline_without_fabricating_it():
    from test_inspector_frontend import browser_assert
    browser_assert("window.NemoTimeline=require("+json.dumps(str(STATIC/'timeline.js'))+");"+r"""
const fields=['row_id','name_index','gpu_start_ns','gpu_end_ns','device_id','context_id','stream_id','correlation_id','api_start_ns','api_end_ns','thread_index','phase_index','api_name_index'];
update.timeline={schema_version:1,available:true,origin:'selected_host_nvtx_start',host_duration_ns:200,span_duration_ns:220,order:'gpu_start_ns,row_id',total_events:1,returned_events:1,truncated:false,names:['raw_kernel<template>'],api_names:['cudaLaunchKernel'],threads:['12345678901234567'],phases:[{id:'forward',start_ns:0,end_ns:200}],fields,events:[[1,0,30,130,0,1,7,11,10,20,0,0,0]]};
source.snapshot({data:JSON.stringify(base)});
assert.ok(get('iteration-timeline').textContent.includes('Iteration Timeline'));
assert.ok(get('timeline-event-detail').textContent.includes('raw_kernel<template>'));
assert.ok(get('timeline-event-detail').textContent.includes('Python 호출 프레임 미수집'));
get('inspector-view').value='capture';get('inspector-view').fire('change');
assert.ok(!get('nsight-content').textContent.includes('CPU thread'));
""")


def test_run_stage_scope_missing_capture_and_same_source_preserves_inspector_dom():
    from test_inspector_frontend import browser_assert
    browser_assert(r"""
assert.ok(get('nsight-content').textContent.includes('Historical capture'));
assert.ok(get('nsight-content').textContent.includes('fixture'));
const old=get('inspector-card');
base.generated_at='2026-10-13T00:00:05Z';
source.snapshot({data:JSON.stringify(base)});
assert.equal(get('inspector-card'),old,'unchanged archive must not rebuild interactions');
get('profile-stage-select').value='pretrain';get('profile-stage-select').fire('change');
assert.ok(get('profile-stage-status').textContent.includes('Reference / history'));
base.runs.push({id:'sft-new',stage:'sft'});
source.snapshot({data:JSON.stringify(base)});
get('run-select').value='sft-new';get('run-select').fire('change');
assert.equal(get('profile-stage-select').value,'sft');
assert.ok(get('profile-stage-status').textContent.includes('sft-new'));
assert.ok(get('nsight-content').textContent.includes('Unavailable capture'));
assert.ok(!get('nsight-content').textContent.includes('gemm<hostile>'));
base.stages.push({id:'mid',name:'MID',run_ids:[]});source.snapshot({data:JSON.stringify(base)});
const stage=get('stages').children.at(-1);stage.fire('click');
assert.equal(get('profile-stage-select').value,'mid');
assert.ok(get('nsight-content').textContent.includes('Unavailable capture'));
get('profile-stage-select').value='pretrain';get('profile-stage-select').fire('change');
assert.ok(get('profile-stage-status').textContent.includes('Reference / history'));
assert.ok(get('nsight-content').textContent.includes('Historical capture'));
""")


def test_kernel_symbol_parts_keep_raw_tokens_and_confidence_separate():
    from test_inspector_frontend import browser_assert
    browser_assert(r"""
update.kernels[0].symbol_analysis={family:'NCCL',parts:[{token:'RING',label:'알고리즘',meaning:'링 형태로 rank를 연결',confidence:'documented',sources:[]},{token:'private_token',label:'미공개 규약',meaning:'확정 불가',confidence:'unknown',sources:[]}],warnings:['shape / Python caller 증거 아님']};
source.snapshot({data:JSON.stringify(base)});
assert.ok(get('inspector-card').textContent.includes('커널명 토큰 해석'));
assert.ok(get('inspector-card').textContent.includes('RING'));
assert.ok(get('inspector-card').textContent.includes('공개 근거 확인'));
assert.ok(get('inspector-card').textContent.includes('미확인'));
assert.equal(get('inspector-card').children.find(c=>c.tagName==='h3').textContent,kernel.name);
""")


TIMELINE_HARNESS = r"""
const timeline=require(TIMELINE_PATH);
const nodes=new Map(), painted=[], labels=[];
const ctx={fillRect(...args){painted.push(args);},fillText(t){labels.push(t);},beginPath(){},moveTo(){},lineTo(){},stroke(){},save(){},restore(){},rect(){},clip(){}};
class Node {
 constructor(tag){this.tagName=tag;this.children=[];this.attrs={};this.handlers={};this.value='';this.own='';this.style={};this.hidden=false;}
 append(...children){this.children.push(...children);}
 replaceChildren(...children){this.children=children;this.own='';}
 setAttribute(k,v){this.attrs[k]=String(v);if(k==='id')nodes.set(v,this);}
 addEventListener(k,fn){this.handlers[k]=fn;}
 fire(k,e={}){this.handlers[k]?.({target:this,preventDefault(){},...e});}
 get textContent(){return this.own+this.children.map(c=>c.textContent).join(' ');}
 set textContent(v){this.own=String(v);this.children=[];}
 getContext(){return ctx;}
 getBoundingClientRect(){return {left:0,top:0,width:1400,height:this.height};}
}
const doc={createElement:tag=>new Node(tag)}, target=new Node('section'), get=id=>nodes.get(id);
const flatten=n=>[n,...n.children.flatMap(flatten)];
const fields=['row_id','name_index','gpu_start_ns','gpu_end_ns','device_id','context_id','stream_id','correlation_id','api_start_ns','api_end_ns','thread_index','phase_index','api_name_index'];
const update={rank:0,update:3,timeline:{schema_version:1,available:true,host_duration_ns:200e6,span_duration_ns:220e6,total_events:120,returned_events:120,truncated:false,names:['exact_raw_kernel<template>','ncclDevKernel_AllGather_RING_LL'],api_names:['cudaLaunchKernel'],threads:['12345678901234567'],phases:[{id:'forward',start_ns:0,end_ns:200e6}],fields,events:Array.from({length:120},(_,i)=>[i,i%2,i*1e6,i*1e6+800000,0,1,i%2+7,i,i*1e6-10000,i*1e6-5000,0,0,0])}};
const options={document:doc,key:'fixture/archive/0/3'};
"""


def timeline_assert(assertions):
    node_assert(TIMELINE_HARNESS.replace('TIMELINE_PATH',json.dumps(str(STATIC/'timeline.js'))) + assertions)


def test_timeline_same_trace_preserves_dom_but_new_data_changes_geometry_and_table():
    timeline_assert(r"""
timeline.render(target,update,options);
const canvas=get('iteration-timeline-canvas');
get('timeline-page-next').fire('click');
const table=flatten(target).find(n=>n.tagName==='tbody');
assert.equal(table.children.length,20);
const geometry=JSON.stringify(painted);
timeline.render(target,JSON.parse(JSON.stringify(update)),options);
assert.equal(get('iteration-timeline-canvas'),canvas,'polling unchanged trace keeps the canvas');
assert.equal(flatten(target).find(n=>n.tagName==='tbody'),table);
update.timeline.names[0]='changed_raw_symbol';
update.timeline.events[100][2]+=500000;update.timeline.events[100][3]+=500000;
painted.length=0;timeline.render(target,update,options);
assert.notEqual(JSON.stringify(painted),geometry);
assert.ok(target.textContent.includes('changed_raw_symbol'));
assert.equal(flatten(target).find(n=>n.tagName==='tbody').children.length,20,'page remains bounded and preserved');
""")


def test_pointer_tooltip_wheel_anchor_drag_pan_filters_and_linked_api():
    timeline_assert(r"""
let selected=null;options.onSelectEvent=e=>selected=e;
timeline.render(target,update,options);
const canvas=get('iteration-timeline-canvas'),note=get('timeline-window-note');
canvas.fire('pointermove',{clientX:262,clientY:110});
const tip=get('timeline-tooltip');
assert.equal(tip.hidden,false,'pointer tooltip, not canvas native title');
assert.ok(tip.textContent.includes('exact_raw_kernel<template>'));
assert.ok(tip.textContent.includes('API Enqueue'));
canvas.fire('click',{clientX:262,clientY:110});
assert.equal(selected.id,0);assert.ok(get('timeline-event-detail').textContent.includes('correlationId'));
assert.ok(get('timeline-event-detail').textContent.includes('Linked API'));
const before=Number(note.attrs['data-end-ns'])-Number(note.attrs['data-start-ns']);
const anchor=before*(800-260)/1110;
canvas.fire('wheel',{clientX:800,clientY:110,deltaY:-1200});
const start=Number(note.attrs['data-start-ns']),end=Number(note.attrs['data-end-ns']);
assert.ok(end-start<before);
assert.ok(Math.abs(start+(end-start)*(800-260)/1110-anchor)<1,'wheel anchored at pointer timestamp');
canvas.fire('pointerdown',{clientX:800,clientY:110,pointerId:1,button:0});
canvas.fire('pointermove',{clientX:900,clientY:110,pointerId:1});
canvas.fire('pointerup',{pointerId:1});
assert.ok(Number(note.attrs['data-start-ns'])<start,'drag pans the time window');
get('timeline-zoom').value='full';get('timeline-zoom').fire('change');
canvas.fire('pointermove',{clientX:262,clientY:110});assert.equal(tip.hidden,false);
const search=get('timeline-search');search.value='ncclDevKernel';search.fire('input');
assert.equal(tip.hidden,true,'filtered or moved geometry invalidates old hover');
assert.equal(flatten(target).find(n=>n.tagName==='tbody').children.length,60);
const stream=get('timeline-stream');stream.value='[0,1,7]';stream.fire('change');
assert.equal(flatten(target).find(n=>n.tagName==='tbody').children.length,0);
search.value='';search.fire('input');
assert.equal(flatten(target).find(n=>n.tagName==='tbody').children.length,60);
assert.ok(note.textContent.includes('Visible events'));
stream.value='';stream.fire('change');
const zoom=get('timeline-zoom');zoom.value='1';zoom.fire('change');
assert.ok(labels.includes('exact_raw_kernel<template>'),'zoomed bars use exact raw names');
canvas.fire('pointerleave');assert.equal(tip.hidden,true);
""")


def test_metric_titles_and_timeline_headers_are_english_without_renaming_symbols():
    html=(STATIC/'index.html').read_text()
    css=(STATIC/'style.css').read_text()
    assert '.timeline-tooltip{position:fixed' in css
    assert 'pointer-events:none' in css
    assert 'touch-action:none' in css
    for title in ['Loss', 'Learning Rate', 'Grad Norm', 'Tokens/s', 'Processed Tokens', 'Loss-bearing Tokens', 'Elapsed', 'ETA', 'Step']:
        assert title in html
    from test_inspector_frontend import browser_assert
    browser_assert(r"""
for(const label of ['Loss','Learning Rate','Grad Norm','Processed Tokens','Loss-bearing Tokens'])assert.ok(get('chart-grid').textContent.includes(label),label);
assert.ok(get('host-facts').textContent.includes('CPU Utilization'));
assert.ok(get('run-facts').textContent.includes('Parameters'));
const flatten=n=>[n,...n.children.flatMap(flatten)];
assert.ok(flatten(get('nsight-content')).some(n=>n.tagName==='th' && n.textContent==='CUDA Duration'));
""")
    timeline_assert(r"""
timeline.render(target,update,options);
const headers=flatten(target).filter(n=>n.tagName==='th').map(n=>n.textContent);
for(const label of ['GPU Start ms','Duration µs','API Enqueue ms','Raw Kernel Name'])assert.ok(headers.includes(label),label);
assert.ok(target.textContent.includes('exact_raw_kernel<template>'));
""")


def test_rlvr_rollout_only_status_and_english_numeric_metrics():
    from test_inspector_frontend import browser_assert
    browser_assert(r"""
base.runs.push({id:'native-rl',stage:'rlvr',status:'observed',training_status:'blocked',training_scope:'rollout_only',native_training_completed:false,metrics:{reward_mean:0,extra:{optimizer_updates:0,private_text:'DO_NOT_DISPLAY'}},rollout_summary:{unique_rollouts:4,all_rank_sampling_calls:512,optimizer_updates:0}});
source.snapshot({data:JSON.stringify(base)});
get('run-select').value='native-rl';get('run-select').fire('change');
assert.ok(get('run-status').textContent.includes('Blocked'));
assert.ok(get('run-status').textContent.includes('rollout_only'));
assert.equal(get('profile-stage-select').value,'rlvr');
const extra=get('run-extra-metrics');
for(const text of ['Rollouts','Optimizer Updates','Reward','4','512','0'])assert.ok(extra.textContent.includes(text),text);
assert.ok(!extra.textContent.includes('DO_NOT_DISPLAY'));
assert.ok(extra.textContent.includes('not training completion'));
assert.ok(get('nsight-content').textContent.includes('Unavailable capture'));
""")


def test_selection_does_not_cause_poll_to_rebuild_same_capture():
    from test_inspector_frontend import browser_assert
    browser_assert(r"""
update.kernels.push({...kernel,name:'selected_second'});
source.snapshot({data:JSON.stringify(base)});
const flatten=n=>[n,...n.children.flatMap(flatten)];
const buttons=flatten(get('nsight-content')).filter(n=>n.tagName==='button');
buttons.find(n=>n.textContent==='selected_second').fire('click');
const card=get('inspector-card'),controls=get('inspector-update');
source.snapshot({data:JSON.stringify(base)});
assert.equal(get('inspector-card'),card,'selected raw symbol does not invalidate archive scaffold');
assert.equal(get('inspector-update'),controls);
assert.ok(card.textContent.includes('selected_second'));
update.kernels[1].name='genuinely_new_symbol';source.snapshot({data:JSON.stringify(base)});
assert.notEqual(get('inspector-update'),controls,'new data invalidates capture presentation');
assert.ok(get('nsight-content').textContent.includes('genuinely_new_symbol'));
""")


def test_timeline_separates_gpu_start_order_from_api_enqueue_order():
    node_assert("const timeline=require("+json.dumps(str(STATIC/'timeline.js'))+");"+r"""
const fields=['row_id','name_index','gpu_start_ns','gpu_end_ns','device_id','context_id','stream_id','correlation_id','api_start_ns','api_end_ns','thread_index','phase_index','api_name_index'];
const update={timeline:{schema_version:1,available:true,origin:'selected_host_nvtx_start',host_duration_ns:200,span_duration_ns:220,order:'gpu_start_ns,row_id',total_events:3,returned_events:3,truncated:false,names:['raw<one>','raw_two'],api_names:['cudaLaunchKernel'],threads:['12345678901234567'],phases:[{id:'forward',start_ns:0,end_ns:200}],fields,events:[[1,0,30,130,0,1,7,11,10,20,0,0,0],[2,1,20,80,0,1,8,12,15,19,0,0,0],[3,1,140,220,0,1,8,12,15,19,0,0,0]]}};
const view=timeline.prepareTimeline(update,'forward');
assert.equal(view.available,true);
assert.deepEqual(view.gpuOrder.map(e=>e.id),[2,1,3]);
assert.deepEqual(view.apiOrder.map(e=>e.id),[1,2,3]);
assert.equal(view.events[0].name,'raw<one>');
assert.equal(view.events[0].thread,'12345678901234567');
assert.equal(view.apiEvents.length,2);
assert.equal(view.gpuLanes.length,2);
assert.equal(view.span,220);
assert.equal(timeline.prepareTimeline(update,'optimizer').events.length,0);
assert.equal(timeline.prepareTimeline({}).available,false);
update.timeline.events[0][6]=null;
assert.ok(timeline.prepareTimeline(update).gpuLanes.some(l=>l.label.includes('미수집')));
""")
