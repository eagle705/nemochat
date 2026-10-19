/* Recorded kernel/API chronology, not a CUDA dependency or critical-path graph. */
(function(root) {
  'use strict';
  const FIELDS=['row_id','name_index','gpu_start_ns','gpu_end_ns','device_id','context_id','stream_id','correlation_id','api_start_ns','api_end_ns','thread_index','phase_index','api_name_index'];
  function prepareTimeline(update,phase='') {
    const t=update?.timeline;
    if (!t || t.available!==true || t.schema_version!==1 || JSON.stringify(t.fields)!==JSON.stringify(FIELDS)) return {available:false,events:[]};
    const phases=t.phases || [], events=[];
    for (const a of t.events || []) {
      const p=a[11]===null ? null : phases[a[11]];
      if (phase && p?.id!==phase) continue;
      events.push({id:a[0],name:t.names[a[1]],start:a[2],end:a[3],device:a[4],context:a[5],stream:a[6],correlation:a[7],apiStart:a[8],apiEnd:a[9],thread:t.threads[a[10]],phase:p?.id || null,apiName:a[12]===null ? null : t.api_names[a[12]],gpuKey:JSON.stringify([a[4],a[5],a[6]])});
    }
    const gpuOrder=[...events].sort((a,b)=>a.start-b.start || a.id-b.id), apiOrder=[...events].sort((a,b)=>a.apiStart-b.apiStart || a.id-b.id);
    const groups=new Map(), lanes=new Map();
    for (const e of events) {
      if (!lanes.has(e.gpuKey)) lanes.set(e.gpuKey,{key:e.gpuKey,kind:'gpu',label:`GPU ${e.device} / ctx ${e.context ?? '미수집'} / stream ${e.stream ?? '미수집'}`});
      const key=JSON.stringify([e.thread,e.correlation,e.apiStart,e.apiEnd,e.apiName]);
      if (!groups.has(key)) groups.set(key,{...e,key,kernels:[]});
      groups.get(key).kernels.push(e.id);
    }
    const gpuLanes=[...lanes.values()].sort((a,b)=>a.label.localeCompare(b.label)), threads=[...new Set(events.map(e=>e.thread))];
    return {available:true,events,gpuOrder,apiOrder,apiEvents:[...groups.values()],gpuLanes,threads,phases,hostDuration:t.host_duration_ns,span:t.span_duration_ns,total:t.total_events,returned:t.returned_events,truncated:t.truncated};
  }
  const states=new Map(), mounted=new WeakMap();
  function render(target,update,options={}) {
    const signature=JSON.stringify([options.key,options.phase,update]);
    if(mounted.get(target)===signature)return;
    mounted.set(target,signature);
    const doc=options.document || target.ownerDocument || root?.document;
    const el=(tag,text,cls)=>{const n=doc.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;return n;};
    const view=prepareTimeline(update,options.phase || '');
    target.replaceChildren();target.className='iteration-timeline';
    target.append(el('h3','Iteration Timeline · Historical capture · Recorded GPU / CPU API / HOST NVTX'));
    if (!view.available) {target.append(el('p','이 캡처의 이벤트 타임라인 미수집 · 기존 집계를 실행 순서로 변환하지 않습니다.','panel-note'));return;}
    const key=options.key || `${update.rank}/${update.update}/${options.phase || ''}`;
    const state=states.get(key) || {order:'gpu',page:0,zoom:'full',position:0,lanePage:0,selected:null,search:'',stream:'',windowWidth:null};states.set(key,state);
    while(states.size>32)states.delete(states.keys().next().value);
    const ms=n=>(n/1e6).toLocaleString('en-US',{maximumFractionDigits:6});
    target.append(el('p',`${view.events.length.toLocaleString()}개 선택 kernel / 전체 ${view.total.toLocaleString()}개 · 가져온 ${view.returned.toLocaleString()}개${view.truncated?' · 이벤트 상한에 의해 잘림':''} · 동일 프로세스 kernel/API 상관관계. CPU API는 연결된 launch만 포함하며 전체 CPU 호출·memcpy·memset·의존성 그래프가 아닙니다.`,'panel-note'));
    target.append(el('p','GPU 시작순은 timestamp 정렬이며 서로 다른 stream의 의존 순서가 아닙니다. CPU enqueue와 실제 GPU 실행은 다르고 겹칠 수 있습니다. 빈 공간을 GPU idle이나 critical path로 확정하지 않습니다. 작은 이벤트는 최소 1px로 그리지만 실제 시간은 상세/목록에 보존합니다.','panel-note'));
    const controls=el('div',undefined,'timeline-controls');target.append(controls);
    function selectControl(id,title,choices,value,fn) {const label=el('label',title),s=el('select');s.setAttribute('id',id);s.setAttribute('aria-label',title);for(const [v,t] of choices){const o=el('option',t);o.value=v;s.append(o);}s.value=value;s.addEventListener('change',()=>fn(s.value));label.append(s);controls.append(label);return s;}
    const order=selectControl('timeline-order','Event order',[['gpu','GPU Start order'],['api','CPU API enqueue order']],state.order,v=>{state.order=v;state.page=0;drawTable();});
    const zoom=selectControl('timeline-zoom','Time window',[['full','Full iteration'],['100','100 ms'],['10','10 ms'],['1','1 ms'],['custom','Pointer zoom']],state.zoom,v=>{if(v==='custom')return;state.zoom=v;state.windowWidth=null;state.position=0;draw();});
    const searchLabel=el('label','Kernel-name search'),search=el('input');search.type='search';search.value=state.search;search.setAttribute('id','timeline-search');search.setAttribute('aria-label','Kernel-name search');search.addEventListener('input',()=>{state.search=search.value;state.page=0;drawTable();draw();});searchLabel.append(search);controls.append(searchLabel);
    selectControl('timeline-stream','GPU stream',[['','All streams'],...view.gpuLanes.map(l=>[l.key,l.label])],state.stream,v=>{state.stream=v;state.page=0;state.lanePage=0;drawTable();draw();});
    const matches=e=>(!state.search || e.name.toLowerCase().includes(state.search.toLowerCase())) && (!state.stream || e.gpuKey===state.stream);
    const pan=el('input');pan.type='range';pan.min='0';pan.max='1000';pan.step='1';pan.value=String(state.position);pan.setAttribute('id','timeline-pan');pan.setAttribute('aria-label','Timeline pan');pan.addEventListener('input',()=>{state.position=Number(pan.value);draw();});controls.append(pan);
    const lanePrev=el('button','Previous lanes'),laneNext=el('button','Next lanes'),laneCount=el('span');lanePrev.type=laneNext.type='button';lanePrev.addEventListener('click',()=>{state.lanePage=Math.max(0,state.lanePage-1);draw();});laneNext.addEventListener('click',()=>{state.lanePage++;draw();});controls.append(lanePrev,laneNext,laneCount);
    const wrap=el('div',undefined,'timeline-canvas-wrap'),canvas=el('canvas');canvas.width=1400;canvas.setAttribute('id','iteration-timeline-canvas');canvas.setAttribute('aria-label','HOST NVTX 구간 / CPU thread별 API / GPU device-context-stream별 kernel · 동일 시간 축');canvas.setAttribute('role','img');wrap.append(canvas);target.append(wrap);
    const ctx=canvas.getContext?.('2d') || null,windowNote=el('p',undefined,'panel-note');windowNote.setAttribute('id','timeline-window-note');target.append(windowNote);
    const tooltip=el('div',undefined,'timeline-tooltip');tooltip.setAttribute('id','timeline-tooltip');tooltip.setAttribute('role','tooltip');tooltip.hidden=true;target.append(tooltip);
    const detail=el('article',undefined,'timeline-event-detail');detail.setAttribute('id','timeline-event-detail');target.append(detail);
    const tableWrap=el('div',undefined,'table-wrap timeline-table'),table=el('table'),thead=el('thead'),head=el('tr'),tbody=el('tbody');
    for(const name of ['Order','Raw Kernel Name','GPU Start ms','Duration µs','API Enqueue ms','Device / Context / Stream','Phase'])head.append(el('th',name));thead.append(head);table.append(thead,tbody);tableWrap.append(table);target.append(tableWrap);
    const paging=el('div',undefined,'timeline-controls'),prev=el('button','Previous 100'),next=el('button','Next 100'),count=el('span');prev.type=next.type='button';next.setAttribute('id','timeline-page-next');prev.addEventListener('click',()=>{state.page=Math.max(0,state.page-1);drawTable();});next.addEventListener('click',()=>{state.page++;drawTable();});paging.append(prev,count,next);target.append(paging);
    const gpuIndex=new Map(view.gpuOrder.map((e,i)=>[e.id,i+1])),apiIndex=new Map(view.apiOrder.map((e,i)=>[e.id,i+1])),eventById=new Map(view.events.map(e=>[e.id,e]));
    let hits=[];
    function eventDetail(e,notify=false) {
      state.selected=e.id;detail.replaceChildren();detail.append(el('h3',e.name),el('p',`GPU Start order #${gpuIndex.get(e.id)} / CPU API enqueue 순 #${apiIndex.get(e.id)} · 이 순번은 stream 간 의존 관계가 아닙니다.`,'panel-note'));
      const dl=el('dl',undefined,'facts');
      for(const [k,v] of [['GPU Interval',`${ms(e.start)} → ${ms(e.end)} ms · ${e.start} → ${e.end} ns`],['GPU Duration',`${ms(e.end-e.start)} ms`],['API Enqueue',`${e.apiName || '명칭 미수집'} · ${ms(e.apiStart)} → ${ms(e.apiEnd)} ms`],['Enqueue → GPU Start',`${ms(e.start-e.apiStart)} ms · timestamp 차이; 큐 원인 미확인`],['GPU / context / stream',`${e.device} / ${e.context ?? '미수집'} / ${e.stream ?? '미수집'}`],['CPU globalTid',e.thread],['CUDA runtime correlationId',e.correlation],['NVTX phase',e.phase || 'API containment 매핑 미확인'],['Source Evidence','프로세스/시간/API 상관관계 · 정확한 Python 호출 프레임 미수집']])dl.append(el('dt',k),el('dd',v));
      detail.append(dl,el('p',`Linked API highlight: ${e.apiName || 'Unknown API'} / correlationId ${e.correlation} / thread ${e.thread} · timestamp correlation, not a dependency edge.`,'panel-note'));if(notify)options.onSelectEvent?.(e);
    }
    function drawTable() {
      tbody.replaceChildren();const sorted=(state.order==='api'?view.apiOrder:view.gpuOrder).filter(matches);
      state.page=Math.min(Math.max(0,Math.ceil(sorted.length/100)-1),state.page);
      const start=state.page*100;
      for(const [i,e] of sorted.slice(start,start+100).entries()) {const tr=el('tr'),td=el('td'),b=el('button',e.name);b.type='button';b.addEventListener('click',()=>{eventDetail(e,true);draw();});td.append(b);tr.append(el('td',start+i+1),td,el('td',ms(e.start)),el('td',(e.end-e.start)/1e3),el('td',ms(e.apiStart)),el('td',`${e.device} / ${e.context ?? '미수집'} / ${e.stream ?? '미수집'}`),el('td',e.phase || '미확인'));tbody.append(tr);}
      count.textContent=`${sorted.length?start+1:0}–${Math.min(start+100,sorted.length)} / ${sorted.length} · 전체 이벤트에서 100개씩 표시`;
      prev.disabled=state.page===0;next.disabled=start+100>=sorted.length;
    }
    let windowStart=0,windowWidth=1,windowBase=0,windowSpan=1,drag=null,suppressClick=false;
    function draw() {
      tooltip.hidden=true;
      const chosenPhase=view.phases.find(p=>p.id===options.phase),base=chosenPhase?.start_ns || 0;
      const upper=chosenPhase ? Math.max(chosenPhase.end_ns,...view.events.map(e=>e.end)) : view.span;
      const span=Math.max(1,upper-base),width=Math.min(span,state.windowWidth || (state.zoom==='full'?span:Number(state.zoom)*1e6)),start=base+Math.max(0,span-width)*state.position/1000,end=start+width;
      windowStart=start;windowWidth=width;windowBase=base;windowSpan=span;pan.value=String(state.position);
      pan.disabled=width===span;
      const lanes=[...view.threads.map((t,i)=>({kind:'api',key:t,label:`CPU thread ${i} · ${t}`})),...view.gpuLanes];
      state.lanePage=Math.min(state.lanePage,Math.max(0,Math.ceil(lanes.length/24)-1));const shown=lanes.slice(state.lanePage*24,state.lanePage*24+24);
      lanePrev.disabled=state.lanePage===0;laneNext.disabled=(state.lanePage+1)*24>=lanes.length;laneCount.textContent=`${shown.length} / ${lanes.length} lanes · page ${state.lanePage+1}`;
      const visible=view.events.filter(e=>matches(e) && e.end>=start && e.start<=end).length;
      windowNote.setAttribute('data-start-ns',start);windowNote.setAttribute('data-end-ns',end);
      windowNote.textContent=`Visible events: ${visible.toLocaleString()} / ${view.events.filter(matches).length.toLocaleString()} filtered / ${view.events.length.toLocaleString()} selected kernels · Window ${ms(start)}–${ms(end)} ms · HOST duration ${ms(view.hostDuration)} ms / kernel tail ${ms(view.span)} ms · Wheel: zoom at pointer · Drag: pan · CPU API and GPU intervals are distinct.`;
      if(!ctx)return;
      canvas.height=90+(shown.length+1)*34;ctx.fillStyle='#0c100f';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.font='12px "NemoChat Korean", monospace';hits=[];
      const left=260,right=1370,pixel=ns=>left+(ns-start)/width*(right-left);
      for(let i=0;i<=8;i++){const x=left+(right-left)*i/8;ctx.strokeStyle='#29352e';ctx.beginPath();ctx.moveTo(x,25);ctx.lineTo(x,canvas.height-15);ctx.stroke();ctx.fillStyle='#9caea2';ctx.fillText(`${ms(start+width*i/8)} ms`,x-18,18);}
      ctx.fillStyle='#9caea2';ctx.fillText('HOST NVTX phases',8,50);
      for(const p of view.phases){if(p.end_ns<start || p.start_ns>end)continue;const x=Math.max(left,pixel(p.start_ns)),w=Math.max(1,Math.min(right,pixel(p.end_ns))-x);ctx.fillStyle='#605035';ctx.fillRect(x,34,w,25);ctx.save();ctx.beginPath();ctx.rect(x,34,w,25);ctx.clip();ctx.fillStyle='#e0b875';ctx.fillText(p.id,x+4,51);ctx.restore();}
      const selectedEvent=view.events.find(e=>e.id===state.selected);
      function bar(e,a,b,y,color,kind){if(b<start || a>end || !matches(e))return;const x=Math.max(left,pixel(a)),w=Math.max(1,Math.min(right,pixel(b))-x);const linked=kind==='api' && selectedEvent && e.thread===selectedEvent.thread && e.correlation===selectedEvent.correlation && e.apiStart===selectedEvent.apiStart && e.apiEnd===selectedEvent.apiEnd;ctx.fillStyle=state.selected===e.id || linked?'#eef2e8':color;ctx.fillRect(x,y+3,w,21);hits.push({e,x,w,y,kind});if(w>90){ctx.save();ctx.beginPath();ctx.rect(x,y+3,w,21);ctx.clip();ctx.fillStyle='#0c100f';ctx.fillText(kind==='api' ? e.apiName || 'Unknown API' : e.name,x+4,y+18);ctx.restore();}}
      shown.forEach((lane,i)=>{const y=68+i*34;ctx.fillStyle='#9caea2';ctx.fillText(lane.label.length>33?lane.label.slice(0,32)+'…':lane.label,8,y+18);ctx.strokeStyle='#1d2922';ctx.beginPath();ctx.moveTo(0,y+30);ctx.lineTo(right,y+30);ctx.stroke();
        if(lane.kind==='api')for(const e of view.apiEvents){if(e.thread!==lane.key)continue;const linked=e.kernels.map(id=>eventById.get(id)).find(matches);if(linked)bar(linked,e.apiStart,e.apiEnd,y,'#b49b68','api');}
        else for(const e of view.events){if(e.gpuKey===lane.key)bar(e,e.start,e.end,y,/nccl/i.test(e.name)?'#d0a55a':'#7bbc92','gpu');}
      });
    }
    const point=event=>{const rect=canvas.getBoundingClientRect();return {x:(event.clientX-rect.left)*canvas.width/rect.width,y:(event.clientY-rect.top)*canvas.height/rect.height};};
    function hit(event){const {x,y}=point(event);return hits.filter(h=>x>=h.x && x<=h.x+h.w && y>=h.y+3 && y<=h.y+24).sort((a,b)=>Math.abs(x-a.x-a.w/2)-Math.abs(x-b.x-b.w/2))[0];}
    function moveWindow(start,width){state.windowWidth=width;state.zoom='custom';zoom.value='custom';state.position=windowSpan>width?Math.max(0,Math.min(1000,(start-windowBase)/(windowSpan-width)*1000)):0;tooltip.hidden=true;draw();}
    canvas.addEventListener('click',event=>{if(suppressClick){suppressClick=false;return;}const h=hit(event);if(h){eventDetail(h.e,true);draw();}});
    canvas.addEventListener('wheel',event=>{const {x}=point(event);if(x<260 || x>1370)return;event.preventDefault();const fraction=(x-260)/1110,anchor=windowStart+windowWidth*fraction,width=Math.max(Math.min(1000,windowSpan),Math.min(windowSpan,windowWidth*Math.exp(Math.max(-2,Math.min(2,event.deltaY*.001)))));moveWindow(anchor-width*fraction,width);},{passive:false});
    canvas.addEventListener('pointerdown',event=>{if(event.button!==0)return;drag={x:point(event).x,start:windowStart,width:windowWidth};suppressClick=false;canvas.setPointerCapture?.(event.pointerId);});
    canvas.addEventListener('pointermove',event=>{if(drag){const delta=point(event).x-drag.x;if(Math.abs(delta)>3)suppressClick=true;moveWindow(drag.start-delta/1110*drag.width,drag.width);return;}const h=hit(event);tooltip.hidden=!h;if(!h)return;const e=h.e;tooltip.textContent=`${e.name}\n${h.kind==='api'?'CPU API':'GPU kernel'} · GPU Start ${ms(e.start)} ms / Duration ${ms(e.end-e.start)} ms\nAPI Enqueue ${e.apiName || 'Unknown API'} ${ms(e.apiStart)}–${ms(e.apiEnd)} ms\nDevice ${e.device} / context ${e.context ?? 'Unknown'} / stream ${e.stream ?? 'Unknown'} / correlationId ${e.correlation}\nRank ${update.rank} / Update ${update.update} / Phase ${e.phase || 'Unknown'} · Historical capture · source: ${options.source || 'see capture Source / path'}\nRuntime correlation only; Python caller unavailable.`;if(tooltip.style){tooltip.style.left=`${Math.max(8,Math.min(event.clientX+14,(root?.innerWidth || 1400)-480))}px`;tooltip.style.top=`${Math.max(8,Math.min(event.clientY+14,(root?.innerHeight || 900)-240))}px`;}});
    canvas.addEventListener('pointerup',event=>{drag=null;canvas.releasePointerCapture?.(event.pointerId);});
    canvas.addEventListener('pointercancel',()=>{drag=null;});
    canvas.addEventListener('pointerleave',()=>{tooltip.hidden=true;});
    drawTable();draw();doc.fonts?.ready?.then(()=>{if(canvas.isConnected)draw();});if(view.events.length)eventDetail(view.events.find(e=>e.id===state.selected) || view.gpuOrder[0]);else detail.append(el('p','선택 HOST phase에 연결된 kernel 이벤트 없음 · GPU idle이라고 추론하지 않습니다.','panel-note'));
  }
  const api={prepareTimeline,render};
  if (typeof module!=='undefined' && module.exports) module.exports=api;
  if (root) root.NemoTimeline=api;
})(typeof window!=='undefined' ? window : null);
