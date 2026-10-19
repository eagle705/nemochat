/* Read-only monitor. Backend facts stay distinct from UI-derived presentation. */
'use strict';
(function (global) {
  const MISSING = '미수집';
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  function validateSnapshot(value) {
    return object(value) && value.schema_version === 1 && Array.isArray(value.runs)
      && value.runs.every(run => object(run) && typeof run.id === 'string' && run.id.length > 0)
      && Array.isArray(value.stages) && value.stages.every(object)
      && Array.isArray(value.insights) && value.insights.every(object)
      && object(value.host) && object(value.collection) && object(value.profiling);
  }
  function formatNumber(value, digits = 3) {
    return finite(value) ? new Intl.NumberFormat('ko-KR', {maximumFractionDigits: digits}).format(value) : MISSING;
  }
  function formatDuration(value) {
    if (!finite(value) || value < 0) return MISSING;
    const seconds = Math.floor(value);
    if (seconds < 60) return `${seconds}초`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}분 ${seconds % 60}초`;
    return `${Math.floor(seconds / 3600)}시간 ${Math.floor(seconds % 3600 / 60)}분`;
  }
  function progressPercent(progress = {}) {
    return finite(progress.step) && finite(progress.total_steps) && progress.total_steps > 0
      ? Math.max(0, Math.min(100, progress.step / progress.total_steps * 100)) : null;
  }
  const PROFILE_STAGES = {pretrain: '사전학습', mid: '중간학습', sft: '지도 미세조정', long: '장문맥', rlvr: '검증 보상 학습', mopd: '다중 교사 증류'};
  function stageProfile(profiling, stage) {
    if (!object(profiling)) return null;
    if (stage === 'unscoped') return profiling;
    return object(profiling.stages) && object(profiling.stages[stage]) ? profiling.stages[stage] : null;
  }
  function createStore(notify = () => {}) {
    const state = {snapshot: null, pending: null, selected: null, paused: false, profileStage: 'pretrain', profilePinned: false};
    function syncProfileStage() {
      const run = state.snapshot?.runs.find(entry => entry && entry.id === state.selected);
      const stage = run?.stage === 'long_context' ? 'long' : run?.stage;
      if (!state.profilePinned && Object.hasOwn(PROFILE_STAGES, stage)) state.profileStage = stage;
    }
    function apply(snapshot) {
      const previousRun=state.snapshot?.runs.find(run=>run.id===state.selected);
      state.snapshot = snapshot;
      if (!snapshot.runs.some(run => run && run.id === state.selected)) { state.selected = [...snapshot.runs].sort((a,b)=>(Date.parse(b.updated_at) || 0)-(Date.parse(a.updated_at) || 0))[0]?.id ?? null; state.profilePinned=false; }
      const nextRun=snapshot.runs.find(run=>run.id===state.selected);
      if(previousRun?.stage!==nextRun?.stage)state.profilePinned=false;
      syncProfileStage();
    }
    return {
      state,
      receive(snapshot) {
        if (!validateSnapshot(snapshot)) return false;
        const current = state.pending || state.snapshot;
        if (current && Date.parse(snapshot.generated_at) < Date.parse(current.generated_at)) return false;
        if (state.paused) state.pending = snapshot;
        else apply(snapshot);
        notify(state);
        return true;
      },
      select(id) {
        if (state.snapshot?.runs.some(run => run && run.id === id)) { state.profilePinned = false; state.selected = id; syncProfileStage(); notify(state); }
      },
      selectProfileStage(stage) {
        if (Object.hasOwn(PROFILE_STAGES, stage) || stage === 'unscoped') { state.profileStage = stage; state.profilePinned = true; notify(state); }
      },
      selectedRun() { return state.snapshot?.runs.find(run => run && run.id === state.selected) || null; },
      setPaused(paused) {
        state.paused = Boolean(paused);
        if (!state.paused && state.pending) { apply(state.pending); state.pending = null; }
        notify(state);
      }
    };
  }
  function chartSeries(history, key, limit = 0) {
    const steps = new Map();
    for (const point of Array.isArray(history) ? history : []) {
      if (object(point) && finite(point.step)) steps.set(point.step, {step: point.step, value: finite(point[key]) ? point[key] : null});
    }
    const points = [...steps.values()].sort((a, b) => a.step - b.step);
    return limit > 0 ? points.slice(-limit) : points;
  }
  function createTransport(deps, onSnapshot, onStatus) {
    let source = null, timer = null, controller = null, stopped = true, live = false, inFlight = false, interval = 5000;
    const clearPoll = () => { if (timer !== null) deps.clearTimeout(timer); timer = null; };
    function accept(value) {
      if (!validateSnapshot(value)) { onStatus('invalid'); return false; }
      const seconds = value.collection.interval_seconds;
      interval = finite(seconds) ? Math.min(60000, Math.max(2000, seconds * 1000)) : 5000;
      onSnapshot(value);
      return true;
    }
    function schedule() {
      if (stopped || live || timer !== null) return;
      timer = deps.setTimeout(() => { timer = null; poll(); }, interval);
    }
    async function poll() {
      if (stopped || inFlight) return;
      inFlight = true;
      let deadline = null;
      try {
        controller = deps.AbortController ? new deps.AbortController() : null;
        if (controller) deadline = deps.setTimeout(() => controller?.abort(), 10000);
        const response = await deps.fetch('/api/snapshot', {cache: 'no-store', signal: controller?.signal});
        if (!response.ok) throw new Error('Snapshot request failed');
        const value = await response.json();
        if (!stopped && accept(value) && !live) onStatus('polling');
      } catch (_error) { if (!stopped && !live) onStatus('offline'); }
      finally {
        if (deadline !== null) deps.clearTimeout(deadline);
        controller = null; inFlight = false; schedule();
      }
    }
    return {
      start() {
        if (!stopped) return;
        stopped = false; onStatus('connecting');
        if (deps.EventSource) {
          try {
            source = new deps.EventSource('/api/events');
            source.onopen = () => { if (stopped) return; live = true; clearPoll(); onStatus('live'); };
            source.addEventListener('snapshot', event => {
              if (stopped) return;
              try {
                if (accept(JSON.parse(event.data))) { live = true; clearPoll(); onStatus('live'); }
                else { live = false; schedule(); }
              } catch (_error) { live = false; onStatus('invalid'); schedule(); }
            });
            source.onerror = () => {
              if (stopped) return;
              live = false; onStatus('reconnecting');
              schedule();
            };
          } catch (_error) { onStatus('reconnecting'); }
        }
        poll();
      },
      stop() { stopped = true; live = false; clearPoll(); controller?.abort(); source?.close(); source = null; }
    };
  }
  function kernelMeaning(name) {
    const symbol = typeof name === 'string' ? name : '';
    const references = [
      [/nccl.*allgather/i, 'NCCL AllGather: 각 rank 입력을 rank 순서로 모아 모든 rank에 전달'],
      [/nccl.*allreduce/i, 'NCCL AllReduce: rank 간 sum/min/max 등의 축약 결과를 모든 rank에 전달'],
      [/nccl.*reducescatter/i, 'NCCL ReduceScatter: 축약 결과의 조각을 rank별로 분배'],
      [/nccl.*alltoall/i, 'NCCL AlltoAll: 모든 rank 사이에 서로 다른 조각 교환'],
      [/gemm|matmul/i, 'GEMM / matmul: 행렬 곱 계열 명칭 · 학습 레이어 미확인'],
      [/FillFunctor/i, 'FillFunctor: 값 채우기 계열 명칭 · 특정 값이나 Python 호출을 증명하지 않음'],
      [/elementwise|vectorized_elementwise|pointwise/i, '원소별 연산 계열 명칭 · 실제 수식 미확인'],
      [/softmax/i, 'Softmax 계열 명칭 · 축 / shape 미확인'],
      [/layer.?norm|rms.?norm|normalization/i, '정규화 계열 명칭 · 방식 / shape 미확인']
    ];
    const meaning = references.find(([pattern]) => pattern.test(symbol))?.[1] || '미분류 · 알려진 명칭과 매칭되지 않음';
    return `${meaning} · 명칭 기반 참고 설명 / 소스 미확인 · 단계 / shape / 병목 추론 없음`;
  }
  function inspectorScope(nsight = {}, choice = {}) {
    const inspector = nsight.inspector;
    const updates = Array.isArray(inspector?.updates) ? inspector.updates.filter(object) : [];
    if (inspector?.available !== true || choice.view === 'capture') return {view: 'capture', kernels: Array.isArray(nsight.kernels) ? nsight.kernels.filter(object) : [], updates: []};
    const rank = updates.some(u => String(u.rank) === String(choice.rank)) ? String(choice.rank) : String(updates[0]?.rank ?? '');
    const ranked = updates.filter(u => String(u.rank) === rank);
    const update = ranked.find(u => String(u.update) === String(choice.update)) || ranked[0];
    const phases = Array.isArray(update?.phases) ? update.phases.filter(object) : [];
    const phase = choice.phase ? phases.find(p => p.id === choice.phase) : null;
    const record = choice.phase ? phase : update;
    return {view: 'measured', rank, update, phase, phases, updates, kernels: Array.isArray(record?.kernels) ? record.kernels.filter(object) : [], scope: inspector.scope};
  }
  const api = {validateSnapshot, formatNumber, formatDuration, progressPercent, createStore, chartSeries, createTransport, stageProfile, kernelMeaning, inspectorScope};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.TrainingMonitor = api;
  if (!global.document) return;

  const doc = global.document;
  const byId = id => doc.getElementById(id);
  const list = value => Array.isArray(value) ? value : [];
  const text = value => value === null || value === undefined || value === '' ? MISSING : (object(value) || Array.isArray(value) ? JSON.stringify(value) : String(value));
  const setText = (id, value) => { byId(id).textContent = text(value); };
  function element(tag, value, className) {
    const node = doc.createElement(tag);
    if (value !== undefined) node.textContent = text(value);
    if (className) node.className = className;
    return node;
  }
  function empty(target, message) { target.replaceChildren(element('p', message, 'empty')); }
  function facts(target, entries) {
    target.replaceChildren();
    for (const [label, value] of entries) target.append(element('dt', label), element('dd', value));
  }
  function table(target, headers, rows, numericColumns = []) {
    const wrap = element('div', undefined, 'table-wrap');
    const grid = element('table');
    const head = element('thead'), headRow = element('tr'), body = element('tbody');
    for (const label of headers) { const cell = element('th', label); cell.scope = 'col'; headRow.append(cell); }
    head.append(headRow);
    for (const row of rows) {
      const tr = element('tr');
      row.forEach((value, index) => tr.append(element('td', value, numericColumns.includes(index) ? 'number' : '')));
      body.append(tr);
    }
    grid.append(head, body); wrap.append(grid); target.append(wrap);
  }
  const statuses = {completed: 'Completed · recorded scope', failed: 'Failed', stale: 'Stale', observed: 'Observed', running: 'Running', blocked: 'Blocked', rollout_only: 'Rollout only', unknown: 'Unknown'};
  const statusText = status => statuses[status] || statuses.unknown;
  const scientific = value => finite(value) ? (value === 0 ? '0' : value.toExponential(3)) : MISSING;
  const numberUnit = (value, unit, digits = 2) => finite(value) ? `${formatNumber(value, digits)} ${unit}` : MISSING;
  const dateText = value => typeof value === 'string' && Number.isFinite(Date.parse(value)) ? new Date(value).toLocaleString('ko-KR', {hour12: false}) : MISSING;
  function ageSeconds(value) {
    const timestamp = Date.parse(value);
    return Number.isFinite(timestamp) ? Math.max(0, (Date.now() - timestamp) / 1000) : null;
  }
  const store = createStore(render);
  let range = 100, renderedSnapshot = null, renderedRun = null, renderedProfileStage = null, renderedRange = null, renderedProfilePinned = null;
  const chartPositions = new Map();
  const openDetails = new Set();
  function profileDetails(key, title) {
    const item = element('details', undefined, 'profile-item');
    const identity = `${store.state.profileStage}:${key}`;
    item.open = openDetails.has(identity);
    item.addEventListener('toggle', () => {
      if (item.open) {
        openDetails.add(identity);
        if (openDetails.size > 500) openDetails.delete(openDetails.values().next().value);
      } else openDetails.delete(identity);
    });
    item.append(element('summary', title));
    return item;
  }
  function render(state) {
    byId('pause-button').textContent = state.paused ? '표시 재개' : '표시 일시정지';
    byId('pause-button').setAttribute('aria-pressed', String(state.paused));
    const notice = byId('display-notice');
    notice.hidden = !state.paused;
    notice.textContent = state.pending ? '표시 일시정지 중 · 수집은 계속됩니다. 재개하면 최신 스냅샷을 표시합니다.' : '표시 일시정지 중 · 학습이나 수집을 멈추지 않습니다.';
    const snapshot = state.snapshot;
    if (!snapshot) return;
    updateFreshness();
    const run = store.selectedRun();
    if (renderedSnapshot === snapshot && renderedRun === state.selected && renderedRange === range && renderedProfileStage === state.profileStage && renderedProfilePinned === state.profilePinned) return;
    const activeChart = doc.activeElement?.classList.contains('chart-svg') ? doc.activeElement.id : null;
    renderedSnapshot = snapshot; renderedRun = state.selected; renderedRange = range; renderedProfileStage = state.profileStage; renderedProfilePinned = state.profilePinned;
    byId('load-notice').hidden = true;
    const picker = byId('run-select'); picker.replaceChildren();
    const runs = list(snapshot.runs).filter(object);
    if (!runs.length) picker.append(element('option', '관측 실행 없음'));
    for (const entry of runs) {
      const option = element('option', `${text(entry.name || entry.id)} · ${statusText(entry.status)}`);
      option.value = entry.id; picker.append(option);
    }
    picker.disabled = !runs.length; picker.value = state.selected || '';
    setText('run-count', runs.length);
    renderStages(snapshot.stages, run);
    setText('run-title', run?.name || run?.id || '관측 실행 없음');
    setText('run-source', run ? `출처 ${text(run.source)} · 기록 시각 ${dateText(run.updated_at)} · ID ${text(run.id)}` : '설정된 경로에서 관측 가능한 학습 아티팩트가 없습니다.');
    const runStatus=run?.training_status || run?.status;
    setText('run-status', `${statusText(runStatus)}${run?.training_scope ? ` · ${run.training_scope}` : ''}`); byId('run-status').dataset.state = runStatus || 'unknown';
    const metrics = run?.metrics || {}, timing = run?.timing || {}, progress = run?.progress || {}, tokens = run?.tokens || {};
    setText('metric-loss', formatNumber(metrics.loss, 5)); setText('metric-lr', scientific(metrics.learning_rate));
    setText('metric-grad', formatNumber(metrics.grad_norm, 5)); setText('metric-speed', formatNumber(timing.tokens_per_second, 2));
    setText('speed-scope', `범위: ${text(timing.scope)}`);
    setText('step-count', `${formatNumber(progress.step, 0)} / ${formatNumber(progress.total_steps, 0)}`);
    const percent = progressPercent(progress);
    if (percent === null) { byId('step-progress').removeAttribute('value'); byId('step-progress').setAttribute('aria-valuetext', '목표 스텝 미수집'); }
    else { byId('step-progress').value = percent; byId('step-progress').setAttribute('aria-valuetext', `${formatNumber(percent, 1)}%`); }
    setText('elapsed', formatDuration(timing.elapsed_seconds)); setText('eta', formatDuration(timing.eta_seconds));
    setText('eta-scope', `범위: ${text(timing.scope)} · 전체 파이프라인 ETA 아님`);
    for (const [id, key] of [['tokens-processed', 'processed'], ['tokens-loss', 'loss_bearing'], ['tokens-source', 'source_read'], ['tokens-target', 'target']]) setText(id, formatNumber(tokens[key], 0));
    const model = run?.model || {}, data = run?.data || {};
    facts(byId('run-facts'), [['Model', model.name], ['Total Parameters', formatNumber(model.parameters_total, 0)], ['Active Parameters', formatNumber(model.parameters_active, 0)], ['Precision', model.precision], ['Container Elapsed', formatDuration(run?.timing?.container_elapsed_seconds)], ['Dataset', data.name], ['Kind', data.kind], ['Data Scope', data.scope], ['Training Stage', PROFILE_STAGES[run?.stage] || run?.stage]]);
    renderExtraMetrics(run);
    renderCharts(run?.history);
    renderHost(snapshot.host);
    const benchmarks = list(run?.benchmarks).filter(object); byId('benchmarks').replaceChildren();
    if (!benchmarks.length) empty(byId('benchmarks'), '선택한 실행의 벤치마크 기록이 없습니다. 점수나 성능을 추정하지 않습니다.');
    else table(byId('benchmarks'), ['Benchmark', 'Recorded Value', 'Measurement Scope'], benchmarks.map(item => [item.name, formatNumber(item.value, 5), item.scope]), [1]);
    renderProfiles(snapshot.profiling);
    renderEvidence(snapshot, run);
    if (activeChart) byId(activeChart)?.focus({preventScroll: true});
  }
  function renderExtraMetrics(run) {
    const target=byId('run-extra-metrics');target.replaceChildren();
    const labels={unique_rollouts:'Rollouts',rollouts:'Rollouts',optimizer_updates:'Optimizer Updates',reward_mean:'Reward',reward:'Reward',all_rank_sampling_calls:'Sampling Calls · all ranks',sampling_calls_per_rank:'Sampling Calls / Rank',rollout_rounds:'Rollout Rounds',scheduler_samples:'Scheduler Samples',teacher_kl:'Teacher KL'};
    const values={...run?.metrics?.extra,...run?.metrics?.extras,...run?.rollout_summary};
    if(finite(run?.metrics?.reward_mean))values.reward_mean=run.metrics.reward_mean;
    const entries=Object.entries(values).filter(([,v])=>finite(v));target.hidden=!entries.length;
    for(const [key,value] of entries){const item=element('div');item.append(element('span',labels[key] || key),element('strong',formatNumber(value,5)));target.append(item);}
    if(run?.training_scope==='rollout_only' || run?.native_training_completed===false)target.append(element('p','Rollout verification only · not training completion. Exit code 0 is not evidence of optimizer updates.','panel-note'));
  }
  function renderStages(stages, run) {
    const target = byId('stages'); target.replaceChildren();
    if (!list(stages).length) return empty(target, '단계 관측 기록 없음');
    list(stages).filter(object).forEach((stage, index) => {
      const observed = stage.status === 'observed';
      const button = element('button', undefined, `stage-button${observed ? ' observed' : ''}${run?.stage === stage.id ? ' active' : ''}`);
      button.type = 'button'; button.setAttribute('aria-pressed', String(run?.stage === stage.id));
      button.append(element('span', String(index + 1).padStart(2, '0'), 'stage-index'));
      const label = element('span'); label.append(element('strong', PROFILE_STAGES[stage.id] || stage.name || stage.id), element('small', observed ? `${list(stage.run_ids).length}개 실행 관측 · 완료 판정 아님` : '미관측')); button.append(label);
      button.addEventListener('click', () => {
        const id = list(stage.run_ids).find(item => store.state.snapshot.runs.some(entry => entry.id === item));
        if (id) store.select(id);
        else { store.selectProfileStage(stage.id === 'long_context' ? 'long' : stage.id); byId('display-notice').hidden = false; byId('display-notice').textContent = `${text(PROFILE_STAGES[stage.id] || stage.name)} 단계의 학습 실행 기록이 없습니다.`; }
      });
      target.append(button);
    });
  }
  function renderHost(host = {}) {
    const cpu = host.cpu || {}, memory = host.memory || {};
    setText('host-name', host.hostname);
    facts(byId('host-facts'), [['CPU', cpu.model], ['Logical CPUs', formatNumber(cpu.logical_count, 0)], ['CPU Utilization', numberUnit(cpu.utilization_percent, '%')], ['RAM Used / Total', `${numberUnit(finite(memory.used_bytes) ? memory.used_bytes / 1073741824 : null, 'GiB')} / ${numberUnit(finite(memory.total_bytes) ? memory.total_bytes / 1073741824 : null, 'GiB')}`]]);
    const target = byId('gpu-list'); target.replaceChildren();
    const gpus = list(host.gpus).filter(object);
    if (!gpus.length) empty(target, 'GPU 수집 정보 없음 · GPU 부재를 의미하지 않습니다.');
    for (const gpu of gpus) {
      const card = element('section', undefined, 'gpu-card'); card.append(element('h3', `GPU ${text(gpu.index)} · ${text(gpu.name)}`));
      const dl = element('dl', undefined, 'facts');
      facts(dl, [['GPU Utilization', numberUnit(gpu.utilization_percent, '%')], ['Memory MiB', `${formatNumber(gpu.memory_used_mb, 0)} / ${formatNumber(gpu.memory_total_mb, 0)}`], ['Temperature / Power', `${numberUnit(gpu.temperature_c, '°C')} / ${numberUnit(gpu.power_w, 'W')}`], ['UUID', gpu.uuid]]);
      card.append(dl); target.append(card);
    }
    setText('host-freshness', `하드웨어 수집: ${dateText(host.sampled_at)}`);
  }
  function svgElement(tag, attrs = {}, content) {
    const node = doc.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (content !== undefined) node.textContent = text(content);
    return node;
  }
  function renderCharts(history) {
    const target = byId('chart-grid'); target.replaceChildren();
    const charts = [
      {id: 'loss', title: 'Loss', keys: ['loss'], labels: ['Loss']},
      {id: 'lr', title: 'Learning Rate', keys: ['learning_rate'], labels: ['Learning Rate'], sci: true},
      {id: 'grad', title: 'Grad Norm', keys: ['grad_norm'], labels: ['Grad Norm']},
      {id: 'tokens', title: 'Recorded Tokens', keys: ['processed_tokens', 'loss_bearing_tokens'], labels: ['Processed Tokens', 'Loss-bearing Tokens']}
    ];
    for (const config of charts) {
      const series = config.keys.map(key => chartSeries(history, key, range));
      const points = series[0], values = series.flatMap(items => items.filter(item => item.value !== null).map(item => item.value));
      const card = element('div', undefined, 'chart-card'), heading = element('div', undefined, 'chart-heading');
      const value = element('span', values.length ? (config.sci ? scientific(points.filter(p => p.value !== null).at(-1)?.value) : formatNumber(points.filter(p => p.value !== null).at(-1)?.value, 4)) : MISSING, 'chart-value');
      heading.append(element('strong', config.title), value); card.append(heading);
      const svg = svgElement('svg', {viewBox: '0 0 360 164', class: 'chart-svg', id: `chart-${config.id}`, role: 'img', 'aria-label': `${config.title} 스텝별 기록. 좌우 방향키로 확인.`, tabindex: '0'});
      svg.append(svgElement('title', {}, `${config.title} · 실제 수집 기록만 표시`));
      const readout = element('div', undefined, 'chart-readout'); readout.setAttribute('aria-live', 'polite');
      if (!values.length) {
        svg.append(svgElement('text', {x: 180, y: 80, 'text-anchor': 'middle'}, '측정 기록 없음'));
        readout.textContent = '미수집 · 그래프를 생성하지 않습니다.';
        card.append(svg, readout); target.append(card); continue;
      }
      let min = Math.min(...values), max = Math.max(...values);
      if (min === max) { const padding = Math.abs(min) * 0.05 || 1; min -= padding; max += padding; }
      const first = points[0].step, last = points.at(-1).step;
      const x = step => first === last ? 192 : 44 + (step - first) / (last - first) * 304;
      const y = val => 127 - (val - min) / (max - min) * 110;
      for (const ratio of [0, 0.5, 1]) {
        const py = 17 + ratio * 110;
        svg.append(svgElement('line', {x1: 44, x2: 348, y1: py, y2: py, class: 'grid-line'}));
        const label = max - ratio * (max - min);
        svg.append(svgElement('text', {x: 38, y: py + 3, 'text-anchor': 'end'}, config.sci ? label.toExponential(1) : new Intl.NumberFormat('ko-KR', {notation: 'compact', maximumFractionDigits: 2}).format(label)));
      }
      svg.append(svgElement('text', {x: 44, y: 148}, `Step ${formatNumber(first, 0)}`), svgElement('text', {x: 348, y: 148, 'text-anchor': 'end'}, formatNumber(last, 0)));
      series.forEach((items, index) => {
        let path = '', connected = false;
        for (const point of items) {
          if (point.value === null) { connected = false; continue; }
          path += `${connected ? 'L' : 'M'}${x(point.step).toFixed(2)},${y(point.value).toFixed(2)} `;
          connected = true;
          svg.append(svgElement('circle', {cx: x(point.step), cy: y(point.value), r: items.length < 5 ? 2.5 : 1.1, fill: index ? 'var(--amber)' : 'var(--green)'}));
        }
        svg.append(svgElement('path', {d: path, class: `series${index ? ' secondary' : ''}`}));
      });
      const cursor = svgElement('line', {x1: 0, x2: 0, y1: 17, y2: 127, class: 'cursor'}); svg.append(cursor);
      let cursorIndex = Math.min(chartPositions.get(config.id) ?? points.length - 1, points.length - 1);
      function inspect(index) {
        cursorIndex = Math.max(0, Math.min(points.length - 1, index)); chartPositions.set(config.id, cursorIndex);
        const point = points[cursorIndex], px = x(point.step); cursor.setAttribute('x1', px); cursor.setAttribute('x2', px);
        readout.textContent = `Step ${formatNumber(point.step, 0)} · ${series.map((items, i) => `${config.labels[i]} ${config.sci ? scientific(items[cursorIndex]?.value) : formatNumber(items[cursorIndex]?.value, 5)}`).join(' / ')}`;
      }
      inspect(cursorIndex);
      svg.addEventListener('pointermove', event => {
        const rect = svg.getBoundingClientRect(); const px = (event.clientX - rect.left) / rect.width * 360;
        let closest = 0;
        points.forEach((point, index) => { if (Math.abs(x(point.step) - px) < Math.abs(x(points[closest].step) - px)) closest = index; });
        inspect(closest);
      });
      svg.addEventListener('keydown', event => {
        if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
          event.preventDefault(); inspect(event.key === 'Home' ? 0 : event.key === 'End' ? points.length - 1 : cursorIndex + (event.key === 'ArrowLeft' ? -1 : 1));
        }
      });
      card.append(svg, readout);
      if (config.keys.length > 1) { const legend = element('div', undefined, 'chart-legend'); config.labels.forEach(label => legend.append(element('span', label))); card.append(legend); }
      target.append(card);
    }
  }
  // Each chart has its own scale and evidence scope; durations are never added across kinds.
  function profileBars(target, id, title, scope, rows, kind) {
    const shown = rows.slice(0, 12);
    if (!shown.length) return;
    const card = element('section', undefined, `profile-chart ${kind}`);
    card.append(element('h3', title), element('p', `${scope} · 독립 축 · ms · 최대 12개 · 호버 / Tab / 탭으로 상세 확인`, 'panel-note'));
    const height = shown.length * 46 + 32;
    const svg = svgElement('svg', {id, viewBox: `0 0 640 ${height}`, class: 'profile-bars', role: 'group', 'aria-label': title});
    const tooltip = element('div', undefined, 'profile-tooltip');
    tooltip.setAttribute('id', `${id}-tooltip`); tooltip.setAttribute('role', 'tooltip'); tooltip.setAttribute('aria-live', 'polite'); tooltip.hidden = true;
    const maximum = Math.max(1, ...shown.map(row => finite(row.value) && row.value >= 0 ? row.value : 0));
    const stage = PROFILE_STAGES[store.state.profileStage] || '단계 미지정';
    let pinned = false, focused = false;
    shown.forEach((row, index) => {
      const y = index * 46;
      const valid = finite(row.value) && row.value >= 0;
      const label = text(row.label);
      const short = label.length > 32 ? `${label.slice(0, 31)}…` : label;
      const group = svgElement('g', {role: 'button', tabindex: '0', 'aria-label': `${label}${row.annotation ? ` · ${row.annotation}` : ''} · ${valid ? numberUnit(row.value, 'ms', 4) : MISSING}`, 'aria-describedby': `${id}-tooltip`, class: 'profile-bar-row'});
      group.append(svgElement('rect', {x: 0, y, width: 640, height: 44, class: 'profile-hit'}));
      group.append(svgElement('text', {x: 4, y: y + 16, class: 'profile-bar-label'}, short));
      if (row.annotation) group.append(svgElement('text', {x: 4, y: y + 34, class: 'profile-bar-annotation'}, row.annotation));
      group.append(svgElement('rect', {x: 250, y: y + 8, width: 310, height: 24, class: 'profile-track'}));
      group.append(svgElement('rect', {x: 250, y: y + 8, width: valid ? row.value / maximum * 310 : 0, height: 24, class: 'profile-bar'}));
      group.append(svgElement('text', {x: 635, y: y + 25, 'text-anchor': 'end'}, valid ? formatNumber(row.value, 3) : MISSING));
      function inspect() {
        tooltip.textContent = `단계: ${stage} · ${row.detail}\n범위: ${scope}`;
        tooltip.hidden = false;
      }
      group.addEventListener('pointerenter', inspect);
      group.addEventListener('pointermove', inspect);
      group.addEventListener('pointerleave', () => { if (!pinned && !focused) tooltip.hidden = true; });
      group.addEventListener('focus', () => { focused = true; inspect(); });
      group.addEventListener('blur', () => { focused = false; if (!pinned) tooltip.hidden = true; });
      group.addEventListener('click', () => { pinned = true; inspect(); row.onSelect?.(); });
      group.addEventListener('keydown', event => {
        if (event.key === 'Escape') { pinned = false; tooltip.hidden = true; }
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); pinned = true; inspect(); row.onSelect?.(); }
      });
      svg.append(group);
    });
    svg.append(svgElement('text', {x: 250, y: height - 7}, '0 ms'), svgElement('text', {x: 560, y: height - 7, 'text-anchor': 'end'}, `${formatNumber(maximum, 3)} ms`));
    card.append(svg, tooltip); target.append(card);
  }
  function profileIntro(target, profile, kind) {
    target.replaceChildren();
    target.append(element('p', `${profile?.available === true ? 'Historical capture · not live profiling' : 'Unavailable capture'} · Source / path: ${text(profile?.source)} · Scope: ${text(profile?.inspector?.scope || profile?.summary?.scope)} · 선택 실행과의 run binding 미검증`, 'profile-source'));
    for (const warning of list(profile?.warnings)) target.append(element('p', `경고: ${text(warning)}`, 'panel-note'));
    if (profile?.available !== true) { target.append(element('p', `${kind} 캡처가 아직 없습니다. 측정값·연산 상관관계를 만들지 않습니다.`, 'empty')); return false; }
    return true;
  }
  const inspectorChoices = new Map();
  function symbolParts(target, decoded) {
    if (!decoded || !list(decoded.parts).length) return;
    const panel=element('section',undefined,'symbol-parts');panel.append(element('h3','커널명 토큰 해석 · 명명규약 근거 / Python 호출 소스 아님'));
    const wrap=element('div',undefined,'table-wrap'),tableNode=element('table'),head=element('thead'),tr=element('tr'),body=element('tbody');
    for(const h of ['원시 토큰','해석 / 한계','신뢰도'])tr.append(element('th',h));head.append(tr);tableNode.append(head,body);
    for(const p of list(decoded.parts).slice(0,24)) {
      const row=element('tr'),meaning=element('td');meaning.append(element('strong',text(p.label)),element('p',text(p.meaning)));
      for(const [i,url] of list(p.sources).slice(0,3).entries()) if(typeof url==='string' && /^https:\/\/(?:docs\.nvidia\.com\/|developer\.nvidia\.com\/|forums\.developer\.nvidia\.com\/|github\.com\/NVIDIA\/|raw\.githubusercontent\.com\/NVIDIA\/)/.test(url)) {const a=element('a',`근거 ${i+1}`);a.href=url;a.target='_blank';a.rel='noopener noreferrer';meaning.append(a);}
      row.append(element('td',text(p.token)),meaning,element('td',({documented:'공개 근거 확인',inferred:'명칭 기반 추정',unknown:'미확인'}[p.confidence] || '미확인')));body.append(row);
    }
    wrap.append(tableNode);panel.append(wrap);for(const warning of list(decoded.warnings))panel.append(element('p',warning,'panel-note'));target.append(panel);
  }
  function sourceBlock(target, source, bridge = false) {
    const verified = source && ((bridge && source.status === 'snapshot_verified' && source.kind === 'explicit_bridge_operation') || (!bridge && source.status === 'host_scope_verified' && source.kind === 'host_scope'));
    if (!verified || typeof source.snippet !== 'string' || !Number.isInteger(source.start_line) || source.start_line < 1 || !Number.isInteger(source.end_line) || source.end_line < source.start_line || !source.file_sha256 || !source.snippet_sha256) {
      target.append(element('p', `소스 ${source?.status === 'rejected' ? '검증 거부' : '미수집'} · 저장된 검증 소스만 표시`, 'panel-note')); return;
    }
    target.append(element('h3', bridge ? '교육용 별도 계측 · 저장 연산 소스 검증 / 네이티브 커널 연결 아님' : '호스트 구간 소스 검증 / 커널 호출 스택 아님'));
    target.append(element('p', `${source.path} · ${text(source.function)} · 파일 SHA256 ${text(source.file_sha256)} · 스니펫 SHA256 ${text(source.snippet_sha256)}`, 'source-identity'));
    const pre = element('pre', undefined, 'source-code');
    pre.textContent = source.snippet.split('\n').slice(0, Math.min(60, source.end_line - source.start_line + 1)).map((line, i) => `${source.start_line + i}  ${line}`).join('\n');
    target.append(pre);
  }
  function renderInspector(target, nsight) {
    const identity = store.state.profileStage;
    const choice = inspectorChoices.get(identity) || {view: 'measured', phase: '', selected: null};
    inspectorChoices.set(identity, choice);
    const selected = inspectorScope(nsight, choice);
    const controls = element('div', undefined, 'inspector-controls');
    function picker(id, label, options, value, change) {
      const wrap = element('label', label), select = element('select'); select.setAttribute('id', id);
      for (const [key, title] of options) { const option = element('option', title); option.value = String(key); select.append(option); }
      select.value = String(value ?? ''); select.disabled = options.length === 0;
      select.addEventListener('change', () => { change(select.value); renderProfiles(store.state.snapshot?.profiling || {}); });
      wrap.append(select); controls.append(wrap);
    }
    picker('inspector-view', '범위', [['measured','측정 업데이트 · 초기화 / 워밍업 제외'],['capture','전체 캡처 · 초기화 포함 진단']], selected.view, v => {choice.view=v;choice.selected=null;});
    if (selected.view === 'measured') {
      choice.rank = selected.rank; choice.update = String(selected.update?.update ?? '');
      if (choice.phase && !selected.phase) choice.phase = '';
      picker('inspector-rank','Rank',[...new Set(selected.updates.map(u=>String(u.rank)))].map(v=>[v,v]),choice.rank,v=>{choice.rank=v;choice.update='';choice.phase='';choice.selected=null;});
      picker('inspector-update','Update',selected.updates.filter(u=>String(u.rank)===choice.rank).map(u=>[u.update,u.update]),choice.update,v=>{choice.update=v;choice.selected=null;});
      picker('inspector-phase','HOST 구간',[['','업데이트 전체'],...selected.phases.map(p=>[p.id,p.name])],choice.phase,v=>{choice.phase=v;choice.selected=null;});
    }
    target.append(controls);
    const scope = selected.view === 'measured' ? `${text(selected.scope)} · Rank ${selected.rank} / Update ${choice.update} / Phase ${choice.phase || '전체'} · CUDA 누적 시간 / wall time 아님` : '전체 캡처 · 초기화 / 워밍업 포함 가능 · CUDA 누적 시간 / wall time 아님';
    target.append(element('p', scope, 'panel-note'));
    target.append(element('p', '연결 신뢰도: 동일 프로세스 NVTX / 런타임 상관관계 · Python 호출 프레임 아님. 정확한 커널 소스에는 Python 호출 스택을 포함한 별도 캡처가 필요합니다.', 'panel-note'));
    const rows = [...selected.kernels].sort((a,b)=>(b.total_ms || 0)-(a.total_ms || 0)).slice(0,64);
    const categoryLabel = kernel => {
      const name = text(kernel.name).toLowerCase();
      if (kernel.category === 'NCCL') {
        for (const [token, label] of [['allgather','값 모으기 · AllGather'],['allreduce','전체 축약 · AllReduce'],['reducescatter','축약 / 분배 · ReduceScatter'],['alltoall','rank 간 데이터 교환 · AllToAll'],['sendrecv','rank 간 송수신 · SendRecv'],['broadcast','값 배포 · Broadcast']]) {
          if (name.includes(token)) return label;
        }
        return 'GPU 간 통신 · NCCL';
      }
      if (name.includes('l2norm')) return '텐서 노름 계산 · L2 norm';
      if (name.includes('bfloat16_copy')) return 'BF16 관련 변환 / 복사';
      return ({GEMM:'행렬 곱 · GEMM', mamba_chunk_scan:'순차 상태 갱신 · Mamba chunk scan', TopK:'상위 K 값 선택 · TopK', permute:'데이터 순열 / 재배치', reduction:'합계 / 축약 연산', fill_copy_cast:'값 채우기 / 복사 / 자료형 변환', opaque_vendor:'라이브러리 커널 · 세부 연산 미확인', unknown:'미분류 연산'}[kernel.category] || kernel.category || '미분류 연산');
    };
    const layout = element('div', undefined, 'inspector-layout'), charts = element('div'), card = element('article', undefined, 'inspector-card'); card.setAttribute('id','inspector-card'); card.setAttribute('aria-live','polite');
    function inspect(kernel) {
      choice.selected = kernel.name; card.replaceChildren();
      card.append(element('h3', kernel.name, 'operator-title'), element('p', categoryLabel(kernel), 'operator-annotation'), element('p', kernel.meaning || kernelMeaning(kernel.name), 'operator-purpose'));
      const dl = element('dl', undefined, 'facts'); facts(dl, [['CUDA Calls',formatNumber(kernel.count,0)],['CUDA Duration / Average',`${numberUnit(kernel.total_ms,'ms',4)} / ${numberUnit(kernel.average_us,'µs',4)}`],['Scope Share',numberUnit(kernel.percent,'%')],['Worker Launches',formatNumber(kernel.worker_launches,0)],['Meaning Evidence',kernel.meaning_evidence || 'symbol_reference'],['Range Correlation',selected.view === 'measured' ? `${selected.rank} / ${choice.update} / ${choice.phase || '업데이트'} · 프로세스 상관관계 / 레이어 확정 아님` : '매핑 미확인']]); card.append(dl);
      const raw = element('details'); raw.append(element('summary','원시 커널 심볼'),element('pre',kernel.name,'source-code')); card.append(raw);
      symbolParts(card,kernel.symbol_analysis);
      card.append(element('p','커널 호출 소스 미수집 · 이 캡처의 정확한 Python 커널 호출 프레임을 표시하지 않습니다.','source-unavailable'));
      if (selected.phase) sourceBlock(card, selected.phase.source);
      else if (selected.view === 'measured') {
        const host = element('details'); host.append(element('summary','인접 호스트 구간 소스 · 커널 호출 스택 아님'));
        for (const phase of selected.phases) { host.append(element('h3',phase.name)); sourceBlock(host,phase.source); } card.append(host);
      }
    }
    profileBars(charts,'profile-kernels','GPU Kernels · Top 12 in Selected Scope',scope,rows.map(kernel=>({label:kernel.name,annotation:categoryLabel(kernel),value:kernel.total_ms,onSelect:()=>inspect(kernel),detail:`원시 커널: ${text(kernel.name)}\nRank ${text(selected.rank)} / Update ${text(choice.update)} / Phase ${choice.phase || '전체'} · 호출 수 ${formatNumber(kernel.count,0)} · CUDA 합계 ${numberUnit(kernel.total_ms,'ms',4)} · 평균 ${numberUnit(kernel.average_us,'µs',4)} · 비율 ${numberUnit(kernel.percent,'%')}\n의미: ${kernel.meaning || kernelMeaning(kernel.name)} · 커널 소스 미확인 · 연결: 프로세스 상관관계 / Python 프레임 아님`})),'cuda');
    const grid = element('div', undefined, 'table-wrap'), tableNode = element('table'), body = element('tbody');
    for (const kernel of rows) {
      const tr = element('tr'), td = element('td'), button = element('button',kernel.name); button.type='button'; button.addEventListener('click',()=>inspect(kernel)); td.append(button); tr.append(td,element('td',categoryLabel(kernel)),element('td',formatNumber(kernel.count,0)),element('td',numberUnit(kernel.total_ms,'ms',4)),element('td',numberUnit(kernel.percent,'%'))); body.append(tr);
    }
    const head = element('thead'), header = element('tr'); for (const label of ['Raw Kernel Name','Annotation','Calls','CUDA Duration','Scope %']) header.append(element('th',label)); head.append(header); tableNode.append(head,body); grid.append(tableNode); charts.append(grid);
    if (rows.length) inspect(rows.find(k=>k.name===choice.selected) || rows[0]); else empty(card,'선택 범위의 커널 없음 · 전체 캡처로 대체하지 않습니다.');
    layout.append(charts,card); target.append(layout);
    if(selected.view==='measured') {
      const timeline=element('section');timeline.setAttribute('id','iteration-timeline');target.append(timeline);
      if(window.NemoTimeline) window.NemoTimeline.render(timeline,selected.update,{document:doc,phase:choice.phase,source:text(nsight.source),key:`${identity}/${text(nsight.source)}/${choice.rank}/${choice.update}/${choice.phase}`,onSelectEvent:e=>{const aggregate=rows.find(k=>k.name===e.name);if(aggregate)inspect(aggregate);}});
      else timeline.append(element('p','타임라인 UI 모듈 미로드 · 실행 순서를 생성하지 않습니다.','panel-note'));
    }
    const hostRows = (choice.phase ? [selected.phase].filter(Boolean) : selected.phases || []).map(p=>({label:p.name,value:p.host_ms,detail:`HOST NVTX ${numberUnit(p.host_ms,'ms',4)} · GPU 커널 시간 아님 · 연결 커널 수 ${formatNumber(p.kernel_count,0)} · 호스트 구간 소스 ${p.source?.status || '미수집'}`}));
    profileBars(target,'profile-phases','HOST Phase Duration · Separate from CUDA',scope,hostRows,'host');
    const candidates = element('div'); candidates.setAttribute('id','inspector-candidates');
    candidates.append(element('h3','우선 조사 대상 · 병목 확정 / 속도 향상 보장 아님'));
    const filtered = selected.view === 'measured' ? list(nsight.inspector.candidates).filter(c=>String(c.rank)===choice.rank && String(c.update)===choice.update && (!choice.phase || c.phase===choice.phase)) : [];
    for (const candidate of filtered.sort((a,b)=>(a.priority || 0)-(b.priority || 0)).slice(0,12)) {
      const item = element('article',undefined,'optimization-card'); item.append(element('h3',`우선순위 ${text(candidate.priority)} · ${text(candidate.title)}`)); const dl=element('dl',undefined,'facts'); facts(dl,[['관측 증거',candidate.evidence],['해석 / 가설',candidate.interpretation],['다음 실험',candidate.action],['소스 신뢰도',candidate.source_confidence || 'unavailable'],['검증 요구','정확성 + 비프로파일 전체 업데이트 A/B']]); item.append(dl); candidates.append(item);
    }
    if (!filtered.length) candidates.append(element('p','이 범위의 조사 후보 없음 · 전체 캡처의 초기화 비용을 측정 업데이트 병목으로 사용하지 않습니다.','panel-note'));
    target.append(candidates);
  }
  let renderedProfileSignature = null;
  function profileSignature(profiling) {
    const choice=inspectorChoices.get(store.state.profileStage);
    return JSON.stringify([store.state.profileStage,store.state.profilePinned,store.state.selected,store.selectedRun()?.stage,profiling,choice ? ['view','rank','update','phase'].map(k=>choice[k]) : null]);
  }
  function renderProfiles(profiling = {}) {
    const signature = profileSignature(profiling);
    if (signature === renderedProfileSignature) return;
    byId('profile-stage-select').value = store.state.profileStage;
    const cards = byId('profile-stage-cards'); cards.replaceChildren();
    for (const [stage, label] of Object.entries(PROFILE_STAGES)) {
      const profile = stageProfile(profiling, stage);
      const available = ['lens', 'nsight', 'bridge'].filter(key => profile?.[key]?.available === true);
      const button = element('button', undefined, 'profile-stage-card'); button.type = 'button'; button.setAttribute('aria-pressed', String(store.state.profileStage === stage));
      button.append(element('strong', label), element('small', available.length ? `${available.join(' / ')} 관측` : '미관측 · 캡처 없음'));
      button.addEventListener('click', () => store.selectProfileStage(stage)); cards.append(button);
    }
    const profile = stageProfile(profiling, store.state.profileStage) || {};
    setText('profile-stage-status', `${store.state.profilePinned ? 'Reference / history view' : 'Selected-stage capture view'} · Selected run: ${text(store.state.selected)} / ${text(store.selectedRun()?.stage)} · Capture stage: ${store.state.profileStage} (${PROFILE_STAGES[store.state.profileStage] || 'Unscoped'}) · Historical archive only; not the selected run’s live CUDA trace. Run binding 미검증; 로그 지표로 CUDA 이벤트를 만들지 않습니다.`);
    const lensTarget = byId('lens-content');
    if (profileIntro(lensTarget, profile.lens, 'Lens')) {
      const spans = list(profile.lens.spans).filter(object);
      profileBars(lensTarget, 'profile-lens', 'Lens Spans · HOST Duration', `출처 ${text(profile.lens.source)} · 개별 호스트 스팬 · 중첩 가능 · GPU 커널 시간 아님`,
        spans.map(span => ({label: span.name, value: span.duration_ms,
          detail: `스팬: ${text(span.name)} · HOST Lens ${numberUnit(span.duration_ms, 'ms', 4)} · GPU 커널 시간 아님\nTrace ${text(span.trace_id)} / Span ${text(span.span_id)} · Rank ${text(span.attributes?.['nv.dl.rank'])} / Update / Phase: 매핑 미확인 · 원시 커널 / 호출 수 / 평균 / 비율: 미수집\n의미: ${text(span.attributes?.['bridge.explanation'])} · 역할: ${text(span.attributes?.['bridge.role'])} · 증거: ${text(span.attributes?.['bridge.evidence'])} · 소스 식별자 미검증`})), 'host');
      if (!spans.length) lensTarget.append(element('p', '가져온 스팬 기록 없음', 'empty'));
      spans.forEach(span => {
        const item = profileDetails(`lens:${text(span.trace_id)}:${text(span.span_id)}:${text(span.name)}`, `${text(span.name)} · ${numberUnit(span.duration_ms, 'ms', 4)}`);
        const dl = element('dl', undefined, 'facts');
        facts(dl, [['Trace ID', span.trace_id], ['Span ID', span.span_id], ['Parent span', span.parent_span_id], ...Object.entries(object(span.attributes) ? span.attributes : {})]);
        item.append(dl); lensTarget.append(item);
      });
    }
    const nsightTarget = byId('nsight-content');
    if (profileIntro(nsightTarget, profile.nsight, 'Nsight')) {
      if (profile.nsight.inspector?.available === true) renderInspector(nsightTarget, profile.nsight);
      else {
      nsightTarget.append(element('p','커널 호출 소스 미수집 · Python 호출 스택이 포함된 별도 캡처가 필요합니다. 현재 캡처에서 소스 줄을 추정하지 않습니다.','source-unavailable'));
      if (object(profile.nsight.summary) && Object.keys(profile.nsight.summary).length) {
        const summary = profile.nsight.summary;
        const dl = element('dl', undefined, 'facts');
        facts(dl, [['Aggregation Scope', summary.scope], ['Capture Kernel Count', formatNumber(summary.kernel_count, 0)], ['CUDA Duration (not wall time)', numberUnit(summary.summed_kernel_ms_not_wall, 'ms')]]);
        nsightTarget.append(dl);
        const selected = object(summary.selected_ranges) ? summary.selected_ranges : {};
        const updates = list(selected.updates).filter(object);
        if (updates.length) {
          nsightTarget.append(element('h3', '선택 업데이트 · 생산자 분석'));
          nsightTarget.append(element('p', '전체 캡처와 별도 범위 · 상관관계는 생산자 보고이며 여기서 재계산하지 않았습니다.', 'panel-note'));
          table(nsightTarget, ['Rank / Update', 'HOST NVTX ms', 'Correlated Kernels', 'Worker Kernels'], updates.map(update => [`${text(update.rank)} / ${text(update.update)}`, formatNumber(update.cpu_nvtx_ms, 3), formatNumber(update.associated_correlated_kernel_count, 0), formatNumber(update.worker_thread_correlated_kernel_count, 0)]), [1, 2, 3]);
          const phases = updates.flatMap(update => list(update.phases).filter(object).map(phase => [`${text(update.rank)} / ${text(update.update)}`, phase.name, formatNumber(phase.cpu_nvtx_ms, 3), formatNumber(phase.associated_correlated_kernel_count, 0)]));
          if (phases.length) {
            const details = profileDetails('nsight:phases', 'Data / Forward / Backward + Sync / Optimizer 구간');
            table(details, ['Rank / Update', 'HOST Phase', 'NVTX ms', 'Correlated Kernels'], phases, [2, 3]);
            nsightTarget.append(details);
          }
        }
      }
      const kernels = list(profile.nsight.kernels).filter(object);
      const scope = `출처 ${text(profile.nsight.source)} · ${text(profile.nsight.summary?.scope)} · 전체 캡처 CUDA 누적 시간 · startup 포함 가능 · 중첩 가능 · wall time 아님`;
      profileBars(nsightTarget, 'profile-kernels', 'CUDA Cumulative Duration · Top 12', scope,
        [...kernels].sort((a, b) => (finite(b.total_ms) ? b.total_ms : -1) - (finite(a.total_ms) ? a.total_ms : -1)).map(kernel => ({label: kernel.name, value: kernel.total_ms,
          detail: `원시 커널: ${text(kernel.name)}\nRank / Update / Phase: 매핑 미확인 · 호출 수 ${formatNumber(kernel.count, 0)} · CUDA 합계 ${numberUnit(kernel.total_ms, 'ms', 4)} · 평균 ${numberUnit(kernel.average_us, 'µs', 4)} · 비율 ${numberUnit(kernel.percent, '%')}\n의미: ${kernelMeaning(kernel.name)}`})), 'cuda');
      const phaseRows = list(profile.nsight.summary?.selected_ranges?.updates).filter(object).flatMap(update => list(update.phases).filter(object).map(phase => ({
        label: `R${text(update.rank)} U${text(update.update)} · ${text(phase.name).split('/').pop()}`, value: phase.cpu_nvtx_ms,
        detail: `Rank ${text(update.rank)} / Update ${text(update.update)} · Phase: ${text(phase.name)}\nHOST NVTX ${numberUnit(phase.cpu_nvtx_ms, 'ms', 4)} · GPU 커널 시간 아님 · 원시 커널 / 호출 수 / 평균 / 비율: 미수집\n의미: 생산자 보고 호스트 구간 · 연결 커널 수 ${formatNumber(phase.associated_correlated_kernel_count, 0)} · CUDA 시간으로 변환하지 않음`
      })));
      profileBars(nsightTarget, 'profile-phases', 'Selected Update Phases · HOST NVTX Duration', '생산자 보고 선택 Rank / Update · Data / Forward / Backward + Sync / Optimizer · GPU 커널 시간 아님', phaseRows, 'host');
      if (!kernels.length) nsightTarget.append(element('p', '선택 범위의 커널 기록 없음', 'empty'));
      else table(nsightTarget, ['Raw Kernel Name', 'Calls', 'CUDA Duration ms', 'Average µs', 'Scope %'], kernels.map(kernel => [kernel.name, formatNumber(kernel.count, 0), formatNumber(kernel.total_ms, 4), formatNumber(kernel.average_us, 4), formatNumber(kernel.percent, 2)]), [1, 2, 3, 4]);
      }
    }
    const bridgeTarget = byId('bridge-content');
    if (profileIntro(bridgeTarget, profile.bridge, 'Bridge')) {
      const operations = list(profile.bridge.operations).filter(object);
      profileBars(bridgeTarget, 'profile-bridge', 'Bridge Operations · Reported HOST Duration', `출처 ${text(profile.bridge.source)} · 별도 연산 계측 · GPU 커널 시간 아님`,
        operations.map(operation => ({label: operation.title || operation.name || operation.id || operation.bridge_id, value: operation.duration_ms,
          detail: `연산: ${text(operation.title || operation.name || operation.id || operation.bridge_id)} · HOST Bridge ${numberUnit(operation.duration_ms, 'ms', 4)} · GPU 커널 시간 아님\n역할: ${text(operation.role)} · 설명: ${text(operation.explanation)} · 증거: ${text(operation.evidence)}\nID ${text(operation.bridge_id || operation.id)} · ${operation.source_evidence?.status === 'snapshot_verified' && operation.source_evidence?.kind === 'explicit_bridge_operation' ? '호스트 snapshot 검증 / 커널 연결 미확인' : '소스 식별자 미검증 / 커널 연결 미확인'} · Rank / Update / Phase: 매핑 미확인 · 원시 커널 / 호출 수 / 평균 / 비율: 미수집`})), 'host');
      if (!operations.length) bridgeTarget.append(element('p', '검증된 연산 식별 기록 없음', 'empty'));
      for (const operation of operations) {
        const label = operation.title || operation.name || operation.operation || operation.operation_id || operation.id || '연산 이름 미수집';
        const item = profileDetails(`bridge:${text(operation.id || operation.operation_id || label)}`, label);
        item.append(element('p', operation.explanation || '설명 미수집', 'operator-purpose'), element('p','교육용 별도 계측 · 네이티브 커널 상관관계 아님','panel-note'));
        sourceBlock(item, operation.source_evidence, true);
        const dl = element('dl', undefined, 'facts'); facts(dl, Object.entries(operation).filter(([key]) => !['source_evidence', 'source'].includes(key))); item.append(dl); bridgeTarget.append(item);
      }
    }
    const optimizations = byId('optimizations'); optimizations.replaceChildren();
    const suggestions = list(profile.optimizations).filter(object);
    if (!suggestions.length) empty(optimizations, '이 단계의 최적화 증거·권고 기록이 아직 없습니다. 프로파일 수집 후 후보 실험을 검토하세요.');
    for (const suggestion of suggestions) {
      const observed = suggestion.confidence === 'observed';
      const item = element('article', undefined, 'optimization-card');
      const heading = element('div', undefined, 'panel-heading'); heading.append(element('h3', suggestion.title), element('span', observed ? '관측 증거' : '후보 가설 · 미측정', 'badge')); item.append(heading);
      const dl = element('dl', undefined, 'facts');
      facts(dl, [['관측 증거', suggestion.evidence], ['후보 가설', suggestion.hypothesis], ['다음 실험', suggestion.next_experiment], ['출처', suggestion.source], ['검증 요구', '정확성 + 비프로파일 전체 업데이트 A/B · 동일 워크로드 비교']]);
      item.append(dl); optimizations.append(item);
    }
    renderedProfileSignature = profileSignature(profiling);
  }
  function renderEvidence(snapshot, run) {
    const collection = snapshot.collection;
    const target = byId('collection-facts'); target.replaceChildren();
    for (const [label, value] of [['스냅샷 생성', dateText(snapshot.generated_at)], ['수집 주기', numberUnit(collection.interval_seconds, '초')], ['수집 비용 (학습 비용 아님)', numberUnit(collection.duration_ms, 'ms', 3)], ['선택 실행 기록', dateText(run?.updated_at)]]) {
      const item = element('div', undefined, 'fact-item'); item.append(element('dt', label), element('dd', value)); target.append(item);
    }
    const provenance = object(run?.provenance) ? run.provenance : {};
    const sourcePanel = byId('run-provenance'); sourcePanel.replaceChildren();
    sourcePanel.append(element('p', `실행 로그 어댑터: ${text(provenance.adapter)} · 스키마 ${text(provenance.schema_version)} · 지표 범위: ${text(provenance.metrics_scope)} · 카운터 범위: ${text(provenance.counter_scope)}`, 'panel-note'));
    const sources = element('details', undefined, 'profile-item');
    sources.append(element('summary', '선택 실행의 실제 입력 파일 · 경로 / 형식 / 수집 상태'));
    const sourceRows = list(provenance.sources).filter(object);
    if (sourceRows.length) table(sources, ['입력 경로', '형식', '상태', '갱신 시각', '읽기 범위'], sourceRows.map(source => [source.path, source.kind, source.available === true ? '파일 관측' : source.available === false ? '파일 미관측' : '상태 미확인', dateText(source.updated_at), source.truncated === true ? '일부 읽기 · 잘림' : source.truncated === false ? '잘림 없음' : '미확인']));
    else sources.append(element('p', '입력 파일 목록 미수집 · 실행 출처를 추정하지 않습니다.', 'empty'));
    sourcePanel.append(sources);
    const warnings = byId('warnings'); warnings.replaceChildren();
    const items = [];
    for (const [source, entries] of [['수집기', collection.warnings], ['호스트', snapshot.host.warnings], ['선택 실행', run?.warnings]]) list(entries).forEach(detail => items.push({title: source, detail, severity: 'warning'}));
    list(snapshot.insights).filter(object).filter(item => !item.run_id || item.run_id === run?.id).forEach(item => items.push(item));
    for (const [source, entries] of Object.entries(snapshot.profiling)) {
      if (object(entries) && source !== 'stages') list(entries.warnings).forEach(detail => items.push({title: source, detail, severity: 'warning'}));
    }
    if (!items.length) empty(warnings, '보고된 경고 없음 · 모든 단계의 실행이나 데이터 완전성을 보증하지 않습니다.');
    for (const item of items) {
      const row = element('div', undefined, `warning-row ${item.severity === 'error' ? 'error' : item.severity === 'info' ? 'info' : ''}`);
      row.append(element('strong', item.title || '경고'), element('p', item.detail)); warnings.append(row);
    }
  }
  function updateFreshness() {
    const snapshot = store.state.snapshot;
    if (!snapshot) return;
    const age = ageSeconds(snapshot.generated_at);
    const seconds = snapshot.collection.interval_seconds;
    const stale = finite(age) && age > Math.max(15, (finite(seconds) ? seconds : 5) * 3);
    const node = byId('snapshot-freshness');
    node.textContent = age === null ? '생성 시각 미수집' : `${formatDuration(age)} 전 생성${stale ? ' · 오래된 스냅샷' : ''}`;
    node.dataset.state = stale ? 'stale' : 'observed';
  }
  byId('run-select').addEventListener('change', event => store.select(event.target.value));
  byId('pause-button').addEventListener('click', () => store.setPaused(!store.state.paused));
  byId('profile-stage-select').addEventListener('change', event => store.selectProfileStage(event.target.value));
  byId('chart-range').addEventListener('change', event => { range = Number(event.target.value); chartPositions.clear(); render(store.state); });
  const tabs = ['lens', 'nsight', 'bridge'];
  function selectTab(key, focus = false) {
    for (const tab of tabs) {
      const selected = key === tab;
      byId(`tab-${tab}`).setAttribute('aria-selected', String(selected)); byId(`tab-${tab}`).tabIndex = selected ? 0 : -1;
      byId(`panel-${tab}`).hidden = !selected;
    }
    if (focus) byId(`tab-${key}`).focus();
  }
  tabs.forEach((key, index) => {
    byId(`tab-${key}`).addEventListener('click', () => selectTab(key));
    byId(`tab-${key}`).addEventListener('keydown', event => {
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        event.preventDefault(); selectTab(event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs.at(-1) : tabs[(index + (event.key === 'ArrowLeft' ? 2 : 1)) % tabs.length], true);
      }
    });
  });
  const connectionLabels = {connecting: '연결 중', live: 'SSE 연결 · 실시간 수신', polling: '폴링 수신 · SSE 재연결 대기', reconnecting: 'SSE 재연결 중 · 폴링 전환', offline: '연결 실패 · 마지막 기록 유지', invalid: '스냅샷 형식 오류 · 마지막 기록 유지'};
  const transport = createTransport({EventSource: global.EventSource, fetch: global.fetch.bind(global), setTimeout: global.setTimeout.bind(global), clearTimeout: global.clearTimeout.bind(global), AbortController: global.AbortController}, snapshot => store.receive(snapshot), status => {
    setText('connection-status', connectionLabels[status] || '상태 미확인'); byId('connection-status').dataset.state = status;
  });
  renderCharts([]);
  renderProfiles({});
  transport.start();
  const freshnessTimer = global.setInterval(updateFreshness, 1000);
  global.addEventListener('pagehide', () => { transport.stop(); global.clearInterval(freshnessTimer); });
})(typeof window !== 'undefined' ? window : globalThis);
