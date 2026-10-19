"""Regression checks from live semantic/source inspector review."""
from test_inspector_frontend import browser_assert


def test_raw_kernel_names_are_primary_with_separate_annotations():
    browser_assert(r"""
const name='ncclDevKernel_AllGather_RING_LL<raw_template>';
update.kernels=[{...kernel,name,category:'NCCL',meaning:'원시 이름 기반 참고 설명'}];
source.snapshot({data:JSON.stringify(base)});
const flatten=n=>[n,...n.children.flatMap(flatten)];
const card=get('inspector-card');
assert.equal(card.children.find(c=>c.tagName==='h3').textContent,name);
assert.ok(card.children.some(c=>c.tagName==='p' && c.textContent==='값 모으기 · AllGather'));
const rendered=flatten(get('nsight-content'));
assert.equal(rendered.find(c=>c.tagName==='button').textContent,name);
assert.ok(rendered.some(c=>c.tagName==='th' && c.textContent==='Annotation'));
const graph=get('profile-kernels');
const row=graph.children.find(c=>c.attrs.role==='button');
assert.ok(row.attrs['aria-label'].startsWith(name));
assert.ok(row.children.filter(c=>c.tagName==='text')[0].textContent.startsWith('ncclDevKernel_AllGather_RING_LL'));
assert.ok(row.children.some(c=>c.tagName==='text' && c.textContent==='값 모으기 · AllGather'));
""")


def test_bridge_hover_distinguishes_verified_snapshot_from_kernel_mapping():
    browser_assert(r"""
const row=get('profile-bridge').children.find(c=>c.attrs.role==='button');
row.fire('pointerenter');
assert.ok(get('profile-bridge-tooltip').textContent.includes('호스트 snapshot 검증'));
assert.ok(get('profile-bridge-tooltip').textContent.includes('커널 연결 미확인'));
""")


def test_source_view_does_not_claim_verification_without_end_line():
    browser_assert(r"""
get('inspector-phase').value='forward';get('inspector-phase').fire('change');
update.phases[0].source={...verified,end_line:null};
source.snapshot({data:JSON.stringify(base)});
assert.ok(!get('inspector-card').textContent.includes('호스트 구간 소스 검증 / 커널 호출 스택 아님'));
""")


def test_native_collective_and_peer_labels_are_distinguishable():
    browser_assert(r"""
update.kernels=[{...kernel,name:'ncclDevKernel_AllGather_RING_LL',category:'NCCL',meaning:'참고 설명'},
{...kernel,name:'ncclDevKernel_SendRecv',category:'NCCL',meaning:'참고 설명'}];
source.snapshot({data:JSON.stringify(base)});
assert.ok(get('nsight-content').textContent.includes('값 모으기 · AllGather'));
assert.ok(get('nsight-content').textContent.includes('rank 간 송수신 · SendRecv'));
""")


def test_source_view_does_not_invent_line_after_verified_end():
    browser_assert(r"""
get('inspector-phase').value='forward';get('inspector-phase').fire('change');
update.phases[0].source={...verified,snippet:code+'\n'};
source.snapshot({data:JSON.stringify(base)});
assert.ok(get('inspector-card').textContent.includes('13  with nvphase'));
assert.ok(!get('inspector-card').textContent.includes('\n14  '));
""")
