"""Bounded live run-root enrollment; fixtures are not GPU training evidence."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def spec():
    return {'runs':[], 'run_roots':[{'path':'runs','format':'bridge','rank_count':8,'prefixes':{'bridge-mid-':'mid','bridge-sft-':'sft'}}]}


def test_discovery_enrolls_new_stage_directory_without_config_edit(tmp_path):
    from discovery import expand_run_roots
    (tmp_path/'runs').mkdir()
    (tmp_path/'runs'/'bridge-mid-02').mkdir()
    (tmp_path/'runs'/'bridge-mid-02'/'requested-config.json').write_text('{}')
    conf,meta=expand_run_roots(tmp_path,spec())
    assert [(r['id'],r['stage']) for r in conf['runs']]==[('bridge-mid-02','mid')]
    (tmp_path/'runs'/'bridge-sft-03').mkdir()
    (tmp_path/'runs'/'bridge-sft-03'/'rank-0-updates.jsonl').write_text('{}\n')
    conf,meta=expand_run_roots(tmp_path,spec())
    assert set((r['id'],r['stage']) for r in conf['runs'])=={('bridge-mid-02','mid'),('bridge-sft-03','sft')}
    assert meta['discovered_runs']==2
    assert spec()['runs']==[]


def test_discovery_rejects_unsafe_roots_and_never_reads_arbitrary_payloads(tmp_path):
    from discovery import expand_run_roots,validate_run_roots
    for root in ['../outside','/etc','.private','runs/../secrets',None,23]:
        bad=spec();bad['run_roots'][0]['path']=root
        with pytest.raises(ValueError):validate_run_roots(tmp_path,bad)
    (tmp_path/'runs').mkdir()
    (tmp_path/'runs'/'bridge-mid-link').symlink_to(tmp_path,target_is_directory=True)
    for name in ['random-sft-01','bridge-sft-empty','.bridge-sft-hidden']:(tmp_path/'runs'/name).mkdir()
    conf,meta=expand_run_roots(tmp_path,spec())
    assert conf['runs']==[]


def test_explicit_run_specs_take_precedence_without_cross_path_id_repair(tmp_path):
    from discovery import expand_run_roots
    p=tmp_path/'runs'/'bridge-mid-02';p.mkdir(parents=True);(p/'requested-config.json').write_text('{}')
    c=spec();c['runs']=[{'id':'bridge-mid-02','path':'explicit','stage':'mid','format':'bridge'}]
    expanded,meta=expand_run_roots(tmp_path,c)
    assert expanded['runs']==c['runs']
    assert meta['warnings']
