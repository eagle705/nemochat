"""Bounded one-level enrollment under explicitly approved artifact roots.

This reads directory/marker metadata only, never checkpoint/data/prompt payloads.
Explicit individual specs win. No general filesystem crawler or trainer hooks.
"""
import copy
import os
from pathlib import Path
import re

MAX_ROOTS=8
MAX_ENTRIES=256
MAX_RUNS=32
STAGES={'pretrain','mid','sft','long','rlvr','mopd'}
FORMATS={'bridge','native','rlvr','canonical'}
MARKERS=('requested-config.json','rank-0-updates.jsonl','rank-0-train-complete.json','rank-0-terminal.json','parent-verification.json','run.json')


def _safe(root,relative):
    if not isinstance(relative,str):raise ValueError('Unsafe discovery path type')
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts or any(p.startswith('.') for p in rel.parts):
        raise ValueError('Unsafe discovery path')
    resolved=(root/rel).resolve()
    if not resolved.is_relative_to(root):raise ValueError('Discovery root escape')
    return resolved


def validate_run_roots(root,config):
    root=Path(root).resolve();roots=config.get('run_roots',[])
    if not isinstance(roots,list) or len(roots)>MAX_ROOTS:raise ValueError('Discovery root count')
    for rule in roots:
        if not isinstance(rule,dict):raise ValueError('Discovery rule type')
        _safe(root,rule['path'])
        if rule.get('format') not in FORMATS or type(rule.get('rank_count',1)) is not int or not 1<=rule.get('rank_count',1)<=64:raise ValueError('Discovery format/ranks')
        prefixes=rule.get('prefixes')
        if not isinstance(prefixes,dict) or not 1<=len(prefixes)<=8:raise ValueError('Discovery prefixes')
        for prefix,stage in prefixes.items():
            if not isinstance(prefix,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,70}',prefix) or stage not in STAGES:raise ValueError('Discovery prefix/stage')
        if rule.get('model_census'):_safe(root,rule['model_census'])
    return config


def expand_run_roots(root,config):
    root=Path(root).resolve();validate_run_roots(root,config);result=copy.deepcopy(config)
    runs=result.setdefault('runs',[]);known={r['id']:r for r in runs}
    meta={'configured_runs':len(runs),'discovered_runs':0,'roots':[],'warnings':[],'scan_truncated':False}
    candidates=[]
    for rule in config.get('run_roots',[]):
        base=_safe(root,rule['path']);meta['roots'].append(rule['path'])
        try:
            with os.scandir(base) as entries:
                for i,e in enumerate(entries):
                    if i>=MAX_ENTRIES:
                        meta['scan_truncated']=True;meta['warnings'].append('Discovery entry bound reached; coverage incomplete');break
                    if e.is_symlink() or not e.is_dir(follow_symlinks=False) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',e.name):continue
                    stages={stage for prefix,stage in rule['prefixes'].items() if e.name.startswith(prefix)}
                    if len(stages)!=1:continue
                    p=Path(e.path)
                    if not any((p/m).is_file() and not (p/m).is_symlink() for m in MARKERS):continue
                    path=str(p.relative_to(root));_safe(root,path)
                    spec={'id':e.name,'name':e.name,'path':path,'stage':next(iter(stages)),'format':rule['format'],'rank_count':rule.get('rank_count',1)}
                    for k in ('model_name','precision','model_census','data_name','data_kind','scope'):
                        if k in rule:spec[k]=copy.deepcopy(rule[k])
                    receipt=p/'container-state.json'
                    if receipt.is_file() and not receipt.is_symlink():spec['terminal_receipt']=str(receipt.relative_to(root))
                    candidates.append((e.stat(follow_symlinks=False).st_mtime_ns,spec))
        except (OSError,ValueError):meta['warnings'].append('Approved discovery root unavailable or changed; no arbitrary paths followed')
    for _,spec in sorted(candidates,key=lambda p:(p[0],p[1]['id']),reverse=True):
        if spec['id'] in known:
            if known[spec['id']]['path']!=spec['path']:meta['warnings'].append('Discovered ID conflicts with explicit spec; explicit path retained')
            continue
        if len(runs)>=MAX_RUNS:
            meta['scan_truncated']=True;meta['warnings'].append('Run bound reached; discovery incomplete');break
        known[spec['id']]=spec;runs.append(spec);meta['discovered_runs']+=1
    return result,meta
