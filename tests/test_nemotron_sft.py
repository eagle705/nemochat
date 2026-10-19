"""CPU controls; real pinned-data integration is recorded separately."""
import importlib
import json
from pathlib import Path
import pytest


def test_pinned_inventory_registers_seed_without_supervision(tmp_path):
    m = importlib.import_module('nemochat.data.nemotron_sft')
    cards = tmp_path / 'discovery'
    cards.mkdir()
    # Handwritten metadata control; actual frozen cards are exercised by CPU receipts.
    for name,revision in m.PINS.items():
        (cards/(name+'.json')).write_text(json.dumps(dict(id='nvidia/'+name,sha=revision,
            cardData={'license':'openmdw-1.1'},files=[{'path':'data/train.jsonl','size':100}])))
    sources = m.pinned_sources(cards)
    assert len({s['dataset'] for s in sources}) == 5
    seed = [s for s in sources if 'Seed' in s['dataset']]
    assert len(seed) == 1 and seed[0]['purpose'] == 'reserved_rl_seed'
    assert all(len(s['revision']) == 40 and s['expected_bytes'] > 0 for s in sources)
    assert all(s['license'] == ['openmdw-1.1'] for s in sources)


def test_bounded_stream_keeps_only_complete_lines_and_counts_read_bytes(tmp_path):
    from nemochat.data.nemotron_sft import collect_jsonl
    raw = b'{"x":1}\n{"x":2}\n{"x":3}\n'
    dest = tmp_path / 'prefix.jsonl'
    result = collect_jsonl(iter([raw[:5], raw[5:19], raw[19:]]), dest, 19, 100)
    assert dest.read_bytes() == b'{"x":1}\n{"x":2}\n'
    assert result['rows'] == 2 and result['retained_bytes'] == 16
    assert result['transferred_bytes'] == 19
    assert result['sampling'] == 'bounded_prefix_not_representative'


def test_stream_rejects_oversized_chunk_instead_of_hiding_transfer(tmp_path):
    from nemochat.data.nemotron_sft import collect_jsonl
    with pytest.raises(ValueError, match='transfer cap'):
        collect_jsonl(iter([b'x' * 100]), tmp_path / 'x', 5, 1)


def test_acquire_real_http_prefix_and_readback(tmp_path):
    from nemochat.data.nemotron_sft import acquire_source
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread
    class Handler(BaseHTTPRequestHandler):
        body = b'{"x":1}\n{"x":2}\n'
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.body)
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        source = dict(dataset='nvidia/Test', revision='a'*40, path='data/x.jsonl',
                      key='test', expected_bytes=16, purpose='sft')
        receipt = acquire_source(source, tmp_path, 16, 1,
                                 base_url=f'http://127.0.0.1:{server.server_port}')
        assert receipt['rows'] == 1
        assert receipt['transferred_bytes'] == 16
        assert (tmp_path/'test.jsonl').read_bytes() == b'{"x":1}\n'
        assert json.loads((tmp_path/'test.receipt.json').read_text()) == receipt
        # Simulate retained complete lines after interrupted transfer: server ignores Range.
        (tmp_path/'test.receipt.json').unlink()
        resumed = acquire_source(source,tmp_path,16,2,resume=True,
                                 base_url=f'http://127.0.0.1:{server.server_port}')
        assert (tmp_path/'test.jsonl').read_bytes() == b'{"x":1}\n{"x":2}\n'
        assert resumed['rows'] == 2 and resumed['transferred_bytes'] == 16
        assert resumed['resumed_source_offset'] == 8
        Handler.body = b'{"x":1}\n'
        with pytest.raises(OSError,match='short pinned'):
            acquire_source(dict(source,key='short'),tmp_path,16,2,
                           base_url=f'http://127.0.0.1:{server.server_port}')
        assert (tmp_path/'short.jsonl').read_bytes() == b'{"x":1}\n'
        assert not (tmp_path/'short.receipt.json').exists()
    finally:
        server.shutdown()
        thread.join()


def test_message_aligned_turn_mask_keeps_false_assistant_context_and_tools():
    from nemochat.data.nemotron_sft import normalize_record
    row = {'messages': [{'role':'user','content':'hi'},
        {'role':'assistant','content':'old','reasoning_content':'SECRET'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'call1','type':'function','function':{'name':'x','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'call1','content':'result'},
        {'role':'assistant','content':'final','tool_calls':None}],
        'tools':[{'type':'function','function':{'name':'x','parameters':{}}}],
        'metadata':{'train_turns_mask':[False,False,True,False,True]}}
    out = normalize_record(row, 'nvidia/Nemotron-SFT-Instruction-Following-Chat-v4')
    assert out['selected'] == [False,False,True,False,True]
    assert out['messages'][1] == {'role':'assistant','content':'old'}
    assert out['messages'][2]['tool_calls'] == row['messages'][2]['tool_calls']
    assert out['messages'][3] == row['messages'][3]
    assert out['tools'] == row['tools']
    assert row['messages'][1]['reasoning_content'] == 'SECRET'
    with pytest.raises(ValueError, match='message-aligned'):
        normalize_record(dict(row, metadata={'train_turns_mask':[True,False,True]}),
                         'nvidia/Nemotron-SFT-Instruction-Following-Chat-v4')
    with pytest.raises(ValueError, match='RL seed'):
        normalize_record({'policy':'x','tools':[]}, 'nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1')


def test_official_template_instrumentation_exact_render_ids_selected_spans():
    from nemochat.data.nemotron_sft import encode_record
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    chars = pre_tokenizers.ByteLevel.alphabet()
    backend = Tokenizer(models.BPE(vocab={c:i for i,c in enumerate(chars)}, merges=[], unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<|im_end|>')
    # Handwritten structural control, not claimed to be official/data evidence.
    tok.chat_template = '''{%- set loop_messages = messages %}
{%- for message in loop_messages %}
{%- if message.role == "assistant" %}
{{- '<|im_start|>assistant\\n' ~ message.content ~ '<|im_end|>\\n' }}
{%- elif message.role == "user" or message.role == "system" %}
{{- '<|im_start|>' ~ message.role ~ '\\n' ~ message.content ~ '<|im_end|>\\n' }}
{%- endif %}
{%- endfor %}'''
    raw = {'messages':[{'role':'user','content':'日本語é'}, {'role':'assistant','content':'old'},
                        {'role':'user','content':'next'}, {'role':'assistant','content':'final'}],
           'metadata':{'train_turns_mask':[False,False,False,True]}}
    row = encode_record(raw, 'nvidia/Nemotron-SFT-Instruction-Following-Chat-v4', tok, 512)
    full = tok.apply_chat_template(raw['messages'], tokenize=True, add_generation_prompt=False, enable_thinking=False)
    # Transformers 5 returns BatchEncoding by default; older pinned hosts return IDs.
    if hasattr(full, 'keys'):
        full = full['input_ids']
    assert row['tokens'][:row['nonpadding_tokens']] == full
    targets = tok.decode([t for t,m in zip(row['tokens'][1:], row['loss_mask']) if m])
    assert 'final<|im_end|>' in targets
    assert 'old' not in targets and 'assistant' not in targets and '日本語' not in targets
    assert sum(row['loss_mask']) > 0
    assert encode_record(raw, 'nvidia/Nemotron-SFT-Instruction-Following-Chat-v4', tok, 8) is None
    # Same full-render path writes actual shifted native arrays, never segments.
    from nemochat.data.nemotron_sft import prepare_subset
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rawdir = root / 'raw'
        rawdir.mkdir()
        rawpath = rawdir/'test.jsonl'
        rawpath.write_text(json.dumps(raw)+'\n')
        receipt = dict(dataset='nvidia/Nemotron-SFT-Instruction-Following-Chat-v4',
                       revision='1cd362945d7a00b136b95b96892bbfea8798686e',
                       key='test',path='data/chat.jsonl',raw_file='test.jsonl',purpose='sft',rows=1)
        (rawdir/'test.receipt.json').write_text(json.dumps(receipt))
        manifest = prepare_subset(rawdir, root/'prepared', tok, set(), quota_per_file=2,
                                  sequence_length=512, heldout_modulus=100)
        assert manifest['status'] == 'blocked_missing_sft_sources'
        assert manifest['train_dialogues'] + manifest['heldout_dialogues'] == 1
        split = 'train' if manifest['train_dialogues'] else 'heldout'
        assert np.load(root/'prepared'/f'{split}-tokens.npy').shape == (1,513)
        assert int(np.load(root/'prepared'/f'{split}-masks.npy').sum()) == sum(row['loss_mask'])
        # A prior pilot heldout remains reserved even with a different main hash ratio.
        manifest2 = prepare_subset(rawdir, root/'expanded', tok, set(), quota_per_file=2,
            sequence_length=512, heldout_modulus=100, reserved_prompt_groups={row['prompt_group']})
        assert manifest2['heldout_dialogues'] == 1 and manifest2['train_dialogues'] == 0
        assert manifest2['prior_reserved_prompt_groups'] == 1
        from nemochat.data.nemotron_sft import audit_prepared
        audit = audit_prepared(rawdir,root/'prepared',tok)
        assert audit['status'] == 'passed' and audit['conversations_verified'] == 1
        masks_path = root/'prepared'/f'{split}-masks.npy'
        masks = np.load(masks_path)
        masks[0,0] = 1
        np.save(masks_path,masks)
        with pytest.raises(ValueError,match='mask'):
            audit_prepared(rawdir,root/'prepared',tok)


def test_data_cli_help_is_real_entrypoint():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, 'sft/data/nemotron_sft.py', '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ['discover', 'acquire', 'prepare'])


def test_native_plan_rejects_base_and_old_format_without_loading_weights():
    import runpy
    native = runpy.run_path('sft/training/native_nemotron_sft.py')
    import yaml
    config = yaml.safe_load(Path('sft/configs/nemochat_baby_nemotron_sft.yaml').read_text())
    manifest = dict(status='passed',schema='nemochat-nemotron-sft-v1',
                    format='official-super35-selected-assistant-v1',sequence_length=4096,
                    train_dialogues=96,heldout_dialogues=8,train_heldout_overlap=0,
                    kmmlu_exact_question_matches=0,kmmlu_reserved_questions=34871,
                    training_repetition=False)
    plan = native['stage_plan'](config,manifest,pilot=True)
    assert plan['updates'] == 12 and plan['global_batch_size'] == 8
    assert plan['source_checkpoint'].endswith('iter_0000391')
    with pytest.raises(ValueError,match='391'):
        native['stage_plan'](dict(config,source_checkpoint='BASE'),manifest,pilot=True)
    with pytest.raises(ValueError,match='official'):
        native['stage_plan'](config,dict(manifest,format='plain-role-segmented-v1'),pilot=True)
    with pytest.raises(ValueError,match='50000'):
        native['stage_plan'](config,manifest,pilot=False)
    with pytest.raises(ValueError,match='prior heldout'):
        native['stage_plan'](config,dict(manifest,train_dialogues=50000,heldout_dialogues=1000),pilot=False)


def test_launch_payload_is_inert_distinct_all8_and_offline_by_default():
    import runpy
    launch = runpy.run_path('sft/training/launch_nemotron_sft.py')
    payload = launch['make_payload']('pilot-control-01', Path('datasets/pilot/manifest.json'), pilot=True)
    assert payload['HostConfig']['DeviceRequests'][0]['DeviceIDs'] == list(map(str,range(8)))
    assert payload['HostConfig']['NetworkMode'] == 'none'
    assert '--pilot' in payload['Cmd']
    assert payload['WorkingDir'] == '/workspace'
    assert all('WANDB_API_KEY' not in value for value in payload['Env'])
    assert payload['HostConfig']['Binds'][0].endswith(':/workspace:ro')
    with pytest.raises(ValueError,match='safe'):
        launch['make_payload']('../bad',Path('datasets/pilot/manifest.json'))


def test_verify_inputs_rejects_forged_pass_without_all_five_pinned_sources(tmp_path):
    import runpy
    native = runpy.run_path('sft/training/native_nemotron_sft.py')
    migration = dict(status='passed',vocab_size=131072,allowed_changed_ids=list(range(18,30)),
        new_revision='new',tokenizer_path='local',datasetReservedCounts={name:dict(status='passed',stored_counts={})
        for name in ('climbmix_train','climbmix_validation','climbmix_calibration','existing_sft_train','existing_sft_heldout')})
    manifest = dict(tokenizer_revision='new',tokenizer_path='local',files={},sources=[],per_source={})
    (tmp_path/'migration.json').write_text(json.dumps(migration))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='five pinned'):
        native['verify_inputs'](tmp_path/'manifest.json',tmp_path/'migration.json')
