"""Strict data-contract rejection controls, separate from fatal template failures."""
import copy
import json
from concurrent.futures import ProcessPoolExecutor
import pytest
import nemochat.data.nemotron_sft as data

DATASET='nvidia/Nemotron-SFT-Agentic-v4'

def row(arguments):
    return {'uuid':'control','messages':[{'role':'user','content':'question'}, {'role':'assistant','content':None,'tool_calls':[{'id':'c','type':'function','function':{'name':'f','arguments':arguments}}]}]}

@pytest.mark.parametrize('arguments,reason,kind', [('null','tool_arguments_not_object','NoneType'), ('[]','tool_arguments_not_object','list'), ('42','tool_arguments_not_object','int'), (None,'tool_arguments_not_object','NoneType'), ('{','tool_arguments_invalid_json','str')])
def test_whole_conversation_rejected_before_tokenizer(arguments,reason,kind):
    original=row(arguments);before=copy.deepcopy(original)
    try:data.normalize_record(original,DATASET)
    except ValueError as error:
        assert type(error).__name__=='SourceSchemaRejection'
        assert error.reason_code==reason
        assert error.details['argument_decoded_type']==kind
        assert error.details['message_index']==1
        assert original==before
    else:pytest.fail('unsupported source arguments escaped normalization')

def reject_in_process(arguments):
    return data.normalize_record(row(arguments),DATASET)

def test_typed_rejection_survives_actual_process_pool():
    import multiprocessing
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as pool:
        future=pool.submit(reject_in_process,'null')
        try:future.result(timeout=20)
        except ValueError as error:
            assert type(error).__name__=='SourceSchemaRejection'
            assert error.reason_code=='tool_arguments_not_object'
            assert error.details['call_index']==0
        else:pytest.fail('missing typed process rejection')

def test_actual_orphan_response_contract_is_typed_without_dropping_context():
    original={'messages':[{'role':'user','content':'question'}, {'role':'tool','tool_call_id':'unknown','content':'result'}, {'role':'assistant','content':'answer'}]}
    before=copy.deepcopy(original)
    try:data.normalize_record(original,DATASET)
    except ValueError as error:
        assert type(error).__name__=='SourceSchemaRejection'
        assert error.reason_code=='orphan_tool_response'
        assert error.details=={'message_index':1}
    else:pytest.fail('orphan response contract was accepted')
    assert original==before


def test_unselected_assistant_tool_arguments_still_validate_whole_context():
    original=row('null');original['messages'].append({'role':'assistant','content':'answer'})
    original['metadata']={'train_turns_mask':[False,False,True]}
    with pytest.raises(data.SourceSchemaRejection) as caught:
        data.normalize_record(original,'nvidia/Nemotron-SFT-Instruction-Following-Chat-v4')
    assert caught.value.reason_code=='tool_arguments_not_object'


def test_json_object_arguments_are_parsed_once_per_normalization(monkeypatch):
    original=row('{"nested":{"x":null},"items":[1,2]}')
    loads=json.loads;seen=[]
    def counting(value,*args,**kwargs):
        seen.append(value);return loads(value,*args,**kwargs)
    monkeypatch.setattr(data.json,'loads',counting)
    normalized=data.normalize_record(original,DATASET,decode_tool_arguments=True)
    assert len(seen)==1
    assert normalized['messages'][1]['tool_calls'][0]['function']['arguments']=={'nested':{'x':None},'items':[1,2]}
    assert original['messages'][1]['tool_calls'][0]['function']['arguments']==seen[0]


def test_normalization_preserves_valid_json_argument_string_and_context():
    original=row('{"nested":{"x":null},"items":[1,2]}')
    normalized=data.normalize_record(original,DATASET)
    assert normalized['messages'][1]['tool_calls']==original['messages'][1]['tool_calls']


def test_producer_quarantines_whole_row_with_offset_and_counts(tmp_path,monkeypatch):
    raw=tmp_path/'raw';raw.mkdir();bad=row('null');good={'uuid':'good','messages':[{'role':'user','content':'next'},{'role':'assistant','content':'answer'}]}
    lines=[json.dumps(r)+'\n' for r in [bad,good]]
    (raw/'source.jsonl').write_text(''.join(lines))
    (raw/'source.receipt.json').write_text(json.dumps(dict(dataset=DATASET,revision=data.PINS['Nemotron-SFT-Agentic-v4'],key='control',path='data/control.jsonl',raw_file='source.jsonl',purpose='sft',rows=2)))
    def encode(record,dataset,tokenizer,length):
        n=data.normalize_record(record,dataset)
        return dict(n,tokens=[1]*(length+1),loss_mask=[1]*length,nonpadding_tokens=length,supervised_tokens=length)
    monkeypatch.setattr(data,'encode_record',encode)
    manifest=data.prepare_subset(raw,tmp_path/'out',None,set(),sequence_length=4)
    assert manifest['per_source']['control']['schema_rejected']==1
    assert manifest['per_source']['control']['accepted']==1
    q=json.loads((tmp_path/'out/source-schema-rejections.jsonl').read_text())
    assert q['source_byte_offset']==0 and q['source_line']==1 and q['source_id']=='control'
    assert q['reason_code']=='tool_arguments_not_object'
    split='train' if manifest['train_dialogues'] else 'heldout'
    retained=json.loads((tmp_path/'out'/f'{split}-dialogues.jsonl').read_text())
    assert retained['source_byte_offset']==len(lines[0].encode()) and retained['source_line']==2
    assert manifest['source_schema_policy']['rejected_records']==1


def test_fatal_encode_valueerror_is_not_quarantined(tmp_path,monkeypatch):
    raw=tmp_path/'raw';raw.mkdir();(raw/'x.jsonl').write_text(json.dumps(row('{}'))+'\n')
    (raw/'x.receipt.json').write_text(json.dumps(dict(dataset=DATASET,revision=data.PINS['Nemotron-SFT-Agentic-v4'],key='x',path='data/x.jsonl',raw_file='x.jsonl',purpose='sft',rows=1)))
    def broken(*args):raise ValueError('instrumentation changed official token IDs')
    monkeypatch.setattr(data,'encode_record',broken)
    with pytest.raises(ValueError,match='instrumentation changed'):
        data.prepare_subset(raw,tmp_path/'out',None,set())
