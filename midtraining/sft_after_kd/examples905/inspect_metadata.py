import json
from pathlib import Path
import torch.distributed.checkpoint as dcp
ROOT=Path('/raid/yoon/nemochat')
SOURCE=ROOT/'results/midtraining/sft-after-kd-runtime05/main/checkpoint/iter_0000905'
OUT=ROOT/'results/midtraining/sft905-examples-01'
m=dcp.FileSystemReader(SOURCE).read_metadata()
keys=list(m.state_dict_metadata)
report={'source':str(SOURCE),'keys':keys,'tensor_count':sum(hasattr(v,'size') for v in m.state_dict_metadata.values()),'metadata_only':True}
(OUT/'source_metadata.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'first_keys':keys[:15],'last_keys':keys[-15:],'root_counts':{r:sum(k.split('.')[0]==r for k in keys) for r in set(k.split('.')[0] for k in keys)},'tensor_count':report['tensor_count']},indent=2))
from transformers import AutoTokenizer
p=ROOT/'datasets/tokenizers/nemotron_super35/fc067b34f6b514014d2f8bddba855d2e4bfdfba6'
t=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
print('TOKENIZER',t.eos_token,t.eos_token_id,t.pad_token,t.pad_token_id)
print('TEMPLATE',t.chat_template)
print('RENDER',repr(t.apply_chat_template([{'role':'user','content':'Why is the sky blue? Answer in one sentence.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)))
