import inspect,json
from pathlib import Path
import lm_eval
from lm_eval.api.samplers import FirstNSampler
# Read HF backend source without triggering GPU-only optional PEFT/TE import.
r=Path(lm_eval.__file__).parent
for x in ['tasks/mmlu/default/_default_template_yaml','tasks/mmlu/default/mmlu_abstract_algebra.yaml','tasks/arc/arc_easy.yaml','tasks/hellaswag/utils.py','tasks/hellaswag/hellaswag.yaml']:
 print(x,(r/x).read_text())
s=(r/'models/huggingface.py').read_text();i=s.index('    def _encode_pair');print(s[i:i+2100]);print(inspect.getsource(FirstNSampler))
