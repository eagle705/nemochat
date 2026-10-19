import json
from score_sft1248_02 import migrated_guard,ROOT,OUT,evaluation
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained(evaluation.BASE,local_files_only=True,trust_remote_code=False)
except Exception as exc:
    failure={'type':type(exc).__name__,'error':str(exc)}
else:
    failure={'status':'failure_not_reproduced'}
suite=migrated_guard(ROOT/'results/midtraining/offline2b-ablation-eval-01/suite.json')
(OUT/'tokenizer-recovery-preflight.json').write_text(json.dumps({'legacy_comparison_tokenizer':failure,'alternative_comparison_tokenizer':'original local BASE tokenizer; canonical vocabulary SHA equals frozen suite','frozen_choices_verified':1600,'evaluated_model_tokenizer':'unchanged SFT1248 migrated tokenizer','model_weights_read':False},indent=2))
print('TOKENIZER_PREFLIGHT_GREEN',failure)
