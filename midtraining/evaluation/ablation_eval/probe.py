from pathlib import Path
import lm_eval,importlib.metadata as im,inspect,json
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.api.task import ConfigurableTask
root=Path(lm_eval.__file__).parent
print('version',getattr(lm_eval,'__version__',None))
for d in im.distributions():
 if 'eval' in d.metadata['Name']:print(d.metadata['Name'],d.version,d.read_text('direct_url.json'))
for pat in ['mmlu/*yaml','arc/*yaml','hellaswag/*yaml']:
 for p in root.glob('tasks/'+pat):
  if p.name in ['mmlu.yaml','_default_template_yaml','arc_challenge.yaml','hellaswag.yaml'] or 'abstract_algebra' in p.name:print(str(p),p.read_text())
print('mmlu files',[str(x) for x in root.glob('tasks/mmlu/**/*')][:30])
print(inspect.signature(evaluator.simple_evaluate));print(inspect.getsource(ConfigurableTask.fewshot_context))
print(inspect.getsource(ConfigurableTask.construct_requests))
