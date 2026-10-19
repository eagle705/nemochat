from pathlib import Path
import importlib.util


def test_six_bilingual_prompts_and_literal_fencing():
    p=Path('/raid/yoon/nemochat/midtraining/publication/hf_kd1000/examples.py')
    assert p.exists(), 'Actual bilingual example generator is missing'
    spec=importlib.util.spec_from_file_location('examples',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert len(m.PROMPTS)==6
    assert sum(x['language']=='en' for x in m.PROMPTS)==3
    assert sum(x['language']=='ko' for x in m.PROMPTS)==3
    text='code\n```python\n  unchanged  \n'
    fenced=m.fenced(text)
    assert fenced.startswith('````text\n')
    assert text in fenced
    assert m.GENERATION==dict(max_new_tokens=64,do_sample=False,num_beams=1,use_cache=True)
