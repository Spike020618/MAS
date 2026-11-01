import argparse,json
from pathlib import Path
from .panel import budget
from .report import counterexamples
from .exposure import audit
from .prepare_data import prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','estimate','exposure','prepare','collect','analyze']);p.add_argument('--root',default='experiments/results');p.add_argument('--out');p.add_argument('--data',default='experiments/results/coordination_game_p1_prepare_v1');p.add_argument('--provider',default='qwen_flash');p.add_argument('--approved-cny',type=float);a=p.parse_args()
    if a.command=='prepare':
        result=prepare(a.out or a.data)
    elif a.command=='collect':
        from .collect import collect
        result=collect(a.data,a.out or 'experiments/results/coordination_game_p1_v1',a.provider,a.approved_cny)
    elif a.command=='analyze':
        from .collect import run_analysis
        result=run_analysis(a.out or 'experiments/results/coordination_game_p1_v1',a.data)
    else:
        result=counterexamples()if a.command=='check'else(budget()if a.command=='estimate'else audit(a.root))
    if a.command not in('collect','analyze') and a.out:
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('output exists; use a new artifact path')
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    elif result is not None:
        print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
