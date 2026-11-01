"""Summarize user-run environment arms; never pool variants as independent data."""
import argparse,json
from pathlib import Path
from experiments.prepare_mechanism_environments import ENVIRONMENTS

def summarize(root):
 out={'scope':'environment-wise descriptive results only; paired variants are not independent samples','environments':{}}
 for env in ENVIRONMENTS:
  p=Path(root)/env/'summary.json'
  out['environments'][env]={'status':'completed','summary':json.loads(p.read_text())}if p.exists()else {'status':'not_scored'}
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs-root',type=Path,required=True);a=p.parse_args();print(json.dumps(summarize(a.runs_root),ensure_ascii=False,indent=2))
