"""Select an already committed acquisition partition; never reshuffles or reads gold."""
import argparse,json
from pathlib import Path
from experiments.consensus_recommendation.common import create,sha

def select(inputs,output,partition):
 inputs=Path(inputs);output=Path(output);tasks=json.loads(inputs.read_text());commit=json.loads((inputs.parent/'split_commitment.json').read_text());splits=commit['task_split']
 if set(splits)!={t['id']for t in tasks}:raise ValueError('commitment/task mismatch')
 if any(t['metadata'].get('acquisition_partition')!=splits[t['id']]for t in tasks):raise ValueError('metadata partition changed')
 subset=[t for t in tasks if splits[t['id']]==partition]
 if not subset:raise ValueError('empty partition; do not substitute another split')
 create(output/'tasks.json',subset);create(output/'split_commitment.json',{**commit,'task_split':{t['id']:partition for t in subset},'parent_commitment_sha256':sha(inputs.parent/'split_commitment.json'),'selection':'unchanged precommitted subset, no new split'})
 return {'tasks':len(subset),'partition':partition,'model_api_calls':0}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--partition',choices=['fit','calibration','diagnostic'],default='diagnostic');a=p.parse_args();print(json.dumps(select(a.inputs,a.output,a.partition),indent=2))
