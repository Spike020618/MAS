"""Read-only input inventory; raw API slots and credentials are never traversed."""
import json
from pathlib import Path
from .data import question_key
from .schema import digest

def audit(root):
    rows=[];ids=set();questions=set();errors=[]
    for path in sorted(Path(root).rglob('inputs.json')):
        try:
            data=json.loads(path.read_text());items=data if isinstance(data,list)else data.get('tasks',[])
            if not isinstance(items,list):raise ValueError('unrecognized inputs schema')
            count=0
            for item in items:
                if not isinstance(item,dict):continue
                id=item.get('id',item.get('_id',item.get('task_id')))
                q=item.get('question',item.get('prompt'))
                if id is not None:ids.add(str(id));ids.add(str(id).removeprefix('e2:qa:'))
                if isinstance(q,str):questions.add(question_key(q))
                count+=1
            rows.append({'path':str(path),'sha256':digest(data),'records':count})
        except Exception as e:errors.append({'path':str(path),'error':type(e).__name__})
    return {'files':rows,'ids':sorted(ids),'question_keys':sorted(questions),'errors':errors,
            'scope':'inputs.json inventory only; additional manifests/raw question mapping and near-duplicate audit required before fresh-test claim',
            'fresh_test_certified':False}
