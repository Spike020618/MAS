from .schema import CheckTask

def propose(relations,member_judgments):
    result=[]
    for id,relation in sorted(relations.items()):
        values={j[id]for j in member_judgments if id in j and j[id]!='unclear'}
        unresolved=any(j.get(id,'unclear')=='unclear'for j in member_judgments)
        if len(values)>1 or unresolved:result.append(CheckTask(id,relation,'current_task',('member_observations',),'disagreement'if len(values)>1 else'unresolved'))
    return result
