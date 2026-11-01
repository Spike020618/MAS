"""Decision-linked review adapter. Existing paper-v2 runs must not resume with this code."""
import copy,json
from . import core

def request(task,state,index,reports=(),action=None,reviewer=0):
 action=copy.deepcopy(action or {})
 req=core.request(task,state,'review',index,reports,action,reviewer=reviewer,assigned_citation='any_used')
 payload=json.loads(req['messages'][1]['content'])
 payload['decision_question']=('Does the supplied evidence justify keeping or replacing the CURRENT ANSWER? Identify the answer-critical premise.' if task['kind']=='qa' else 'Does the supplied evidence justify replacing one current Top10 candidate with one supplied outside candidate? Compare BOTH candidates against the CURRENT user request, not historical-user preferences.')
 payload['decision_effect_contract']={'choice':'keep or revise','reason':'connect the judged premise to the current decision; refutation alone need not imply revision','candidate_pair':'exactly [inside_ID, outside_ID] for a recommendation revision; null otherwise'}
 req['messages'][1]['content']=json.dumps(payload,ensure_ascii=False,sort_keys=True)
 req['messages'][0]['content']+=' Also return decision_effect with choice, reason and candidate_pair. Evaluate the CURRENT decision explicitly. For revise, provide an evidence-grounded legal patch; for keep, patch must be null. Keep and revise refer ONLY to the native answer or ranking, never to rewriting reasons or claim status. For keep, patch MUST be null even when a premise was refuted. For revise, patch MUST change the current native decision; setting the same answer is not revision. Before returning check choice/patch consistency. A refuted premise does not automatically imply flipping an answer. A recommendation swap requires evidence about comparative fit, not merely an unsupported claim about one movie.'
 return req

def validate(task,state,report,index,action):
 # Linked reviews accept a revision resting on re-reading initial evidence; the basis is recorded instead of rejected so real change attempts are not discarded. Non-linked contracts keep the strict assigned-citation rule.
 refs=core.validate_report(task,state,report,'review',index,action,require_assigned=False)
 effect=report.get('decision_effect')
 if not isinstance(effect,dict)or effect.get('choice')not in ('keep','revise')or not isinstance(effect.get('reason'),str)or not effect['reason'].strip():raise ValueError('explicit decision effect required')
 patch=report.get('patch')
 if (effect['choice']=='revise')!=(patch is not None):raise ValueError('decision effect and patch disagree')
 if task['kind']=='rec'and patch:
  if effect.get('candidate_pair')!=[patch['a'],patch['b']]:raise ValueError('comparison pair must match local swap')
 if patch and task['kind']=='qa'and patch['value']==state['decision']:raise ValueError('revision must change current answer')
 cited={r['source_id']for r in refs};assigned=set(action.get('source_ids')or [])
 basis='initial' if not cited&assigned else('assigned' if assigned<=cited else 'joint')
 return refs,basis
