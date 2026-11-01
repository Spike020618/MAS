"""Native ReDial entity-linked, source-scoped shared materials (no query labels).

Identity links come from CRSLab training utterance annotations and CURRENT prefix
annotations only. No fuzzy title matching, MovieLens, questionnaires or test file.
KG relations retain published numeric IDs because this release has no name map.
"""
import argparse,json,re,hashlib
from collections import Counter,defaultdict
from pathlib import Path
from urllib.parse import unquote
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'experiments/data/raw/crslab_redial_native/released'
OUT=ROOT/'experiments/results/native_entity_evidence_v1'
REF=re.compile(r'@([0-9]+)')


def title(uri):return unquote(uri.removeprefix('<http://dbpedia.org/resource/').removesuffix('>')).replace('_',' ')

def fingerprint(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def merge_dialog(dialog):
    merged=[]
    for utt in dialog:
        if merged and merged[-1]['role']==utt['role']:
            for field in ('text','movies'):merged[-1][field]+=utt[field]
        else:merged.append({'role':utt['role'],'text':list(utt['text']),'movies':list(utt['movies'])})
    return merged


def constraints(utterances,entities):
    out=[]
    for u in utterances:
        refs=set(REF.findall(' '.join(u['text'])));uris=set(m for m in u['movies'] if m in entities)
        if refs and len(refs)==len(uris):out.append((refs,uris))
    return out


def infer_links(conditions,seed=None):
    """Only singleton identity constraints or one remaining bijection; conflicts unresolved."""
    candidates=defaultdict(set)
    if seed:
        for ref,uri in seed.items():candidates[ref].add(uri)
    for refs,uris in conditions:
        if len(refs)==len(uris)==1:candidates[next(iter(refs))].update(uris)
    for _ in range(12):
        mapping={ref:next(iter(uris)) for ref,uris in candidates.items() if len(uris)==1}
        additions=defaultdict(set)
        for refs,uris in conditions:
            unresolved=refs-set(mapping)
            if len(unresolved)!=1:continue
            resolved={mapping[r] for r in refs if r in mapping}
            # Cardinality is essential: aliases or omitted annotations cannot imply a bijection.
            if len(resolved)!=len(refs)-1 or not resolved<=uris:continue
            remaining=uris-resolved
            if len(remaining)==1:additions[next(iter(unresolved))].update(remaining)
        changed=False
        for ref,uris in additions.items():
            before=len(candidates[ref]);candidates[ref].update(uris);changed|=len(candidates[ref])!=before
        if not changed:break
    return ({ref:next(iter(uris)) for ref,uris in candidates.items() if len(uris)==1},
            {ref:sorted(uris) for ref,uris in candidates.items() if len(uris)>1})


def canonical(text,links,entities):
    def replace(match):
        ref=match.group(1);uri=links.get(ref)
        if uri is None:return match.group(0)
        return title(uri)+' [native_movie_id='+str(entities[uri])+'; redial_ref=@'+ref+']'
    return REF.sub(replace,text)


class NativeEvidence:
    def __init__(self,data_dir=DATA):
        self.data_dir=Path(data_dir)
        self.entities=json.loads((self.data_dir/'entity2id.json').read_text());self.names={v:title(k) for k,v in self.entities.items()}
        self.items=json.loads((self.data_dir/'movie_ids.json').read_text());self.item_set=set(self.items);self.item_index={m:i for i,m in enumerate(self.items)}
        self.kg=json.loads((self.data_dir/'dbpedia_subkg.json').read_text())
        self.train_raw=json.loads((self.data_dir/'train_data.json').read_text())
        # valid file is indexed for serving prefixes, never used in fits or global link induction.
        self.valid_raw=json.loads((self.data_dir/'valid_data.json').read_text())
        self.conversations={str(c['conv_id']):c for c in self.train_raw+self.valid_raw}
        cond=constraints([u for c in self.train_raw for u in c['dialog']],self.entities)
        self.links,self.conflicts=infer_links(cond)
        self.training=[];self.popularity=Counter();self.association=defaultdict(Counter);self.association_denominator=Counter()
        for conv in self.train_raw:
            cid=str(conv['conv_id']);merged=merge_dialog(conv['dialog']);prefix=[];seen=set()
            for i,u in enumerate(merged):
                targets=[self.entities[m] for m in u['movies'] if m in self.entities]
                if prefix and u['role']=='Recommender' and targets:
                    text='\n'.join(prefix)
                    row={'id':cid+':'+str(i),'conversation_id':cid,'text':canonical(text,self.links,self.entities),'observed_targets':targets}
                    self.training.append(row);self.popularity.update(targets)
                    for prior in seen:
                        self.association_denominator[prior]+=1
                        for target,count in Counter(targets).items():self.association[prior][target]+=count/len(targets)
                prefix.append(u['role']+': '+' '.join(u['text']));seen.update(targets)
        self.vectorizer=TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True,dtype=np.float32)
        self.train_matrix=self.vectorizer.fit_transform([r['text'] for r in self.training])
        rr=[];cc=[];vv=[]
        for i,r in enumerate(self.training):
            for m,count in Counter(r['observed_targets']).items():rr.append(i);cc.append(self.item_index[m]);vv.append(count/len(r['observed_targets']))
        self.target_matrix=csr_matrix((np.array(vv,dtype=np.float32),(rr,cc)),shape=(len(self.training),len(self.items)))
        self.train_conversation_ids=np.array([r['conversation_id'] for r in self.training])
        self._augmented={}

    def visible_prefix(self,case_id):
        cid,cutoff=case_id.rsplit(':',1);cutoff=int(cutoff)
        if cid not in self.conversations:raise ValueError('case must identify a released train/valid prefix')
        # Stop at the requested prefix: current/future response annotations are not inspected.
        prefix=[];current=None;merged_index=-1
        for utt in self.conversations[cid]['dialog']:
            if current is None or utt['role']!=current['role']:
                merged_index+=1
                if merged_index>=cutoff:break
                current={'role':utt['role'],'text':[],'movies':[]};prefix.append(current)
            current['text']+=utt['text'];current['movies']+=utt['movies']
        if len(prefix)!=cutoff:raise ValueError('invalid merged-turn prefix index')
        return prefix

    def augment(self,row,history_limit=3,kg_limit=6):
        key=fingerprint({'row':row,'history_limit':history_limit,'kg_limit':kg_limit})
        if key in self._augmented:return self._augmented[key]
        visible=self.visible_prefix(row['id']);original='\n'.join(u['role']+': '+' '.join(u['text']) for u in visible)
        if original!=row['text']:raise ValueError('serving text differs from author released visible prefix')
        local_links,local_conflicts=infer_links(constraints(visible,self.entities),self.links)
        for ref in local_conflicts:local_links.pop(ref,None)
        text=canonical(original,local_links,self.entities)
        query=self.vectorizer.transform([text]);sim=(query@self.train_matrix.T).toarray().ravel()
        cid=row['id'].rsplit(':',1)[0];sim[self.train_conversation_ids==cid]=-1
        order=np.argsort(-sim,kind='stable');order=[int(i) for i in order[:50] if sim[i]>0]
        weights=sim[order]
        neighbor_scores=(weights@self.target_matrix[order]).ravel() if order else np.zeros(len(self.items))
        source={};mentioned=set()
        for i,u in enumerate(visible):
            utt_text=canonical(' '.join(u['text']),local_links,self.entities)
            mids=[str(self.entities[m]) for m in u['movies'] if m in self.entities]
            mentioned.update(int(m) for m in mids)
            source['dialogue:'+str(i)]={'source_id':'dialogue:'+str(i),'kind':'dialogue','role':u['role'].lower(),'origin_case_id':row['id'],
                'text':utt_text,'allowed_use':'current_expression' if u['role']=='Seeker' else 'dialogue_context','movie_ids':mids,
                'current_user_fact_authority':u['role']=='Seeker','semantic_entailment_certified':False}
        neighbors=[]
        for idx in order[:history_limit]:
            train=self.training[idx];sid='history:'+train['id'];targets=[{'movie_id':str(m),'title':self.names[m]} for m in train['observed_targets']]
            evidence_text='OTHER training conversation prefix: '+train['text']+'\nObserved next recommender movie mentions: '+json.dumps(targets,ensure_ascii=False)+'. These are not the current seeker preferences or verified likes.'
            card={'source_id':sid,'kind':'historical_prediction_basis','role':'other_conversation','origin_case_id':train['id'],'text':evidence_text,
                  'allowed_use':'prediction_basis','movie_ids':[x['movie_id'] for x in targets],'current_user_fact_authority':False,'semantic_entailment_certified':False}
            source[sid]=card;neighbors.append({'source_id':sid,'origin_case_id':train['id'],'text':train['text'],'observed_targets':targets,'similarity':round(float(sim[idx]),6)})
        features={}
        for movie in row['candidates']:
            mid=str(movie['movie_id']);m=int(mid)
            if m not in self.item_set:raise ValueError('candidate is not a native released movie')
            association=sum(self.association[prior].get(m,0)/max(1,self.association_denominator[prior]) for prior in mentioned)/max(1,len(mentioned))
            linked=[]
            for relation,tail in self.kg.get(mid,[]):
                if tail==m or tail not in self.names:continue
                linked.append({'relation_id':relation,'entity_id':tail,'entity_name':self.names[tail]})
                if len(linked)>=kg_limit:break
            feature={'movie_id':mid,'neighbor_score':round(float(neighbor_scores[self.item_index[m]]),6),'association_score':round(float(association),6),
                     'training_count':self.popularity[m],'kg_neighbors':linked}
            features[mid]=feature
            sid='train-score:'+mid
            source[sid]={'source_id':sid,'kind':'training_aggregate','origin_case_id':'released_train_aggregate','allowed_use':'prediction_basis','movie_id':mid,
                'text':'Movie '+mid+' ('+self.names[m]+'): training-neighbor weighted score='+str(feature['neighbor_score'])+'; prior-mentioned-item association='+str(feature['association_score'])+'; training next-recommender mention count='+str(feature['training_count'])+'. Predictive analogy only; not current preference, popularity is not liking.',
                'current_user_fact_authority':False,'semantic_entailment_certified':False}
            sid='kg:'+mid
            source[sid]={'source_id':sid,'kind':'knowledge_graph','origin_case_id':'crslab_released_dbpedia_subkg','allowed_use':'catalog_attribute','movie_id':mid,
                'text':'Movie '+mid+' ('+self.names[m]+') published DBpedia neighbors: '+json.dumps(linked,ensure_ascii=False)+'. relation_id is the original numeric KG relation; its name is unavailable, so no actor/director/genre relationship is asserted.',
                'current_user_fact_authority':False,'semantic_entailment_certified':False}
        refs=REF.findall(original);resolved=[ref for ref in refs if ref in local_links]
        out={'id':row['id'],'text':text,'canonical_text':text,'original_text':original,'candidates':row['candidates'],'sources':source,
             'candidate_evidence':features,'training_neighbors':neighbors,
             'mapping_audit':{'reference_occurrences':len(refs),'resolved_occurrences':len(resolved),'unresolved_refs':sorted(set(refs)-set(resolved)),
                 'current_prefix_conflicting_refs':local_conflicts,'global_training_links':len(self.links),'global_conflicting_refs':len(self.conflicts),
                 'identity_rule':'training/current-prefix singleton or uniquely remaining bijection only; ambiguous links unresolved',
                 'query_current_response_available':False,'same_conversation_training_neighbors_excluded':True,'training_only_fits':True}}
        self._augmented[key]=out;return out


def run(output=OUT,inputs=None):
    inputs=Path(inputs) if inputs else ROOT/'experiments/results/native_redial_benchmark_v1/single_llm.inputs.json'
    rows=json.loads(inputs.read_text());builder=NativeEvidence();augmented=[builder.augment(row) for row in rows]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'fixed32.inputs.json').write_text(json.dumps(augmented,ensure_ascii=False,indent=2)+'\n')
    audit={'inputs_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_contexts':len(rows),'training_contexts':len(builder.training),'global_training_links':len(builder.links),'global_conflicting_refs':len(builder.conflicts),
        'reference_occurrences':sum(a['mapping_audit']['reference_occurrences'] for a in augmented),
        'resolved_occurrences':sum(a['mapping_audit']['resolved_occurrences'] for a in augmented),
        'contexts_with_unresolved_refs':sum(bool(a['mapping_audit']['unresolved_refs']) for a in augmented),
        'same_conversation_neighbor_count':sum(n['origin_case_id'].rsplit(':',1)[0]==a['id'].rsplit(':',1)[0] for a in augmented for n in a['training_neighbors']),
        'test_or_query_label_files_read':False,'movie_lens_used':False,'questionnaires_used':False,
        'data_sha256':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ('train_data.json','valid_data.json','entity2id.json','movie_ids.json','dbpedia_subkg.json')},
        'limitations':['numeric KG relation identity preserved; relation names not inferred','no semantic preference or satisfaction certification','release input annotations may themselves have errors; no manual truth claim']}
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps(audit,indent=2));return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs');p.add_argument('--output',default=str(OUT));a=p.parse_args();run(a.output,a.inputs)
