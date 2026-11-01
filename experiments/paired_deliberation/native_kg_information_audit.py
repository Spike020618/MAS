"""Label-free audit of released anonymous KG edges, without inventing relations."""
import collections,json,re
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,sha
DATA=ROOT/'experiments/data/raw/crslab_redial_native/released'
OUT=ROOT/'experiments/results/native_kg_information_audit_v1'
TERMS={'comedy','horror','action','romance','romantic','science','fiction','thriller','animation','animated','drama','adventure','fantasy','mystery'}


def analyze(root,kg,names):
    rows=json.loads((root/'inputs.json').read_text());first=collections.Counter();all_names=collections.Counter();relations=collections.Counter();missed=[];counts=collections.Counter();unique=set()
    for row in rows:
        seeker=' '.join(s['text'] for s in row['sources'].values() if s.get('role')=='seeker')
        tokens=set(re.findall('[a-z]+',seeker.lower()))&TERMS
        for movie,feature in row['candidate_evidence'].items():
            initial=feature['kg_neighbors'];full=[{'relation_id':r,'entity_id':tail,'entity_name':names[tail]} for r,tail in kg.get(movie,[]) if tail!=int(movie) and tail in names]
            counts['candidate_occurrences']+=1;unique.add(movie);first.update(e['entity_name']for e in initial);all_names.update(e['entity_name']for e in full);relations.update(str(e['relation_id'])for e in initial)
            a=[e for e in initial if set(re.findall('[a-z]+',e['entity_name'].lower()))&tokens]
            b=[e for e in full if set(re.findall('[a-z]+',e['entity_name'].lower()))&tokens]
            counts['first3_preference_word_overlap_candidates']+=bool(a);counts['all_edges_preference_word_overlap_candidates']+=bool(b)
            if b and not a:missed.append({'case_id':row['id'],'movie_id':movie,'query_preference_terms':sorted(tokens),'omitted_matching_edges':b})
    return {'contexts':len(rows),'unique_candidate_movies':len(unique),**dict(counts),'truncation_lost_overlap_candidate_count':len(missed),
        'common_first3_entities':first.most_common(25),'common_all_neighbor_entities':all_names.most_common(25),'first3_relation_id_frequency':dict(relations),
        'omitted_overlap_examples':missed,'lexical_vocabulary':sorted(TERMS),
        'interpretation':'word overlap with named adjacent entities only; does not identify relation semantics or prove preference entailment'}


def run():
    kg=json.loads((DATA/'dbpedia_subkg.json').read_text());names={value:key[1:-1].removeprefix('http://dbpedia.org/resource/').replace('_',' ') for key,value in json.loads((DATA/'entity2id.json').read_text()).items()}
    result={'source_sha256':sha(Path(__file__)),'no_labels_read':True,'frozen_experiment_inputs_unchanged':True,
      'roots':{name:analyze(ROOT/'experiments/results'/name,kg,names) for name in ('native_open_evidence_screen_v2','native_open_evidence_transfer_v1')},
      'decision':'do not build a new selector merely to reorder anonymous people/studio edges; sparse type-word coverage does not show abundant genre facts hidden beyond first3',
      'limitations':['anonymous relation IDs cannot be called director/starring/genre without a relation map','lexical match is not semantic relevance or liking','all-neighbor overlap is an upper bound only for this explicit word vocabulary']}
    create(OUT/'audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return result
if __name__=='__main__':run()
