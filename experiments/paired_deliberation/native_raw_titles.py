"""Train-only exact ReDial reference titles for native numeric placeholders.

Native IDs stay unchanged. Only raw conversationId and movieMentions metadata
are used; no messages, questionnaires, rawtest, ratings or query labels.
"""
import collections,json,re,unicodedata
from pathlib import Path
from experiments.consensus_recommendation.common import ROOT,create,sha
DATA=ROOT/'experiments/data/raw/crslab_redial_native/released'
RAW=ROOT/'experiments/data/raw/redial/train.jsonl'
OUT=ROOT/'experiments/results/native_raw_title_metadata_v1'


def normalized_title(title):return ' '.join(unicodedata.normalize('NFKC',title).split()).casefold()


def extract_title_metadata(rows,allowed_cids):
    titles=collections.defaultdict(collections.Counter);provenance=collections.defaultdict(set);accepted=0;empty=0
    for row in rows:
        cid=str(row.get('conversationId'))
        if cid not in allowed_cids:continue
        accepted+=1;mentions=row.get('movieMentions')
        if not isinstance(mentions,dict):empty+=1;continue
        for ref,title in mentions.items():
            if not isinstance(title,str) or not title.strip() or not str(ref).isdigit():continue
            titles[str(ref)][title]+=1;provenance[str(ref)].add(cid)
    return titles,provenance,{'official_train_overlap_conversations':accepted,'empty_or_non_dict_movie_mentions':empty}


class NativeRawTitles:
    def __init__(self):
        entities=json.loads((DATA/'entity2id.json').read_text());self.entities={v:k for k,v in entities.items()};self.items=json.loads((DATA/'movie_ids.json').read_text())
        cids={str(c['conv_id']) for c in json.loads((DATA/'train_data.json').read_text())}
        with RAW.open() as stream:self.titles,self.provenance,self.counts=extract_title_metadata((json.loads(line)for line in stream),cids)
    def lookup(self,movie_id):
        movie_id=int(movie_id)
        if movie_id not in self.items:raise ValueError('not a native released item')
        entity=self.entities[movie_id];result={'movie_id':str(movie_id),'native_entity_key':entity,'native_identity_unchanged':True,'linked':False}
        if not entity.isdigit():result['reason']='not_numeric_native_placeholder';return result
        candidates=self.titles.get(entity,{})
        if not candidates:result['reason']='no_title_in_allowed_training_metadata';return result
        grouped=collections.defaultdict(list)
        for title in candidates:grouped[normalized_title(title)].append(title)
        if len(grouped)!=1:result['reason']='conflicting_raw_titles_for_exact_reference';result['raw_title_variants']=sorted(candidates);return result
        chosen=min(candidates,key=lambda t:(-candidates[t],t))
        result.update(linked=True,redial_reference_id=entity,title=chosen,raw_title_variants=sorted(candidates),
            source_kind='raw_training_movie_identity_metadata',origin='original_raw_train_intersection_official_train_only',
            source_id='raw-title:'+str(movie_id),supporting_training_conversations=len(self.provenance[entity]),
            semantic_preference_certified=False,used_current_or_future_query_mentions=False)
        return result


def run():
    linker=NativeRawTitles();records=[linker.lookup(m)for m in linker.items];bare=[r for r in records if r['native_entity_key'].isdigit()]
    report={'source_sha256':sha(Path(__file__)),'native_items':len(linker.items),'numeric_native_placeholders':len(bare),'resolved_bare_placeholders':sum(r['linked']for r in bare),
        'reasons':dict(collections.Counter(r.get('reason','linked') for r in bare)),**linker.counts,
        'raw_train_sha256':sha(RAW),'allowed_official_train_sha256':sha(DATA/'train_data.json'),
        'used_fields':['conversationId','movieMentions'],'messages_or_questionnaires_used':False,'raw_test_used':False,'query_labels_used':False,
        'current_frozen_inputs_modified':False,'title_only_metadata_no_genre_join':True,
        'future_feature_rule':'new feature version, same native IDs and all-method permissions; no candidate filtering or outcome-derived title matching'}
    create(OUT/'native_placeholder_titles.json',bare);create(OUT/'audit.json',report);print(json.dumps(report,indent=2));return report
if __name__=='__main__':run()
