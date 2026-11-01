"""Strict MovieLens GENRE METADATA links, never native labels or item filtering.

Only canonical DBpedia URIs with explicit (YEAR[_film]) and unique normalized
TITLE+YEAR on BOTH sides may link. No fuzzy, alternative title, inferred year,
rating or preference data. Existing native experiment inputs stay unchanged.
"""
import argparse,collections,csv,json,re,unicodedata
from pathlib import Path
from urllib.parse import unquote
from experiments.consensus_recommendation.common import ROOT,create,sha
DATA=ROOT/'experiments/data/raw/crslab_redial_native/released'
MOVIES=ROOT/'experiments/data/raw/movielens/ml-25m/movies.csv'
OUT=ROOT/'experiments/results/native_external_genres_probe_v1'


def normalize_title(text):
    text=' '.join(unicodedata.normalize('NFKC',text).split())
    article=re.search(r', (The|A|An)$',text,re.I)
    if article:text=article.group(1)+' '+text[:article.start()]
    return text.casefold()


def native_key(entity):
    prefix='<http://dbpedia.org/resource/'
    if not entity.startswith(prefix) or not entity.endswith('>'):return None,'not_dbpedia_uri'
    name=unquote(entity[len(prefix):-1]);match=re.fullmatch(r'(.+)_\((\d{4})(?:_film)?\)',name)
    if not match:return None,'no_explicit_simple_uri_year'
    return (normalize_title(match.group(1).replace('_',' ')),match.group(2)),None


def external_key(title):
    match=re.fullmatch(r'(.+) \((\d{4})\)',title)
    return (normalize_title(match.group(1)),match.group(2)) if match else None


class StrictNativeGenres:
    def __init__(self,entities,items,metadata_rows):
        self.ids=list(items);self.entities={value:key for key,value in entities.items()};self.native=collections.defaultdict(list);self.external=collections.defaultdict(list);self.keys={};self.reasons={}
        for native_id in items:
            key,reason=native_key(self.entities[native_id]);self.keys[native_id]=key
            if key:self.native[key].append(native_id)
            else:self.reasons[native_id]=reason
        for row in metadata_rows:
            key=external_key(row['title'])
            if key:self.external[key].append({'movieId':row['movieId'],'title':row['title'],'genres':row['genres']})
    def lookup(self,native_id):
        native_id=int(native_id)
        if native_id not in self.keys:raise ValueError('native item outside released universe')
        key=self.keys[native_id];result={'native_movie_id':str(native_id),'native_entity':self.entities[native_id],'linked':False,
            'identity_rule':'unique exact normalized title+explicit URI year; no fuzzy/year inference','native_target_identity_unchanged':True}
        if key is None:result['reason']=self.reasons[native_id];return result
        result['normalized_title'],result['year']=key
        if len(self.native[key])!=1:result['reason']='ambiguous_native_title_year';return result
        matches=self.external.get(key,[])
        if not matches:result['reason']='no_exact_external_title_year';return result
        if len(matches)!=1:result['reason']='ambiguous_external_title_year';return result
        row=matches[0]
        if not row['genres'] or row['genres']=='(no genres listed)':result['reason']='genre_missing';return result
        result.update(linked=True,external_metadata_movie_id=row['movieId'],external_original_title=row['title'],genres=row['genres'].split('|'),
            feature_version='external-feature-movielens25m-strict-title-year-v1',source_id='external-genre:'+str(native_id),
            source_kind='external_catalog_metadata',allowed_use='catalog_attribute',semantic_preference_certified=False)
        return result


def run(output=OUT):
    # Only movies.csv is opened; ratings, tags, links and query labels are never read.
    entities=json.loads((DATA/'entity2id.json').read_text());items=json.loads((DATA/'movie_ids.json').read_text())
    with MOVIES.open(newline='',encoding='utf-8') as stream:metadata=list(csv.DictReader(stream))
    linker=StrictNativeGenres(entities,items,metadata);linked={m:linker.lookup(m) for m in items};counts=collections.Counter(r.get('reason','linked') for r in linked.values())
    contexts={}
    for name in ('native_open_evidence_screen_v2','native_open_evidence_transfer_v1'):
        rows=json.loads((ROOT/'experiments/results'/name/'inputs.json').read_text());mids=[int(m['movie_id']) for row in rows for m in row['candidates']];unique=set(mids)
        contexts[name]={'contexts':len(rows),'candidate_occurrences':len(mids),'linked_occurrences':sum(linked[m]['linked'] for m in mids),
            'candidate_occurrence_coverage':sum(linked[m]['linked'] for m in mids)/len(mids),'unique_movies':len(unique),'unique_linked_movies':sum(linked[m]['linked'] for m in unique)}
    source=ROOT/'experiments/results/native_open_evidence_screen_v2/inputs.json';old=json.loads(source.read_text());probe=[linked[int(m['movie_id'])] for m in old[0]['candidates'][:10]]
    result={'source_sha256':sha(Path(__file__)),'native_item_count':len(items),'native_identity_and_candidate_universe_unchanged':True,
        'strict_linked_with_genres':sum(row['linked']for row in linked.values()),'coverage':sum(row['linked']for row in linked.values())/len(items),
        'reasons':dict(counts),'contexts':contexts,'probe10_count':10,'probe10_linked':sum(r['linked']for r in probe),
        'native_entity_sha256':sha(DATA/'entity2id.json'),'native_item_sha256':sha(DATA/'movie_ids.json'),'metadata_sha256':sha(MOVIES),
        'metadata_only_columns':['movieId','title','genres'],'ratings_or_labels_read':False,'current_frozen_inputs_modified':False,
        'risk_notes':['unique exact title/year is reproducible identity evidence, not manual verification',
            'public datasets can contain wrong or different release years; strict mismatch remains missing',
            'titles with translations/aliases/country-qualified film suffixes are deliberately not linked',
            'MovieLens genre taxonomy is external metadata, not ReDial ground truth or current preference',
            'feature availability may be biased toward year-disambiguated films; do not filter evaluation items',
            'any later use must grant all methods same access and mark an external-feature experiment version']}
    output=Path(output);create(output/'probe10.json',probe);create(output/'audit.json',result);print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(OUT));a=p.parse_args();run(a.output)
