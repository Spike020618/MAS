"""Ten-movie official metadata feasibility probe; never sends user dialogue.

Explicit movie URI identities and named RDF predicates only. No model APIs,
keys, whole-catalog crawl or inferred relation names. Current experiment inputs
are not modified. Endpoint partial-result headers are retained and flagged.
"""
import datetime,hashlib,json,time
from pathlib import Path
import requests
from experiments.consensus_recommendation.common import ROOT,create,sha

OUT=ROOT/'experiments/results/native_dbpedia_metadata_probe_v1'
ENDPOINT='https://dbpedia.org/sparql'
DOCS='https://www.dbpedia.org/resources/sparql/'
PROPERTIES={key:'http://dbpedia.org/ontology/'+key for key in ('genre','director','starring','abstract')}


def batch_query(movies):
    values=' '.join('<'+m['uri']+'>' for m in movies)
    predicates=' '.join('<'+uri+'>' for uri in PROPERTIES.values())
    return ('SELECT ?movie ?property ?value WHERE { VALUES ?movie { '+values+' } VALUES ?property { '+predicates+' } '
            '?movie ?property ?value . FILTER(!isLiteral(?value) || lang(?value)="" || langMatches(lang(?value),"en")) } '
            'ORDER BY ?movie ?property ?value LIMIT 2000')


def fetch(root,name,query):
    started=time.monotonic();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    request={'endpoint':ENDPOINT,'method':'GET','params':{'query':query,'format':'application/sparql-results+json','timeout':20000},
             'headers':{'Accept':'application/sparql-results+json','User-Agent':'MAS-Native-Metadata-Feasibility/1.0'},
             'dialogue_transmitted':False,'keys_used':False,'started_utc':stamp}
    create(root/(name+'.request.json'),request)
    try:
        response=requests.get(ENDPOINT,params=request['params'],headers=request['headers'],timeout=(10,30),allow_redirects=False)
        raw={'status_code':response.status_code,'response_url':response.url,'headers':dict(response.headers),
             'body':response.text,'elapsed_seconds':time.monotonic()-started,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except requests.RequestException as exc:
        raw={'status_code':None,'transport_error':type(exc).__name__+': '+str(exc),'elapsed_seconds':time.monotonic()-started,
             'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    create(root/(name+'.response.json'),raw);return raw


def parse(raw,movies):
    per={m['uri']:{**m,'properties':{name:[] for name in PROPERTIES}} for m in movies}
    status=raw.get('status_code');rows=[];error=None
    if status==200:
        try:rows=json.loads(raw['body'])['results']['bindings']
        except (ValueError,KeyError,TypeError) as exc:error='unparseable SPARQL response: '+str(exc)
    else:error='HTTP '+str(status) if status is not None else raw.get('transport_error')
    by_property={uri:key for key,uri in PROPERTIES.items()};unexpected=[]
    for row in rows:
        movie=row.get('movie',{}).get('value');predicate=row.get('property',{}).get('value');value=row.get('value')
        if movie not in per or predicate not in by_property or not isinstance(value,dict):unexpected.append(row);continue
        per[movie]['properties'][by_property[predicate]].append(value)
    headers={key.lower():value for key,value in raw.get('headers',{}).items()}
    partial=bool(headers.get('x-sparql-maxrows') or headers.get('x-sql-state') or 'incomplete' in headers.get('x-sql-message','').lower())
    for movie in per.values():movie['missing_properties']=[name for name,values in movie['properties'].items() if not values]
    return {'http_status':status,'parse_error':error,'partial_result_headers_detected':partial,'bindings':len(rows),
            'coverage':{name:sum(bool(row['properties'][name]) for row in per.values()) for name in PROPERTIES},
            'movies':list(per.values()),'unexpected_bindings':unexpected,'no_cross_identity_merge':True}


def run(root=OUT):
    root=Path(root);source=ROOT/'experiments/results/native_open_evidence_screen_v2/inputs.json'
    rows=json.loads(source.read_text());entity_file=ROOT/'experiments/data/raw/crslab_redial_native/released/entity2id.json'
    inverse={value:key[1:-1] for key,value in json.loads(entity_file.read_text()).items()}
    movies=[{'movie_id':m['movie_id'],'title':m['title'],'uri':inverse[int(m['movie_id'])]} for m in rows[0]['candidates'][:10]]
    assert len(movies)==10 and all(m['uri'].startswith('http://dbpedia.org/resource/') and not any(c in m['uri'] for c in '<>\n\r ') for m in movies)
    plan={'scope':'ten movies, old8 candidates, public native-ID metadata feasibility only; no labels or dialogue sent',
        'movie_selection':'first10 candidate IDs in first old8 input, no outcomes used','movies':movies,'official_documentation':DOCS,
        'source_input_sha256':sha(source),'native_identity_sha256':sha(entity_file),'source_sha256':sha(Path(__file__)),
        'properties':PROPERTIES,'endpoint':ENDPOINT,'max_network_calls':2,'no_automatic_retry':True,
        'partial_results_policy':'HTTP200 does not imply completeness; preserve all headers and flag DBpedia anytime/maxrows markers'}
    create(root/'plan.json',plan)
    path=root/'batch10.response.json';raw=json.loads(path.read_text()) if path.exists() else fetch(root,'batch10',batch_query(movies))
    result=parse(raw,movies)
    if result['http_status']!=200 or result['parse_error']:
        path=root/'single_diagnostic.response.json';single=json.loads(path.read_text()) if path.exists() else fetch(root,'single_diagnostic',batch_query(movies[:1]))
        result['single_diagnostic']=parse(single,movies[:1])
    create(root/'metadata.json',result)
    summary={key:result[key] for key in ('http_status','parse_error','partial_result_headers_detected','bindings','coverage')}
    if 'single_diagnostic' in result:summary['single_diagnostic']={key:result['single_diagnostic'][key] for key in ('http_status','parse_error','bindings','coverage')}
    summary.update(dialogue_transmitted=False,keys_used=False,model_api_calls=0,whole_catalog_downloaded=False)
    create(root/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2));return summary

if __name__=='__main__':run()
