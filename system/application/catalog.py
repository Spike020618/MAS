"""Source-backed catalog with lexical, tag and preference retrieval."""
import hashlib,json,math,re,threading
from pathlib import Path
from system.game_consensus_recsys.domain import GENRE_ALIASES as BASE_GENRES
GENRE_ALIASES={**BASE_GENRES,"恐怖":"horror"}
def normalize_genres(values):
    return {GENRE_ALIASES.get(str(x).strip().lower(),str(x).strip().lower()) for x in values}
from system.game_consensus_recsys.movie_data import tokens
ROOT=Path(__file__).resolve().parents[2]
ALIASES={'星际穿越':'Interstellar','盗梦空间':'Inception','机器人总动员':'WALL·E','肖申克的救赎':'Shawshank Redemption','泰坦尼克号':'Titanic','玩具总动员':'Toy Story','疯狂动物城':'Zootopia','千与千寻':'Spirited Away','楚门的世界':'Truman Show','阿凡达':'Avatar','寻梦环游记':'Coco','头脑特工队':'Inside Out','奥本海默':'Oppenheimer','流浪地球':'Wandering Earth','霸王别姬':'Farewell My Concubine'}
CONCEPTS={'温暖':['heartwarming','friendship'],'轻松':['comedy','feel-good'],'烧脑':['mind-bending','twist ending'],'太空':['space','sci-fi'],'家庭':['family','children'],'悬疑':['mystery','suspense'],'机器人':['robots','robot'],'爱情':['romance'],'科幻':['sci-fi'],'动画':['animation'],'感人':['emotional','touching']}

class Catalog:
    def __init__(self,path=None):
        self.path=Path(path or ROOT/'experiments/data/processed/movie_catalog.jsonl')
        self.lock=threading.RLock();self.load()
    def load(self):
        rows=[]
        if self.path.exists():
            with self.path.open()as f:rows=[json.loads(x)for x in f if x.strip()]
        else:
            from system.game_consensus_recsys.catalog import DEFAULT_MOVIES
            rows=[{'movie_id':'demo:'+m.movie_id,'title':m.title,'year':m.year,'genres':m.genres,'description':m.description,'evidence_refs':{'metadata':'bundled_demo_catalog'}}for m in DEFAULT_MOVIES]
        self.movies={str(x['movie_id']):x for x in rows}
        self.version=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()[:12]
        self.texts={id:self.text(m)for id,m in self.movies.items()}
    def text(self,m):
        return ' '.join([m.get('title',''),m.get('description',''),' '.join(m.get('genres',[])),' '.join(x['tag']for x in m.get('semantic_tags',[]))]).lower()
    def detail(self,id):
        if id not in self.movies:raise KeyError('影片不存在')
        m=self.movies[id]
        return {'id':id,'title':m['title'],'year':m.get('year'),'genres':m.get('genres',[]),'description':m.get('description',''),
                'tags':m.get('semantic_tags',[])[:12],'rating':m.get('rating_mean'),'rating_scale':5 if m.get('rating_mean')is not None else None,
                'rating_count':m.get('rating_count'),'rating_source':'MovieLens' if m.get('rating_mean')is not None else None,
                'poster':m.get('poster_url') if str(m.get('poster_url','')).startswith('https://')else None,
                'links':{'IMDb':'https://www.imdb.com/title/'+m['imdb_id']+'/'}if re.fullmatch(r'tt\d+',str(m.get('imdb_id','')))else {},
                'sources':m.get('evidence_refs',{}),'catalog_version':self.version}
    def search(self,query,constraints,profile,limit=24):
        expanded=query
        for k,v in ALIASES.items():
            if k in query:expanded+=' '+v
        ts=tokens(expanded)-{'推荐','电影','几部','一些','不要','喜欢','只要','换几','几部','看看','找一','一部'}
        concepts={v for k,vs in CONCEPTS.items()if k in query for v in vs}
        preferred=normalize_genres(constraints.get('preferred_genres',[]))
        required=normalize_genres(constraints.get('required_genres',[]));excluded=normalize_genres(constraints.get('excluded_genres',[]))
        exclude_ids=set(profile.get('seen',[])+profile.get('disliked',[])+constraints.get('excluded_movie_ids',[]))
        liked=set(profile.get('favorites',[]));likedgenres=set()
        for id in liked:
            if id in self.movies:likedgenres.update(normalize_genres(self.movies[id].get('genres',[])))
        similar=any(w in query for w in ('类似','相似','像'))
        anchors=[m for m in self.movies.values()if len(m['title'])>3 and m['title'].lower()in expanded.lower()]if similar else[]
        anchor_ids={str(m['movie_id'])for m in anchors}
        anchor_tags={t['tag'].lower()for m in anchors for t in m.get('semantic_tags',[])}
        anchor_genres={g for m in anchors for g in normalize_genres(m.get('genres',[]))}
        ranked=[]
        for id,m in self.movies.items():
            genres=normalize_genres(m.get('genres',[]));year=m.get('year');text=self.texts[id]
            if id in anchor_ids or id in exclude_ids or excluded&genres or not required<=genres:continue
            if constraints.get('min_year')and(not year or year<constraints['min_year']):continue
            if constraints.get('max_year')and(not year or year>constraints['max_year']):continue
            hits=sum(t in text for t in ts if len(t)>1);taghits=sum(t in text for t in concepts)
            titlehit=any(v.lower() in m['title'].lower()for k,v in ALIASES.items()if k in query)
            pref=len(preferred&genres);affinity=len(likedgenres&genres)
            simtags={t['tag'].lower()for t in m.get('semantic_tags',[])}
            similarity=.25*len(anchor_tags&simtags)/max(1,len(anchor_tags|simtags))+.15*len(anchor_genres&genres)/max(1,len(anchor_genres|genres))
            pop=math.log1p(m.get('rating_count',0))/14
            score=hits*.13+taghits*.12+pref*.24+affinity*.04+pop*.08+(0 if similar else titlehit*2)+similarity
            ranked.append((score,id,{'关键词匹配':round(hits*.13+titlehit*2,3),'主题标签匹配':round(taghits*.12,3),'偏好匹配':round(pref*.24+affinity*.04,3),'资料覆盖':round(pop*.08,3),'相似影片':round(similarity,3)}))
        ranked.sort(key=lambda x:(-x[0],x[1]))
        return [dict(self.detail(id),retrieval_score=round(score,4),retrieval_components=parts)for score,id,parts in ranked[:limit]]
    def import_rows(self,rows):
        if not isinstance(rows,list)or not 1<=len(rows)<=1000:raise ValueError('每次导入1—1000条影片记录')
        for m in rows:
            if not isinstance(m,dict)or type(m.get('movie_id'))not in(str,int)or not str(m['movie_id']).strip()or not isinstance(m.get('title'),str)or not m['title'].strip()or not isinstance(m.get('genres'),list)or not isinstance(m.get('evidence_refs'),dict)or not m['evidence_refs']:raise ValueError('每条记录需要movie_id/title/genres/evidence_refs')
            if m.get('year')is not None and(type(m['year'])is not int or not 1800<=m['year']<=2200):raise ValueError('年份无效')
            if any(not isinstance(g,str)for g in m['genres']):raise ValueError('类型必须是字符串列表')
            if any(not isinstance(v,str)for v in m['evidence_refs'].values()):raise ValueError('资料来源必须是字符串')
            if m.get('rating_mean')is not None and(type(m['rating_mean'])not in(int,float)or not 0<=m['rating_mean']<=5):raise ValueError('目录评分必须是0—5的数字')
            if type(m.get('rating_count',0))is not int or m.get('rating_count',0)<0:raise ValueError('评分人数无效')
            if not isinstance(m.get('semantic_tags',[]),list)or any(not isinstance(t,dict)or not isinstance(t.get('tag'),str)for t in m.get('semantic_tags',[])):raise ValueError('主题标签格式无效')
            if not isinstance(m.get('description',''),str):raise ValueError('简介格式无效')
        with self.lock:
            merged=dict(self.movies)
            for m in rows:merged[str(m['movie_id'])]=m
            self.path.parent.mkdir(parents=True,exist_ok=True)
            tmp=self.path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(m,ensure_ascii=False)+'\n'for m in merged.values()));tmp.replace(self.path);self.load()
        return {'count':len(self.movies),'version':self.version}

def parse_turn(query,previous,preferences,last_ids=()):
    c={**preferences,**previous}
    # Long-term preferences remain soft signals even when a session has its own preferences.
    blocked=normalize_genres(previous.get('excluded_genres',[]))
    c['preferred_genres']=[g for g in dict.fromkeys(preferences.get('preferred_genres',[])+previous.get('preferred_genres',[])) if not normalize_genres([g]) & blocked]
    for key in ('preferred_genres','excluded_genres','required_genres','excluded_movie_ids'):c[key]=list(c.get(key,[]))
    if re.search('重新开始|清除条件|不限类型',query):c={k:[]for k in ('preferred_genres','excluded_genres','required_genres','excluded_movie_ids')}
    for zh,en in GENRE_ALIASES.items():
        if zh not in query:continue
        prefix=re.split('[，。；,]',query[:query.rfind(zh)])[-1]
        cues=re.findall('不要|排除|不看|不喜欢|别推荐|只要|必须|只看|喜欢|想看|偏好',prefix)
        cue=cues[-1]if cues else''
        if cue in ('不要','排除','不看','不喜欢','别推荐'):key='excluded_genres';opposite=('preferred_genres','required_genres')
        elif cue in ('只要','必须','只看'):key='required_genres';opposite=('excluded_genres',)
        else:key='preferred_genres';opposite=('excluded_genres',)
        c[key]=sorted(set(c.get(key,[])+[en]))
        for k in opposite:c[k]=[x for x in c.get(k,[])if x!=en]
    yr=re.search(r'(19\d{2}|20\d{2})\s*[至到—-]\s*(19\d{2}|20\d{2})',query)
    if yr:c.update(min_year=int(yr[1]),max_year=int(yr[2]))
    else:
        yr=re.search(r'(19\d{2}|20\d{2})\s*年?\s*(以后|之后|以来|以前|之前)',query)
        if yr:c['max_year'if yr[2]in('以前','之前')else'min_year']=int(yr[1])
    if '不限年份'in query:c.pop('min_year',None);c.pop('max_year',None)
    if re.search('换一批|换几部|都看过|别的|其他几部',query):c['excluded_movie_ids']=sorted(set(c.get('excluded_movie_ids',[])+list(last_ids)))
    return c
