"""Application orchestration: persistent sessions, tasks, feedback and pairing."""
import base64,json,math,os,re,secrets,shutil,subprocess,tempfile,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .store import Store
from .catalog import Catalog,parse_turn,ROOT
from .engine import EngineRequest,get_engine,PROVIDERS,METHODS,Cancelled,UnknownCall

class Service:
    def __init__(self,data_dir=None,catalog=None,engine=None):
        self.data=Path(data_dir or os.environ.get('MAS_SYSTEM_DATA',ROOT/'output/system_app'));self.data.mkdir(parents=True,exist_ok=True)
        self.store=Store(self.data/'application.sqlite');self.catalog=catalog or Catalog(self.data/'catalog.jsonl' if (self.data/'catalog.jsonl').exists() else None);self.engine=engine or get_engine()
        self.pool=ThreadPoolExecutor(max_workers=3);self.lock=threading.RLock();self.active=set()
    def bootstrap(self):
        with self.lock:
            first=not self.store.list('profile');r=self.store.profile();r['profile']['admin']=first
            self.store.put('profile',r['profile']['id'],r['profile']['id'],r['profile']);return r
    def config(self):
        return {'engine':self.engine.capabilities(),'providers':[{'id':id,'model':p['model'],'ready':id=='offline'or bool(os.environ.get(p['key']))}for id,p in PROVIDERS.items()],
                'catalog':{'count':len(self.catalog.movies),'version':self.catalog.version},'image':{'ocr':shutil.which('swift')is not None and os.uname().sysname=='Darwin','scope':'本机文字识别；无文字的场景图需补充描述'},'methods':self.engine.capabilities()['methods']}
    def home(self,owner):
        p=self.store.get('profile',owner,owner)
        prefs=p.get('preferences',{})
        has_taste=bool(prefs.get('preferred_genres') or prefs.get('excluded_genres') or p.get('favorites') or p.get('seen') or p.get('disliked'))
        personal=self.catalog.search('',prefs,p,8) if has_taste else []
        popular=self.catalog.search('',{}, {},24)
        return {'personal':personal,'popular':popular[:8],'has_taste':has_taste}
    def session(self,owner):
        id=uuid.uuid4().hex;s={'id':id,'owner':owner,'title':'新对话','messages':[],'constraints':{},'last_ids':[],'created':time.time()}
        self.store.put('session',id,owner,s);return s
    def submit(self,owner,payload):
        query=payload.get('query','')
        if not isinstance(query,str)or not query.strip()or len(query)>3000:raise ValueError('请输入1—3000字的需求')
        sid=payload.get('session_id');s=self.store.get('session',sid,owner)if sid else self.session(owner)
        method=payload.get('method','game');provider=payload.get('provider','offline')
        if method not in self.engine.capabilities()['methods'] or provider not in PROVIDERS:raise ValueError('未知方法或模型')
        if provider!='offline'and not os.environ.get(PROVIDERS[provider]['key']):raise ValueError('该模型尚未配置密钥')
        calls=payload.get('max_calls',3);budget=payload.get('max_cny',.1);top=payload.get('top_k',6)
        if type(calls)is not int or not 0<=calls<=3 or type(top)is not int or not 1<=top<=12 or type(budget)not in(int,float)or not .001<=budget<=2:raise ValueError('预算范围：0—3次核查、0.001—2元、1—12条结果')
        reference=self.store.get('task',payload['reference_task_id'],owner)if payload.get('reference_task_id')else None
        p=self.store.get('profile',owner,owner)
        constraints=parse_turn(query,s['constraints'],p['preferences'],s['last_ids'])
        extra=payload.get('constraints',{})
        if not isinstance(extra,dict):raise ValueError('筛选条件格式错误')
        for key in ('preferred_genres','required_genres','excluded_genres'):
            if key in extra:
                if not isinstance(extra[key],list)or any(not isinstance(x,str)for x in extra[key]):raise ValueError('类型条件格式错误')
                constraints[key]=extra[key]
        for key in ('min_year','max_year'):
            if key in extra:
                if extra[key]is not None and(type(extra[key])is not int or not 1800<=extra[key]<=2200):raise ValueError('年份无效')
                constraints[key]=extra[key]
        if constraints.get('min_year')and constraints.get('max_year')and constraints['min_year']>constraints['max_year']:raise ValueError('开始年份不能晚于结束年份')
        from .catalog import normalize_genres
        conflict=normalize_genres(constraints.get('required_genres',[]))&normalize_genres(constraints.get('excluded_genres',[]))
        if conflict:raise ValueError('必选和排除类型冲突，请调整筛选条件')
        with self.lock:
            if any(t['session_id']==s['id']and t['status']in('running','queued')for t in self.store.list('task',owner)):raise ValueError('当前会话仍有任务运行，可取消后继续')
            # Keep prior substantive query for short follow-ups.
            effective=query
            if s.get('base_query')and len(query)<35 and any(w in query for w in ('换','不要','只要','类似','以后','之前','轻松','其他','年')):effective=s['base_query']+'；本轮要求：'+query
            else:s['base_query']=query
            if any(w in query for w in ('上一部','第一部','这部'))and s['last_ids']and'类似'in query:
                effective='推荐和'+self.catalog.detail(s['last_ids'][0])['title']+'类似的电影；'+query
            explanation_source=None
            if re.search('为什么|推荐理由|推荐依据',query):
                for msg in reversed(s['messages']):
                    if msg['role']=='assistant'and msg.get('task_id'):
                        candidate=self.store.get('task',msg['task_id'],owner)
                        if candidate.get('result')and candidate['result'].get('movies'):
                            explanation_source=candidate['id'];break
            id=uuid.uuid4().hex
            task={'id':id,'owner':owner,'session_id':s['id'],'status':'queued','query':effective,'display_query':query,'constraints':constraints,
                  'profile_snapshot':p,'provider':provider,'method':method,'max_calls':calls,'max_cny':budget,'top_k':top,'events':[],'receipts':{},'result':None,'created':time.time(),'cancel':False,'explanation_source':explanation_source}
            s['constraints']=constraints;s['title']=s['title']if s['messages']else query[:30];s['messages'].append({'role':'user','text':query,'task_id':id,'time':time.time()})
            if reference:
                if reference['status']!='complete':raise ValueError('只能对已完成任务执行同状态比较')
                for key in ('query','display_query','constraints','profile_snapshot','provider','max_calls','max_cny','top_k','candidates','catalog_version'):
                    task[key]=reference[key]
                task['explanation_source']=None
                s['constraints']=task['constraints']
                task['comparison_group']=reference.get('comparison_group',reference['id'])
            else:task['comparison_group']=id
            task['engine_id']=self.engine.capabilities()['id']
            self.store.put('session',s['id'],owner,s);self.store.put('task',id,owner,task);self.active.add(id);self.pool.submit(self.execute,id,owner)
        return task
    def execute(self,id,owner):
        def mutate(fn):
            with self.lock:
                t=self.store.get('task',id,owner);fn(t);self.store.put('task',id,owner,t);return t
        def emit(stage,message,detail):
            mutate(lambda t:t['events'].append({'seq':len(t['events'])+1,'time':time.time(),'stage':stage,'message':message,'detail':detail}))
        def checkpoint(op,slot,value):
            if op=='read':return self.store.get('task',id,owner)['receipts'].get(slot)
            mutate(lambda t:t['receipts'].__setitem__(slot,value))
        def cancelled():return self.store.get('task',id,owner)['cancel']
        try:
            t=mutate(lambda t:t.update(status='running'))
            if t.get('engine_id')!=self.engine.capabilities()['id']:raise ValueError('引擎版本已变化，请创建新任务')
            emit('planning','正在整理需求与偏好',{'constraints':t['constraints']})
            if not t.get('candidates'):
                candidates=self.catalog.search(t['query'],t['constraints'],t['profile_snapshot'],24)
                t=mutate(lambda t:t.update(candidates=candidates,catalog_version=self.catalog.version))
            emit('retrieval','候选检索完成',{'count':len(t['candidates']),'catalog_version':t['catalog_version'],'channels':['关键词','主题标签','用户偏好']})
            if cancelled():raise Cancelled()
            r=EngineRequest(query=t['query'],constraints=t['constraints'],profile=t['profile_snapshot'],candidates=t['candidates'],method=t['method'],provider=t['provider'],max_calls=t['max_calls'],max_cny=t['max_cny'],top_k=t['top_k'])
            if t.get('explanation_source'):
                original=self.store.get('task',t['explanation_source'],owner)['result']
                movies=original['movies'];chosen=movies[:1]if any(x in t['display_query']for x in ('第一','这部','上一部'))else movies[:3]
                text=' '.join(m['title']+'：'+m.get('explanation','依据目录与偏好匹配')for m in chosen)
                result={**original,'changes':[],'allocation':[],'usage':{'calls':0,'tokens':0,'estimated_cny':0,'reserved_cny':0},'summary':text,'explanation_source':t['explanation_source']}
                emit('explanation','根据已有推荐记录解释依据',{'source_task':t['explanation_source'],'candidate_ids':[m['id']for m in chosen]})
            else:result=self.engine.run(r,emit,checkpoint,cancelled)
            if result.get('schema')!='mas.engine.result.v1' or not isinstance(result.get('movies'),list):raise ValueError('引擎返回不兼容的结果')
            ids=[m.get('id')for m in result['movies']]
            allowed={m['id']for m in t['candidates']}
            if t.get('explanation_source'):allowed={m['id']for m in original['movies']}
            if len(ids)>t['top_k']or len(ids)!=len(set(ids))or not set(ids)<=allowed:raise ValueError('引擎返回了范围外或重复候选')
            if any(not isinstance(result.get(k),list)for k in ('reports','changes','allocation'))or not isinstance(result.get('summary'),str):raise ValueError('引擎返回缺少标准过程字段')
            usage=result.get('usage',{})
            if type(usage.get('calls'))is not int or not 0<=usage['calls']<=t['max_calls']:raise ValueError('引擎调用数不符合预算合同')
            if not isinstance(usage.get('estimated_cny'),(int,float))or not math.isfinite(usage['estimated_cny'])or not 0<=usage['estimated_cny']<=t['max_cny']:raise ValueError('引擎费用不符合预算合同')
            if cancelled():raise Cancelled()
            with self.lock:
                mutate(lambda t:t.update(status='complete',result=result,finished=time.time()))
                s=self.store.get('session',t['session_id'],owner);s['last_ids']=[x['id']for x in result['movies']]
                s['messages'].append({'role':'assistant','text':result['summary'],'task_id':id,'time':time.time()});self.store.put('session',s['id'],owner,s)
            emit('complete','推荐已完成',{'usage':result['usage']})
        except Cancelled:mutate(lambda t:t.update(status='cancelled',finished=time.time()))
        except UnknownCall as e:mutate(lambda t:t.update(status='needs_review',error=str(e),finished=time.time()))
        except Exception as e:
            # Never return provider exception bodies or credential-bearing request objects.
            mutate(lambda t:t.update(status='failed',error=str(e)if isinstance(e,(ValueError,KeyError))else type(e).__name__,finished=time.time()))
        finally:
            with self.lock:self.active.discard(id)
    def task(self,owner,id):
        t=self.store.get('task',id,owner)
        # Provider raw responses retained server-side for receipt replay only.
        return {k:v for k,v in t.items()if k not in ('receipts','profile_snapshot','candidates','owner')}
    def resume(self,owner,id):
        with self.lock:
            t=self.store.get('task',id,owner)
            if id in self.active or t['status']not in('interrupted','cancelled','failed'):raise ValueError('该任务不能恢复')
            if any(x['status']=='intent'for x in t['receipts'].values()):raise ValueError('存在完成状态未知的请求，不能自动补发；请创建新任务')
            t.update(status='queued',cancel=False);self.store.put('task',id,owner,t);self.active.add(id);self.pool.submit(self.execute,id,owner)
        return self.task(owner,id)
    def feedback(self,owner,data):
        id=data.get('movie_id');self.catalog.detail(id);action=data.get('action')
        if action not in('favorite','unfavorite','seen','unseen','dislike','undislike'):raise ValueError('未知反馈类型')
        with self.lock:
            p=self.store.get('profile',owner,owner);field={'favorite':'favorites','unfavorite':'favorites','seen':'seen','unseen':'seen','dislike':'disliked','undislike':'disliked'}[action]
            p[field]=[x for x in p[field]if x!=id]
            if not action.startswith('un'):p[field].append(id)
            self.store.put('profile',owner,owner,p)
        return p
    def preferences(self,owner,data):
        if not isinstance(data,dict):raise ValueError('偏好格式错误')
        for k,v in data.items():
            if k not in('preferred_genres','excluded_genres')or not isinstance(v,list)or any(not isinstance(x,str)for x in v):raise ValueError('偏好字段无效')
        p=self.store.get('profile',owner,owner);p['preferences']=data;self.store.put('profile',owner,owner,p);return p
    def pair(self,owner,sid):
        self.store.get('session',sid,owner)
        return {'token':self.store.token(owner,'session:'+sid,900),'expires_in':900,'session_id':sid}
    def ocr(self,data):
        try:raw=base64.b64decode(data['image'].split(',')[-1],validate=True)
        except Exception:raise ValueError('图片编码错误')from None
        if len(raw)>5_000_000:raise ValueError('图片不能超过5MB')
        if not (raw.startswith(b'\x89PNG')or raw.startswith(b'\xff\xd8')or raw.startswith(b'GIF8')):raise ValueError('请上传PNG、JPEG或GIF图片')
        if not self.config()['image']['ocr']:return {'text':'','status':'unavailable','message':'本机未配置文字识别，请填写图片描述'}
        with tempfile.TemporaryDirectory()as tmp:
            path=Path(tmp)/'image';path.write_bytes(raw)
            run=subprocess.run(['swift',str(Path(__file__).with_name('ocr.swift')),str(path)],capture_output=True,text=True,timeout=60)
        if run.returncode:raise ValueError('图片文字识别失败，请换清晰图片或填写描述')
        return {'text':run.stdout.strip()[:3000],'status':'complete','message':'识别到的文字可编辑后用于搜索；未进行人物或无文字场景识别'}
