"""Local persistent profiles, sessions and resumable jobs. Tokens are never logged."""
import hashlib, json, secrets, sqlite3, threading, time, uuid
from pathlib import Path

class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db=sqlite3.connect(str(path),check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.lock=threading.RLock()
        with self.db:
            self.db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS objects(kind TEXT,id TEXT,owner TEXT,data TEXT,updated REAL,PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS tokens(hash TEXT PRIMARY KEY,owner TEXT,scope TEXT,expires REAL);
            ''')
        # Interrupted external requests must not be implicitly resubmitted.
        for task in self.list('task'):
            if task['status'] in ('running','queued'):
                task['status']='interrupted';self.put('task',task['id'],task['owner'],task)
    def put(self,kind,id,owner,data):
        with self.lock,self.db:
            self.db.execute('INSERT OR REPLACE INTO objects VALUES(?,?,?,?,?)',(kind,id,owner,json.dumps(data,ensure_ascii=False),time.time()))
    def get(self,kind,id,owner=None):
        with self.lock:
            r=self.db.execute('SELECT * FROM objects WHERE kind=? AND id=?',(kind,id)).fetchone()
        if not r or (owner is not None and r['owner']!=owner):raise KeyError('记录不存在')
        return json.loads(r['data'])
    def list(self,kind,owner=None):
        with self.lock:
            rows=self.db.execute('SELECT * FROM objects WHERE kind=?'+(' AND owner=?' if owner else '')+' ORDER BY updated DESC',(kind,owner)if owner else(kind,)).fetchall()
        return [json.loads(r['data'])for r in rows]
    def delete(self,kind,id,owner):
        self.get(kind,id,owner)
        with self.lock,self.db:self.db.execute('DELETE FROM objects WHERE kind=? AND id=? AND owner=?',(kind,id,owner))
    def token(self,owner,scope='profile',ttl=365*86400):
        token=secrets.token_urlsafe(32)
        with self.lock,self.db:self.db.execute('INSERT INTO tokens VALUES(?,?,?,?)',(hashlib.sha256(token.encode()).hexdigest(),owner,scope,time.time()+ttl))
        return token
    def auth(self,token):
        with self.lock:r=self.db.execute('SELECT * FROM tokens WHERE hash=?',(hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if not r or r['expires']<time.time():raise PermissionError('请重新建立本地会话')
        return dict(r)
    def profile(self):
        id=uuid.uuid4().hex
        p={'id':id,'preferences':{'preferred_genres':[],'excluded_genres':[]},'favorites':[],'seen':[],'disliked':[]}
        self.put('profile',id,id,p)
        return {'token':self.token(id),'profile':p}
