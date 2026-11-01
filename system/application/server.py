"""Same-origin HTTP application with per-profile bearer tokens and scoped pairing."""
import argparse,json,mimetypes,os,socket
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from .service import Service
from .catalog import ROOT

class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,service=None):
        self.service=service or Service();super().__init__(address,Handler)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass # URLs may contain user text; use persisted task diagnostics.
    def send(self,status,data):
        body=json.dumps(data,ensure_ascii=False).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):self.handle_route('GET')
    def do_POST(self):self.handle_route('POST')
    def handle_route(self,method):
        try:
            parsed=urlparse(self.path);path=parsed.path;q=parse_qs(parsed.query);svc=self.server.service
            if not path.startswith('/api/'):
                if method!='GET':return self.send(405,{'error':'不支持该操作'})
                root=ROOT/'system/frontend/frontend/dist';rel=path.lstrip('/')or'index.html';file=(root/rel).resolve()
                if not file.is_relative_to(root.resolve())or not file.is_file():return self.send(404,{'error':'页面不存在；请先构建前端'})
                body=file.read_bytes();self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(file)[0]or'application/octet-stream');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            data={}
            if method=='POST':
                length=int(self.headers.get('Content-Length','0'))
                if length>7_000_000:raise ValueError('请求过大')
                data=json.loads(self.rfile.read(length)or b'{}')
                if not isinstance(data,dict):raise ValueError('请求必须是JSON对象')
                origin=self.headers.get('Origin')
                if origin and urlparse(origin).netloc!=self.headers.get('Host'):raise PermissionError('只接受同源请求')
            if path=='/api/bootstrap'and method=='POST':return self.send(200,svc.bootstrap())
            auth=svc.store.auth(self.headers.get('Authorization','').removeprefix('Bearer '));owner=auth['owner'];scope=auth['scope']
            limited=scope.startswith('session:');sid=scope.removeprefix('session:')
            if limited:
                allowed=path in('/api/config','/api/ocr')or(path=='/api/tasks'and method=='POST'and data.get('session_id')==sid)or(path=='/api/sessions/'+sid and method=='GET')or(path.startswith('/api/tasks/')and(method=='GET'or(path.endswith('/cancel')and method=='POST')))
                if not allowed:raise PermissionError('扫码入口仅能操作绑定会话')
            if path=='/api/config':return self.send(200,svc.config())
            if path=='/api/home'and method=='GET':return self.send(200,svc.home(owner))
            if path=='/api/profile':
                p=svc.preferences(owner,data)if method=='POST'else svc.store.get('profile',owner,owner);return self.send(200,p)
            if path=='/api/feedback'and method=='POST':return self.send(200,svc.feedback(owner,data))
            if path=='/api/sessions':return self.send(200,svc.session(owner)if method=='POST'else svc.store.list('session',owner))
            if path.startswith('/api/sessions/'):
                return self.send(200,svc.store.get('session',path.split('/')[3],owner))
            if path=='/api/pair'and method=='POST':
                result=svc.pair(owner,data['session_id']);host=self.headers.get('Host','localhost')
                if host.startswith(('localhost:','127.0.0.1:')):
                    sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                    try:sock.connect(('8.8.8.8',80));host=sock.getsockname()[0]+':'+str(self.server.server_port)
                    except OSError:pass
                    finally:sock.close()
                result['url']='http://'+host+'/#/mobile?token='+result['token']+'&session='+result['session_id'];return self.send(200,result)
            if path=='/api/tasks':return self.send(200,svc.task(owner,svc.submit(owner,data)['id'])if method=='POST'else [svc.task(owner,t['id'])for t in svc.store.list('task',owner)])
            if path.startswith('/api/tasks/'):
                parts=path.split('/');id=parts[3];t=svc.task(owner,id)
                if limited and t['session_id']!=sid:raise PermissionError('任务不属于绑定会话')
                if len(parts)>4 and method=='POST':
                    if parts[4]=='resume':return self.send(200,svc.resume(owner,id))
                    if parts[4]=='cancel':
                        with svc.lock:
                            raw=svc.store.get('task',id,owner);raw['cancel']=True;svc.store.put('task',id,owner,raw)
                        return self.send(200,svc.task(owner,id))
                return self.send(200,t)
            if path=='/api/movie':return self.send(200,svc.catalog.detail(q.get('id',[''])[0]))
            if path=='/api/catalog':
                if method=='POST':
                    if not svc.store.get('profile',owner,owner).get('admin'):raise PermissionError('资料导入需要本机管理员身份')
                    # Write to an application-owned copy; leave research inputs untouched.
                    svc.catalog.path=svc.data/'catalog.jsonl';return self.send(200,svc.catalog.import_rows(data.get('movies')))
                return self.send(200,svc.catalog.search(q.get('q',[''])[0],{},svc.store.get('profile',owner,owner),24))
            if path=='/api/ocr'and method=='POST':return self.send(200,svc.ocr(data))
            if path=='/api/research':
                from system.game_consensus_recsys.research_view import inventory,sealed_case
                return self.send(200,sealed_case(q['stage'][0],int(q.get('index',['0'])[0]))if'stage'in q else inventory())
            return self.send(404,{'error':'接口不存在'})
        except PermissionError as e:self.send(403,{'error':str(e)})
        except KeyError:self.send(404,{'error':'记录不存在或无权访问'})
        except (ValueError,TypeError,IndexError)as e:self.send(400,{'error':str(e)})
        except RuntimeError as e:self.send(409,{'error':str(e)})
        except Exception:self.send(500,{'error':'操作失败，请检查本地服务日志和任务状态'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8010);args=parser.parse_args()
    server=Server((args.host,args.port));print(f'影视协作系统 http://{args.host}:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();server.service.pool.shutdown(wait=False,cancel_futures=True)
if __name__=='__main__':main()
