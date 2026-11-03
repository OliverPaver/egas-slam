#!/usr/bin/env python3
"""Local-only EGAS-SLAM capture viewer. Serve completed captures or follow a running capture."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import mimetypes
import webbrowser

PROJECT = Path(__file__).resolve().parents[1]

def read_frames(root):
    result=[]
    file=root/'frames.jsonl'
    if file.exists():
        for line in file.read_text().splitlines():
            try: result.append(json.loads(line))
            except json.JSONDecodeError: continue # writer may be between append and flush
    return result

def serve(runs, port=8765, open_browser=False):
    roots={}
    for value in runs:
        p=Path(value).expanduser().resolve()
        if any(x in p.parents for x in (Path('/media'),Path('/mnt'),Path('/run/media'))):
            raise ValueError('Use local capture storage')
        root=p if p.name=='demo' else p/'map/demo'
        roots[p.parent.parent.name if p.name=='demo' else p.name]=root
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            url=urlparse(self.path)
            if url.path=='/api/catalog':
                self.send_json([{'id':k,'frames':len(read_frames(v))} for k,v in roots.items()]);return
            if url.path=='/api/frames':
                key=parse_qs(url.query).get('run',[''])[0]
                if key not in roots:self.send_error(404);return
                self.send_json(read_frames(roots[key]));return
            if url.path.startswith('/data/'):
                parts=unquote(url.path).split('/',3)
                if len(parts)!=4 or parts[2] not in roots:self.send_error(404);return
                base=roots[parts[2]];target=(base/parts[3]).resolve()
            else:
                base=PROJECT/'viewer'
                target=(base/('index.html' if url.path=='/' else unquote(url.path).lstrip('/'))).resolve()
            if not target.is_relative_to(base) or not target.is_file():self.send_error(404);return
            self.send_response(200)
            self.send_header('Content-Type',mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length',str(target.stat().st_size))
            self.send_header('Cache-Control','no-cache' if target.suffix in ('.json','.jsonl','.html','.js') else 'max-age=3600')
            self.end_headers()
            try:
                with target.open('rb') as f:
                    import shutil
                    shutil.copyfileobj(f,self.wfile)
            except (BrokenPipeError,ConnectionResetError):pass
        def send_json(self,value):
            data=json.dumps(value).encode();self.send_response(200)
            self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'EGAS viewer: http://127.0.0.1:{server.server_port}',flush=True)
    if open_browser:webbrowser.open(f'http://127.0.0.1:{server.server_port}')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='*',help='Local capture runs; defaults to the prepared room0 and office demonstrations')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--open',action='store_true')
    a=p.parse_args()
    if not a.runs:
        root=Path.home()/'egas_workspace/runs'
        a.runs=[root/'demo_replica_room0_full',root/'demo_tum_office']
    serve(a.runs,a.port,a.open)
