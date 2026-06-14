#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, os, threading, time
from pathlib import Path
import requests

PART_SIZE=160*1024*1024
TLS=threading.local()

def session():
    if not hasattr(TLS,'s'):
        TLS.s=requests.Session(); TLS.s.trust_env=False
    return TLS.s

def head(url):
    r=session().head(url,allow_redirects=True,timeout=60); r.raise_for_status()
    size=int(r.headers.get('x-linked-size') or r.headers['content-length'])
    return size,r.url

def fetch(task):
    url,p,start,end=task; want=end-start+1
    for attempt in range(12):
        have=p.stat().st_size if p.exists() else 0
        if have==want: return want
        if have>want: p.unlink(); have=0
        try:
            headers={'Range':f'bytes={start+have}-{end}'}
            with session().get(url,headers=headers,stream=True,timeout=(30,120)) as r:
                if r.status_code not in (200,206): r.raise_for_status()
                if have and r.status_code==200:
                    p.unlink(missing_ok=True); have=0; continue
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('ab' if have else 'wb') as f:
                    for chunk in r.iter_content(4*1024*1024):
                        if chunk: f.write(chunk)
            if p.stat().st_size==want: return want
        except Exception as e:
            print(f'[retry] {p.name} attempt={attempt+1} have={p.stat().st_size if p.exists() else 0}/{want} error={e}',flush=True)
            time.sleep(min(30,2*(attempt+1)))
    raise RuntimeError(f'failed part {p}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--workers',type=int,default=8); ap.add_argument('files',nargs='+'); a=ap.parse_args()
    out=Path(a.output_dir); tmp=out/'._____temp'; out.mkdir(parents=True,exist_ok=True); tmp.mkdir(exist_ok=True)
    all_tasks=[]; infos=[]
    for name in a.files:
        url=f'https://hf-mirror.com/{a.repo}/resolve/main/{name}'
        size,final_url=head(url); final=out/name
        if final.exists() and final.stat().st_size==size:
            print(f'[skip] {name} complete size={size}',flush=True); continue
        tasks=[]
        for start in range(0,size,PART_SIZE):
            end=min(size-1,start+PART_SIZE-1); part=tmp/f'{name}_{start}_{end}'; tasks.append((final_url,part,start,end))
        reused=sum(p.stat().st_size for _,p,_,_ in tasks if p.exists())
        print(f'[plan] {name} size={size} reused={reused} parts={len(tasks)}',flush=True)
        all_tasks.extend(tasks); infos.append((name,size,final,tasks))
    if all_tasks:
        done=0; total=sum(e-s+1 for _,_,s,e in all_tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            for n in ex.map(fetch,all_tasks):
                done+=n; print(f'[progress] completed_parts_bytes={done}/{total}',flush=True)
    for name,size,final,tasks in infos:
        staging=final.with_suffix(final.suffix+'.merging')
        with staging.open('wb') as w:
            for _,part,start,end in tasks:
                if part.stat().st_size != end-start+1: raise RuntimeError(f'invalid part {part}')
                with part.open('rb') as r:
                    while True:
                        b=r.read(16*1024*1024)
                        if not b: break
                        w.write(b)
        if staging.stat().st_size!=size: raise RuntimeError(f'merged size mismatch {name}')
        os.replace(staging,final); print(f'[complete] {name} size={size}',flush=True)
if __name__=='__main__': main()
