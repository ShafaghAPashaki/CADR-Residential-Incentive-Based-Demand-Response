"""Refresh frozen source manifests after pre-training calibration changes.

Old results are not hashed as source, and previous manifests are excluded to
avoid a self-referential hash. This does not update or touch checkpoints.
"""
from pathlib import Path
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]

def digest(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def static_file(p:Path)->bool:
    r=p.relative_to(ROOT).parts
    if '__pycache__' in r or p.suffix=='.pyc':return False
    if p.name in ('source_hashes.json','benchmark_source_hashes.json'):return False
    if len(r)>1 and r[0]=='results' and r[1] not in ('forecast','calibration','final_diagnostics'):
        return False
    return True

def main():
    bench=ROOT/'benchmark'
    b={p.relative_to(bench).as_posix():digest(p) for p in sorted(bench.rglob('*')) if p.is_file() and static_file(p)}
    (bench/'benchmark_source_hashes.json').write_text(json.dumps(b,indent=2,sort_keys=True)+'\n')
    a={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and static_file(p)}
    # Include the benchmark manifest itself in top-level source manifest.
    a['benchmark/benchmark_source_hashes.json']=digest(bench/'benchmark_source_hashes.json')
    (ROOT/'source_hashes.json').write_text(json.dumps(a,indent=2,sort_keys=True)+'\n')
    print(f'Refreshed {len(a)} root static-source hashes, {len(b)} benchmark hashes')

if __name__=='__main__':main()
