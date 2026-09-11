"""Snapshot persistence with independent source caches and secret-safe failures."""
from pathlib import Path
import json
import os
from market_refresh import refresh, atomic_json

BASE=Path(__file__).resolve().parent
CACHE=BASE/'assets/data/latest_market_snapshot.json'
RUNTIME=Path(os.environ.get('JPM_CACHE_DIR',str(BASE/'assets/data/runtime')))

def load_cached_snapshot():
    path=RUNTIME/'snapshot.json'
    if os.environ.get('JPM_OFFLINE')=='1' or not path.exists():
        path=CACHE
    try:
        snapshot=json.loads(path.read_text(encoding='utf-8'))
        if not all(key in snapshot for key in ['jpm_close','risk_free_rate','vix_close','historical_volatility_20d']):
            raise ValueError('Incomplete snapshot')
    except (OSError,ValueError):
        snapshot=json.loads(CACHE.read_text(encoding='utf-8'))
    snapshot['update_status']='cached_fallback'
    return snapshot

def online_snapshot():
    return refresh(load_cached_snapshot(),RUNTIME/'sources')

def update_snapshot():
    previous=load_cached_snapshot()
    try:
        snapshot=online_snapshot()
        if snapshot.get('feature_context') and snapshot.get('update_status')!='cached_fallback':
            RUNTIME.mkdir(parents=True,exist_ok=True)
            atomic_json(RUNTIME/'snapshot.json', snapshot)
        return snapshot
    except Exception as exc:
        previous['online_error']=type(exc).__name__
        return previous

if __name__=='__main__':
    value=update_snapshot()
    print(json.dumps({k:value.get(k) for k in ['as_of_date','update_status','source_status']},indent=2))
    if value.get('update_status')=='cached_fallback':
        raise SystemExit(1)
