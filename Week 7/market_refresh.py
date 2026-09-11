"""Refresh dated source histories and rebuild the complete observable feature row.

FRED credentials are read from the environment or a local .env file, never saved
to snapshots. Both Treasury series are retained: DGS10 matches model training,
while DGS1 supplies the one-year contract discount rate.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import os
import time
import tempfile
import numpy as np
import pandas as pd
from observable_features import build_observable_features

BASE = Path(__file__).resolve().parent

def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,suffix='.tmp',delete=False) as stream:
        json.dump(value,stream,indent=2)
        temp=Path(stream.name)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)

def fred_key():
    key = os.environ.get('FRED_API_KEY')
    if key:
        return key
    for parent in [BASE, *BASE.parents]:
        file = parent / '.env'
        if file.is_file():
            for line in file.read_text(encoding='utf-8-sig').splitlines():
                name, sep, value = line.partition('=')
                if sep and name.strip() == 'FRED_API_KEY':
                    return value.strip().strip('\"\'')
    return None

def download(url, timeout=12, attempts=2):
    # Never propagate exceptions containing authenticated URLs to the UI/logs.
    error = 'NetworkError'
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers={'User-Agent':'Mozilla/5.0'}), timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            error = type(exc).__name__
            if attempt + 1 < attempts:
                time.sleep(0.4)
    raise RuntimeError(error)

def fetch_jpm():
    chart = json.loads(download('https://query1.finance.yahoo.com/v8/finance/chart/JPM?range=5y&interval=1d'))['chart']['result'][0]
    dates = pd.to_datetime(chart['timestamp'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
    quote = chart['indicators']['quote'][0]
    raw_close = pd.Series(quote['close'],dtype=float)
    adjusted = chart['indicators'].get('adjclose',[{}])[0].get('adjclose',quote['close'])
    factor = pd.Series(adjusted,dtype=float) / raw_close
    data = pd.DataFrame({'Date':dates,'Close':adjusted,'Volume':quote['volume']})
    for name in ['Open','High','Low']:
        data[name] = pd.Series(quote[name.lower()],dtype=float)*factor
    today = pd.Timestamp.now(tz='America/New_York').tz_localize(None).normalize()
    data = data[data['Date'] < today].dropna().sort_values('Date')
    if len(data)<100:
        raise ValueError('Insufficient JPM feature history')
    return {'history':data.to_json(orient='records',date_format='iso')}

def fetch_vix():
    frame=pd.read_csv(BytesIO(download('https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv')))
    frame=frame.rename(columns={'DATE':'Date','CLOSE':'VIX_Close'})[['Date','VIX_Close']]
    frame['Date']=pd.to_datetime(frame['Date'],format='%m/%d/%Y')
    return {'history':frame.to_json(orient='records',date_format='iso')}

def fetch_rate(series):
    start=(pd.Timestamp.now()-pd.DateOffset(years=5,days=7)).date().isoformat()
    key=fred_key()
    if key:
        query=urlencode({'series_id':series,'api_key':key,'file_type':'json','observation_start':start})
        payload=json.loads(download('https://api.stlouisfed.org/fred/series/observations?'+query))
        frame=pd.DataFrame(payload['observations'])[['date','value']]
        method='FRED authenticated API'
    else:
        frame=pd.read_csv(BytesIO(download(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}')))
        method='FRED public CSV'
    frame.columns=['Date',series]
    frame['Date']=pd.to_datetime(frame['Date'])
    frame[series]=pd.to_numeric(frame[series],errors='coerce')
    frame=frame.dropna()
    if frame.empty:
        raise ValueError('Empty Treasury series')
    return {'history':frame.to_json(orient='records',date_format='iso'),'method':method}

def refresh(previous, cache_dir):
    cache_dir=Path(cache_dir)
    cache_dir.mkdir(parents=True,exist_ok=True)
    jobs={'JPM':fetch_jpm,'VIX':fetch_vix,'DGS10':lambda:fetch_rate('DGS10'),'DGS1':lambda:fetch_rate('DGS1')}
    sources={}; status={}
    def run(name,fn):
        path=cache_dir/f'{name}.json'
        try:
            value=fn()
            value['retrieved_at_utc']=datetime.now(timezone.utc).isoformat()
            atomic_json(path,value)
            return name,value,{'status':'online_success'}
        except Exception as exc:
            try:
                value=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
            except (OSError, ValueError):
                value=None
            return name,value,{'status':'cached_source' if value else 'unavailable','error':type(exc).__name__}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,k,v) for k,v in jobs.items()]
        for future in futures:
            name,value,item=future.result(); sources[name]=value; status[name]=item
    if any(sources[name] is None for name in ['JPM','VIX','DGS10']):
        result=dict(previous)
        result.update(update_status='cached_fallback',source_status=status,online_error='Required source history unavailable')
        return result
    frames={}
    for name,value in sources.items():
        if value is not None:
            frames[name]=pd.read_json(BytesIO(value['history'].encode()),orient='records')
            frames[name]['Date']=pd.to_datetime(frames[name]['Date']).dt.normalize()
            frames[name]=frames[name].sort_values('Date').drop_duplicates('Date')
            status[name]['as_of_date']=str(frames[name]['Date'].max().date())
    # A common completed date prevents pairing today's spot with old features.
    as_of=min(frames[k]['Date'].max() for k in ['JPM','VIX','DGS10'])
    data=frames['JPM'][frames['JPM']['Date']<=as_of].copy()
    for key in ['VIX','DGS10']:
        data=pd.merge_asof(data,frames[key],on='Date',direction='backward')
    data=data.rename(columns={'DGS10':'Treasury_Rate'})
    data=build_observable_features(data)
    data['Log_Moneyness']=np.log(data['Close']/150)
    metadata_path=BASE/'assets/models/model_bundle_metadata.json'
    if not metadata_path.exists():
        metadata_path=BASE.parent/'Week 6/trained_models/model_bundle_metadata.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    required=metadata['pricing_features']
    complete=data.dropna(subset=required)
    if complete.empty:
        raise ValueError('No complete feature row after source alignment')
    row=complete.iloc[-1]
    if not np.isfinite(row[required].astype(float)).all():
        raise ValueError('Non-finite model features')
    rate=float(row['Treasury_Rate_Decimal']); rate_source='DGS10 fallback'
    if 'DGS1' in frames:
        rates=frames['DGS1'][frames['DGS1']['Date']<=row['Date']]
        if not rates.empty:
            rate=float(rates.iloc[-1]['DGS1'])/100; rate_source='DGS1'
    context={'feature_as_of_date':str(row['Date'].date()),'pricing_rate':rate,
             'values':{name:float(row[name]) for name in required}}
    return {'as_of_date':context['feature_as_of_date'],'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
            'mode':'online' if all(v['status']=='online_success' for v in status.values()) else 'mixed_sources',
            'update_status':'online_success' if all(v['status']=='online_success' for v in status.values()) else 'partial_refresh',
            'jpm_close':float(row['Close']),'vix_close':float(row['VIX_Close']),
            'risk_free_rate':rate,'pricing_rate_source':rate_source,
            'historical_volatility_20d':float(row['Rolling_Volatility_20D']),
            'feature_context':context,'source_status':status,
            'source_dates':{k:v.get('as_of_date') for k,v in status.items()},'online_error':None,
            'feature_note':'All features rebuilt at a common completed date; ML Treasury input remains DGS10.'}
