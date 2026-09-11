"""Reproducibility checks for point-in-time features and source failures."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'Week 8/pricing_tool'
sys.path.insert(0,str(APP))
from observable_features import build_observable_features
import market_refresh

class FeatureTests(unittest.TestCase):
    def test_future_prices_do_not_change_past_features(self):
        market=pd.read_csv(ROOT/'Week2/processed_data/market_data_processed.csv',parse_dates=['Date'])
        original=build_observable_features(market)
        changed=market.copy()
        changed.loc[changed.Date>='2022-01-01','Close']*=10
        revised=build_observable_features(changed)
        pd.testing.assert_frame_equal(original.loc[original.Date<'2022-01-01'],revised.loc[revised.Date<'2022-01-01'])

    def test_training_and_packaged_features_match(self):
        market=pd.read_csv(ROOT/'Week2/processed_data/market_data_processed.csv',parse_dates=['Date'])
        saved=pd.read_csv(ROOT/'Week 5/model_results/ml_dataset.csv',parse_dates=['Date']).set_index('Date')
        actual=build_observable_features(market).set_index('Date').loc[saved.index]
        names=json.loads((APP/'assets/models/model_bundle_metadata.json').read_text())['volatility_features']
        np.testing.assert_allclose(actual[names],saved[names],atol=1e-12,rtol=1e-10)

    def test_retry_errors_do_not_disclose_urls(self):
        with patch.object(market_refresh,'urlopen',side_effect=RuntimeError('https://example.test?api_key=SECRET')):
            with self.assertRaisesRegex(RuntimeError,'^RuntimeError$'):
                market_refresh.download('https://example.test?api_key=SECRET',attempts=1)

    def test_single_source_cache(self):
        # Synthetic source histories are used only as test fixtures, never as data.
        dates=pd.bdate_range('2020-01-01',periods=800)
        x=np.arange(len(dates)); close=100+0.1*x+np.sin(x)
        jpm=pd.DataFrame({'Date':dates,'Close':close,'High':close+1,'Low':close-1,'Open':close-.1,'Volume':10000+x%23})
        sources={'JPM':{'history':jpm.to_json(orient='records',date_format='iso')},
                 'VIX':{'history':pd.DataFrame({'Date':dates,'VIX_Close':20+np.cos(x)}).to_json(orient='records',date_format='iso')}}
        for name in ['DGS1','DGS10']:
            sources[name]={'history':pd.DataFrame({'Date':dates,name:4+np.sin(x)/10}).to_json(orient='records',date_format='iso')}
        with tempfile.TemporaryDirectory() as directory:
            for name,value in sources.items():
                (Path(directory)/f'{name}.json').write_text(json.dumps(value))
            with patch.object(market_refresh,'fetch_jpm',return_value=sources['JPM']),patch.object(market_refresh,'fetch_vix',return_value=sources['VIX']),patch.object(market_refresh,'fetch_rate',side_effect=lambda name: (_ for _ in ()).throw(TimeoutError('temporary')) if name=='DGS1' else sources[name]):
                result=market_refresh.refresh({},directory)
            self.assertEqual(result['source_status']['DGS1']['status'],'cached_source')
            self.assertEqual(result['source_status']['JPM']['status'],'online_success')
            self.assertEqual(result['update_status'],'partial_refresh')
            self.assertEqual(len(result['feature_context']['values']),19)
            self.assertEqual(result['feature_context']['feature_as_of_date'],dates[-1].date().isoformat())

    def test_corrupt_cache_and_all_sources_down(self):
        previous={'as_of_date':'2024-12-30','jpm_close':231.08}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'JPM.json').write_text('not json')
            with patch.object(market_refresh,'fetch_jpm',side_effect=TimeoutError),patch.object(market_refresh,'fetch_vix',side_effect=TimeoutError),patch.object(market_refresh,'fetch_rate',side_effect=TimeoutError):
                result=market_refresh.refresh(previous,directory)
            self.assertEqual(result['jpm_close'],previous['jpm_close'])
            self.assertEqual(result['update_status'],'cached_fallback')

if __name__=='__main__':
    unittest.main()
