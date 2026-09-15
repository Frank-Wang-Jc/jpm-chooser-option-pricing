"""Regression tests for negative live predictions and structural extrapolation."""
from pathlib import Path
import sys, json, pickle, unittest
import numpy as np
import pandas as pd
import joblib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Week 8/pricing_tool'))
from pricing_engine import PRICING_MODEL, METADATA, FEATURE_CONTEXT, price_contract
from bounded_pricing import BoundedChooserRegressor

class DirectPricingTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([FEATURE_CONTEXT['values']])[METADATA['pricing_features']]

    def test_grid_bounds_and_finite(self):
        frame=pd.concat([self.frame()]*600,ignore_index=True)
        frame['Close']=np.geomspace(.01,1e7,len(frame))
        frame['Log_Moneyness']=np.log(frame.Close/150)
        for rate in [-.05,0,.0428,.30]:
            frame['Treasury_Rate_Decimal']=rate
            for vol in [.01,.1422,.5,3.0]:
                frame['Rolling_Volatility_20D']=vol
                p=PRICING_MODEL.predict(frame)
                lo,hi=PRICING_MODEL.price_bounds(frame)
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p>=lo-1e-8)&(p<=hi+1e-8)).all())
                # With other inputs fixed, very distant moneyness loses time value.
                self.assertLess(abs(p[-1]-lo[-1]),1e-5)

    def test_screenshot_input_has_real_third_price(self):
        r=price_contract(353.56,rate=.0428,volatility=.1422)
        self.assertGreater(r['approach2_direct_price'],200)
        self.assertLessEqual(r['direct_price_bounds'][0],r['approach2_direct_price'])
        self.assertIn('Close',r['out_of_training_range'])

    def test_actual_pricing_rate_separate_from_ml_rate(self):
        f=self.frame(); f['Close']=353.56; f['Log_Moneyness']=np.log(353.56/150)
        f['Treasury_Rate_Decimal']=.0528
        lower,_=PRICING_MODEL.price_bounds(f,pricing_rate=.0428)
        expected=abs(353.56*np.exp(-.0233)-150*np.exp(-.0428))
        self.assertAlmostEqual(lower[0],expected,places=10)

    def test_serialization_matches(self):
        f=self.frame(); expected=PRICING_MODEL.predict(f)
        for directory in [ROOT/'Week 6/trained_models',ROOT/'Week 8/pricing_tool/assets/models']:
            with open(directory/'best_direct_pricing_model.pkl','rb') as stream: model=pickle.load(stream)
            np.testing.assert_allclose(model.predict(f),expected,rtol=0,atol=1e-12)
            np.testing.assert_allclose(joblib.load(directory/'best_direct_pricing_model.joblib').predict(f),expected,rtol=0,atol=1e-12)

    def test_no_invalid_feature_silent_zero(self):
        f=self.frame(); f.loc[0,'Close']=np.nan
        with self.assertRaises(ValueError): PRICING_MODEL.predict(f)

    def test_contract_restriction_retained(self):
        self.assertFalse(price_contract(353.56,strike=160)['ml_available'])

if __name__=='__main__': unittest.main()
