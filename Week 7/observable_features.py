"""Shared historical/live feature formulas. All windows end at the row date."""
import numpy as np

def build_observable_features(market):
    data = market.copy().sort_values('Date').reset_index(drop=True)
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Rolling_Volatility_20D'] = data['Daily_Return'].rolling(20).std(ddof=1) * np.sqrt(252)
    for window in (20, 50):
        data[f'Moving_Average_{window}D'] = data['Close'].rolling(window).mean()
    data['Price_Momentum_20D'] = data['Close'].pct_change(20)
    data['Interest_Rate_Momentum'] = data['Treasury_Rate'].diff()
    data['VIX_Return'] = data['VIX_Close'].pct_change()
    data['VIX_JPM_Correlation_20D'] = data['Daily_Return'].rolling(20).corr(data['VIX_Return'])
    data['Abs_Return_1D'] = data['Daily_Return'].abs()
    for window in (5, 10, 60):
        data[f'Rolling_Volatility_{window}D'] = data['Log_Return'].rolling(window).std(ddof=1) * np.sqrt(252)
    data['Treasury_Rate_Decimal'] = data['Treasury_Rate'] / 100
    data['MA20_Gap'] = data['Close'] / data['Moving_Average_20D'] - 1
    data['MA50_Gap'] = data['Close'] / data['Moving_Average_50D'] - 1
    data['Intraday_Range'] = (data['High'] - data['Low']) / data['Close']
    data['Overnight_Gap'] = data['Open'] / data['Close'].shift(1) - 1
    data['Volume_ZScore_20D'] = (data['Volume'] - data['Volume'].rolling(20).mean()) / data['Volume'].rolling(20).std(ddof=1)
    return data
