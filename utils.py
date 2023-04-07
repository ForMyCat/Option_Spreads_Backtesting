import pandas as pd
import numpy as np
import yfinance as yf

def OHLC_info_by_range(stock_code,start_date,end_date,interval = '1d'):
    #define the ticker symbol
    tickerSymbol = stock_code

    loc_tickerDf = yf.download(tickerSymbol, start= start_date, end= end_date, interval = interval)
    print("Start date:", start_date)
    print("End date:", end_date)

    loc_tickerDf.reset_index(inplace = True)
    loc_tickerDf['Date'] = pd.to_datetime(loc_tickerDf['Date']).dt.tz_localize(None)
    loc_tickerDf.dropna(inplace = True)         
    loc_tickerDf.reset_index(inplace = True)
    return loc_tickerDf

def calculate_actual_earn(x):
    if x.isCALL:
        buy_itm = x['Price@Expiration'] > x.BuyStrike
        sell_itm = x['Price@Expiration'] > x.SellStrike
        actual_earn = x.Premium + buy_itm * (x['Price@Expiration'] - x.BuyStrike) - sell_itm * (x['Price@Expiration'] - x.SellStrike)
    else:
        buy_itm = x['Price@Expiration'] < x.BuyStrike
        sell_itm = x['Price@Expiration'] < x.SellStrike
        actual_earn = x.Premium + buy_itm * (x.BuyStrike - x['Price@Expiration']) - sell_itm * (x.SellStrike - x['Price@Expiration'])

    return actual_earn