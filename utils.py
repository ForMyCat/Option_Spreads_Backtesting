import pandas as pd
import numpy as np
import yfinance as yf

def OHLC_info_by_day(stock_code,query_days=60,interval = '1d'):
    today = date.today() + timedelta(days = 2)

    #define the ticker symbol
    tickerSymbol = stock_code

    #get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)

    #get the historical prices for this ticker
    start_date = (today - timedelta(days=query_days)).strftime("%Y-%m-%d")
    end_date= today.strftime("%Y-%m-%d")

    loc_tickerDf = tickerData.history(interval = interval, start= start_date, end= end_date)
    # print("Start date:", start_date)
    # print("End date:", end_date)
    loc_tickerDf.dropna(inplace = True)
    loc_tickerDf.reset_index(inplace = True)
    return loc_tickerDf

def OHLC_info_by_range(stock_code,start_date,end_date,interval = '1d'):

    #define the ticker symbol
    tickerSymbol = stock_code

    loc_tickerDf = yf.download(tickerSymbol, start= start_date, end= end_date, interval = interval)
    print("Start date:", start_date)
    print("End date:", end_date)

    loc_tickerDf.reset_index(inplace = True)
    loc_tickerDf['Date'] = pd.to_datetime(loc_tickerDf['Date']).dt.tz_localize(None)
    loc_tickerDf.rename(columns = {'Date':'DATE', 'Close':'CLOSE'}, inplace = True)
    loc_tickerDf['DATE'] =  pd.to_datetime(loc_tickerDf.DATE,utc = False)

    loc_tickerDf.dropna(inplace = True)         
    loc_tickerDf.reset_index(inplace = True)
    return loc_tickerDf

def calculate_actual_earn(x):
    if x.isCALL:
        buy_itm = x['PRICE@EXPIRE'] > x.BUY_STRIKE
        sell_itm = x['PRICE@EXPIRE'] > x.SELL_STRIKE
        actual_earn = x.PREMIUM + buy_itm * (x['PRICE@EXPIRE'] - x.BUY_STRIKE) - sell_itm * (x['PRICE@EXPIRE'] - x.SELL_STRIKE)
    else:
        buy_itm = x['PRICE@EXPIRE'] < x.BUY_STRIKE
        sell_itm = x['PRICE@EXPIRE'] < x.SELL_STRIKE
        actual_earn = x.PREMIUM + buy_itm * (x.BUY_STRIKE - x['PRICE@EXPIRE']) - sell_itm * (x.SELL_STRIKE - x['PRICE@EXPIRE'])

    return actual_earn


def return_cum_earn_list(_df):
    cum_earn = 0
    cum_earn_list = list()
    for i in _df.ACTUAL_EARN:
        cum_earn += i
        cum_earn_list.append(cum_earn)
    return cum_earn_list

def prep_op_st_df(_ticker = 'SPY'):
    path = 'Spreads_Data\\' + _ticker + '\\' + _ticker +'_Options_EOD_2010_2022.csv'
    # For Mac, use the following line
    # path = 'Options_Data/' + _ticker + '/' + _ticker +'_Options_EOD_2010_2022.csv'
    df = pd.read_csv(path)
    op = pd.DataFrame()
    op['QUOTE_TIME_UTC'] = pd.to_datetime(df.QUOTE_UNIXTIME,unit='s',utc = False) #convert date from UNIX to UTC
    op['EXPIRE_UTC'] = pd.to_datetime(df.EXPIRE_UNIX,unit='s',utc = False)

    op['QUOTE_TIME_EST'] = op['QUOTE_TIME_UTC'].apply(lambda x: x.date()) #convert date from UTC to EST
    op['EXPIRE_EST'] = op['EXPIRE_UTC'].apply(lambda x: x.date())

    for i in ['STRIKE','DTE','C_IV','C_BID','C_ASK',
                     'P_IV','P_BID','P_ASK']:
        op[i]= df[i]
    op.drop(columns = ['QUOTE_TIME_UTC','EXPIRE_UTC'], inplace = True)

    df_stock = OHLC_info_by_range(_ticker,'2010-01-01','2022-12-31',interval = '1d')
    st = df_stock.copy()

    return op, st

