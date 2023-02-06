import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import talib as ta

class backtest:
	def __init__(self , _spread_data = (pd.DataFrame(), pd.DataFrame()), _stock_data = pd.DataFrame()):
		self.call_spreads, self.put_spreads = _spread_data[0].copy(), _spread_data[1].copy()
		self.stock_data = _stock_data.copy()

		self.DTE = self.call_spreads.head(1).DTE
		
		self.call_spreads['QUOTE_TIME_EST'] = pd.to_datetime(self.call_spreads.QUOTE_TIME_EST).dt.tz_localize(None)
		self.put_spreads['QUOTE_TIME_EST'] = pd.to_datetime(self.put_spreads.QUOTE_TIME_EST).dt.tz_localize(None)
		self.add_technical_indicators()

		self.call_spreads = self.call_spreads.sort_values(by = ['QUOTE_TIME_EST','SELL_STRIKE'], ascending = [True, True]).copy()
		self.put_spreads = self.put_spreads.sort_values(by = ['QUOTE_TIME_EST','SELL_STRIKE'], ascending = [True, True]).copy()

		self.call_satisfied = pd.DataFrame()
		self.put_satisfied = pd.DataFrame()

		self.call_cum_return = list()
		self.put_cum_return = list()

	def set_parm(self, parms , start_date, end_date):

		parm0 = parms[0] if parms[0] != None else -200
		parm1 = parms[1] if parms[1] != None else -1
		parm2 = parms[2] if parms[2] != None else 0
		parm3 = parms[3] if parms[3] != None else 0
		parm4 = parms[4] if parms[4] != None else 0
		parm5 = parms[5] if parms[5] != None else 200
		parm6 = parms[6] if parms[6] != None else 1
		parm7 = parms[7] if parms[7] != None else 0		

		self.start_date = pd.to_datetime(start_date) if start_date != None else max(min(self.call_spreads.QUOTE_TIME_EST), min(self.put_spreads.QUOTE_TIME_EST)) 
		self.end_date = pd.to_datetime(end_date) if end_date != None else min(max(self.call_spreads.QUOTE_TIME_EST), max(self.put_spreads.QUOTE_TIME_EST))

		self.min_EXPECTED_EARN = parm0
		self.min_EARN_RATIO = parm1

		self.min_SELL_OTM_PROB = parm2
		self.min_BUY_OTM_PROB = parm3

		self.min_width = parm4
		self.max_width = parm5

		self.max_trades_per_day = parm6

		self.min_PREMIUM = parm7


	def go(self):
		self.call_satisfied, self.call_cum_return = self.filter_spreads(self.call_spreads)
		self.put_satisfied, self.put_cum_return = self.filter_spreads(self.put_spreads)
		print(f'Done backtesting from %s to %s!'%(self.start_date, self.end_date))
		 

	def filter_spreads(self, df_):
		df_ = df_.copy()

		df_ = df_.loc[(df_.QUOTE_TIME_EST >= self.start_date) & (df_.QUOTE_TIME_EST <= self.end_date)]

		df_ = df_.loc[(df_.EXPECTED_EARN >= self.min_EXPECTED_EARN)]
		df_ = df_.loc[(df_.EXPECTED_EARN_RATIO >= self.min_EARN_RATIO)]
		df_ = df_.loc[(df_.SELL_OTM_PROB >= self.min_SELL_OTM_PROB)]
		df_ = df_.loc[(df_.BUY_OTM_PROB >= self.min_BUY_OTM_PROB)]
		df_['WIDTH'] = abs(df_.SELL_STRIKE - df_.BUY_STRIKE)
		df_ = df_.loc[(df_.WIDTH >= self.min_width) & (df_.WIDTH <= self.max_width)]
		df_ = df_.loc[df_.PREMIUM >= self.min_PREMIUM]

		df_ = df_.sort_values(by = ['QUOTE_TIME_EST','PREMIUM'], ascending = [True, False]).groupby('QUOTE_TIME_EST').head(self.max_trades_per_day).copy()
		df_['ACTUAL_EARN'] = df_.apply(lambda x: utils.calculate_actual_earn(x), axis = 1)
		cum_return = utils.return_cum_earn_list(df_)
		df_['CUM_EARN'] = cum_return
		df_['WIN'] = df_['ACTUAL_EARN'] > 0

		df_ = df_.copy()
		return df_, cum_return



	def draw_result(self, _st, _show = 'BOTH'):
		fig,ax1 = plt.subplots()

		fig.set_size_inches(16, 8, forward=True)
		# make a plot
		ax1.plot(_st.DATE, _st.CLOSE,color="grey")
		# set x-axis label
		ax1.set_xlabel("Year", fontsize = 14)
		# set y-axis label
		ax1.set_ylabel("Stock",
		              color="black",
		              fontsize=14)
		ax2 = ax1.twinx()

		call_by_day = (self.call_satisfied.groupby('QUOTE_TIME_EST').agg({'ACTUAL_EARN':'sum'}))
		put_by_day = (self.put_satisfied.groupby('QUOTE_TIME_EST').agg({'ACTUAL_EARN':'sum'}))
		call_by_day.reset_index(inplace = True)
		put_by_day.reset_index(inplace = True)

		if _show == 'BOTH':
			ax2.plot(call_by_day.QUOTE_TIME_EST, utils.return_cum_earn_list(call_by_day),color="green")
			ax2.plot(put_by_day.QUOTE_TIME_EST, utils.return_cum_earn_list(put_by_day),color="red")
			ax2.legend(['Call Credit Spreads', 'Put Credit Spreads'],loc='upper left')
		
		if _show == 'CALL' or _show == 'C':
			ax2.plot(call_by_day.QUOTE_TIME_EST, utils.return_cum_earn_list(call_by_day),color="green")
			ax2.legend(['Call Credit Spreads'],loc='upper left')

		if _show == 'PUT' or _show == 'P':
			ax2.plot(put_by_day.QUOTE_TIME_EST, utils.return_cum_earn_list(put_by_day),color="red")
			ax2.legend(['Put Credit Spreads'],loc='upper left')


		ax2.set_ylabel("Spreads",color="black")
		plt.title("Strategy Return VS Stock Return",fontsize=16)
		ax1.legend(['Stock'],loc='center left')
		plt.show()

	def win_rate(self):
		call_win = (self.call_satisfied.ACTUAL_EARN >= 0).sum()
		put_win = (self.put_satisfied.ACTUAL_EARN >= 0).sum()

		call_win_rate = call_win/(self.call_satisfied.shape[0])
		put_win_rate = put_win/(self.put_satisfied.shape[0])

		print('Call:', (self.call_satisfied.shape[0]), 'trades, win rate:', round(call_win_rate,4), 'profits:', round(self.call_satisfied.iloc[-1].CUM_EARN,2))
		print('Put:', (self.put_satisfied.shape[0]), 'trades, win rate:', round(put_win_rate,4), 'profits:', round(self.put_satisfied.iloc[-1].CUM_EARN,2))

		return round(call_win_rate,4), round(put_win_rate,4)

	def add_technical_indicators(self):
		# technical indicators include: HIST_VOLATILITY, RSI(length = DTE)

		# Calculate 180d historical volitality
		returns = np.log(self.stock_data['CLOSE']/self.stock_data['CLOSE'].shift(1))
		returns.fillna(0, inplace=True)
		volatility = returns.rolling(window=180).std()*np.sqrt(252)
		self.stock_data['HIST_VOLATILITY'] = volatility


		# Add RSI related to DTE
		self.stock_data['RSI'] = ta.RSI(self.stock_data['CLOSE'], timeperiod = self.DTE)

		# Add three EMA data
		self.stock_data['EMA_50'] = ta.EMA(self.stock_data['CLOSE'], timeperiod = 50)
		self.stock_data['EMA_252'] = ta.EMA(self.stock_data['CLOSE'], timeperiod = 252)
		self.stock_data['EMA_DTE'] = ta.EMA(self.stock_data['CLOSE'], timeperiod = self.DTE)

		self.call_spreads = pd.merge(self.call_spreads, self.stock_data[['DATE','HIST_VOLATILITY','RSI','EMA_252','EMA_50','EMA_DTE']], left_on=  ['QUOTE_TIME_EST'],
                   right_on= ['DATE'], 
                   how = 'left')

		self.put_spreads = pd.merge(self.put_spreads, self.stock_data[['DATE','HIST_VOLATILITY','RSI','EMA_252','EMA_50','EMA_DTE']], left_on=  ['QUOTE_TIME_EST'],
			right_on= ['DATE'], 
			how = 'left')

		self.put_spreads.drop(columns = 'DATE', inplace = True)
		self.call_spreads.drop(columns = 'DATE', inplace = True)
		self.put_spreads.dropna(inplace = True)
		self.call_spreads.dropna(inplace = True)

	def report(self, opion_type = 'BOTH'):

		call_win = (self.call_satisfied.ACTUAL_EARN >= 0).sum()
		put_win = (self.put_satisfied.ACTUAL_EARN >= 0).sum()

		if opion_type in ['BOTH','C','CALL']:
			print('Credit Call:')
			print()
			print('Number of Positions:', (self.call_satisfied).shape[0])
			print('Win Rate:', round((self.call_satisfied.ACTUAL_EARN >= 0).sum()/(self.call_satisfied.shape[0]),4))
			print('Total Return:', round(self.call_satisfied.ACTUAL_EARN.sum(),2))
			print('Max Collateral:', self.call_satisfied.WIDTH.max())
			print('Max Drawdown:', round(self.call_satisfied.CUM_EARN.min(),2),'Date:', self.call_satisfied.loc[self.call_satisfied.CUM_EARN.idxmin()].QUOTE_TIME_EST)
			print('Max Individual Loss:', round(self.call_satisfied.ACTUAL_EARN.min(),2),'id:', self.call_satisfied.ACTUAL_EARN.idxmin())
			print('Max Individual Gain:', round(self.call_satisfied.ACTUAL_EARN.max(),2),'id:', self.call_satisfied.ACTUAL_EARN.idxmax())
			print('Average Collateral:', round(self.call_satisfied.WIDTH.mean(),2))
			print('Average Gain:', round(self.call_satisfied.ACTUAL_EARN.mean(),2))
			print('Average Expectation:', round(self.call_satisfied.SELL_OTM_PROB.mean(),2))

		print()

		if opion_type in ['BOTH','P','PUT']:
			print('Credit Put:')
			print()
			print('Number of Positions:', (self.put_satisfied).shape[0])
			print('Win Rate:', round((self.put_satisfied.ACTUAL_EARN >= 0).sum()/(self.put_satisfied.shape[0]),4))
			print('Total Return:', round(self.put_satisfied.ACTUAL_EARN.sum(),2))
			print('Max Collateral:', self.put_satisfied.WIDTH.max())
			print('Max Drawdown:', round(self.put_satisfied.CUM_EARN.min(),2),'Date:', self.put_satisfied.loc[self.put_satisfied.CUM_EARN.idxmin()].QUOTE_TIME_EST)
			print('Max Individual Loss:', round(self.put_satisfied.ACTUAL_EARN.min(),2),'id:', self.put_satisfied.ACTUAL_EARN.idxmin())
			print('Max Individual Gain:', round(self.put_satisfied.ACTUAL_EARN.max(),2),'id:', self.put_satisfied.ACTUAL_EARN.idxmax())
			print('Average Collateral:', round(self.put_satisfied.WIDTH.mean(),2))
			print('Average Gain:', round(self.put_satisfied.ACTUAL_EARN.mean(),2))
			print('Average Expectation:', round(self.put_satisfied.SELL_OTM_PROB.mean(),2))




