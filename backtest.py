import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import talib as ta
from datetime import timedelta

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

	def set_parm(self, parms = (None,None,None,None,None,None,10,None,None,None), start_date = None , end_date = None ):

		parm0 = parms[0] 
		parm1 = parms[1] 
		parm2 = parms[2] 
		parm3 = parms[3] 
		parm4 = parms[4] 
		parm5 = parms[5] 
		parm6 = parms[6] 
		parm7 = parms[7] 	
		parm8 = parms[8]
		parm9 = parms[9]

		self.min_EXPECTED_EARN = parm0
		self.min_EARN_RATIO = parm1
		self.min_SELL_OTM_PROB = parm2
		self.min_BUY_OTM_PROB = parm3
		self.min_width = parm4
		self.max_width = parm5
		self.max_trades_per_day = parm6
		self.min_PREMIUM = parm7
		self.max_distance_ratio = parm8
		if (parm9 != None) or (parm9 != []):
			self.skip_date = [pd.to_datetime(i) for i in parm9] 

		self.start_date = pd.to_datetime(start_date) if start_date != None else max(min(self.call_spreads.QUOTE_TIME_EST), min(self.put_spreads.QUOTE_TIME_EST)) 
		self.end_date = pd.to_datetime(end_date) if end_date != None else min(max(self.call_spreads.QUOTE_TIME_EST), max(self.put_spreads.QUOTE_TIME_EST))



	def go(self):
		self.call_satisfied, self.call_cum_return = self.filter_spreads(self.call_spreads)
		self.put_satisfied, self.put_cum_return = self.filter_spreads(self.put_spreads)
		self.combo_satisfied = pd.concat([self.call_satisfied,self.put_satisfied],ignore_index = True)
		self.combo_satisfied.sort_values(by = 'QUOTE_TIME_EST', ascending= True,inplace = True)
		self.combo_satisfied.CUM_EARN = self.combo_satisfied.ACTUAL_EARN.cumsum()
		print(f'Done backtesting from %s to %s!'%(self.start_date, self.end_date))

	def combinational_earn(self):
		# Calculate total cumulative return
		a = self.combo_satisfied[['QUOTE_TIME_EST','ACTUAL_EARN']].copy()
		
		a = (a.groupby('QUOTE_TIME_EST').agg({'ACTUAL_EARN':'sum'}))
		a.reset_index(inplace = True)
		a['CUM_EARN'] = a.ACTUAL_EARN.cumsum()

		return a
		 

	def filter_spreads(self, df_):
		df_ = df_.copy()

		df_ = df_.loc[(df_.QUOTE_TIME_EST >= self.start_date) & (df_.QUOTE_TIME_EST <= self.end_date)]

		df_['STRIKE_DISTANCE'] = abs(df_.SELL_STRIKE - df_.CURRENT_PRICE)
		df_['STRIKE_DISTANCE_RATIO'] = df_['STRIKE_DISTANCE']/df_.CURRENT_PRICE
		if self.max_distance_ratio != None: df_ = df_.loc[(df_.STRIKE_DISTANCE_RATIO <= self.max_distance_ratio)]
		if self.min_EXPECTED_EARN != None: df_ = df_.loc[(df_.EXPECTED_EARN >= self.min_EXPECTED_EARN)]
		if self.min_EARN_RATIO != None: df_ = df_.loc[(df_.EXPECTED_EARN_RATIO >= self.min_EARN_RATIO)]
		if self.min_SELL_OTM_PROB != None: df_ = df_.loc[(df_.SELL_OTM_PROB >= self.min_SELL_OTM_PROB)]
		if self.min_BUY_OTM_PROB != None: df_ = df_.loc[(df_.BUY_OTM_PROB >= self.min_BUY_OTM_PROB)]

		df_['WIDTH'] = abs(df_.SELL_STRIKE - df_.BUY_STRIKE)
		if self.max_width != None: df_ = df_.loc[(df_.WIDTH <= self.max_width)]
		if self.min_width != None: df_ = df_.loc[(df_.WIDTH >= self.min_width)]
		if self.min_PREMIUM != None: df_ = df_.loc[df_.PREMIUM >= self.min_PREMIUM]

		df_['EXPIRE_TIME'] = df_.QUOTE_TIME_EST + pd.Timedelta(int(self.DTE),'D')
		

		if (self.skip_date != None) and (self.skip_date != []): 
			for i in self.skip_date:
				df_ = df_.loc[(df_.EXPIRE_TIME != i)]

		df_ = df_.sample(frac=1).sort_values(by = ['QUOTE_TIME_EST'], ascending = [True]).groupby('QUOTE_TIME_EST').head(self.max_trades_per_day).copy()
		# df_ = df_.sort_values(by = ['QUOTE_TIME_EST','EXPECTED_EARN_RATIO'], ascending = [True, False]).groupby('QUOTE_TIME_EST').head(self.max_trades_per_day).copy()
		df_['ACTUAL_EARN'] = df_.apply(lambda x: utils.calculate_actual_earn(x), axis = 1)
		cum_return = df_.ACTUAL_EARN.cumsum()
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

		if _show in ['BOTH','Both','both']:
			ax2.plot(call_by_day.QUOTE_TIME_EST, call_by_day.ACTUAL_EARN.cumsum(),color="green")
			ax2.plot(put_by_day.QUOTE_TIME_EST, put_by_day.ACTUAL_EARN.cumsum(),color="red")
			ax2.legend(['Call Credit Spreads', 'Put Credit Spreads'],loc='upper left')
		
		if _show in ['CALL','Call','C','call','c']:
			ax2.plot(call_by_day.QUOTE_TIME_EST, call_by_day.ACTUAL_EARN.cumsum(),color="green")
			ax2.legend(['Call Credit Spreads'],loc='upper left')

		if _show in ['PUT','Put','P','put','p']:
			ax2.plot(put_by_day.QUOTE_TIME_EST, put_by_day.ACTUAL_EARN.cumsum(),color="red")
			ax2.legend(['Put Credit Spreads'],loc='upper left')

		if _show in ['COMBO','combo','Combo']:
			ax2.plot(self.combinational_earn().QUOTE_TIME_EST, self.combinational_earn().CUM_EARN,color="purple")
			ax2.legend(['Combo Earn'],loc='upper left')

		if _show in ['ALL','All','all']:
			ax2.plot(self.combinational_earn().QUOTE_TIME_EST, self.combinational_earn().CUM_EARN,color="purple")
			ax2.plot(call_by_day.QUOTE_TIME_EST, call_by_day.ACTUAL_EARN.cumsum(),color="green")
			ax2.plot(put_by_day.QUOTE_TIME_EST, put_by_day.ACTUAL_EARN.cumsum(),color="red")
			ax2.legend(['Combo Earn','Call Credit Spreads', 'Put Credit Spreads'],loc='upper left')



		ax2.set_ylabel("Spreads",color="black")
		plt.title("Strategy Return VS Stock Return",fontsize=16)
		ax1.legend(['Stock'],loc='center left')
		plt.show()

	def win_rate(self):
		call_win = (self.call_satisfied.ACTUAL_EARN >= 0).sum()
		put_win = (self.put_satisfied.ACTUAL_EARN >= 0).sum()
		total_win = (self.combo_satisfied.ACTUAL_EARN >= 0).sum()

		call_win_rate = call_win/(self.call_satisfied.shape[0])
		put_win_rate = put_win/(self.put_satisfied.shape[0])
		total_win_rate = total_win/(self.combo_satisfied.shape[0])

		print('Call:', (self.call_satisfied.shape[0]), 'trades, win rate:', round(call_win_rate,4), 'profits:', round(self.call_satisfied.iloc[-1].CUM_EARN,2))
		print('Put:', (self.put_satisfied.shape[0]), 'trades, win rate:', round(put_win_rate,4), 'profits:', round(self.put_satisfied.iloc[-1].CUM_EARN,2))
		print('Total', (self.combo_satisfied.shape[0]), 'trades, win rate:', round(total_win_rate,4), 'profit:',round(self.combo_satisfied.ACTUAL_EARN.sum(),2))

		return round(call_win_rate,4), round(put_win_rate,4)

	def add_technical_indicators(self):
		# technical indicators include: HIST_VOLATILITY, RSI(length = DTE)

		# Calculate 150d historical volitality
		returns = np.log(self.stock_data['CLOSE']/self.stock_data['CLOSE'].shift(1))
		returns.fillna(0, inplace=True)
		volatility = returns.rolling(window=150).std()*np.sqrt(252)
		self.stock_data['HIST_VOLATILITY'] = volatility


		# Add RSI related to DTE
		self.stock_data['RSI'] = ta.RSI(self.stock_data['CLOSE'], timeperiod = self.DTE)

		# Add four EMA data
		self.stock_data['EMA_7'] = ta.EMA(self.stock_data['CLOSE'], timeperiod = 7)
		self.stock_data['EMA_50'] = ta.EMA(self.stock_data['CLOSE'], timeperiod = 50)
		self.stock_data['EMA_252'] = ta.EMA(self.stock_data['CLOSE'], timeperiod = 252)
		self.stock_data['EMA_DTE'] = ta.EMA(self.stock_data['CLOSE'], timeperiod = self.DTE)

		# Add trend indicator, bull:1, bear:0, TREND_REVERSAL: crossover
		self.stock_data['TREND'] = 0.0
		self.stock_data['TREND'] = np.where(self.stock_data['EMA_DTE'] > self.stock_data['EMA_50'], 1.0, 0.0)
		self.stock_data['TREND_REVERSAL'] = self.stock_data['TREND'].diff()

		self.call_spreads = pd.merge(self.call_spreads, self.stock_data[['DATE','HIST_VOLATILITY','RSI','EMA_252','EMA_50','EMA_7','EMA_DTE','TREND','TREND_REVERSAL']], left_on=  ['QUOTE_TIME_EST'],
                   right_on= ['DATE'], 
                   how = 'left')

		self.put_spreads = pd.merge(self.put_spreads, self.stock_data[['DATE','HIST_VOLATILITY','RSI','EMA_252','EMA_50','EMA_7','EMA_DTE','TREND','TREND_REVERSAL']], left_on=  ['QUOTE_TIME_EST'],
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

		if opion_type in ['COMBO','Combo','combo']:
			print('Combo Strategy:')
			print()
			print('Number of Positions:', (self.combo_satisfied).shape[0])
			print('Win Rate:', round((self.combo_satisfied.ACTUAL_EARN >= 0).sum()/(self.combo_satisfied.shape[0]),4))
			print('Total Return:', round(self.combo_satisfied.ACTUAL_EARN.sum(),2))
			print('Max Collateral:', self.combo_satisfied.WIDTH.max())
			print('Max Drawdown:', round(self.combo_satisfied.CUM_EARN.min(),2),'Date:', self.combo_satisfied.loc[self.combo_satisfied.CUM_EARN.idxmin()].QUOTE_TIME_EST)
			print('Max Individual Loss:', round(self.combo_satisfied.ACTUAL_EARN.min(),2),'id:', self.combo_satisfied.ACTUAL_EARN.idxmin())
			print('Max Individual Gain:', round(self.combo_satisfied.ACTUAL_EARN.max(),2),'id:', self.combo_satisfied.ACTUAL_EARN.idxmax())
			print('Average Collateral:', round(self.combo_satisfied.WIDTH.mean(),2))
			print('Average Gain:', round(self.combo_satisfied.ACTUAL_EARN.mean(),2))
			print('Average Expectation:', round(self.combo_satisfied.SELL_OTM_PROB.mean(),2))

	def technical_indicator_search(self, opion_type = 'BOTH', _min_iv_ratio = None):
		min_iv_ratio = _min_iv_ratio if _min_iv_ratio != None else 0
		self.go()
		if opion_type in ['BOTH','C','CALL']:
			# self.call_satisfied = self.call_satisfied.loc[(self.call_satisfied.HIST_VOLATILITY) <= min_iv_ratio]
			self.call_satisfied = self.call_satisfied.loc[(self.call_satisfied.SELL_ATM_IV/self.call_satisfied.HIST_VOLATILITY) >= min_iv_ratio]
			self.call_satisfied = self.call_satisfied.loc[self.call_satisfied.TREND == 0]

		if opion_type in ['BOTH','P','PUT']:
			# self.put_satisfied = self.put_satisfied.loc[(self.put_satisfied.HIST_VOLATILITY) <= min_iv_ratio]
			self.put_satisfied = self.put_satisfied.loc[(self.put_satisfied.SELL_ATM_IV/self.put_satisfied.HIST_VOLATILITY) >= min_iv_ratio]
			self.put_satisfied = self.put_satisfied.loc[self.put_satisfied.TREND == 1]

		self.combo_satisfied = pd.concat([self.call_satisfied,self.put_satisfied],ignore_index = True)
		self.put_satisfied['CUM_EARN'] = self.put_satisfied.ACTUAL_EARN.cumsum()
		self.call_satisfied['CUM_EARN'] = self.call_satisfied.ACTUAL_EARN.cumsum()



