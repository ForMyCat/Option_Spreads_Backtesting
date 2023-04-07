import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import talib as ta
import datetime 
from datetimerange import DateTimeRange
import math
import time

class backtest:
	def __init__(self , _spread_data = (pd.DataFrame(), pd.DataFrame()), _DTE = 7 ,_stock_data = pd.DataFrame()):
		call_spreads = _spread_data[0].copy()
		put_spreads = _spread_data[1].copy()

		call_spreads['EXPIRE_EST'] = pd.to_datetime(call_spreads['EXPIRE_EST']).dt.tz_localize(None)
		put_spreads['EXPIRE_EST'] = pd.to_datetime(put_spreads['EXPIRE_EST']).dt.tz_localize(None)
		call_spreads['QUOTE_TIME_EST'] = pd.to_datetime(call_spreads['QUOTE_TIME_EST']).dt.tz_localize(None)
		put_spreads['QUOTE_TIME_EST'] = pd.to_datetime(put_spreads['QUOTE_TIME_EST']).dt.tz_localize(None)
		call_spreads['COLLATERAL'] = abs(call_spreads.SELL_STRIKE - call_spreads.BUY_STRIKE)
		put_spreads['COLLATERAL'] = abs(put_spreads.SELL_STRIKE - put_spreads.BUY_STRIKE)
		call_spreads['spread_idx'] = list(zip(call_spreads.SELL_STRIKE, call_spreads.BUY_STRIKE,call_spreads.EXPIRE_EST))
		put_spreads['spread_idx'] = list(zip(put_spreads.SELL_STRIKE, put_spreads.BUY_STRIKE,put_spreads.EXPIRE_EST))

		self.call_spreads = call_spreads.loc[call_spreads.DTE == _DTE].copy()
		self.put_spreads = put_spreads.loc[put_spreads.DTE == _DTE].copy()

		self.stock_data = _stock_data.copy()

		self.all_calls = pd.DataFrame()
		self.all_puts = pd.DataFrame()

		for i in range(_DTE,0,-1):
			c = call_spreads.loc[call_spreads.DTE == i].copy()
			p = put_spreads.loc[put_spreads.DTE == i].copy()
			if (c.shape[0] == 0) or (p.shape[0] == 0):
				print('Missing data for DTE:',i,'Please load in more data.')
			self.all_calls = pd.concat([self.all_calls,c],ignore_index = 1)
			self.all_puts = pd.concat([self.all_puts,p],ignore_index = 1)

		self.DTE = _DTE
		
		self.add_technical_indicators()

		self.call_spreads = self.call_spreads.sort_values(by = ['QUOTE_TIME_EST','SELL_STRIKE'], ascending = [True, True]).reset_index(drop = True).copy()
		self.put_spreads = self.put_spreads.sort_values(by = ['QUOTE_TIME_EST','SELL_STRIKE'], ascending = [True, True]).reset_index(drop = True).copy()

		self.call_satisfied = pd.DataFrame()
		self.put_satisfied = pd.DataFrame()

		self.trades_history_dict = {'Date':[], 'Spreads Type':[], 'Spread index':[], 'Trades Type':[], 'Initial Funds':[], 'Funds After':[], 'Stock Price':[],
			      'Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)':[], 'Profit':[],'# of strategies holding':[0]}
		self.trades_history = pd.DataFrame()
		print('Done initializing!')

	def set_parm(self, parms = (None,None,None,None,None,None,10,None,None,None)):

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



	def set_portfolio(self, parms = (10000, None, None, 0.1), start_date = None , end_date = None ):
		self.init_fund = parms[0]
		self.stop_profit = parms[1]
		self.stop_loss = parms[2]
		self.position_size = parms[3]

		self.start_date = pd.to_datetime(start_date) if start_date != None else max(min(self.call_spreads.QUOTE_TIME_EST), min(self.put_spreads.QUOTE_TIME_EST)) 
		self.end_date = pd.to_datetime(end_date) if end_date != None else min(max(self.call_spreads.QUOTE_TIME_EST), max(self.put_spreads.QUOTE_TIME_EST))

	def go(self, spread_type = 'P',use_technical_indicator = False, _min_iv_ratio = None):
		time0 = time.time()
		ptime0 = time.process_time()
		self.call_satisfied = self.filter_spreads(self.call_spreads)
		self.put_satisfied = self.filter_spreads(self.put_spreads)

		if use_technical_indicator:
			min_iv_ratio = _min_iv_ratio if _min_iv_ratio != None else 0

			# self.call_satisfied = self.call_satisfied.loc[(self.call_satisfied.HIST_VOLATILITY) <= min_iv_ratio]
			self.call_satisfied = self.call_satisfied.loc[(self.call_satisfied.SELL_ATM_IV/self.call_satisfied.HIST_VOLATILITY) >= min_iv_ratio]
			self.call_satisfied = self.call_satisfied.loc[self.call_satisfied.TREND == 0]

			# self.put_satisfied = self.put_satisfied.loc[(self.put_satisfied.HIST_VOLATILITY) <= min_iv_ratio]
			self.put_satisfied = self.put_satisfied.loc[(self.put_satisfied.SELL_ATM_IV/self.put_satisfied.HIST_VOLATILITY) >= min_iv_ratio]
			self.put_satisfied = self.put_satisfied.loc[self.put_satisfied.TREND == 1]

		# self.combo_satisfied = pd.concat([self.call_satisfied,self.put_satisfied],ignore_index = True)
		# self.combo_satisfied.sort_values(by = 'QUOTE_TIME_EST', ascending= True,inplace = True)
		self.call_satisfied.reset_index(drop = True, inplace = True)
		self.put_satisfied.reset_index(drop = True, inplace = True)
		# self.combo_satisfied.reset_index(drop = True, inplace = True)

		# self.generate_results(spread_type = spread_type)
		# self.trades_history = pd.DataFrame(self.trades_history_dict)
		time1 = time.time()
		ptime1 = time.process_time()
		print(f'Done backtesting from %s to %s!'%(self.start_date, self.end_date))
		print('Total Time:', round(time1-time0,2), 'seconds, CPU Time:', round(ptime1-ptime0,2), 'seconds.')

	def generate_results(self, spread_type = 'PUT'):
		if spread_type in ['PUT','Put','P','put','p']:
			spreadss = self.put_satisfied.copy()
			all_spreadss = self.all_puts.copy()


		if spread_type in ['CALL','Call','C','call','c']:
			spreadss = self.call_satisfied.copy()
			all_spreadss = self.all_calls.copy()

		self.trades_history_dict = {'Date':[], 'Spreads Type':[], 'Spread index':[], 'Trades Type':[], 'Initial Funds':[], 'Funds After':[], 'Stock Price':[],
			      'Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)':[], 'Profit':[], '# of strategies holding':[0]}
		self.trades_history = pd.DataFrame()

		time_range = DateTimeRange(min(spreadss.QUOTE_TIME_EST),max(spreadss.EXPIRE_EST))
		hold_dict = dict()
		fund = self.init_fund
		stop_loss = self.stop_loss
		stop_profit = self.stop_profit
		for value in time_range.range(datetime.timedelta(days=1)):
			stock_price_today = self.stock_data.loc[self.stock_data.DATE == value]
			if spreadss.loc[spreadss.QUOTE_TIME_EST == value].shape[0]:
				hold = (spreadss.loc[spreadss.QUOTE_TIME_EST == value]).iloc[0]
				open_quantity = math.floor(fund * self.position_size/ (hold.COLLATERAL*100))
				if open_quantity == 0:
					pass
				else:
					collateral = hold.COLLATERAL* open_quantity * 100
					
					premium = hold.PREMIUM
					hold_idx = hold.name
					sp_type = 'C' if hold.isCALL else 'P'
					sp_info = (hold.spread_idx, open_quantity, collateral, premium, hold_idx)
					hold_dict[hold.EXPIRE_EST] = sp_info

					self.trades_history_dict['Date'].append(value)
					self.trades_history_dict['Spread index'].append(hold_idx)
					self.trades_history_dict['Spreads Type'].append(sp_type)
					self.trades_history_dict['Trades Type'].append('Open')
					self.trades_history_dict['Initial Funds'].append(fund)
					self.trades_history_dict['Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)'].append(sp_info)
					self.trades_history_dict['Stock Price'].append(round(stock_price_today.iloc[0].CLOSE,2))
					self.trades_history_dict['Profit'].append(0)
					self.trades_history_dict['# of strategies holding'].append(self.trades_history_dict['# of strategies holding'][-1] + 1)
					
					fund -= collateral
					self.trades_history_dict['Funds After'].append(fund)

					# print(value, 'take collateral:',collateral , 'remain:', fund, 'quantity:', open_quantity)
		#             print(colla_dict.keys())
			check_dict = hold_dict.copy()

			for i, j in check_dict.items():
				spread_today = all_spreadss.loc[(all_spreadss.spread_idx == j[0]) & (all_spreadss.QUOTE_TIME_EST == value)]
				if spread_today.shape[0]:
					spread_today = spread_today.iloc[0]
					premium_today = spread_today.PREMIUM
					if premium_today >= stop_loss * j[3]:

						self.trades_history_dict['Date'].append(value)
						self.trades_history_dict['Spread index'].append(j[4])
						self.trades_history_dict['Spreads Type'].append(sp_type)
						self.trades_history_dict['Trades Type'].append('Stop Loss')
						self.trades_history_dict['Initial Funds'].append(fund)
						
						self.trades_history_dict['Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)'].append(j)
						self.trades_history_dict['Stock Price'].append(round(stock_price_today.iloc[0].CLOSE,2))
						self.trades_history_dict['Profit'].append(-(stop_loss - 1) * j[3] * 100 * j[1])

						fund += j[2] - (stop_loss - 1) * j[3] * 100 * j[1]

						self.trades_history_dict['Funds After'].append(fund)
						hold_dict.pop(i, 'No Key found')
						self.trades_history_dict['# of strategies holding'].append(self.trades_history_dict['# of strategies holding'][-1] - 1)

		
						# print('Stop Loss:', j , 'on:', value, 'current stock price:', spread_today.CURRENT_PRICE,
						# 	'current spread price:',premium_today, 'loss:',-1 * (stop_loss - 1) * j[3] * 100 * j[1], 
						# 	'release collateral:', j[2] - (stop_loss - 1) * j[3] * 100 * j[1])
					
			if value in list(hold_dict.keys()):
				close_spread, close_quantity, release_collateral = spreadss.loc[hold_dict[value][4]], hold_dict[value][1], hold_dict[value][2]  
				profit = close_spread.ACTUAL_EARN * 100 * close_quantity
				if profit < 0:
					if profit <= -1 * (stop_loss - 1) * close_spread.PREMIUM * 100 * close_quantity:
						# print('before stop',profit, 'after stop',(stop_loss - 1) * close_spread.PREMIUM * 100 * close_quantity)
						profit = -1 * (stop_loss - 1) * close_spread.PREMIUM * 100 * close_quantity

				self.trades_history_dict['Date'].append(value)
				self.trades_history_dict['Spread index'].append(hold_dict[value][4])
				self.trades_history_dict['Spreads Type'].append(sp_type)
				self.trades_history_dict['Trades Type'].append('Close')
				self.trades_history_dict['Initial Funds'].append(fund)
				
				self.trades_history_dict['Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)'].append(hold_dict[value])
				self.trades_history_dict['Stock Price'].append(round(stock_price_today.iloc[0].CLOSE,2))
				self.trades_history_dict['Profit'].append(profit)	

				fund += release_collateral + profit

				self.trades_history_dict['Funds After'].append(fund)
				self.trades_history_dict['# of strategies holding'].append(self.trades_history_dict['# of strategies holding'][-1] - 1)
				hold_dict.pop(value, 'No Key found')

		self.trades_history_dict['# of strategies holding'].pop(0)		
				# print(value, 'release collateral:',release_collateral , 'profit:', profit,'remain:', fund)
		
				

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


		if (self.skip_date != None) and (self.skip_date != []): 
			for i in self.skip_date:
				df_ = df_.loc[(df_.EXPIRE_EST != i)]

		df_ = df_.sample(frac=1).sort_values(by = ['QUOTE_TIME_EST'], ascending = [True]).groupby('QUOTE_TIME_EST').head(self.max_trades_per_day).copy()
		# df_ = df_.sort_values(by = ['QUOTE_TIME_EST','EXPECTED_EARN_RATIO'], ascending = [True, False]).groupby('QUOTE_TIME_EST').head(self.max_trades_per_day).copy()
		df_['ACTUAL_EARN'] = df_.apply(lambda x: utils.calculate_actual_earn(x), axis = 1)
		df_['WIN'] = df_['ACTUAL_EARN'] > 0

		df_ = df_.copy()
		return df_

	def win_rate(self):
		call_win = (self.call_satisfied.ACTUAL_EARN >= 0).sum()
		put_win = (self.put_satisfied.ACTUAL_EARN >= 0).sum()
		# total_win = (self.combo_satisfied.ACTUAL_EARN >= 0).sum()

		call_win_rate = call_win/(self.call_satisfied.shape[0])
		put_win_rate = put_win/(self.put_satisfied.shape[0])
		# total_win_rate = total_win/(self.combo_satisfied.shape[0])

		print('Call:', (self.call_satisfied.shape[0]), 'trades, win rate:', round(call_win_rate,4))
		print('Put:', (self.put_satisfied.shape[0]), 'trades, win rate:', round(put_win_rate,4))
		# print('Total', (self.combo_satisfied.shape[0]), 'trades, win rate:', round(total_win_rate,4))

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


	def report(self):
		open_sp = self.trades_history.loc[self.trades_history['Trades Type'] == 'Open']
		close_sp = self.trades_history.loc[(self.trades_history['Trades Type'] == 'Close') | (self.trades_history['Trades Type'] == 'Stop Loss')]
		close_sp = close_sp.reset_index(drop = True)
		cum_fund = self.init_fund + close_sp.Profit.cumsum()


		total_return = self.trades_history.Profit.sum()
		total_return_ratio = total_return/self.init_fund

		close_sp['year'] = close_sp.Date.dt.year
		years = close_sp.Date.dt.year.unique()
		annual_return_ratio = list()
		for yr in years:
			yr_df = close_sp.loc[close_sp.year == yr]
			starting_funds = yr_df.loc[(yr_df['# of strategies holding'] == 0) | (yr_df['# of strategies holding'] == 1)].iloc[0]['Funds After']
			yr_profit_ratio = round(yr_df.Profit.sum()/starting_funds,4)
			yr_days = ((yr_df.iloc[-1].Date - yr_df.iloc[0].Date).days)
			annual_return_ratio.append(yr_profit_ratio* 365 / yr_days) 

		
		annual_return_std = np.std(annual_return_ratio)
		sharpe = ((total_return_ratio/((self.end_date - self.start_date).days/365)) - 0.01)/annual_return_std

		i = np.argmax(np.maximum.accumulate(cum_fund) - cum_fund) # end of the period
		j = np.argmax(cum_fund[:i]) # start of period
		maxdd = (cum_fund[j] - cum_fund[i])/cum_fund[j]

		def find_collateral(row):
			return row['Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)'][2]
		
		close_sp['Collateral'] = close_sp.apply(lambda x: find_collateral(x), axis = 1)

		print('Report:')
		print()
		print('Number of Positions:', (open_sp).shape[0])
		print('Win Rate:', round(((self.trades_history.Profit > 0).sum())/(open_sp).shape[0],4))
		print('Sharpe Ratio:', round(sharpe,2))
		print('Total Return:', round(total_return,2))
		print('Total Return Ratio:', round(total_return_ratio,4))
		
		# yr_return = [(x,y) for x, y in zip(years, annual_return_ratio)]
		# print('Return by Year:',yr_return)
		print('Max Collateral:', (open_sp['Initial Funds'] - open_sp['Funds After']).max())
		print('Max Drawdown:', round(maxdd,3),'Date:', close_sp.loc[i].Date)
		print('Max Individual Loss:', round(self.trades_history.Profit.min(),2),'id:', self.trades_history.loc[self.trades_history.Profit.idxmin(),'Spread index'])
		print('Max Individual Gain:', round(self.trades_history.Profit.max(),2),'id:', self.trades_history.loc[self.trades_history.Profit.idxmax(),'Spread index'])
		print('Average Collateral:', round((open_sp['Initial Funds'] - open_sp['Funds After']).mean(),2))
		print('Average Gain:', round(self.trades_history.Profit.sum()/open_sp.shape[0],2))
		print('Average Return Ratio:', round((close_sp.Profit/close_sp['Collateral']).mean(),4))
		print('Average Expectation Return Ratio:', round(self.put_satisfied.EXPECTED_EARN_RATIO.mean(),4))
		print('Average Expected Win Rate:', round(self.put_satisfied.SELL_OTM_PROB.mean(),2))

		fig,ax1 = plt.subplots()

		fig.set_size_inches(16, 8, forward=True)
		# make a plot
		ax1.plot(self.stock_data.DATE, self.stock_data.CLOSE,color="grey")
		# set x-axis label
		ax1.set_xlabel("Year", fontsize = 14)
		# set y-axis label
		ax1.set_ylabel("Stock",
		              color="black",
		              fontsize=14)
		ax1.legend(['Stock'],loc='upper left')
		ax2 = ax1.twinx()

		ax2.plot(close_sp.Date, cum_fund, color='orange')
		ax2.plot([close_sp.loc[i].Date, close_sp.loc[j].Date], [cum_fund[i], cum_fund[j]], 'o', color='red', markersize=10)

		ax2.legend(['Funds'],loc='upper right')
		ax2.set_ylabel('Funds',color="black",
		              fontsize=14)
		plt.title("Strategy Return VS Stock Return",fontsize=16)
		plt.show()

		zeroed_stock = self.stock_data.dropna().copy()
		zeroed_stock.CLOSE /= zeroed_stock.iloc[0]['CLOSE']

		zeroed_sp = cum_fund.copy()
		zeroed_sp /= self.init_fund

		fig2 = plt.figure(2)
		fig2.set_figwidth(16)
		fig2.set_figheight(8)
		plt.plot(zeroed_stock.DATE, zeroed_stock.CLOSE, color='grey', label = 'Stock')
		plt.plot(close_sp.Date, zeroed_sp, color="orange", label = 'Spread')
		plt.legend(loc='upper left')
		plt.title("Standardized Strategy Return VS Stock Return",fontsize=16)
		plt.show()
		





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

		# self.combo_satisfied = pd.concat([self.call_satisfied,self.put_satisfied],ignore_index = True)



