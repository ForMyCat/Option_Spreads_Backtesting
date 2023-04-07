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
	def __init__(self , _spread_data = pd.DataFrame(), _DTE = 7 ,_stock_data = pd.DataFrame()):
		time0 = time.time()

		self.DTE = _DTE
		spreads = _spread_data.copy()
		spreads['ExpirationDate'] = pd.to_datetime(spreads['ExpirationDate']).dt.tz_localize(None)
		spreads['QuoteDate'] = pd.to_datetime(spreads['QuoteDate']).dt.tz_localize(None)
		spreads['Collateral'] = abs(spreads.SellStrike - spreads.BuyStrike)
		spreads['Spread_idx'] = list(zip(spreads.SellStrike, spreads.BuyStrike,spreads.ExpirationDate))

		self.all_spreads = spreads.copy()
		self.stock_data = _stock_data.copy()
		
		self.add_technical_indicators()

		self.all_spreads = self.all_spreads.sort_values(by = ['QuoteDate','SellStrike'], ascending = [True, True]).reset_index(drop = True)
		self.DTE_spreads = self.all_spreads.loc[self.all_spreads.Dte == _DTE].copy()

		self.spreads_satisfied = pd.DataFrame()

		self.trades_history_dict = {'Date':[], 'Spreads Type':[], 'Spread index':[], 'Trades Type':[], 'Initial Funds':[], 'Funds After':[], 'Stock Price':[],
			      'Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)':[], 'Profit':[],'# of strategies holding':[0]}
		self.trades_history = pd.DataFrame()
		time1 = time.time()
		print('Done initializing! Time:', round(time1 - time0, 2), 'seconds.')

	def set_parm(self, parms = (None,None,None,None,None,None,1,None,None,None)):

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

		self.min_Expected_Earn = parm0
		self.min_Expected_Earn_Ratio = parm1
		self.Sell_OTM_Prob_Range = parm2
		self.Buy_OTM_Prob_Range = parm3
		self.min_Width = parm4
		self.max_Width = parm5
		self.max_trades_per_day = parm6
		self.min_Premium = parm7
		self.max_Distance_Ratio = parm8
		if (parm9 != None) or (parm9 != []):
			self.Skip_Date = [pd.to_datetime(i) for i in parm9] 



	def set_portfolio(self, parms = (10000, None, None, 0.1, None), _backtest_start_date = None , _backtest_end_date = None ):
		self.init_fund = parms[0]
		self.take_profit = parms[1]
		self.stop_loss = parms[2]
		self.position_size = parms[3]
		self.max_position = parms[4]

		self.backtest_start_date = pd.to_datetime(_backtest_start_date) if _backtest_start_date != None else self.start_date
		self.backtest_end_date = pd.to_datetime(_backtest_end_date) if _backtest_end_date != None else self.end_date

	def find_spreads(self, _use_technical_indicator = False, _min_iv_ratio = None, _start_date = None, _end_date = None):
		time0 = time.time()
		ptime0 = time.process_time()
		self.start_date = pd.to_datetime(_start_date) if _start_date != None else min(self.DTE_spreads.QuoteDate)
		self.end_date = pd.to_datetime(_end_date) if _end_date != None else max(self.DTE_spreads.ExpirationDate)

		if _use_technical_indicator:
			min_iv_ratio = _min_iv_ratio if _min_iv_ratio != None else 0

			self.spreads_satisfied = self.DTE_spreads.loc[(self.DTE_spreads.Sell_IV/self.DTE_spreads.Hist_Volatility) >= min_iv_ratio]
			self.spreads_satisfied = self.spreads_satisfied.loc[self.spreads_satisfied.Trend == 1]
			self.spreads_satisfied = self.filter_spreads(self.spreads_satisfied)
		else:
			self.spreads_satisfied = self.filter_spreads(self.DTE_spreads)
		

		print(f'Done searching spreads from %s to %s!'%(self.start_date, self.end_date))
		print('Total Time:', round(time.time()-time0,2), 'seconds, CPU Time:', round(time.process_time()-ptime0,2), 'seconds.')


	def generate_results(self):
		time0 = time.time()

		# Create dictionary to store options price history for fast backtesting
		spreadss = self.spreads_satisfied.loc[(self.spreads_satisfied.QuoteDate >= self.backtest_start_date) & (self.spreads_satisfied.ExpirationDate <= self.backtest_end_date)].copy()
		all_spreadss = self.all_spreads.loc[(self.all_spreads.QuoteDate >= self.backtest_start_date) 
				      & (self.all_spreads.ExpirationDate <= self.backtest_end_date)
					  & (self.all_spreads.Dte < self.DTE)].copy()
		
		# trade_spreads_track = dict()
		# for i, j in zip(list(spreadss.Spread_idx), list(spreadss.QuoteDate)):
		# 	op_idx = i
		# 	spread_price = all_spreadss.loc[(all_spreadss.Spread_idx == op_idx)]
		# 	# for QD, PR in zip(list(spread_price.QuoteDate), list(spread_price.Premium)):
		# 	trade_spreads_track[op_idx] = dict(zip(list(spread_price.QuoteDate), list(spread_price.Premium)))

		def zip_cols(_df):
			list0 = list(_df['QuoteDate'])
			list1 = list(_df['Premium'])
			return zip(list0, list1)
	
		inner_dict_list = [None] * len(spreadss)
		for i, k in enumerate(spreadss.Spread_idx):
			inner_dict_list[i] = dict(zip_cols(all_spreadss.loc[(all_spreadss.Spread_idx == k), ['QuoteDate','Premium']])) 
		trade_spreads_track = dict(zip(list(spreadss.Spread_idx), inner_dict_list))
		
		print('Done Done Preparing for Backtesting',round(time.time()-time0,2), 'seconds')


		self.trades_history_dict = {'Date':[], 'Spreads Type':[], 'Spread index':[], 'Trades Type':[], 'Initial Funds':[], 'Funds After':[], 'Stock Price':[],
			      'Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)':[], 'Profit':[], '# of strategies holding':[0]}
		self.trades_history = pd.DataFrame()

		time_range = DateTimeRange(self.backtest_start_date, self.backtest_end_date + datetime.timedelta(days=self.DTE))
		hold_dict = dict()
		fund = self.init_fund
		self.min_fund = fund
		stop_loss = self.stop_loss
		take_profit = self.take_profit

		for value in time_range.range(datetime.timedelta(days=1)):
			if fund < self.min_fund:
				self.min_fund = fund

			stock_today = self.stock_data.loc[self.stock_data.Date == value]
			if len(stock_today) > 0:
				stock_price_today = round(stock_today.iloc[0].Close,2)
			
			if spreadss.loc[spreadss.QuoteDate == value].shape[0]:
				hold = (spreadss.loc[spreadss.QuoteDate == value]).iloc[0]

				open_quantity = min(math.floor(fund * self.position_size/ (hold.Collateral*100)), math.floor((fund - (1 - self.max_position)*self.init_fund)/ (hold.Collateral*100)))

				if open_quantity == 0:
					pass
				else:
					collateral = hold.Collateral* open_quantity * 100
					
					premium = hold.Premium
					hold_idx = hold.name
					sp_type = 'C' if hold.isCALL else 'P'
					sp_info = (hold.Spread_idx, open_quantity, collateral, round(premium,2), hold_idx)
					hold_dict[hold.Spread_idx] = sp_info

					self.trades_history_dict['Date'].append(value)
					self.trades_history_dict['Spread index'].append(hold_idx)
					self.trades_history_dict['Spreads Type'].append(sp_type)
					self.trades_history_dict['Trades Type'].append('Open')
					self.trades_history_dict['Initial Funds'].append(fund)
					self.trades_history_dict['Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)'].append(sp_info)
					self.trades_history_dict['Stock Price'].append(stock_price_today)
					self.trades_history_dict['Profit'].append(0)
					self.trades_history_dict['# of strategies holding'].append(self.trades_history_dict['# of strategies holding'][-1] + 1)
					
					fund -= collateral
					self.trades_history_dict['Funds After'].append(fund)

					# print(value, 'take collateral:',collateral , 'remain:', fund, 'quantity:', open_quantity, 'at',hold.Collateral,'each', 'premium each:', premium)
		#             print(colla_dict.keys())
			check_dict = hold_dict.copy()

			for i, j in check_dict.items():
				spread_today = trade_spreads_track[i].copy()
				if value in spread_today.keys():
					premium_today = spread_today[value]
					if (stop_loss != None):
						if (premium_today >= stop_loss * j[3]):
							self.trades_history_dict['Date'].append(value)
							self.trades_history_dict['Spread index'].append(j[4])
							self.trades_history_dict['Spreads Type'].append(sp_type)
							self.trades_history_dict['Trades Type'].append('Stop Loss')
							self.trades_history_dict['Initial Funds'].append(fund)
							
							self.trades_history_dict['Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)'].append(j)
							self.trades_history_dict['Stock Price'].append(stock_price_today)
							# profit =  (j[3] - premium_today) * 100 * j[1] # Use EOD premuim to stop loss
							profit =  (1 - stop_loss) * j[3] * 100 * j[1] # Assuming we are able to cap loss at max stop-loss 
							self.trades_history_dict['Profit'].append(profit)
							fund += j[2] + profit
							# fund += j[2] - (stop_loss - 1) * j[3] * 100 * j[1]

							self.trades_history_dict['Funds After'].append(fund)
							hold_dict.pop(i, 'No Key found')
							self.trades_history_dict['# of strategies holding'].append(self.trades_history_dict['# of strategies holding'][-1] - 1)

					if (take_profit != None):
						if (premium_today <= (1 - take_profit) * j[3]):

							self.trades_history_dict['Date'].append(value)
							self.trades_history_dict['Spread index'].append(j[4])
							self.trades_history_dict['Spreads Type'].append(sp_type)
							self.trades_history_dict['Trades Type'].append('Take Profit')
							self.trades_history_dict['Initial Funds'].append(fund)
							
							self.trades_history_dict['Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)'].append(j)
							self.trades_history_dict['Stock Price'].append(stock_price_today)
							self.trades_history_dict['Profit'].append(take_profit * j[3] * 100 * j[1])

							fund += j[2] + take_profit * max(0, j[3]) * 100 * j[1]

							self.trades_history_dict['Funds After'].append(fund)
							hold_dict.pop(i, 'No Key found')
							self.trades_history_dict['# of strategies holding'].append(self.trades_history_dict['# of strategies holding'][-1] - 1)

		
						# print('Stop Loss:', j , 'on:', value, 'current stock price:', stock_price_today,
						# 	'current spread price:',premium_today, 'loss:',-1 * (stop_loss - 1) * j[3] * 100 * j[1], 
						# 	'release collateral:', j[2] - (stop_loss - 1) * j[3] * 100 * j[1])
				elif value == i[2]:
					close_spread, close_quantity, release_collateral = spreadss.loc[hold_dict[i][4]], hold_dict[i][1], hold_dict[i][2]
					close_premium = close_spread.Premium
					profit = close_spread.Actual_Earn * 100 * close_quantity


					## On the last day we cannot close or open any contracts.

					# if (stop_loss != None):
					# 	if profit <= -1 * (stop_loss - 1) * close_premium * 100 * close_quantity:
					# 		# print('before stop',profit, 'after stop', -1 * (stop_loss - 1) * close_premium * 100 * close_quantity)
					# 		profit = -1 * (stop_loss - 1) * close_premium * 100 * close_quantity
					# if (take_profit != None):
					# 	if (profit > close_premium * take_profit * 100 * close_quantity):
					# 		profit = close_premium * take_profit * 100 * close_quantity


					self.trades_history_dict['Date'].append(value)
					self.trades_history_dict['Spread index'].append(hold_dict[i][4])
					self.trades_history_dict['Spreads Type'].append(sp_type)
					self.trades_history_dict['Trades Type'].append('Close')
					self.trades_history_dict['Initial Funds'].append(fund)
					
					self.trades_history_dict['Option Info: ((Sell Strike, buy Strike, Expiry), Quantity, Collateral, Premium, idx)'].append(hold_dict[i])
					self.trades_history_dict['Stock Price'].append(stock_price_today)
					self.trades_history_dict['Profit'].append(profit)	

					fund += release_collateral + profit

					self.trades_history_dict['Funds After'].append(fund)
					self.trades_history_dict['# of strategies holding'].append(self.trades_history_dict['# of strategies holding'][-1] - 1)
					hold_dict.pop(value, 'No Key found')
					# print(value, 'release collateral:',release_collateral , 'profit:', profit,'remain:', fund)

		self.trades_history_dict['# of strategies holding'].pop(0)
		self.trades_history = pd.DataFrame(self.trades_history_dict)	
		print('Done backtesting:', round(time.time() - time0,2), 'seconds')
				

	def filter_spreads(self, df_):
		df_ = df_.copy()

		df_ = df_.loc[(df_.QuoteDate >= self.start_date) & (df_.ExpirationDate <= self.end_date)]

		df_ = df_.loc[df_.SellPrice > df_.BuyPrice]
		
		# print('a', df_.shape)

		df_['Strike_Distance'] = abs(df_.SellStrike - df_.UnderlyingPrice)
		df_['Strike_Distance_Ratio'] = df_['Strike_Distance']/df_.UnderlyingPrice

		# print('b', df_.shape)

		if self.max_Distance_Ratio != None: df_ = df_.loc[(df_.Strike_Distance_Ratio <= self.max_Distance_Ratio)]
		if self.min_Expected_Earn != None: df_ = df_.loc[(df_.ExpectedEarn >= self.min_Expected_Earn)]
		if self.min_Expected_Earn_Ratio != None: df_ = df_.loc[(df_.ExpectedEarnRatio >= self.min_Expected_Earn_Ratio)]


		# print('c', df_.shape)
		min_sell_prob = self.Sell_OTM_Prob_Range[0] if self.Sell_OTM_Prob_Range != None else None
		max_sell_prob = self.Sell_OTM_Prob_Range[1] if self.Sell_OTM_Prob_Range != None else None


		min_buy_prob = self.Buy_OTM_Prob_Range[0] if self.Buy_OTM_Prob_Range != None else None
		max_buy_prob = self.Buy_OTM_Prob_Range[1] if self.Buy_OTM_Prob_Range != None else None

		if min_sell_prob != None: df_ = df_.loc[(df_.Sell_OTM_Prob >= min_sell_prob)]
		if max_sell_prob != None: df_ = df_.loc[(df_.Sell_OTM_Prob <= max_sell_prob)]

		# print('d', df_.shape)

		if min_buy_prob != None: df_ = df_.loc[(df_.Buy_OTM_Prob >= min_buy_prob)]
		if max_buy_prob != None: df_ = df_.loc[(df_.Buy_OTM_Prob <= max_buy_prob)]

		# print('e', df_.shape)


		df_['Width'] = abs(df_.SellStrike - df_.BuyStrike)
		if self.max_Width != None: df_ = df_.loc[(df_.Width <= self.max_Width)]
		if self.min_Width != None: df_ = df_.loc[(df_.Width >= self.min_Width)]
		if self.min_Premium != None: df_ = df_.loc[df_.Premium >= self.min_Premium]

		# print('f', df_.shape)

		if (self.Skip_Date != None) and (self.Skip_Date != []): 
			for i in self.Skip_Date:
				df_ = df_.loc[(df_.ExpirationDate != i)]

		df_ = df_.sample(frac=1).sort_values(by = ['QuoteDate'], ascending = [True]).groupby('QuoteDate').head(self.max_trades_per_day).copy()
		# df_ = df_.sort_values(by = ['QUOTE_TIME_EST','EXPECTED_EARN_RATIO'], ascending = [True, False]).groupby('QUOTE_TIME_EST').head(self.max_trades_per_day).copy()
		df_['Actual_Earn'] = df_.apply(lambda x: utils.calculate_actual_earn(x), axis = 1)
		df_['Win'] = df_['Actual_Earn'] > 0

		return df_.copy()

	def win_rate(self):
		spreads_win = (self.spreads_satisfied.Actual_Earn > 0).sum()

		spreads_win_rate = spreads_win/(self.spreads_satisfied.shape[0])

		print('Total:', (self.spreads_satisfied.shape[0]), 'satisfied spreads, win rate:', round(spreads_win_rate,4))

		return round(spreads_win_rate,4)

	def add_technical_indicators(self):
		# technical indicators include: Hist_Volatility, RSI(length = DTE), EMA

		# Calculate 150d historical volitality
		returns = np.log(self.stock_data['Close']/self.stock_data['Close'].shift(1))
		returns.fillna(0, inplace=True)
		volatility = returns.rolling(window=150).std()*np.sqrt(252)
		self.stock_data['Hist_Volatility'] = volatility

		# Add RSI related to DTE
		if self.DTE == 1:
			self.stock_data['RSI'] = ta.RSI(self.stock_data['Close'], timeperiod = 7)
			self.stock_data['EMA_DTE'] = ta.EMA(self.stock_data['Close'], timeperiod = 7)
		else:
			self.stock_data['RSI'] = ta.RSI(self.stock_data['Close'], timeperiod = self.DTE)
			self.stock_data['EMA_DTE'] = ta.EMA(self.stock_data['Close'], timeperiod = self.DTE)

		# Add four EMA data
		self.stock_data['EMA_7'] = ta.EMA(self.stock_data['Close'], timeperiod = 7)
		self.stock_data['EMA_50'] = ta.EMA(self.stock_data['Close'], timeperiod = 50)
		self.stock_data['EMA_252'] = ta.EMA(self.stock_data['Close'], timeperiod = 252)
		

		# Add trend indicator, bull:1, bear:0, TREND_REVERSAL: crossover
		self.stock_data['Trend'] = 0.0
		self.stock_data['Trend'] = np.where(self.stock_data['EMA_DTE'] > self.stock_data['EMA_50'], 1.0, 0.0)
		self.stock_data['Trend_Reversal'] = self.stock_data['Trend'].diff()


		self.all_spreads = pd.merge(self.all_spreads, self.stock_data[['Date','Hist_Volatility','RSI','EMA_252','EMA_50','EMA_7','EMA_DTE','Trend','Trend_Reversal']], left_on=  ['QuoteDate'],
			right_on= ['Date'], 
			how = 'left')

		self.all_spreads.drop(columns = 'Date', inplace = True)
		self.all_spreads.dropna(inplace = True)


	def report(self):
		open_sp = self.trades_history.loc[self.trades_history['Trades Type'] == 'Open']
		close_sp = self.trades_history.loc[(self.trades_history['Trades Type'] == 'Close') | (self.trades_history['Trades Type'] == 'Stop Loss') | 
				     (self.trades_history['Trades Type'] == 'Take Profit')]
		close_sp = close_sp.reset_index(drop = True)
		cum_fund = self.init_fund + close_sp.Profit.cumsum()


		total_return = self.trades_history.Profit.sum()
		total_return_ratio = total_return/self.init_fund

		close_sp['year'] = close_sp.Date.dt.year
		years = close_sp.Date.dt.year.unique()
		annual_return_ratio = list()
		prev_yr_funds = self.init_fund
		for yr in years:
			yr_df = close_sp.loc[close_sp.year == yr]
			starting_funds = prev_yr_funds
			prev_yr_funds += yr_df.Profit.sum()
			print('Year',yr,'Return:',round(yr_df.Profit.sum(),2),'Starting:',starting_funds)
			yr_profit_ratio = round(yr_df.Profit.sum()/starting_funds,4)
			annual_return_ratio.append(yr_profit_ratio)

		print('Annual return rate:',annual_return_ratio)
		
		annual_return_std = np.std(annual_return_ratio)
		print('Annual return std:', annual_return_std)
		# sharpe = ((total_return_ratio/((self.end_date - self.start_date).days/365)))/annual_return_std
		sharpe = (np.mean(annual_return_ratio))/annual_return_std

		end_dd = np.argmax(np.maximum.accumulate(cum_fund) - cum_fund) # end of the period
		start_dd = np.argmax(cum_fund[:end_dd]) # start of period
		maxdd = (cum_fund[start_dd] - cum_fund[end_dd])/cum_fund[start_dd]

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
		print('Funds not utilized:',round(self.min_fund,2))
		
		print('Max Collateral:', (open_sp['Initial Funds'] - open_sp['Funds After']).max())
		print('Max Drawdown:', round(maxdd,3),'Date:', close_sp.loc[start_dd].Date, '-', close_sp.loc[end_dd].Date)
		print('Max Individual Loss:', round(self.trades_history.Profit.min(),2),'id:', self.trades_history.loc[self.trades_history.Profit.idxmin(),'Spread index'])
		print('Max Individual Gain:', round(self.trades_history.Profit.max(),2),'id:', self.trades_history.loc[self.trades_history.Profit.idxmax(),'Spread index'])
		
		print('Average Collateral:', round((open_sp['Initial Funds'] - open_sp['Funds After']).mean(),2))
		print('Average Gain:', round(self.trades_history.Profit.sum()/open_sp.shape[0],2))
		print('Average Annual Return:', round(np.mean(annual_return_ratio),2))
		print('Average Return Ratio:', round((close_sp.Profit/close_sp['Collateral']).mean(),4))
		print('Average Expectation Return Ratio:', round(self.spreads_satisfied.ExpectedEarnRatio.mean(),4))
		print('Average Expected Win Rate:', round(self.spreads_satisfied.Sell_OTM_Prob.mean(),2))

		fig,ax1 = plt.subplots()

		fig.set_size_inches(16, 8, forward=True)
		# make stock plot
		stock_plot = self.stock_data.loc[(self.stock_data.Date >= self.backtest_start_date) & (self.stock_data.Date <= self.backtest_end_date)]
		ax1.plot(stock_plot.Date, stock_plot.Close,color="grey")
		ax1.set_xlabel("Year", fontsize = 14)
		ax1.set_ylabel("Stock",
		              color="black",
		              fontsize=14)
		ax1.legend(['Stock'],loc='upper left')
		ax2 = ax1.twinx()

		ax2.plot(close_sp.Date, cum_fund, color='orange')
		ax2.plot([close_sp.loc[end_dd].Date, close_sp.loc[start_dd].Date], [cum_fund[end_dd], cum_fund[start_dd]], 'o', color='red', markersize=10)

		ax2.legend(['Funds'],loc='upper right')
		ax2.set_ylabel('Funds',color="black",
		              fontsize=14)
		plt.title("Strategy Return VS Stock Return",fontsize=16)
		plt.show()

		zeroed_stock = stock_plot.dropna().copy()
		zeroed_stock.Close /= zeroed_stock.iloc[0]['Close']

		zeroed_sp = cum_fund.copy()
		zeroed_sp /= self.init_fund

		fig2 = plt.figure(2)
		fig2.set_figwidth(16)
		fig2.set_figheight(8)
		plt.plot(zeroed_stock.Date, zeroed_stock.Close, color='grey', label = 'Stock')
		plt.plot(close_sp.Date, zeroed_sp, color="orange", label = 'Spread')
		plt.legend(loc='upper left')
		plt.title("Standardized Strategy Return VS Stock Return",fontsize=16)
		plt.show()
		