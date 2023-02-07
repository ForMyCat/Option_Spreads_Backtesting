import pandas as pd
import numpy as np
import datetime
from datetime import date, timezone
from datetime import timedelta
import scipy
import time

class spreads_prep:
	def __init__(self , _option_data, _stock_data):
		self.start_date, self.end_date, self.df_aligned = self.preproc_data(_stock_data, _option_data)
		self.calculate_probability()
		self.all_calls = None
		self.all_puts = None

	def preproc_data(self, st, op):

		time_0 = time.time()
		ptime_0 = time.process_time()

		options_clean = op.copy()
		all_stock = st.copy()

		# Exclude Options with 0 DTE Since It Will Cause Probability Calculation Error
		options_clean = options_clean.loc[options_clean.DTE != 0].reset_index(drop = True)

		#Exclude Options with 0 Transaction
		# for i in ['C_VOLUME','P_VOLUME']:
		# 	options_clean[i] = pd.to_numeric(options_clean[i],errors='coerce')
		# options_clean = options_clean.loc[(options_clean.C_VOLUME != 0) & (options_clean.P_VOLUME != 0)].reset_index(drop = True)
		
		#Convert All Data Columns to Numeric
		for i in ['STRIKE','DTE','C_IV','C_BID','C_ASK','P_IV','P_BID','P_ASK']:
			options_clean[i] = pd.to_numeric(options_clean[i],errors='coerce')

		# options_clean = options_clean.reset_index(drop = True)
		options_clean = options_clean.dropna().reset_index(drop = True)


		#Line up stock data with option data
		all_stock.index = pd.to_datetime(all_stock.DATE).dt.tz_localize(None)
		options_clean.index = pd.to_datetime(options_clean.EXPIRE_EST).dt.tz_localize(None)

		start_date =  max((all_stock.index).min(), (options_clean.index).min()).date()
		end_date =  min((all_stock.index).max(), (options_clean.index).max()).date()

		# Join Option and Stock Price by Expiry Date to create PRICE@EXPIRE Column
		# df_aligned = options_clean.join(all_stock['CLOSE']).copy()
		df_aligned = options_clean.join(all_stock['CLOSE']).dropna().copy()
		df_aligned.rename(columns = {'CLOSE':'PRICE@EXPIRE'}, inplace = True)


		# Join Option and Stock Price by Quote Date to create CURRENT_PRICE Column
		df_aligned.index = pd.to_datetime(df_aligned.QUOTE_TIME_EST)
		# df_aligned = df_aligned.join(all_stock['CLOSE']).copy()
		df_aligned = df_aligned.join(all_stock['CLOSE']).dropna().copy()
		df_aligned.rename(columns = {'CLOSE':'CURRENT_PRICE'}, inplace = True)
		# df_aligned.drop(columns = ['QUOTE_TIME_UTC','EXPIRE_UTC', 'EXPIRE_EST'], axis = 1, inplace = True)

		# Drop options with zero IV
		df_aligned =df_aligned.loc[(df_aligned.P_IV != 0) & (df_aligned.C_IV != 0)].copy()

		df_aligned.reset_index(inplace = True, drop = True)

		time_1 = time.time()
		ptime_1 = time.process_time()
		print(f'Done Cleaning Data! CPU time: %.2f seconds, Total time: %.2f seconds'%(ptime_1 - ptime_0, time_1 - time_0))
		
		return start_date, end_date, df_aligned.copy()

	def calculate_probability(self):
		time_2 = time.time()
		ptime_2 = time.process_time()

		temp_df = self.df_aligned.copy()
		temp_df['STRIKE_DISTANCE'] = abs(temp_df['STRIKE'] - temp_df['CURRENT_PRICE'])

		#Find ATM IV Data For Each Date
		atm_options = temp_df.sort_values(by = ['QUOTE_TIME_EST','STRIKE_DISTANCE']).groupby('QUOTE_TIME_EST').head(1).copy()
		atm_options.rename(columns = {'P_IV':'P_ATM_IV','C_IV':'C_ATM_IV'}, inplace = True)

		#Join ATM IV Data By Date
		temp_df = temp_df.set_index('QUOTE_TIME_EST').join(atm_options.set_index('QUOTE_TIME_EST')[['P_ATM_IV','C_ATM_IV']]).reset_index()
		p_std = np.sqrt((temp_df['DTE']) * (temp_df['P_ATM_IV']**2) / 252)
		c_std = np.sqrt((temp_df['DTE']) * (temp_df['C_ATM_IV']**2) / 252)
		
		price_to_strike_ratio = (np.log(temp_df['CURRENT_PRICE']/ temp_df['STRIKE']))

		temp_df['P_ITM_PROB'] = 1 - scipy.stats.norm.cdf((price_to_strike_ratio)/p_std)
		temp_df['C_ITM_PROB'] = scipy.stats.norm.cdf(price_to_strike_ratio/c_std)

		temp_df['P_OTM_PROB'] = 1 - temp_df['P_ITM_PROB']
		temp_df['C_OTM_PROB'] = 1 - temp_df['C_ITM_PROB']

		self.df_aligned = temp_df.copy()
		time_3 = time.time()
		ptime_3 = time.process_time()

		print(f'Done Calculating Probability! CPU time: %.2f seconds, Total time: %.2f seconds'%(ptime_3 - ptime_2, time_3 - time_2))


	def spread_search(self, DTE, start_date, end_date):
		time_0 = time.time()
		ptime_0 = time.process_time()

		print(f'Start searching for spreads from %s to %s'%(start_date,end_date))
		df_search = self.df_aligned.copy()
		df_search = df_search.loc[(df_search['QUOTE_TIME_EST'] >= start_date) & (df_search['QUOTE_TIME_EST'] <= end_date)].copy()
		df_search = df_search.loc[(df_search['DTE']).round(2) == DTE].copy()
		df_search = df_search.sort_values(by='QUOTE_TIME_EST').reset_index(drop=True).copy()
		# date_range = pd.date_range(start_date, end_date, freq = 'D')
		all_date = df_search['QUOTE_TIME_EST'].unique()

		# cc = 0

		all_puts = pd.DataFrame()
		all_calls = pd.DataFrame()
		# calls_list = list()
		# puts_list = list()

		for date in all_date:
			time_4 = time.time()
			ptime_4 = time.process_time()

			single_day_options = df_search.loc[df_search['QUOTE_TIME_EST'] == date].copy()
			
			single_day_options = single_day_options.sort_values(by='STRIKE', ascending = True).reset_index(drop=True).copy()

			single_day_call_spreads = {'QUOTE_TIME_EST':[],'SELL_PRICE':[],'BUY_PRICE':[], 'DTE':[], 'isCALL':[],
										'SELL_STRIKE':[],'BUY_STRIKE':[],'PREMIUM':[],'MAX_LOSS':[],'SELL_ATM_IV':[],'BUY_ATM_IV':[],'SELL_IV':[],'BUY_IV':[],
										'EXPECTED_EARN':[],'EXPECTED_EARN_RATIO':[],'BUY_OTM_PROB':[],'SELL_OTM_PROB':[],'CURRENT_PRICE':[],'PRICE@EXPIRE':[]}

			single_day_put_spreads = {'QUOTE_TIME_EST':[],'SELL_PRICE':[],'BUY_PRICE':[], 'DTE':[], 'isCALL':[],
										'SELL_STRIKE':[],'BUY_STRIKE':[],'PREMIUM':[],'MAX_LOSS':[],'SELL_ATM_IV':[],'BUY_ATM_IV':[],'SELL_IV':[],'BUY_IV':[],
										'EXPECTED_EARN':[],'EXPECTED_EARN_RATIO':[],'BUY_OTM_PROB':[],'SELL_OTM_PROB':[],'CURRENT_PRICE':[],'PRICE@EXPIRE':[]}
			# if cc >=50:
			# 	break

			# cc += 1
			for i in single_day_options.index:

				for j in single_day_options.index:
					if i >= j:
						continue
					else:
						op_low = single_day_options.iloc[i].copy()
						op_high = single_day_options.iloc[j].copy()

						if (op_low.STRIKE > op_low.CURRENT_PRICE): # Call spread: lower strike is sell, in this case sell op_low

							earn = self.calculate_expected_earn('CALL', sell = op_low, buy = op_high)
							earn_ratio = earn/ abs(op_low.STRIKE - op_high.STRIKE)

							single_day_call_spreads['QUOTE_TIME_EST'].append(date)
							single_day_call_spreads['DTE'].append(op_low.DTE)
							single_day_call_spreads['isCALL'].append(1)
							single_day_call_spreads['SELL_STRIKE'].append(op_low.STRIKE)
							single_day_call_spreads['SELL_ATM_IV'].append(op_low.C_ATM_IV)
							single_day_call_spreads['BUY_ATM_IV'].append(op_high.C_ATM_IV)
							single_day_call_spreads['SELL_IV'].append(op_low.C_IV)
							single_day_call_spreads['BUY_IV'].append(op_high.C_IV)
							single_day_call_spreads['SELL_PRICE'].append(op_low.C_BID)
							single_day_call_spreads['BUY_PRICE'].append(op_high.C_ASK)
							single_day_call_spreads['BUY_STRIKE'].append(op_high.STRIKE)
							single_day_call_spreads['PREMIUM'].append(op_low.C_BID - op_high.C_ASK)
							single_day_call_spreads['MAX_LOSS'].append(op_high.STRIKE - op_low.STRIKE - (op_low.C_BID - op_high.C_ASK))
							single_day_call_spreads['EXPECTED_EARN'].append(earn)
							single_day_call_spreads['EXPECTED_EARN_RATIO'].append(earn_ratio)
							single_day_call_spreads['SELL_OTM_PROB'].append(op_low.C_OTM_PROB)
							single_day_call_spreads['BUY_OTM_PROB'].append(op_high.C_OTM_PROB)
							single_day_call_spreads['CURRENT_PRICE'].append(op_high.CURRENT_PRICE)
							single_day_call_spreads['PRICE@EXPIRE'].append(op_high['PRICE@EXPIRE'])
							

						if (op_high.STRIKE < op_high.CURRENT_PRICE): # Put spread: higher strike is sell, in this case sell op_high

							earn = self.calculate_expected_earn('PUT', sell = op_high, buy = op_low)
							earn_ratio = earn/ abs(op_low.STRIKE - op_high.STRIKE)

							single_day_put_spreads['QUOTE_TIME_EST'].append(date)
							single_day_put_spreads['DTE'].append(op_high.DTE)
							single_day_put_spreads['isCALL'].append(0)
							single_day_put_spreads['SELL_STRIKE'].append(op_high.STRIKE)
							single_day_put_spreads['SELL_ATM_IV'].append(op_high.P_ATM_IV)
							single_day_put_spreads['BUY_ATM_IV'].append(op_low.P_ATM_IV)
							single_day_put_spreads['SELL_IV'].append(op_high.P_IV)
							single_day_put_spreads['BUY_IV'].append(op_low.P_IV)
							single_day_put_spreads['SELL_PRICE'].append(op_high.P_BID)									
							single_day_put_spreads['BUY_PRICE'].append(op_low.P_ASK)
							single_day_put_spreads['BUY_STRIKE'].append(op_low.STRIKE)
							single_day_put_spreads['PREMIUM'].append(op_high.P_BID - op_low.P_ASK)
							single_day_put_spreads['MAX_LOSS'].append(op_high.STRIKE - op_low.STRIKE - (op_high.P_BID - op_low.P_ASK))
							single_day_put_spreads['EXPECTED_EARN'].append(earn)
							single_day_put_spreads['EXPECTED_EARN_RATIO'].append(earn_ratio)
							single_day_put_spreads['SELL_OTM_PROB'].append(op_high.P_OTM_PROB)
							single_day_put_spreads['BUY_OTM_PROB'].append(op_low.P_OTM_PROB)
							single_day_put_spreads['CURRENT_PRICE'].append(op_high.CURRENT_PRICE)
							single_day_put_spreads['PRICE@EXPIRE'].append(op_high['PRICE@EXPIRE'])


			# Save the best call/put spread of the day in a dictionary
			# print(single_day_options.iloc[i].QUOTE_TIME_EST)
			df_single_day_call_spreads = pd.DataFrame(single_day_call_spreads)
			df_single_day_put_spreads = pd.DataFrame(single_day_put_spreads)
			# df_single_day_call_spreads = df_single_day_call_spreads.sort_values(by='EXPECTED_EARN', ascending = False).reset_index(drop=True)
			# df_single_day_put_spreads = df_single_day_put_spreads.sort_values(by='EXPECTED_EARN', ascending = False).reset_index(drop=True)

			# if (len(df_single_day_call_spreads) > 0):
			# 	best_call_spread_today = df_single_day_call_spreads.head(1)
			# 	best_call_spread_dict[date] = best_call_spread_today
			# print('Num of call spreads today:',len(df_single_day_call_spreads))

			# if (len(df_single_day_put_spreads) > 0):
			# 	best_put_spread_today = df_single_day_put_spreads.head(1)
			# 	best_put_spread_dict[date] = best_put_spread_today
			# print('Num of put spreads today:',len(df_single_day_put_spreads))

			# calls_list.append(df_single_day_put_spreads)
			# puts_list.append(df_single_day_call_spreads)

			all_puts = pd.concat((all_puts,df_single_day_put_spreads),ignore_index = True)
			all_calls = pd.concat((all_calls,df_single_day_call_spreads),ignore_index = True)

			time_5 = time.time()
			ptime_5 = time.process_time()

		time_1 = time.time()
		ptime_1 = time.process_time()
		print(f'Done Calculating All Spreads with DTE = %s! CPU time: %.2f seconds, Total time: %.2f seconds'%(str(DTE), ptime_1 - ptime_0, time_1 - time_0))
		# return pd.DataFrame(best_call_spread_dict).T.copy(), pd.DataFrame(best_put_spread_dict).T.copy()
		return all_calls, all_puts


	def calculate_expected_earn(self, op_type, sell, buy):

		expected_earn = -1

		if op_type in ['c', 'C', 'call', 'Call', 'CALL']:
			premiun = sell.C_BID - buy.C_ASK
			max_loss = buy.STRIKE - sell.STRIKE
			expected_earn = premiun - sell.C_ITM_PROB * max_loss

		elif op_type in ['p', 'P', 'put', 'Put', 'PUT']:
			premiun = sell.P_BID - buy.P_ASK
			max_loss = sell.STRIKE - buy.STRIKE
			expected_earn = premiun - sell.P_ITM_PROB * max_loss

		return expected_earn




	def get_all_spreads(self, _DTE = 14, _start_date = None , _end_date = None):
		if _start_date == None:
			_start_date = self.start_date

		if _end_date == None:
			_end_date = self.end_date 

		self.all_calls, self.all_puts = self.spread_search(_DTE, _start_date, _end_date)
		return self.all_calls, self.all_puts

