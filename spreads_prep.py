import pandas as pd
import numpy as np
import datetime
from datetime import date, timezone
from datetime import timedelta
import scipy
import time
import gc
import os, glob

class spreads_prep:
	def __init__(self , _option_data, _stock_data, DTE = [1,2,3,4,5,6,7]):
		self.preproc_data(_stock_data, _option_data)

		time_2 = time.time()
		ptime_2 = time.process_time()
		self.tk = _option_data.iloc[0].Symbol
		self.calls = self.calculate_probability(self.calls)
		self.puts = self.calculate_probability(self.puts)
		time_3 = time.time()
		ptime_3 = time.process_time()
		print(f'Done Calculating Probability! CPU time: %.2f seconds, Total time: %.2f seconds'%(ptime_3 - ptime_2, time_3 - time_2))

		for d in DTE:
			time_0 = time.time()
			ptime_0 = time.process_time()
			call_spreads = self.spread_search(d, self.calls)
			put_spreads = self.spread_search(d, self.puts)
			call_spreads.to_csv('SpreadsData\\' + self.tk + '\\DTE'+ str(d)+ '_' + self.tk +'_CallSpreads.csv')
			put_spreads.to_csv('SpreadsData\\' + self.tk + '\\DTE'+ str(d)+ '_' + self.tk +'_PutSpreads.csv')
			del call_spreads
			del put_spreads
			gc.collect()
			time_1 = time.time()
			ptime_1 = time.process_time()
			print(f'Done Calculating Spreads with DTE = %s! CPU time: %.2f seconds, Total time: %.2f seconds'%(str(d), ptime_1 - ptime_0, time_1 - time_0))

	def preproc_data(self, st, op):

		time_0 = time.time()
		ptime_0 = time.process_time()

		op_clean = op.copy()
		op_clean['QuoteDate'] = pd.to_datetime(op_clean.QuoteDate)
		op_clean['ExpirationDate'] = pd.to_datetime(op_clean.ExpirationDate)
		op_clean = op_clean.loc[op_clean.ExpirationDate.dt.date < datetime.date.today()]

		# Map expiration date to the cloest previous closing date
		date_map = dict()
		market_date = set(st.Date)
		op_exp_date = list(set(op_clean['ExpirationDate']))

		def expire_on_market_date(op_exp,market_open):
			return op_exp in market_open

		for i in op_exp_date:
			checking_date = i 
			while not expire_on_market_date(checking_date,market_date):
				checking_date -= timedelta(days = 1)
			date_map[i] = checking_date

		op_clean['ExpirationDate'] = op_clean['ExpirationDate'].map(date_map)

		op_clean['Dte'] = (op_clean.ExpirationDate - op_clean.QuoteDate).dt.days.astype(int)
		op_clean['key'] = list(zip(op_clean.StrikePrice, op_clean.ExpirationDate))

		#Convert Columns to Numeric
		for i in ['BidSize','OpenInterest', 'Volume']:
			op_clean[i] =  pd.to_numeric(op_clean[i], errors='coerce',downcast = 'integer')

		op_clean = op_clean.loc[op_clean.Volume != 0]

		for i in ['AskPrice','BidPrice','LastPrice', 'StrikePrice','UnderlyingPrice','ImpliedVolatility','Delta','Gamma', 'Vega', 'Rho', 'Theta']:
			op_clean[i] =  pd.to_numeric(op_clean[i], errors='coerce')

		op_clean = op_clean.dropna().reset_index(drop = True)

		# # Exclude Options with 0 DTE Since It Will Cause Probability Calculation Error
		# op_zero = op_clean.loc[op_clean['Dte'] == 0]
		# op_not_zero = op_clean.loc[op_clean['Dte'] > 0]

		self.start_date =  max((st.Date).min(), (op_clean.QuoteDate).min())
		self.end_date =  min((st.Date).max(), (op_clean.ExpirationDate).max())

		# Join Option and Stock Price by Expiry Date to create PRICE@EXPIRE Column
		st = st.copy()
		st['Price@Expiration'] = st['Close']
		st.index = st.Date
		op_clean.index = op_clean.ExpirationDate

		exp_date = set(op_clean.ExpirationDate)
		exp_date.update(st.Date)
		exp_df = pd.DataFrame(exp_date, columns = ['Date'])
		exp_df = exp_df.sort_values(by = 'Date').reset_index(drop = True)
		exp_df.index = exp_df.Date
		exp_df = exp_df.join(st['Price@Expiration'])
		exp_df = exp_df.ffill()
		op_aligned = op_clean.join(exp_df['Price@Expiration'])
		op_aligned.dropna(inplace = True)
		op_aligned.reset_index(inplace = True, drop = True)

		self.calls = op_aligned.loc[op_aligned['PutCall'] == 'call'].reset_index(drop = True).copy()
		self.puts = op_aligned.loc[op_aligned['PutCall'] == 'put'].reset_index(drop = True).copy()

		self.op_aligned = op_aligned.copy()

		time_1 = time.time()
		ptime_1 = time.process_time()
		print(f'Done Cleaning Data! CPU time: %.2f seconds, Total time: %.2f seconds'%(ptime_1 - ptime_0, time_1 - time_0))
		

	def calculate_probability(self, options):
		
		temp_df = options.copy()

		temp_df['Strike_Distance'] = abs(temp_df['StrikePrice'] - temp_df['UnderlyingPrice'])

		#Find ATM IV Data For Each Date
		atm_options = temp_df.sort_values(by = ['QuoteDate','Strike_Distance']).groupby(['QuoteDate','Dte']).head(1).copy()
		atm_options.rename(columns = {'ImpliedVolatility':'ATM_IV'}, inplace = True)

		#Join ATM IV Data By Date
		temp_df = pd.merge(temp_df, atm_options[['QuoteDate','Dte','ATM_IV']],  how='left', left_on=['QuoteDate','Dte'], right_on = ['QuoteDate','Dte'])
		
		_std = np.sqrt((temp_df['Dte']) * (temp_df['ATM_IV']**2) / 252)
		
		price_to_strike_ratio = (np.log(temp_df['UnderlyingPrice']/ temp_df['StrikePrice']))

		if temp_df.iloc[0].PutCall == 'call':
			temp_df['ITM_Prob'] = scipy.stats.norm.cdf(price_to_strike_ratio/_std)
		elif temp_df.iloc[0].PutCall == 'put':
			temp_df['ITM_Prob'] = 1 - scipy.stats.norm.cdf((price_to_strike_ratio)/_std)
		else:
			print('Missing Option Type.')
			return

		temp_df['OTM_Prob'] = 1 - temp_df['ITM_Prob']

		return temp_df.copy()


	def spread_search(self, DTE, options):
		start_date = self.start_date
		end_date = self.end_date
		# print(f'Start searching for spreads from %s to %s'%(start_date,end_date))
		op_cols = ['QuoteDate','Symbol', 'ExpirationDate', 'AskPrice', 'BidPrice','Dte'
           ,'PutCall', 'StrikePrice','UnderlyingPrice','OTM_Prob','ImpliedVolatility','ATM_IV','Price@Expiration']
		
		df_search = options[op_cols].copy()
		df_search = df_search.loc[(df_search['QuoteDate'] >= start_date) & (df_search['ExpirationDate'] <= end_date)]
		df_search = df_search.loc[df_search.Dte == DTE]
		df_search = df_search.sort_values(by=['QuoteDate','StrikePrice']).reset_index(drop=True)

		all_date = df_search['QuoteDate'].unique()
		isCall = 1 if options.iloc[0].PutCall == 'call' else 0

		all_spreads = pd.DataFrame()
		
		def calculate_expected_earn(sell, buy):
			expected_earn = -999
			premiun = sell.BidPrice - buy.AskPrice
			max_loss = abs(buy.StrikePrice - sell.StrikePrice)
			expected_earn = premiun - (1 - sell.OTM_Prob) * max_loss

			return expected_earn
		# calls_list = list()
		# puts_list = list()
		
		spreads_dict = {'QuoteDate':[],'ExpirationDate':[],'SellPrice':[],'BuyPrice':[], 'Dte':[], 'isCALL':[],
								'SellStrike':[],'BuyStrike':[],'Premium':[],'MaxLoss':[],'ATM_IV':[],'Sell_IV':[],'Buy_IV':[],
								'ExpectedEarn':[],'ExpectedEarnRatio':[],'Buy_OTM_Prob':[],'Sell_OTM_Prob':[],'UnderlyingPrice':[]
							,'Price@Expiration':[]}

		for date in all_date:
			single_day_options = df_search.loc[df_search['QuoteDate'] == date].copy()

			single_day_options = single_day_options.sort_values(by='StrikePrice', ascending = True).reset_index(drop=True).copy()

			single_day_op_index = list(single_day_options.index)
			single_day_op_index.sort()
			
			for i in single_day_op_index:

				for j in single_day_op_index:
					if i >= j:
						continue
					else:
						op_low = single_day_options.loc[i].copy()
						op_high = single_day_options.loc[j].copy()
						
						if isCall:
							op_sell, op_buy = op_low, op_high
						else:
							op_sell, op_buy = op_high, op_low
							
	#                     if (op_low.STRIKE > op_low.CURRENT_PRICE): # Call spread: lower strike is sell, in this case sell op_low

						earn = calculate_expected_earn(sell = op_sell, buy = op_buy)
						earn_ratio = earn/ (abs(op_sell.StrikePrice - op_buy.StrikePrice) + 0.0000001)

						spreads_dict['QuoteDate'].append(date)
						spreads_dict['ExpirationDate'].append(op_sell.ExpirationDate)
						spreads_dict['Dte'].append(op_sell.Dte)
						spreads_dict['isCALL'].append(isCall)
						spreads_dict['SellStrike'].append(op_sell.StrikePrice)
						spreads_dict['ATM_IV'].append(op_sell.ATM_IV)
						spreads_dict['Sell_IV'].append(op_sell.ImpliedVolatility)
						spreads_dict['Buy_IV'].append(op_buy.ImpliedVolatility)
						spreads_dict['SellPrice'].append(op_sell.BidPrice)
						spreads_dict['BuyPrice'].append(op_buy.AskPrice)
						spreads_dict['BuyStrike'].append(op_buy.StrikePrice)
						spreads_dict['Premium'].append(op_sell.BidPrice - op_buy.AskPrice)
						spreads_dict['MaxLoss'].append(abs(op_buy.StrikePrice - op_sell.StrikePrice) - (op_sell.BidPrice - op_buy.AskPrice))
						spreads_dict['ExpectedEarn'].append(earn)
						spreads_dict['ExpectedEarnRatio'].append(earn_ratio)
						spreads_dict['Sell_OTM_Prob'].append(op_sell.OTM_Prob)
						spreads_dict['Buy_OTM_Prob'].append(op_buy.OTM_Prob)
						spreads_dict['UnderlyingPrice'].append(op_buy.UnderlyingPrice)
						spreads_dict['Price@Expiration'].append(op_buy['Price@Expiration'])

		all_spreads = pd.DataFrame(spreads_dict)

		return all_spreads


	def get_all_spreads(self, _DTE = 14, _start_date = None , _end_date = None):
		if _start_date == None:
			_start_date = self.start_date

		if _end_date == None:
			_end_date = self.end_date 

		self.all_calls, self.all_puts = self.spread_search(_DTE, _start_date, _end_date)
		return self.all_calls, self.all_puts


