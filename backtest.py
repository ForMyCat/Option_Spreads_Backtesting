import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class backtest:
	def __init__(self , _spread_data = (pd.DataFrame(), pd.DataFrame())):
		self.call_spreads, self.put_spreads = _spread_data[0], _spread_data[1]

		self.call_spreads = self.call_spreads.sort_values(by = ['QUOTE_TIME_EST','SELL_STRIKE'], ascending = [True, True]).copy()
		self.put_spreads = self.put_spreads.sort_values(by = ['QUOTE_TIME_EST','SELL_STRIKE'], ascending = [True, True]).copy()

		self.call_satisfied = pd.DataFrame()
		self.put_satisfied = pd.DataFrame()


		self.call_cum_return = list()
		self.put_cum_return = list()

	def set_parm(self,parms = (0, 0, 0.5, 0.5, 0, 30, 1)):
		self.min_EXPECTED_EARN = parms[0]
		self.min_EARN_RATIO = parms[1]

		self.min_SELL_OTM_PROB = parms[2]
		self.min_BUY_OTM_PROB = parms[3]

		self.min_width = parms[4]
		self.max_width = parms[5]

		self.max_trades_per_day = parms[6]

	def go(self):
		self.call_satisfied, self.call_cum_return = self.filter_spreads(self.call_spreads)
		self.put_satisfied, self.put_cum_return = self.filter_spreads(self.put_spreads)
		print('Done backtesting!')
		 

	def filter_spreads(self, df_):
		df_ = df_.copy()

		df_ = df_.loc[(df_.EXPECTED_EARN >= self.min_EXPECTED_EARN)]
		df_ = df_.loc[(df_.EXPECTED_EARN_RATIO >= self.min_EARN_RATIO)]
		df_ = df_.loc[(df_.SELL_OTM_PROB >= self.min_SELL_OTM_PROB)]
		df_ = df_.loc[(df_.BUY_OTM_PROB >= self.min_BUY_OTM_PROB)]
		df_['WIDTH'] = abs(df_.SELL_STRIKE - df_.BUY_STRIKE)
		df_ = df_.loc[(df_.WIDTH >= self.min_width) & (df_.WIDTH <= self.max_width)]

		df_ = df_.sort_values(by = ['QUOTE_TIME_EST','PREMIUM'], ascending = [True, False]).groupby('QUOTE_TIME_EST').head(self.max_trades_per_day).copy()
		df_['ACTUAL_EARN'] = df_.apply(lambda x: utils.calculate_actual_earn(x), axis = 1)
		cum_return = utils.return_cum_earn_list(df_)
		df_['CUM_EARN'] = cum_return
		df_['WIN'] = df_['ACTUAL_EARN'] > 0

		df_ = df_.copy()
		return df_, cum_return



	def draw_result(self, st):
		fig,ax1 = plt.subplots()

		fig.set_size_inches(16, 8, forward=True)
		# make a plot
		ax1.plot(st.DATE, st.CLOSE,color="grey")
		# set x-axis label
		ax1.set_xlabel("Year", fontsize = 14)
		# set y-axis label
		ax1.set_ylabel("Stock",
		              color="black",
		              fontsize=14)


		# make a plot with different y-axis using second axis object
		ax2 = ax1.twinx()

		# print(st.DATE)
		# print(self.call_satisfied.QUOTE_TIME_EST)

		

		call_by_day = (self.call_satisfied.groupby('QUOTE_TIME_EST').agg({'ACTUAL_EARN':'sum'}))
		put_by_day = (self.put_satisfied.groupby('QUOTE_TIME_EST').agg({'ACTUAL_EARN':'sum'}))
		call_by_day.reset_index(inplace = True)
		put_by_day.reset_index(inplace = True)


		ax2.plot(pd.to_datetime(call_by_day.QUOTE_TIME_EST).dt.tz_localize(None), utils.return_cum_earn_list(call_by_day),color="green")
		ax2.plot(pd.to_datetime(put_by_day.QUOTE_TIME_EST).dt.tz_localize(None), utils.return_cum_earn_list(put_by_day),color="red")
		ax2.set_ylabel("Spreads",color="black")
		plt.title("Strategy Return VS Stock Return",fontsize=16)
		ax1.legend(['Stock'],loc='center left')
		ax2.legend(['Call Credit Spreads', 'Put Credit Spreads'],loc='upper left')

		plt.show()

	def win_rate(self):
		call_win = (self.call_satisfied.ACTUAL_EARN >= 0).sum()
		put_win = (self.put_satisfied.ACTUAL_EARN >= 0).sum()

		call_win_rate = call_win/(self.call_satisfied.shape[0])
		put_win_rate = put_win/(self.put_satisfied.shape[0])

		print('Call:', (self.call_satisfied.shape[0]), 'trades, win rate:', round(call_win_rate,4))
		print('Put:', (self.put_satisfied.shape[0]), 'trades, win rate:', round(put_win_rate,4))

		return round(call_win_rate,4), round(put_win_rate,4)
