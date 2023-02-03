



---
# backtest.backtest

## class backtest.backtest(_spread_date = tuple of length 2 (call spreads, put spreads))
> Parameters:
1. _spread_date: a length 2 tuple contains all (call, put) spreads  
> Attributes:
1. call_spreads, put_spreads: store initial spreads data
   
2. call_satisfied, put_satisfied: store spreads that satisfy params, 
   avaliable after backtest.go()
3. call_cum_return, put_cum_return: cumulative return on selected spreads, avaliable after backtest.go()

> Methods:
1. set_parm(parms = (-9999, -1.0, 0, 0, 0, 999, 99999))
   - Set the parameters for creating strategy. Using the default parameter will result in returning all spreads at backtest.go().
   - parms: a length 7 tuple contains parameter for creating strategy:(
     1. min_EXPECTED_EARN, 
     2. min_EARN_RATIO,
     3. min_SELL_OTM_PROB, 
     4. min_BUY_OTM_PROB,
     5. min_width,
     6. max_width,
     7. max_trades_per_day)

&nbsp;

1. go()
   - Call after setting strategy parameters with set_parm()
   - Will set self.call_satisfied and self.put_satisfied according to strategy.

&nbsp;

3. draw_result(st):
   - Draw the actual return plot for both call/put spreads and the stock price.
   - st: a pandas df containing stock OHLC 


&nbsp;

4. win_rate()
   - return: float call win rate, put win rate


&nbsp;

5. filter_spreads(df_)
   - A helper function used by backtest.go().
   - df_: a pandas df containing either the original call/put spreads data
   - return: a pandas df containing the spreads that satisfy the strategy