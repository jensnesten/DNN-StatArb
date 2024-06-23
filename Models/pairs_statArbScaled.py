#%%
from backtesting import Backtest, Strategy, _stats
from backtesting.lib import crossover
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd
import time
from lightweight_charts import Chart


data = pd.read_csv('../Data/SP500_DJIA_2020_clean.csv', index_col=0, parse_dates=True)
SP500 = pd.DataFrame(data)

start_time = time.time()

class StatArb(Strategy):
    lookback = 30
    size = 2
    stop_loss = size * 0.03525
    max_position = size * 7
    position_num = 0
    zscore_threshold = 1.0
    bidask_spread = 0.4 / 2
    init_cash = 100000
    scaling_factor = 0.05
    scaling_threshold = 0.1

    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index).astype(float)
        close2 = pd.Series(self.data.Close2, index=self.data.index).astype(float)
        self.spread = self.I(lambda:  1000 * close/close[-2] - close2/close2[-2])
        
    def next(self):
        price = self.data["Close"][-1]
        spread = self.spread[-self.lookback:]
        spread_mean = spread.mean()
        spread_std = spread.std()
        zscore = (spread[-1] - spread_mean) / spread_std
        self.scale_model()
        
        # Debugging statements
        #print(f"\nPrice: {price}, Spread: {spread[-1]}, Z-Score: {zscore}, \nSpread Mean: {spread_mean}, Spread Std: {spread_std}")
        
        if zscore > self.zscore_threshold and len(self.trades) < self.max_position:
            #print(f"\nSELL SIGNAL: Z-Score {zscore} > {self.zscore_threshold}")
            self.sell(sl=price + ( self.bidask_spread + self.stop_loss), size=self.size)

        elif zscore < -self.zscore_threshold and len(self.trades) < self.max_position:
            #print(f"\nBUY SIGNAL: Z-Score {zscore} < -{self.zscore_threshold}")
            self.buy(sl=price - (self.bidask_spread + self.stop_loss), size=self.size)

        elif abs(zscore) < self.zscore_threshold / 2:
            if self.position.is_long:
                #print(f"\nCLOSE LONG: Z-Score {zscore} < {self.zscore_threshold / 8}")
                self.position.close()
                
            elif self.position.is_short:
                #print(f"\nCLOSE SHORT: Z-Score {zscore} < {self.zscore_threshold / 8}")
                self.position.close()
        
        
    # Scaling function
    def scale_model(self):    
        if self.equity > self.init_cash * (1 + self.scaling_threshold):
            self.size += 1
            self.init_cash = self.equity
            #self.stop_loss = self.stop_loss * (1 + self.scaling_factor)
            print(f"Scaling up: New size: {self.size}, New stop loss: {self.stop_loss}")

bt = Backtest(SP500, StatArb,
              cash=100000, commission=0.0,
              exclusive_orders=False, trade_on_close=True, margin=0.05)

output = bt.run()
print(output)
bt.plot()

# Save the equity curve to a csv file
equity = pd.DataFrame(output['_equity_curve'])
equity.iloc[:, :1].to_csv('equity_curve.csv')




end_time = time.time()  # Step 3
runtime = end_time - start_time  # Step 4
#print(f"Runtime: {runtime} seconds")  # Step 5
