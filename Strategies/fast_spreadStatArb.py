#%%
from backtesting import Backtest, Strategy
import warnings 
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd
import pytz

data = pd.read_csv('SP500.csv', index_col=0, parse_dates=True)
SP500 = pd.DataFrame(data)

class StatArb(Strategy):
    lookback = 30
    size = 1
    stop_loss = size * 0.075
    max_position = 3
    zscore_threshold = 1.0
    num = 2


    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index).astype(float)
        self.spread = self.I(lambda: close.diff())

    def next(self):
        price = self.data["Close"][-1]
        spread = self.spread[-self.lookback:]
        spread_mean = spread.mean()
        spread_std = spread.std()
        zscore = (spread[-1] - spread_mean) / spread_std


        # Debugging statements
        #print(f"\nPrice: {price}, Spread: {spread[-1]}, Z-Score: {zscore}, \nSpread Mean: {spread_mean}, Spread Std: {spread_std}")

        if zscore > self.zscore_threshold and len(self.trades) < self.max_position:
            #print(f"\nSELL SIGNAL: Z-Score {zscore} > {self.zscore_threshold}")
            self.buy(sl=price - 0.075, size=self.size)

        elif zscore < -self.zscore_threshold and len(self.trades) < self.max_position:
            #print(f"\nBUY SIGNAL: Z-Score {zscore} < -{self.zscore_threshold}")
            self.sell(sl=price + 0.075, size=self.size)

        elif abs(zscore) < self.zscore_threshold / self.num:
            if self.position.is_long:
                #print(f"\nCLOSING LONG: Z-Score {zscore} < {self.zscore_threshold / 8}")
                self.position.close()
                
            elif self.position.is_short:
                #print(f"\nCLOSING SHORT: Z-Score {zscore} < {self.zscore_threshold / 8}")
                self.position.close()

bt = Backtest(SP500, StatArb,
              cash=10000, commission=0.0,
              exclusive_orders=True, margin=0.05)

output = bt.run()
print(output)
bt.plot()
