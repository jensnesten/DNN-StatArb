#%%
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import time

# calc run  time
start_time = time.time()


data = pd.read_csv('../Data/SP500_year.csv', index_col=0, parse_dates=True)
SP500 = pd.DataFrame(data)

class StatArb(Strategy):
    lookback = 20
    size = 5
    stop_loss = 0.0705
    max_position = 3
    position_num = 0
    zscore_threshold = 1.0
    bidask_spread = 0.4 / 2
    init_cash = 100000
    scaling_factor = 0.05
    scaling_threshold = 0.1

    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index).astype(float)
        close2 = pd.Series(self.data.Close2, index=self.data.index).astype(float)
        self.spread = self.I(lambda: close/close[-2] - close2/close2[-2])
        
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
            self.sell(sl=price + ( self.bidask_spread + self.stop_loss), size=self.size)
        elif zscore < -self.zscore_threshold and len(self.trades) < self.max_position:
            self.buy(sl=price - (self.bidask_spread + self.stop_loss), size=self.size)

        elif abs(zscore) < self.zscore_threshold / 2:
            if self.position.is_long:
                self.position.close()
                
            elif self.position.is_short:
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
              exclusive_orders=False, trade_on_close=True)

output = bt.run()
print(output)
print("--- %s seconds ---" % (time.time() - start_time))

equity = pd.DataFrame(output['_equity_curve'])
equity.iloc[:, :1].to_csv('eq.csv')

eq = pd.read_csv('eq.csv', parse_dates=[0], index_col=0)

eq['Equity'] = (eq['Equity'] - eq['Equity'].iloc[0]) / eq['Equity'].iloc[0] * 100
data['Close'] = (data['Close'] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
plt.figure(figsize=(15, 8))

eq['Equity'].plot(label='Strategy')
data['Close'].plot(label='SP500')

plt.title('Equity Curve')
plt.xlabel('Time')
plt.ylabel('Equity Return (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




#bt.plot()