#%%
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd
import numpy as np
data = pd.read_csv('./Data/SP500_DJIA_2m_clean.csv', index_col=0, parse_dates=True)
SP500 = pd.DataFrame(data)

class StatArb(Strategy):
    lookback = 30
    size = 1
    stop_loss = size * 0.03525
    max_position = 3
    position_num = 0
    bidask_spread = 0.4 / 2

    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index).astype(float)
        close2 = pd.Series(self.data.Close2, index=self.data.index).astype(float)
        
        # Handle NaN values by forward filling and then backward filling
        close.replace([np.inf, -np.inf], np.nan, inplace=True)
        close2.replace([np.inf, -np.inf], np.nan, inplace=True)
        close.fillna(method='ffill', inplace=True)
        close.fillna(method='bfill', inplace=True)
        close2.fillna(method='ffill', inplace=True)
        close2.fillna(method='bfill', inplace=True)
        
        # Ensure both series have the same length and are aligned
        self.close = close
        self.close2 = close2
        
        self.spread = self.I(lambda: 1000 * self.close - self.close2)
        self.logspread = np.log(self.close) - np.log(self.close2)
        
        # Handle NaN values in log spread
        self.logspread.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.logspread.fillna(method='ffill', inplace=True)
        self.logspread.fillna(method='bfill', inplace=True)
        
        # Initialize and train the HMM
        self.hmm = GaussianHMM(n_components=2, covariance_type='full', n_iter=1000)
        self.hmm.fit(self.logspread.values.reshape(-1, 1))
        
        # Predict hidden states
        self.hidden_states = self.hmm.predict(self.logspread.values.reshape(-1, 1))
        self.hidden_states_series = pd.Series(self.hidden_states, index=self.logspread.index)
        
    def next(self):
        price = self.data["Close"][-1]
        
        # Get the current hidden state
        current_state = self.hidden_states_series[-1]
        
        # Debugging statements
        print(f"\nPrice: {price}, Hidden State: {current_state}")

        if current_state == 1 and len(self.trades) < self.max_position:
            print(f"\nSELL SIGNAL: Hidden State {current_state}")
            self.sell(sl=price + (self.bidask_spread + self.stop_loss), size=self.size)

        elif current_state == 0 and len(self.trades) < self.max_position:
            print(f"\nBUY SIGNAL: Hidden State {current_state}")
            self.buy(sl=price - (self.bidask_spread + self.stop_loss), size=self.size)

        elif self.position:
            if self.position.is_long and current_state == 1:
                print(f"\nCLOSE LONG: Hidden State {current_state}")
                self.position.close()
                
            elif self.position.is_short and current_state == 0:
                print(f"\nCLOSE SHORT: Hidden State {current_state}")
                self.position.close()

bt = Backtest(SP500, StatArb,
              cash=10000, commission=0.0,
              exclusive_orders=False, trade_on_close=True, margin=0.05)

output = bt.run()
print(output)
bt.plot()
