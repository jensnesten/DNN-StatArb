from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import time
from backtesting.test import SMA

data = pd.read_csv('../Data/GOOG.csv', index_col=0, parse_dates=True)
SP500 = pd.DataFrame(data)

# calc run  time
start_time = time.time()


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()


bt = Backtest(SP500, SmaCross,
              cash=10000, commission=.002,
              exclusive_orders=True)

output = bt.run()
#bt.plot()
print(output)
print("--- %s seconds ---" % (time.time() - start_time))