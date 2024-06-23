import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv('../Data/SP500_DJIA_2020_clean.csv', index_col=0, parse_dates=True)
SP500 = pd.DataFrame(data)

class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(4, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

"""class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x"""

    
model = DeepNN()
model.load_state_dict(torch.load('model.pth'))
model.eval() 
scaler = joblib.load('scaler.pkl')

class MLStatArb(Strategy):
    lookback = 13
    size = 5
    stop_loss = 0.0705
    max_position = 5
    position_num = 0
    bidask_spread = 0.4 / 2
    init_cash = 100000
    scaling_factor = 0.05
    zscore_threshold = 1.0
    scaling_threshold = 0.1

    def init(self):
        self.model = model  # Use the loaded model
        self.scaler = scaler  # Use the loaded scaler
        close = pd.Series(self.data.Close, index=self.data.index).astype(float)
        close2 = pd.Series(self.data.Close2, index=self.data.index).astype(float)
        spread = close / close.shift(1) - close2 / close2.shift(1)
        spread_mean = spread.rolling(window=self.lookback).mean()
        spread_std = spread.rolling(window=self.lookback).std()
        zscore = (spread - spread_mean) / spread_std

        self.spread = self.I(lambda: spread)
        self.spread_mean = self.I(lambda: spread_mean)
        self.spread_std = self.I(lambda: spread_std)
        self.zscore = self.I(lambda: zscore)

    def next(self):
        price = self.data["Close"][-1]
        spread = self.spread[-1]
        spread_mean = self.spread_mean[-1]
        spread_std = self.spread_std[-1]
        zscore = self.zscore[-1]
        self.scale_model()

        features = np.array([[spread, spread_mean, spread_std, zscore]])
        features = self.scaler.transform(features)  # Standardize the features
        features = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(features)
            _, signal = torch.max(output.data, 1)
            signal = signal.item()  # Convert to 0, 1, 2

        if signal == 2 and len(self.trades) < self.max_position:  # Sell signal
            self.sell(sl=price + (self.bidask_spread + self.stop_loss), size=self.size)

        elif signal == 0 and len(self.trades) < self.max_position:  # Buy signal
            self.buy(sl=price - (self.bidask_spread + self.stop_loss), size=self.size)

        elif signal == 1:  # Hold signal
            if self.position.is_long:
                self.position.close()
        
            elif self.position.is_short:
                self.position.close()

    def scale_model(self):
        if self.equity > self.init_cash * (1 + self.scaling_threshold):
            self.size += 1
            self.init_cash = self.equity
            print(f"Scaling up: New size: {self.size}")

bt = Backtest(data, MLStatArb,
              cash=100000, commission=0.0,
              exclusive_orders=False, trade_on_close=True, margin=0.05)

output = bt.run()
print(output)
#bt.plot()

equity = pd.DataFrame(output['_equity_curve'])
equity.iloc[:, :1].to_csv('eq.csv')

eq = pd.read_csv('eq.csv', parse_dates=[0], index_col=0)

eq['Equity'] = (eq['Equity'] - eq['Equity'].iloc[0]) / eq['Equity'].iloc[0] * 100
data['Close'] = (data['Close'] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100
plt.figure(figsize=(15, 8))

eq['Equity'].plot(label='Strategy')
data['Close'].plot(label='SP500')

plt.title('Equity Curve nn')
plt.xlabel('Time')
plt.ylabel('Equity Return (%)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
