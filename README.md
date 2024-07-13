# DNN-StatArb
Deep learning applied to middle frequency statistical arbitrage trading strategies. For backtesting we will be using backtrader (https://github.com/mementum/backtrader), a simple backtesting engine written in python. 
## Requirements
```bash
pip install -r requirements.txt
```
## Base model

The base model is a statistical arbitrage pairs trading strategy (Strategies/pairs_statArb.py), that leverages the mean reverting principle. In OUP we simulate the mean reverting 
behaviour of this pair as an Ornstein-Uhlenbeck process. The strategy is based on calculating the spread return of the closing price of both assets, and from that deriving a z-score:
```python
def next(self):
        price = self.data["Close"][-1]
        spread = self.spread[-self.lookback:]
        spread_mean = spread.mean()
        spread_std = spread.std()
        zscore = (spread[-1] - spread_mean) / spread_std
```
We then capitalise on its mean reverting behaviour by exploiting temporary mispricings of the pair, which in this case is the SP500 and DJIA. We define the trading logic based on the aforementioned parameters:

```python
if zscore > self.zscore_threshold and len(self.trades) < self.max_position:
            self.sell(sl=price + ( self.bidask_spread + self.stop_loss), size=self.size)

        elif zscore < -self.zscore_threshold and len(self.trades) < self.max_position:
            self.buy(sl=price - (self.bidask_spread + self.stop_loss), size=self.size)

        elif abs(zscore) < self.zscore_threshold / 2:
            if self.position.is_long:
                self.position.close()
                
            elif self.position.is_short:
                self.position.close()

```
As we're not trading both instruments simultaneously, it goes without saying that this is not a beta-neutral optimal pairs trading strategy: despite it exhibiting similar characteristics in equity development.
## Results
![Figure_3](https://github.com/jensnesten/DNN-StatArb/assets/42718681/587590e9-78f8-45be-81ff-50e64cabe90d)

From here, we introduce a simple scaling function that enables the model to scale non-linearly while maintaining the desired margin impact relative to current equity:

```python
def scale_model(self):    
    if self.equity > self.init_cash * (1 + self.scaling_threshold):
        self.size += 1
        self.init_cash = self.equity

```

## Implementation of feed-forward neural network
Now that our basemodel is established, we then utilize the universal approximating abilities of feed-forward neural networks, to increase the quality of our trading signal. We first train the neural network in Models/FFNN_train.py, with the features initialised as the spread return, standard deviance of the spread return, mean of the spread return and the Z-score. The training script is currently set to use Apple Silicon GPU: change if you have a CUDA GPU available for faster training:

```python
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

```
Our neural network has a pyramidial architecture, with a maximum of 512 neurons in the first layers. This design is used to progressively reduce the dimensionality of the data as it moves through the network, allowing the network to focus on more abstract and high-level features. In total we have 10 layers: 5 fully connected, 4 batch normalization and 1 dropout to avoid overfitting. The model found in Models/model.pth and Models/scaler.plk, is trained on 4 years of 1 minute intraday data of US500 (SP500) and US30 (DJIA) with a loss of around 0.04.

## Results 
![Figure_5](https://github.com/jensnesten/DNN-StatArb/assets/42718681/86b962e7-4cf9-40e2-bf2d-19d3b881f0ed)

