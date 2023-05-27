import backtrader as bt
import datetime
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input

# Define the Supertrend strategy
class SupertrendStrategy(bt.Strategy):
    params = (
        ('period', 10),
        ('multiplier', 3.0),
    )

    def __init__(self):
        self.supertrend = bt.indicators.SuperTrend(self.data, period=self.params.period, multiplier=self.params.multiplier)

    def next(self):
        if self.position.size == 0:
            if self.data.close[0] > self.supertrend[0]:
                self.buy()
        elif self.position.size > 0:
            if self.data.close[0] < self.supertrend[0]:
                self.sell()

# Define the live data feed
class LiveFeed(bt.feeds.GenericCSVData):
    params = (
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),
    )

# Create a cerebro instance
cerebro = bt.Cerebro()

# Add the Supertrend strategy to cerebro
cerebro.addstrategy(SupertrendStrategy)

# Create a data feed
data_path = 'path_to_live_data.csv'  # Replace with your live data source
data_feed = LiveFeed(dataname=data_path)

# Add the data feed to cerebro
cerebro.adddata(data_feed)

# Set initial capital
cerebro.broker.setcash(100000)

# Set position size
cerebro.addsizer(bt.sizers.FixedSize, stake=10)  # Replace stake with your desired position size

# Initialize P&L list and datetime list
pnl_list = []
datetime_list = []

# Define Dash application
app = dash.Dash(__name__)

# Dash layout
app.layout = html.Div([
    html.H1('Live Trading Dashboard'),
    dcc.Graph(id='market-data-graph'),
    dcc.Graph(id='pnl-graph'),
    dcc.Interval(id='update-interval', interval=1000, n_intervals=0)
])

# Callback function to update the market data graph
@app.callback(Output('market-data-graph', 'figure'),
              [Input('update-interval', 'n_intervals')])
def update_market_data_graph(n):
    # Retrieve the latest market data
    latest_data = cerebro.datas[0].lines.datetime.get(0, len(cerebro.datas[0])-1)
    latest_prices = cerebro.datas[0].lines.close.get(0, len(cerebro.datas[0])-1)

    # Create the market data graph
    figure = {
        'data': [
            {'x': latest_data, 'y': latest_prices, 'type': 'line', 'name': 'Market Data'}
        ],
        'layout': {
            'title': 'Live Market Data'
        }
    }
    return figure

# Callback function to update the P&L graph
@app.callback(Output('pnl-graph', 'figure'),
              [Input('update-interval', 'n_intervals')])
def update_pnl_graph(n):
    # Retrieve the latest P&L and datetime
    pnl = cerebro.broker.getvalue()
    dt = cerebro.datas[0].lines.datetime.get(0, len(cerebro.datas[0])-1)

    # Append the latest P&L and datetime to the lists
    pnl_list.append(pnl)
    datetime_list.append(dt)

    # Create the P&L graph
    figure = {
        'data': [
            {'x': datetime_list, 'y': pnl_list, 'type': 'line', 'name': 'P&L'}
        ],
        'layout': {
            'title': 'Live P&L'
        }
    }
    return figure

# Function to run the live strategy
def run_live_strategy():
    cerebro.run()

# Run the live strategy in a separate thread
import threading
live_strategy_thread = threading.Thread(target=run_live_strategy)
live_strategy_thread.start()

# Run the Dash application
if __name__ == '__main__':
    app.run_server(debug=True)
