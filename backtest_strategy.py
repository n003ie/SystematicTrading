import pandas as pd
import backtrader as bt
import dash
import dash_core_components as dcc
import dash_html_components as html

# Define a Dash application
app = dash.Dash(__name__)

# Define a Strategy class
class SupertrendStrategy(bt.Strategy):
    params = (
        ('period', 7),
        ('multiplier', 3.0),
        ('trailing_stop_percent', 0.05),
        ('martingale_factor_method', 'fixed'),
        ('martingale_factor', 2),
        ('martingale_factor_increment', 1),
        ('martingale_factor_max', 10),
    )

    def __init__(self):
        self.supertrend = bt.indicators.SuperTrend(
            period=self.params.period,
            multiplier=self.params.multiplier,
        )
        self.trailing_stop = self.params.trailing_stop_percent * self.supertrend.lines.supertrend

        self.martingale_factor_method = self.params.martingale_factor_method
        self.martingale_factor = self.params.martingale_factor
        self.martingale_factor_increment = self.params.martingale_factor_increment
        self.martingale_factor_max = self.params.martingale_factor_max

        self.current_order_size = 0

    def calculate_martingale_factor(self):
        if self.martingale_factor_method == 'fixed':
            return self.martingale_factor

        elif self.martingale_factor_method == 'incremental':
            return self.martingale_factor + (self.martingale_factor_increment * len(self.position))

        elif self.martingale_factor_method == 'exponential':
            return self.martingale_factor ** len(self.position)

    def next(self):
        if self.data.close[0] > self.supertrend.lines.supertrend[0]:
            if self.position.size == 0:
                self.current_order_size = 1
                self.buy(size=self.current_order_size)
        elif self.data.close[0] < self.supertrend.lines.supertrend[0]:
            if self.position.size > 0:
                self.sell(size=self.position.size)

        if self.position:
            self.sell(exectype=bt.Order.Stop, price=self.trailing_stop[0])

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.current_order_size = 1
            elif order.issell():
                self.martingale_factor = min(self.calculate_martingale_factor(), self.martingale_factor_max)
                self.current_order_size = self.calculate_martingale_factor()
                self.buy(size=self.current_order_size)

# Create a function to run the backtest
def run_backtest(data_path):
    # Load historical data into a Pandas DataFrame
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data = data.loc['2023-01-01':'2023-05-26']  # Filter data for the specified date range

    # Create a cerebro instance
    cerebro = bt.Cerebro()

    # Pass the strategy to cerebro
    cerebro.addstrategy(SupertrendStrategy)

    # Create a data feed
    data_feed = bt.feeds.PandasData(dataname=data)

    # Add the data feed to cerebro
    cerebro.adddata(data_feed)

    # Set initial capital
    cerebro.broker.setcash(100000)

    # Set position size
    cerebro.addsizer(bt.sizers.PercentSizer, percents=10)

    # Add a trailing stop analyzer
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    # Run the backtest
    results = cerebro.run()

    # Print the final portfolio value
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Retrieve and print trade statistics
    trade_stats = results[0].analyzers.trade_analyzer.get_analysis()
    print('Total trades:', trade_stats['total'])
    print('Total closed trades:', trade_stats['closed'])
    print('Winning trades:', trade_stats['won'])
    print('Losing trades:', trade_stats['lost'])
    
    # Retrieve cumulative P&L
    pnl = [x.broker.getvalue() for x in results]

    # Plot cumulative P&L using Dash
    app.layout = html.Div(children=[
        dcc.Graph(
            id='pnl-graph',
            figure={
                'data': [
                    {'x': data.index, 'y': pnl, 'type': 'line', 'name': 'Cumulative P&L'}
                ],
                'layout': {
                    'title': 'Cumulative P&L'
                }
            }
        )
    ])

    # Run the Dash application
    app.run_server(debug=True)

# Run the backtest with your data file
data_file = 'path_to_your_data_file.csv'
run_backtest(data_file)
