import matplotlib.pyplot as plt
import pandas as pd
import pmdarima as pm
import yfinance as yf
from prophet import Prophet


def getMarketData(bond_future_symbol, start_date, end_date):
    bond_future_data = yf.download(bond_future_symbol, start=start_date, end=end_date)
    return bond_future_data


def runPredictSARMIA():
    # Define the bond future symbol and date range
    bond_future_symbol = 'ZN=F'  # Example: 30-Year U.S. Treasury Bond Futures
    start_date = '2022-01-01'
    end_date = '2023-9-9'

    # Fetch bond future price data from Yahoo Finance
    bond_future_data = getMarketData(bond_future_symbol, start_date, end_date)

    # Extract the closing prices
    closing_prices = bond_future_data['Close']

    # Plot the closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(closing_prices, label='Closing Prices')
    plt.title(f'{bond_future_symbol} Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Perform SARIMAX forecasting
    # configure SARIMAX parameters based on your requirements
    model = pm.auto_arima(closing_prices, seasonal=True, m=12, stepwise=True, suppress_warnings=True,
                          error_action="ignore")

    # Print the SARIMAX model summary
    print(model.summary())

    # Forecast future prices
    n_forecast_periods = 12  # Example: forecast for the next 12 periods
    forecast, conf_int = model.predict(n_periods=n_forecast_periods, return_conf_int=True)

    # Create a DataFrame for the forecast results
    forecast_dates = pd.date_range(start=closing_prices.index[-1], periods=n_forecast_periods + 1, inclusive='right')
    forecast_df = pd.DataFrame(
        {'Date': forecast_dates, 'Forecast': forecast, 'Lower_CI': conf_int[:, 0], 'Upper_CI': conf_int[:, 1]})
    forecast_df.set_index('Date', inplace=True)

    # Plot the forecasted prices
    plt.figure(figsize=(12, 6))
    plt.plot(closing_prices, label='Historical Prices')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red')
    plt.fill_between(forecast_df.index, forecast_df['Lower_CI'], forecast_df['Upper_CI'], color='pink', alpha=0.5,
                     label='95% CI')
    plt.title(f'{bond_future_symbol} Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def runPredictProphet():
    # Define the bond future symbol and date range
    bond_future_symbol = 'ZN=F'  # Example: 30-Year U.S. Treasury Bond Futures
    start_date = '2022-01-01'
    end_date = '2023-9-9'

    # Fetch bond future price data from Yahoo Finance
    bond_future_data = getMarketData(bond_future_symbol, start_date, end_date)

    # Extract the closing prices and prepare the data for Prophet
    df = bond_future_data[['Close']].reset_index()
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize and fit the Prophet model
    model = Prophet(seasonality_mode='multiplicative',
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=True
                    )
    model.fit(df)

    # Create a future dataframe for forecasting
    future = model.make_future_dataframe(periods=365)  # Forecast for the next 365 days

    # Generate forecasts
    forecast = model.predict(future)

    # Plot the forecasted prices
    fig = model.plot(forecast)
    plt.title(f'{bond_future_symbol} Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


if __name__ == "__main__":
    #runPredictSARMIA()
    runPredictProphet()
