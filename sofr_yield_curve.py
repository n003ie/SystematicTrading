import blpapi
import numpy as np

# Bloomberg session setup
sessionOptions = blpapi.SessionOptions()
sessionOptions.setServerHost("localhost")  # Replace with your Bloomberg Terminal API server host
sessionOptions.setServerPort(8194)  # Bloomberg Terminal API server port
session = blpapi.Session(sessionOptions)
session.start()

# Define the Bloomberg tickers for SOFR futures and swaps
sofr_futures_tickers = ["SR1 Comdty", "SR3 Comdty"]  # 1-month and 3-month futures two years out
sofr_swaps_tickers = ["US05YFRA Curncy", "US10YFRA Curncy", "US30YFRA Curncy"]  # 5Y, 10Y, and 30Y swaps

# Initialize dictionaries to store responses
sofr_futures_responses = {}
sofr_swaps_responses = {}

# Fetch SOFR futures data
for ticker in sofr_futures_tickers:
    sofr_futures_request = session.createRequest("ReferenceDataRequest")
    sofr_futures_request.getElement("securities").appendValue(ticker)
    sofr_futures_request.getElement("fields").appendValue("PX_LAST")
    session.sendRequest(sofr_futures_request)

# Fetch SOFR swap data
for ticker in sofr_swaps_tickers:
    sofr_swaps_request = session.createRequest("ReferenceDataRequest")
    sofr_swaps_request.getElement("securities").appendValue(ticker)
    sofr_swaps_request.getElement("fields").appendValue("PX_LAST")
    session.sendRequest(sofr_swaps_request)

# Process responses
while True:
    event = session.nextEvent()
    if event.eventType() == blpapi.Event.RESPONSE:
        for msg in event:
            if msg.hasElement("securityData"):
                securityData = msg.getElement("securityData")
                fieldData = securityData.getElement("fieldData")
                ticker = msg.correlationIds()[0].value()
                if ticker in sofr_futures_tickers:
                    sofr_futures_responses[ticker] = fieldData.getElementAsFloat("PX_LAST")
                elif ticker in sofr_swaps_tickers:
                    sofr_swaps_responses[ticker] = fieldData.getElementAsFloat("PX_LAST")
    if len(sofr_futures_responses) == len(sofr_futures_tickers) and len(sofr_swaps_responses) == len(sofr_swaps_tickers):
        break

# Close the Bloomberg session
session.stop()

# Calculate SOFR rates for futures
sofr_rates_futures = {}
for ticker, price in sofr_futures_responses.items():
    maturity = int(ticker[2])  # Extract the maturity (1 or 3 months)
    sofr_rate = 100 - price  # Calculate SOFR rate (assuming futures prices are quoted in percentage terms)
    sofr_rates_futures[f"{maturity}M"] = sofr_rate / 100  # Convert to decimal form

# Calculate SOFR rates for swaps
sofr_rates_swaps = {}
for ticker, price in sofr_swaps_responses.items():
    maturity = int(ticker[2:4])  # Extract the maturity (5, 10, or 30 years)
    sofr_rate = 100 - price  # Calculate SOFR rate (assuming swaps prices are quoted in percentage terms)
    sofr_rates_swaps[f"{maturity}Y"] = sofr_rate / 100  # Convert to decimal form

# Create a term structure yield curve (linear interpolation)
maturities = [1 / 12, 3 / 12, 5, 10, 30]  # Maturities in years (1 month, 3 months, 5 years, 10 years, 30 years)
sofr_rates = [sofr_rates_futures["1M"], sofr_rates_futures["3M"], sofr_rates_swaps["5Y"], sofr_rates_swaps["10Y"], sofr_rates_swaps["30Y"]]
sofr_curve = np.interp(maturities, [1 / 12, 3 / 12, 5, 10, 30], sofr_rates)

# Estimated 10-year TSY bond price using the SOFR yield curve
tsy_10yr_price = 100 / ((1 + sofr_curve[3]) ** maturities[3])

# Print SOFR rates and estimated 10-year TSY bond price
print("SOFR Rates (in decimal form):")
for maturity, rate in zip(maturities, sofr_curve):
    print(f"{maturity}-Year: {rate:.6f}")
print("Estimated 10-Year TSY Bond Price:", tsy_10yr_price)
