import QuantLib as ql

# Constants
spot_price = 109.5  # Current tsy bond price
futures_price = 97.03125  # Current bond future price
strike_price = 109.5  # Option strike price
maturity = 3/ 360  # Time to maturity in years
volatility = 0.07  # Annualized volatility (as a decimal)
interest_rate = 0.05  # Annualized risk-free interest rate (as a decimal)

# QuantLib setup
today = ql.Date(28, ql.August, 2023)
ql.Settings.instance().evaluationDate = today
calendar = ql.NullCalendar()
day_count = ql.Actual360()

# Initialize QuantLib Black-Scholes process for option
u_option = ql.SimpleQuote(spot_price)
r_option = ql.SimpleQuote(interest_rate)
sigma_option = ql.SimpleQuote(volatility)
risk_free_curve_option = ql.FlatForward(today, ql.QuoteHandle(r_option), day_count)
volatility_curve_option = ql.BlackConstantVol(today, calendar, ql.QuoteHandle(sigma_option), day_count)
process_option = ql.BlackScholesProcess(ql.QuoteHandle(u_option), ql.YieldTermStructureHandle(risk_free_curve_option), ql.BlackVolTermStructureHandle(volatility_curve_option))

# Create a European call option
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
exercise = ql.EuropeanExercise(today + ql.Period(int(maturity * 365), ql.Days))
option = ql.EuropeanOption(payoff, exercise)

# Initialize option pricing engine
engine_option = ql.AnalyticEuropeanEngine(process_option)
option.setPricingEngine(engine_option)

# Calculate option premium/payoff
option_premium = option.NPV()

# Calculate the basis (futures price - spot price)
basis = futures_price - spot_price

# Identify potential anomalies
if option_premium - basis > 0.01:  # Example threshold for anomaly detection
    print("Potential anomaly detected!")

    # Strategy: Basis trading
    if basis > 0:  # If basis is positive, short futures and long bonds
        print("Execute strategy: Short futures and long bonds")
    else:  # If basis is negative, long futures and short bonds
        print("Execute strategy: Long futures and short bonds")
else:
    print("No anomaly detected.")
