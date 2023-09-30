import math

import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np


def calculate_greeks_black(calc_date, strike, spot, yield_curve, volatility, option_maturity_date):
    #calendar = ql.UnitedStates()
    #bussiness_convention = ql.ModifiedFollowing
    #settlement_days = 0
    #day_count = ql.ActualActual()
    #interest_rate = 0.00105

    #yield_curve = ql.FlatForward(calc_date,
    #                             interest_rate,
    #                            day_count,
    #                             ql.Compounded,
    #                             ql.Continuous)

    ql.Settings.instance().evaluationDate = calc_date
    flavor = ql.Option.Call

    discount = yield_curve.discount(option_maturity_date)
    strikepayoff = ql.PlainVanillaPayoff(flavor, strike)
    T = yield_curve.dayCounter().yearFraction(calc_date,
                                              option_maturity_date)
    stddev = volatility * math.sqrt(T)

    black = ql.BlackCalculator(strikepayoff,
                               spot,
                               stddev,
                               discount)
    delta, vega, theta, gamma, rho = black.delta(spot), black.vega(T), black.theta(spot, T), black.gamma(spot), black.rho(T)
    print("%-20s: %4.4f" % ("Option Price", black.value()))
    print("%-20s: %4.4f" % ("Delta", black.delta(spot)))
    print("%-20s: %4.4f" % ("Gamma", black.gamma(spot)))
    print("%-20s: %4.4f" % ("Theta", black.theta(spot, T)))
    print("%-20s: %4.4f" % ("Vega", black.vega(T)))
    print("%-20s: %4.4f" % ("Rho", black.rho(T)))
    breakeven_price = strike + delta
    breakeven_yield = yield_curve.zeroRate(option_maturity_date, ql.Actual360(), ql.Continuous, ql.Annual).rate()

    print("%-20s: %4.4f" % ("Breakeven Price", breakeven_price))
    print(strike, spot, delta, vega, theta, gamma, rho, breakeven_price)
    return delta, vega, theta, gamma, rho, breakeven_price


# Function to calculate option Greeks for a given strike
def calculate_greeks(today, strike, underlying_price, yield_curve, volatility, option_expiry):
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual360()
    calendar = ql.NullCalendar()

    u = ql.SimpleQuote(underlying_price)
    sigma = ql.SimpleQuote(volatility)

    risk_free_curve = ql.YieldTermStructureHandle(yield_curve)
    volatility_curve = ql.BlackConstantVol(today, calendar, ql.QuoteHandle(sigma), day_count)

    process = ql.BlackScholesProcess(ql.QuoteHandle(u), risk_free_curve,
                                     ql.BlackVolTermStructureHandle(volatility_curve))

    option_type = ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    exercise = ql.EuropeanExercise(option_expiry)
    option = ql.EuropeanOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    delta = option.delta()
    vega = option.vega()
    theta = option.theta()
    gamma = option.gamma()
    rho = option.rho()
    premium = option.NPV()
    breakeven_price = strike + premium#for a call
    breakeven_yield = yield_curve.zeroRate(option_expiry, day_count, ql.Continuous, ql.Annual).rate()
    print(strike, underlying_price, premium, delta, vega, theta, gamma, rho, breakeven_price)
    return delta, vega, theta, gamma, rho, breakeven_price


def run():
    # Input parameters
    underlying_price = 109.25
    option_expiry = ql.Date(30, ql.August, 2023)
    strike_range = [108, 108.25, 108.5, 108.75, 109, 109.25, 109.5, 109.75, 110, 110.25, 110.5, 110.75,
                    112]  # range(108, 111)
    volatility = 0.0611

    # Create a yield curve (Term Structure)
    today = ql.Date(28, ql.August, 2023)
    calendar = ql.NullCalendar()
    # Define the quoted rate
    quoted_rate = 0.05

    # Create a Quote object for the rate
    rate_quote = ql.QuoteHandle(ql.SimpleQuote(quoted_rate))

    # Define the rest of the parameters
    tenor = ql.Period(3, ql.Months)
    settlement_days = 0  # Adjust this according to your specific settlement convention
    calendar = ql.NullCalendar()  # Use the appropriate calendar
    business_day_convention = ql.ModifiedFollowing  # Use the appropriate convention
    compoundable = False  # Depending on your requirements
    day_counter = ql.Actual360()  # Use the appropriate day counter

    # Create the DepositRateHelper
    rate_helper = ql.DepositRateHelper(rate_quote, tenor, settlement_days, calendar, business_day_convention,
                                       compoundable, day_counter)
    # Add the rate helper to the list of rate helpers
    rate_helpers = [rate_helper]
    yield_curve = ql.PiecewiseFlatForward(today, rate_helpers, ql.Actual360())

    # Initialize lists to store Greek values
    delta_values = []
    vega_values = []
    theta_values = []
    gamma_values = []
    rho_values = []
    breakeven_values = []
    # Calculate Greeks for each strike
    for strike in strike_range:
        #delta, vega, theta, gamma, rho, breakeven_price = calculate_greeks(today, strike, underlying_price, yield_curve, volatility,
        #                                                option_expiry)

        delta, vega, theta, gamma, rho, breakeven_price = calculate_greeks(today, strike, underlying_price, yield_curve, volatility,
                                                          option_expiry)

        # Append values to lists
        delta_values.append(delta)
        vega_values.append(vega)
        theta_values.append(theta)
        gamma_values.append(gamma)
        rho_values.append(rho)
        breakeven_values.append(breakeven_price)

    # Create a dashboard
    plt.figure(figsize=(12, 8))

    # Delta
    plt.subplot(2, 3, 1)
    plt.plot(strike_range, delta_values)
    plt.title('Delta')
    plt.xlabel('Strike')
    plt.ylabel('Delta')

    # Vega
    plt.subplot(2, 3, 2)
    plt.plot(strike_range, vega_values)
    plt.title('Vega')
    plt.xlabel('Strike')
    plt.ylabel('Vega')

    # Theta
    plt.subplot(2, 3, 3)
    plt.plot(strike_range, theta_values)
    plt.title('Theta')
    plt.xlabel('Strike')
    plt.ylabel('Theta')

    # Gamma
    plt.subplot(2, 3, 4)
    plt.plot(strike_range, gamma_values)
    plt.title('Gamma')
    plt.xlabel('Strike')
    plt.ylabel('Gamma')

    # Rho
    plt.subplot(2, 3, 5)
    plt.plot(strike_range, rho_values)
    plt.title('Rho')
    plt.xlabel('Strike')
    plt.ylabel('Rho')

    plt.subplot(2, 3, 6)
    plt.plot(strike_range, breakeven_values)
    plt.title('Breakeven')
    plt.xlabel('Strike')
    plt.ylabel('Breakeven')

    plt.tight_layout()
    plt.show()

    # Suggest the best option strike based on Delta
    best_strike_index = np.argmax(np.abs(delta_values))
    best_strike = strike_range[best_strike_index]
    print(f"Suggested Best Strike: {best_strike} (Max Absolute Delta)")


if __name__ == "__main__":
    run()
