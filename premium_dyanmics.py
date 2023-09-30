import QuantLib as ql
import matplotlib.pyplot as plt


def premium_at_different_spots(today=ql.Date(25, ql.August, 2023),
                               spot_price=109.5,
                               strike_price=109.5,
                               maturity=1 / 120,
                               volatility=0.07,
                               interest_rate=0.05,
                               call_put=ql.Option.Put):
    '''
    # Constants
    spot_price = 109.5  # Current bond future price
    strike_price = 109.5  # Option strike price
    maturity = 1 / 120  # Time to maturity in years
    volatility = 0.07  # Annualized volatility (as a decimal)
    interest_rate = 0.05  # Annualized risk-free interest rate (as a decimal)

    # QuantLib setup
    today =

    '''
    ql.Settings.instance().evaluationDate = today
    calendar = ql.NullCalendar()
    day_count = ql.Actual360()
    option_type = call_put

    # Create a European option
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.EuropeanExercise(today + ql.Period(int(maturity * 365), ql.Days))
    option = ql.EuropeanOption(payoff, exercise)

    # Set up the QuantLib Black-Scholes process
    u = ql.SimpleQuote(spot_price)
    r = ql.SimpleQuote(interest_rate)
    sigma = ql.SimpleQuote(volatility)
    risk_free_curve = ql.FlatForward(today, ql.QuoteHandle(r), day_count)
    volatility_curve = ql.BlackConstantVol(today, calendar, ql.QuoteHandle(sigma), day_count)
    process = ql.BlackScholesProcess(ql.QuoteHandle(u), ql.YieldTermStructureHandle(risk_free_curve),
                                     ql.BlackVolTermStructureHandle(volatility_curve))

    # Calculate option prices for a range of spot prices
    # spot_price_range = [107.75, 108, 108.25, 108.5, 108.75, 109, 109.25, 109.5, 109.75, 110, 110.25, 110.5, 110.75,
    #                    112]  # np.linspace(120, 150, 31)  # Adjust the range as needed
    min_spot_range = spot_price - 6 * (.25)
    spot_price_range = [min_spot_range + i * (.25) for i in range(13)]
    print(spot_price_range)
    option_prices = []

    for spot_prc in spot_price_range:
        u.setValue(spot_prc)
        engine = ql.AnalyticEuropeanEngine(process)
        option.setPricingEngine(engine)
        option_price = option.NPV()
        option_prices.append(option_price)

    # Plot the option price vs. spot price
    plt.figure(figsize=(10, 6))
    plt.plot(spot_price_range, option_prices, label='Option Price')
    plt.xlabel('Spot Price')
    plt.ylabel('Option Price')
    plt.title('Option Price vs. Spot Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def probabilities_at_different_strike():
    import QuantLib as ql

    # Constants
    spot_price = 109.5  # Current bond future price
    strike_price = 109.5  # Option strike price
    maturity = 1 / 120  # Time to maturity in years
    volatility = 0.07  # Annualized volatility (as a decimal)
    interest_rate = 0.05  # Annualized risk-free interest rate (as a decimal)

    # List of strike prices
    strike_prices = [107.75, 108, 108.25, 108.5, 108.75, 109, 109.25, 109.5, 109.75, 110, 110.25, 110.5, 110.75,
                     112]  # np.linspace(120, 150, 31)  # Adjust the range as needed

    # Initialize QuantLib
    today = ql.Date(28, ql.August, 2023)
    ql.Settings.instance().evaluationDate = today
    calendar = ql.NullCalendar()
    day_count = ql.Actual360()

    # Initialize QuantLib Black-Scholes process
    u = ql.SimpleQuote(spot_price)
    r = ql.SimpleQuote(interest_rate)
    sigma = ql.SimpleQuote(volatility)
    risk_free_curve = ql.FlatForward(today, ql.QuoteHandle(r), day_count)
    volatility_curve = ql.BlackConstantVol(today, calendar, ql.QuoteHandle(sigma), day_count)
    process = ql.BlackScholesProcess(ql.QuoteHandle(u), ql.YieldTermStructureHandle(risk_free_curve),
                                     ql.BlackVolTermStructureHandle(volatility_curve))

    # Initialize option pricing engine
    engine = ql.AnalyticEuropeanEngine(process)

    # Calculate probabilities for each strike
    probabilities = []

    for strike_price in strike_prices:
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
        exercise = ql.EuropeanExercise(today + ql.Period(int(maturity * 365), ql.Days))
        option = ql.EuropeanOption(payoff, exercise)
        option.setPricingEngine(engine)

        option_price = option.NPV()
        probability = option.delta() if option_price > 0 else 0.0
        probabilities.append(probability)

    # Print probabilities
    for strike, probability in zip(strike_prices, probabilities):
        print(f"Strike Price: {strike}, Probability: {probability:.4f}")


if __name__ == "__main__":
    #probabilities_at_different_strike()
    premium_at_different_spots()
