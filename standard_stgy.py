import matplotlib.pyplot as plt
import numpy as np
'''
Call Option Purchase:
Useful When: You are bullish on the bond future and expect its price to rise.
Best Identifying Factor: High confidence in a future price increase.

Put Option Purchase:
Useful When: You are bearish on the bond future and expect its price to fall.
Best Identifying Factor: Strong conviction that future prices will decline.

Straddle:
Useful When: You anticipate significant price volatility but are unsure about the direction (up or down).
Best Identifying Factor: Anticipated market volatility without a clear trend.

Strangle (Low):
Useful When: You expect moderate price movement but want to hedge against extreme price moves.
Best Identifying Factor: Moderate expected volatility with a bias towards a certain direction (up or down).

Strangle (High):
Useful When: Similar to the low strangle, but with a higher strike price. Useful when expecting higher price volatility.

Bull Call Spread:
Useful When: You are moderately bullish and want to reduce the cost of buying a call option.
Best Identifying Factor: Moderate bullish outlook.

Bear Call Spread:
Useful When: You are moderately bearish and want to reduce the cost of buying a call option.
Best Identifying Factor: Moderate bearish outlook.

Butterfly Spread:
Useful When: You anticipate very low volatility and expect the bond future to remain stable.
Best Identifying Factor: Low expected price movement.

Digital Call Option:
Useful When: You want a binary payout based on whether the future price exceeds a certain level.
Best Identifying Factor: Strong belief in a specific price level to be reached.

Digital Put Option:
Useful When: Similar to digital call, but you want a binary payout if the future price falls below a certain level.

'''

def check_strategy():
    # Bond future price
    future_price = 109.75

    # Strike prices for different strategies
    strike_price_call = 110.0
    strike_price_put = 109.0
    strike_price_straddle = 110.0
    strike_price_strangle_low = 108.0
    strike_price_strangle_high = 112.0
    strike_price_bull_call = 110.0
    strike_price_bull_put = 108.0
    strike_price_bear_call = 112.0
    strike_price_bear_put = 108.0
    strike_price_fly_low = 108.0
    strike_price_fly_mid = 110.0
    strike_price_fly_high = 112.0

    # Create an array of underlying prices (for the x-axis)
    underlying_prices = np.arange(100, 120, 0.5)

    # Calculate payoffs for each strategy
    call_payoff = np.maximum(underlying_prices - strike_price_call, 0)
    put_payoff = np.maximum(strike_price_put - underlying_prices, 0)
    straddle_payoff = call_payoff + put_payoff
    strangle_low_payoff = np.maximum(underlying_prices - strike_price_strangle_low, 0) - np.maximum(underlying_prices - strike_price_strangle_high, 0)
    bull_call_payoff = np.maximum(underlying_prices - strike_price_bull_call, 0)
    bull_put_payoff = np.maximum(strike_price_bull_put - underlying_prices, 0)
    bear_call_payoff = np.maximum(underlying_prices - strike_price_bear_call, 0)
    bear_put_payoff = np.maximum(strike_price_bear_put - underlying_prices, 0)
    fly_low_payoff = np.maximum(underlying_prices - strike_price_fly_low, 0) - np.maximum(underlying_prices - strike_price_fly_mid, 0) + np.maximum(underlying_prices - strike_price_fly_high, 0)
    digital_call_payoff = np.where(underlying_prices > strike_price_call, 1, 0)
    digital_put_payoff = np.where(underlying_prices < strike_price_put, 1, 0)

    # Calculate breakeven points for each strategy
    call_breakeven = strike_price_call
    put_breakeven = strike_price_put
    straddle_breakeven = strike_price_call + straddle_payoff[0]  # Straddle breakeven is at the initial option premium
    strangle_low_breakeven = strike_price_strangle_low - strangle_low_payoff[0]  # Strangle breakevens are at the initial option premiums
    strangle_high_breakeven = strike_price_strangle_high + strangle_low_payoff[0]
    bull_call_breakeven = strike_price_call + bull_call_payoff[0]
    bull_put_breakeven = strike_price_bull_put - bull_put_payoff[0]
    bear_call_breakeven = strike_price_bear_call + bear_call_payoff[0]
    bear_put_breakeven = strike_price_bear_put - bear_put_payoff[0]
    fly_low_breakeven = strike_price_fly_low - fly_low_payoff[0]
    fly_high_breakeven = strike_price_fly_high + fly_low_payoff[0]
    digital_call_breakeven = strike_price_call
    digital_put_breakeven = strike_price_put

    # Create a dictionary of strategies and their payoffs
    strategies = {
        "Call Option Purchase": call_payoff,
        "Put Option Purchase": put_payoff,
        "Straddle": straddle_payoff,
        "Strangle (Low)": strangle_low_payoff,
        "Strangle (High)": -strangle_low_payoff,  # Opposite sign
        "Bull Call Spread": bull_call_payoff - bull_put_payoff,
        "Bear Call Spread": bear_call_payoff - bear_put_payoff,
        "Butterfly Spread": fly_low_payoff,
        "Digital Call Option": digital_call_payoff,
        "Digital Put Option": -digital_put_payoff,  # Opposite sign
    }

    # Create a plot for each strategy
    plt.figure(figsize=(12, 8))
    for strategy, payoff in strategies.items():
        plt.plot(underlying_prices, payoff, label=strategy)

    # Plot breakeven points
    plt.axvline(call_breakeven, color='r', linestyle='--', label='Call Breakeven')
    plt.axvline(put_breakeven, color='g', linestyle='--', label='Put Breakeven')
    plt.axvline(straddle_breakeven, color='b', linestyle='--', label='Straddle Breakeven')
    plt.axvline(strangle_low_breakeven, color='m', linestyle='--', label='Strangle Low Breakeven')
    plt.axvline(strangle_high_breakeven, color='c', linestyle='--', label='Strangle High Breakeven')
    plt.axvline(bull_call_breakeven, color='y', linestyle='--', label='Bull Call Breakeven')
    plt.axvline(bull_put_breakeven, color='orange', linestyle='--', label='Bull Put Breakeven')
    plt.axvline(bear_call_breakeven, color='purple', linestyle='--', label='Bear Call Breakeven')
    plt.axvline(bear_put_breakeven, color='brown', linestyle='--', label='Bear Put Breakeven')
    plt.axvline(fly_low_breakeven, color='pink', linestyle='--', label='Butterfly Low Breakeven')
    plt.axvline(fly_high_breakeven, color='lime', linestyle='--', label='Butterfly High Breakeven')
    plt.axvline(digital_call_breakeven, color='gray', linestyle='--', label='Digital Call Breakeven')
    plt.axvline(digital_put_breakeven, color='olive', linestyle='--', label='Digital Put Breakeven')

    plt.xlabel('Underlying Price')
    plt.ylabel('Payoff')
    plt.title('Bond Future Option Payoff Diagrams')
    plt.legend()
    plt.grid(True)
    plt.show()
