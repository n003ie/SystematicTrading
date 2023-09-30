import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yfinance as yf
from dateutil.relativedelta import *
from hmmlearn import hmm
from scipy.stats import pearsonr
from sklearn.mixture import GaussianMixture
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

symbols = ['ZN=F', 'DX-Y.NYB', 'CL=F', 'GC=F', 'NQ=F']  # , 'RX=F', 'ZN=F', '^TNX'

def reg_coef(x, y, label=None, color=None, cmap=None, **kwargs):
    ax = plt.gca()
    r, p = pearsonr(x, y)
    norm = plt.Normalize(-1, 1)
    cmap = cmap if not cmap is None else plt.cm.coolwarm
    ax.annotate(f'{r:.2f}', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', fontsize=30,
                bbox={'facecolor': cmap(norm(r)), 'alpha': 0.5, 'pad': 20})
    ax.set_axis_off()

def get_market_data(start_date, end_date, returnChange=False):
    closing_prices = yf.download(symbols, start=start_date, end=end_date)['Adj Close'].rename(
        columns={'ZN=F': 'ZN', 'DX-Y.NYB': 'DXY', 'CL=F': 'CL', 'GC=F': 'GC', 'NQ=F': 'NQ'})
    if returnChange:
        return closing_prices.diff().dropna()
    else:
        return closing_prices.dropna()


def multiple_regression(start_date, end_date):
    #5 day of prediction period

    ooSampleED = end_date
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')+relativedelta(weeks=-1)
    ooSampleSD  = end_date
    returns = get_market_data(start_date, end_date.strftime('%Y-%m-%d'))

    sns.set(font_scale=1)
    sns.set_style("white")
    for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        plt.rcParams[param] = 'cornflowerblue'
    plt.rcParams['font.family'] = 'Verdana'
    g = sns.PairGrid(returns, height=2)
    g.map_diag(plt.hist, color='turquoise')
    g.map_lower(plt.scatter, color='fuchsia')
    g.map_upper(reg_coef, cmap=plt.get_cmap('PiYG'))
    plt.setp(g.axes, xticks=[], yticks=[])

    g.fig.suptitle('ZN Correlations', fontsize=30)
    g.fig.tight_layout()
    plt.show()

    plot_acf(returns['ZN'].diff().dropna(), lags=90)
    plt.show()

    y = returns[['ZN']][1:]
    x = returns[['ZN', 'DXY', 'CL', 'GC', 'NQ']].shift(1).dropna().rename(
        columns={'ZN': 'Lag1ZN'})

    dta = y.merge(x, on='Date')
    print(dta)
    formula = 'ZN ~ DXY + CL + GC + NQ + DXY * CL * GC * NQ'  # add interaction terms later
    mod1 = smf.glm(formula=formula, data=dta, family=sm.families.Gaussian()).fit()
    print(mod1.summary())

    formula = 'ZN ~ Lag1ZN + DXY + CL + DXY * CL + DXY * GC + CL * GC + DXY * NQ'  # add interaction terms later
    mod1 = smf.glm(formula=formula, data=dta, family=sm.families.Gaussian()).fit()
    print(mod1.summary())
    ooSampleReturns = get_market_data(ooSampleSD, ooSampleED)
    print(ooSampleReturns)
    test_data = ooSampleReturns[['ZN', 'DXY', 'CL', 'GC', 'NQ']].shift(1).dropna().rename(
                        columns={'ZN': 'Lag1ZN'})
    print(test_data)
    print(mod1.predict(test_data))


def hmm_regimes(n_states, start_date, end_date):
    returns = get_market_data(start_date, end_date)
    n_samples = returns.shape[0]
    print(returns)
    returns_array = returns.to_numpy().reshape(-1, len(symbols))

    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full")
    model.fit(returns_array)
    # need to add the regime to the data frame to interpret
    hidden_states = model.predict(returns_array)
    transition_matrix = model.transmat_

    plt.figure(figsize=(12, 6))
    for i in range(n_states):
        plt.plot(np.arange(n_samples), returns, label=f"Regime {i + 1}", linestyle='-', linewidth=1)
    plt.title("Market Regimes Over Time")
    plt.xlabel("Time")
    plt.ylabel("Returns")
    plt.legend()
    plt.show()

    # Print transition matrix
    print("Transition Matrix:")
    print(transition_matrix)


def gmm_regimes(n_components, start_date, end_date):
    returns = get_market_data(start_date, end_date)
    n_samples = returns.shape[0]
    print(returns)
    # Fit the GMM model and get regime assignments
    returns_array = 1e4 * returns.to_numpy().reshape(-1, len(symbols))
    print(returns_array)
    model = GaussianMixture(n_components=n_components, covariance_type="full", random_state=6, verbose=2)
    model.fit(returns_array)
    regime_assignments = model.predict(returns_array)
    print('regime_assignments')
    print(regime_assignments)

    min_regime_length = min(np.bincount(regime_assignments))
    # Plot each regime separately
    time_axis = np.arange(regime_assignments.shape[0])
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, regime_assignments, label="Regimes", linestyle='-', linewidth=1)
    for i in range(n_components):
        regime_indices = np.where(regime_assignments == i)[0]
        regime_data = returns_array[regime_indices, -1]
        print(regime_data)
        time_axis = np.arange(regime_data.shape[0])
        # plt.plot(time_axis, regime_data, label=f"Regime {i + 1}", linestyle='-', linewidth=1)

    # Create a dictionary to store majority regimes for each market
    majority_regimes = {}

    # Iterate through each market (column) in your returns data
    for i, market_name in enumerate(symbols):
        market_data = returns_array[:, i]  # Extract data for the current market
        market_regime_counts = np.bincount(regime_assignments)  # Count occurrences of each regime
        majority_regime = np.argmax(market_regime_counts)  # Find the regime with the highest count
        majority_regimes[market_name] = majority_regime  # Assign the majority regime to the market

    # Print or access the majority regimes for each market
    for market_name, majority_regime in majority_regimes.items():
        print(f"Market: {market_name}, Majority Regime: {majority_regime}")

    plt.title("Market Regimes Over Time (GMM)")
    plt.xlabel("Time")
    plt.ylabel("Returns")
    plt.legend()
    plt.show()

    # Print transition matrix
    # need to add the regime to the data frame to interpret
    transition_matrix = np.zeros((n_components, n_components))
    for i in range(1, len(regime_assignments)):
        transition_matrix[regime_assignments[i - 1], regime_assignments[i]] += 1
    transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)

    print("Transition Matrix:")
    print(transition_matrix)


def markov_regression(n_regimes, start_date, end_date):
    returns = get_market_data(start_date, end_date)
    # n_samples = returns.shape[0]
    # returns_array = returns.to_numpy().reshape(-1, len(symbols))
    model = MarkovRegression(returns['ZN=F'].to_numpy().reshape(-1, 1), k_regimes=n_regimes, trend='ct',
                             exog=returns[['DX-Y.NYB', 'CL=F', 'GC=F', 'NQ=F']].to_numpy().reshape(-1,
                                                                                                   len(symbols) - 1),
                             switching_variance=True)
    res = model.fit()
    print(res.summary())


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2023-09-19'
    regimes = 4
    # hmm_regimes(regimes, start_date, end_date)
    # gmm_regimes(regimes, start_date, end_date)
    # markov_regression(regimes, start_date, end_date)
    multiple_regression(start_date, end_date)
