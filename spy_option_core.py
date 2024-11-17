import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from pandas import DataFrame
from scipy.stats import norm, skew
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OptionMetrics:
    """Dataclass to store option pricing and risk metrics"""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class OptionsPricingEngine:
    """Advanced options pricing engine using multiple models"""

    def __init__(self, risk_free_rate: float = 0.05, div_yield: float = 0.0):
        self.rf = risk_free_rate
        self.div = div_yield

    def black_scholes(self, S: float, K: float, T: float, sigma: float,
                      option_type: str) -> OptionMetrics:
        """
        Calculate option metrics using Black-Scholes model with dividend yield
        """
        if T <= 0:
            intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
            return OptionMetrics(intrinsic, 0, 0, 0, 0, 0)

        d1 = (np.log(S / K) + (self.rf - self.div + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * np.exp(-self.div * T) * norm.cdf(d1) - K * np.exp(-self.rf * T) * norm.cdf(d2)
            delta = np.exp(-self.div * T) * norm.cdf(d1)
        else:
            price = K * np.exp(-self.rf * T) * norm.cdf(-d2) - S * np.exp(-self.div * T) * norm.cdf(-d1)
            delta = -np.exp(-self.div * T) * norm.cdf(-d1)

        gamma = np.exp(-self.div * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * sigma * np.exp(-self.div * T) * norm.pdf(d1)) / (2 * np.sqrt(T)) - \
                self.rf * K * np.exp(-self.rf * T) * norm.cdf(d2 if option_type == 'call' else -d2)
        vega = S * np.exp(-self.div * T) * np.sqrt(T) * norm.pdf(d1)
        rho = K * T * np.exp(-self.rf * T) * norm.cdf(d2 if option_type == 'call' else -d2)

        return OptionMetrics(price, delta, gamma, theta, vega, rho)

    def heston_model(self, S: float, K: float, T: float, v0: float, theta: float,
                     kappa: float, sigma: float, rho: float, option_type: str) -> float:
        """
        Implement Heston stochastic volatility model using characteristic function

        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to expiration in years
        v0 : float
            Initial variance
        theta : float
            Long-term variance
        kappa : float
            Mean reversion speed
        sigma : float
            Volatility of variance
        rho : float
            Correlation between stock and variance processes
        option_type : str
            'call' or 'put'

        Returns:
        --------
        float
            Option price
        """

        def char_function(phi: complex, S: float, K: float, T: float, v0: float,
                          theta: float, kappa: float, sigma: float, rho: float) -> complex:
            # Define auxiliary parameters
            a = kappa * theta
            u = -0.5
            # Market price of volatility risk lambda is typically set to 0
            # in practice as it's difficult to estimate
            lambda_ = 0
            b = kappa + lambda_

            # Calculate characteristic function components
            d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 - sigma ** 2 * (2 * u * phi * 1j - phi ** 2))
            g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)

            # Calculate exponents
            C = (self.rf - self.div) * phi * 1j * T + \
                a / sigma ** 2 * ((b - rho * sigma * phi * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
            D = (b - rho * sigma * phi * 1j + d) / sigma ** 2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

            # Return characteristic function
            return np.exp(C + D * v0 + 1j * phi * np.log(S))

        # Integration parameters
        N = 100
        phi = np.linspace(0.0001, 100, N)

        # Calculate option price using numerical integration
        integrand = np.zeros(N, dtype=complex)
        for i in range(N):
            integrand[i] = np.exp(-1j * phi[i] * np.log(K)) * char_function(phi[i], S, K, T, v0, theta, kappa, sigma,
                                                                            rho) / (1j * phi[i])

        price = S * np.exp(-self.div * T) - K * np.exp(-self.rf * T) / np.pi * \
                np.trapz(np.real(integrand), phi)

        return max(0, price if option_type == 'call' else price + K * np.exp(-self.rf * T) - S * np.exp(-self.div * T))

class LSTMOptionsPricing(nn.Module):
    """LSTM-based deep learning model for options pricing"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class AdvancedOptionsScreener:
    """Advanced options screening and analysis system"""

    def __init__(self, ticker: str = "SPY", lookback_days: int = 252):
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.spy = yf.Ticker(ticker)
        self.pricing_engine = OptionsPricingEngine()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._initialize_model()

    def _initialize_model(self) -> LSTMOptionsPricing:
        """Initialize and configure the LSTM model"""
        model = LSTMOptionsPricing(input_size=5, hidden_size=64).to(self.device)
        return model

    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch and process historical price data with advanced metrics"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        hist_data = self.spy.history(start=start_date, end=end_date)
        hist_data['returns'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))

        # Calculate various volatility measures
        hist_data['realized_vol'] = hist_data['returns'].rolling(30).std() * np.sqrt(252)
        hist_data['parkinson_vol'] = np.sqrt(252 / (4 * np.log(2))) * \
                                     (np.log(hist_data['High'] / hist_data['Low'])).rolling(30).std()

        # Add technical indicators
        hist_data['rsi'] = self._calculate_rsi(hist_data['Close'])
        hist_data['bbands'] = self._calculate_bollinger_bands(hist_data['Close'])

        return hist_data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Bands percentage"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        return (prices - sma) / (2 * std)

    def calculate_implied_volatility(self, price: float, S: float, K: float,
                                     T: float, option_type: str) -> float:
        """Calculate implied volatility using Newton-Raphson method with safeguards"""

        def objective(sigma):
            return self.pricing_engine.black_scholes(S, K, T, sigma, option_type).price - price

        sigma = 0.3  # Initial guess
        max_iter = 100
        tolerance = 1e-5

        for _ in range(max_iter):
            diff = objective(sigma)
            if abs(diff) < tolerance:
                return sigma

            vega = self.pricing_engine.black_scholes(S, K, T, sigma, option_type).vega
            if abs(vega) < 1e-10:
                return sigma

            new_sigma = sigma - diff / vega

            # Add bounds and dampening
            if new_sigma <= 0.001:
                sigma = 0.001
            elif new_sigma > 5:
                sigma = 5
            else:
                sigma = 0.4 * sigma + 0.6 * new_sigma

        return sigma

    def calculate_risk_neutral_density(self, options_df: pd.DataFrame,
                                       current_price: float) -> pd.DataFrame:
        """Calculate risk-neutral probability density using option prices"""
        calls = options_df[options_df['option_type'] == 'call'].sort_values('strike')

        # Use cubic spline interpolation for smooth second derivatives
        spline = interp1d(calls['strike'], calls['lastPrice'], kind='cubic', fill_value='extrapolate')

        # Generate fine grid for better numerical accuracy
        strikes = np.linspace(calls['strike'].min(), calls['strike'].max(), 1000)
        prices = spline(strikes)

        # Calculate second derivative
        d2C_dK2 = np.gradient(np.gradient(prices, strikes), strikes)

        T = calls['days_to_expiry'].iloc[0] / 365
        rnd = np.exp(self.pricing_engine.rf * T) * d2C_dK2

        # Normalize density
        rnd = rnd / np.trapz(rnd, strikes)

        return pd.DataFrame({
            'strike': strikes,
            'density': rnd
        })

    def calculate_all_greeks(self, row: pd.Series, current_price: float) -> pd.Series:
        """
        Calculate all option Greeks with robust error handling

        Parameters:
        -----------
        row : pd.Series
            Single option record containing strike, expiry, etc.
        current_price : float
            Current underlying price

        Returns:
        --------
        pd.Series
            Series containing all calculated Greeks
        """
        try:
            # Basic parameters
            S = current_price
            K = row['strike']
            T = row['days_to_expiry'] / 365
            sigma = row['impliedVolatility']
            r = self.pricing_engine.rf
            q = self.pricing_engine.div  # Dividend yield

            if T <= 0 or sigma <= 0:
                return pd.Series({
                    'delta': 0, 'gamma': 0, 'theta': 0,
                    'vega': 0, 'rho': 0, 'charm': 0,
                    'vanna': 0, 'vomma': 0
                })

            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            # Standard normal functions
            n_d1 = norm.pdf(d1)
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)

            if row['option_type'] == 'put':
                N_d1 = N_d1 - 1
                N_d2 = N_d2 - 1

            # First-order Greeks
            delta = np.exp(-q * T) * N_d1
            theta = (-S * sigma * np.exp(-q * T) * n_d1 / (2 * np.sqrt(T)) -
                     r * K * np.exp(-r * T) * N_d2 + q * S * np.exp(-q * T) * N_d1)

            # Second-order Greeks
            gamma = np.exp(-q * T) * n_d1 / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-q * T) * np.sqrt(T) * n_d1 / 100  # Divided by 100 for better scaling

            # Additional Greeks
            rho = K * T * np.exp(-r * T) * N_d2 / 100  # Divided by 100 for better scaling

            # Cross Greeks
            charm = -np.exp(-q * T) * (
                        n_d1 * (2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)))
            vanna = -np.exp(-q * T) * n_d1 * d2 / sigma
            vomma = vega * d1 * d2 / sigma

            return pd.Series({
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho,
                'charm': charm,
                'vanna': vanna,
                'vomma': vomma
            })

        except Exception as e:
            print(f"Warning: Error calculating Greeks: {str(e)}")
            return pd.Series({
                'delta': 0, 'gamma': 0, 'theta': 0,
                'vega': 0, 'rho': 0, 'charm': 0,
                'vanna': 0, 'vomma': 0
            })

    def analyze_options_chain(self, min_volume: int = 100, min_open_interest: int = 500) -> Dict:
        """Comprehensive options chain analysis with enhanced Greeks"""
        # Get available expiration dates
        expiration_dates = self.spy.options
        current_price = self.spy.history(period='1d')['Close'].iloc[-1]

        # Combine calls and puts for all expiration dates
        all_options = []
        for expiry in expiration_dates:
            try:
                chain = self.spy.option_chain(expiry)
                calls = chain.calls.assign(option_type='call', expiry=expiry)
                puts = chain.puts.assign(option_type='put', expiry=expiry)
                all_options.extend([calls, puts])
            except Exception as e:
                print(f"Warning: Error fetching data for expiry {expiry}: {str(e)}")
                continue

        if not all_options:
            raise ValueError("No valid options data found")

        options_df = pd.concat(all_options, ignore_index=True)

        # Calculate days to expiry
        options_df['days_to_expiry'] = pd.to_datetime(options_df['expiry']).dt.date.apply(
            lambda x: (x - datetime.now().date()).days
        )

        # Calculate implied volatility
        options_df['impliedVolatility'] = options_df.apply(
            lambda row: self.calculate_implied_volatility(
                row['lastPrice'], current_price, row['strike'],
                row['days_to_expiry'] / 365, row['option_type']
            ),
            axis=1
        )

        # Calculate all Greeks
        greeks_df = options_df.apply(
            lambda row: self.calculate_all_greeks(row, current_price),
            axis=1
        )

        # Add Greeks to main dataframe
        options_df = pd.concat([options_df, greeks_df], axis=1)

        # Calculate Greeks sensitivity
        options_df['delta_sensitivity'] = options_df['gamma'].abs()
        options_df['vega_sensitivity'] = options_df['vanna'].abs()
        options_df['theta_sensitivity'] = options_df['charm'].abs()

        # Filter by volume and open interest
        options_df = options_df[
            (options_df['volume'] >= min_volume) &
            (options_df['openInterest'] >= min_open_interest)
            ]

        # Calculate additional metrics
        options_df = self.calculate_additional_metrics(options_df, current_price)

        # Add risk metrics based on Greeks
        options_df['gamma_risk_score'] = self.calculate_gamma_risk(options_df)
        options_df['vega_risk_score'] = self.calculate_vega_risk(options_df)
        options_df['theta_risk_score'] = self.calculate_theta_risk(options_df)

        # Generate analysis summary
        summary = self.generate_summary(options_df)

        return {
            'options_data': options_df,
            'current_price': current_price,
            'expiration_dates': expiration_dates,
            'summary': summary
        }

    def calculate_gamma_risk(self, options_df: pd.DataFrame) -> pd.Series:
        """Calculate gamma risk score based on multiple factors"""
        print(options_df['gamma'])
        gamma_exposure = options_df['gamma'] * options_df['openInterest'] * 100  # Contract multiplier

        # Normalize gamma exposure
        gamma_score = (gamma_exposure - gamma_exposure.mean()) / gamma_exposure.std()

        # Adjust for time decay
        time_factor = np.exp(-options_df['days_to_expiry'] / 365)
        print(gamma_score)
        return gamma_score * time_factor

    def calculate_vega_risk(self, options_df: pd.DataFrame) -> pd.Series:
        """Calculate vega risk score based on multiple factors"""
        vega_exposure = options_df['vega'] * options_df['openInterest'] * 100

        # Normalize vega exposure
        vega_score = (vega_exposure - vega_exposure.mean()) / vega_exposure.std()

        # Adjust for implied volatility level
        vol_factor = options_df['impliedVolatility'] / options_df['impliedVolatility'].mean()

        return vega_score * vol_factor

    def calculate_theta_risk(self, options_df: pd.DataFrame) -> pd.Series:
        """Calculate theta risk score based on multiple factors"""
        theta_exposure = options_df['theta'] * options_df['openInterest'] * 100

        # Normalize theta exposure
        theta_score = (theta_exposure - theta_exposure.mean()) / theta_exposure.std()

        # Adjust for time to expiration
        time_factor = 1 / np.sqrt(options_df['days_to_expiry'])

        return theta_score * time_factor

    def calculate_additional_metrics(self, options_df: pd.DataFrame,
                                     current_price: float) -> pd.DataFrame:
        """Calculate additional analysis metrics with enhanced risk measures"""
        try:
            # Original metrics
            options_df['volume_oi_ratio'] = options_df['volume'] / options_df['openInterest'].replace(0, np.nan)
            options_df['dollar_volume'] = options_df['volume'] * options_df['lastPrice']

            # Get historical volatility
            try:
                hist_data = self.fetch_historical_data()
                hist_vol = hist_data['realized_vol'].iloc[-1]
            except Exception:
                hist_vol = options_df['impliedVolatility'].mean()

            # Volatility metrics
            options_df['iv_percentile'] = options_df.groupby('days_to_expiry')['impliedVolatility'].transform(
                lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')
            )
            options_df['iv_hv_ratio'] = options_df['impliedVolatility'] / hist_vol

            # Risk metrics with error handling
            options_df['probability_itm'] = options_df.apply(
                lambda row: self.calculate_probability_itm(row, current_price)
                if not pd.isna(row['impliedVolatility']) else np.nan,
                axis=1
            )

            # Enhanced risk metrics using Greeks
            options_df['total_risk_score'] = (
                    options_df['gamma_risk_score'] * 0.4 +
                    options_df['vega_risk_score'] * 0.3 +
                    options_df['theta_risk_score'] * 0.3
            )

            # Strategy scores
            options_df['directional_score'] = (
                    options_df['delta'].abs() * 0.5 +
                    options_df['gamma'].abs() * 0.3 +
                    options_df['charm'].abs() * 0.2
            )

            options_df['volatility_score'] = (
                    options_df['vega'].abs() * 0.4 +
                    options_df['vomma'].abs() * 0.3 +
                    options_df['vanna'].abs() * 0.3
            )

            options_df['income_score'] = (
                    options_df['theta'].abs() * 0.6 +
                    options_df['charm'].abs() * 0.4
            )
            # Value metrics
            options_df['value_score'] = (
                    (10 - options_df['iv_percentile']) * 0.3 +
                    options_df['volume_oi_ratio'].fillna(0) * 0.2 +
                    (options_df['probability_itm'].fillna(0.5) * 0.5)
            )

            # Clean up any infinite values
            options_df = options_df.replace([np.inf, -np.inf], np.nan)

            return options_df

        except Exception as e:
            print(f"Warning: Error in calculate_additional_metrics: {str(e)}")
            return options_df

    def calculate_probability_itm(self, row: pd.Series, current_price: float) -> float:
        """Calculate probability of option expiring in-the-money with error handling"""
        try:
            if pd.isna(row['impliedVolatility']) or row['days_to_expiry'] <= 0:
                return 0.5

            T = row['days_to_expiry'] / 365
            sigma = row['impliedVolatility']

            d1 = (np.log(current_price / row['strike']) +
                  (self.pricing_engine.rf - self.pricing_engine.div + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

            prob = norm.cdf(d1 if row['option_type'] == 'call' else -d1)

            # Ensure probability is between 0 and 1
            return max(0, min(1, prob))

        except Exception as e:
            print(f"Warning: Error calculating ITM probability: {str(e)}")
            return 0.5  # Return 50% probability as fallback

    def generate_summary(self, options_df: pd.DataFrame) -> Dict:
        """Generate enhanced summary with opportunity analysis"""
        summary = {
            'total_volume': options_df['volume'].sum(),
            'total_open_interest': options_df['openInterest'].sum(),
            'put_call_ratio': len(options_df[options_df['option_type'] == 'put']) /
                              max(1, len(options_df[options_df['option_type'] == 'call'])),
            'volume_put_call_ratio': options_df[options_df['option_type'] == 'put']['volume'].sum() /
                                     max(1, options_df[options_df['option_type'] == 'call']['volume'].sum()),
            'avg_implied_volatility': options_df['impliedVolatility'].mean(),
            'iv_skew': self.calculate_volatility_skew(options_df),
            'top_opportunities': self.get_top_opportunities(options_df),
            'opportunity_metrics': self.get_opportunity_metrics(options_df)
        }

        # Add volatility surface metrics if available
        vol_surface_metrics = self.analyze_volatility_surface(options_df)
        if vol_surface_metrics:
            summary.update(vol_surface_metrics)

        return summary

    def find_atm_strike(self, options_df: pd.DataFrame) -> Optional[float]:
        """
        Find the at-the-money strike price with robust error handling

        Parameters:
        -----------
        options_df : pd.DataFrame
            DataFrame containing options data with 'strike' column

        Returns:
        --------
        Optional[float]
            ATM strike price, or None if cannot be determined
        """
        try:
            if options_df.empty:
                return None

            current_price = self.spy.history(period='1d')['Close'].iloc[-1]

            # Calculate absolute difference from current price
            strike_diff = (options_df['strike'] - current_price).abs()

            # Find strike with minimum difference
            atm_strike = options_df.loc[strike_diff.idxmin(), 'strike']

            return float(atm_strike)

        except Exception as e:
            print(f"Warning: Error finding ATM strike: {str(e)}")
            return None

    def calculate_volatility_skew(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and analyze volatility skew across strikes and expirations
        with robust error handling
        """
        skew_data = []

        try:
            for expiry in options_df['expiry'].unique():
                expiry_options = options_df[options_df['expiry'] == expiry]

                # Find ATM strike
                atm_strike = self.find_atm_strike(expiry_options)
                if atm_strike is None:
                    continue

                # Calculate ATM volatility
                atm_options = expiry_options[expiry_options['strike'] == atm_strike]
                if atm_options.empty:
                    continue

                atm_vol = atm_options['impliedVolatility'].mean()

                # Calculate 25-delta put and call volatilities
                put_25d = self.interpolate_by_delta(
                    expiry_options[expiry_options['option_type'] == 'put'],
                    -0.25
                )
                call_25d = self.interpolate_by_delta(
                    expiry_options[expiry_options['option_type'] == 'call'],
                    0.25
                )

                if put_25d is not None and call_25d is not None:
                    skew_data.append({
                        'expiry': expiry,
                        'days_to_expiry': expiry_options['days_to_expiry'].iloc[0],
                        'atm_vol': atm_vol,
                        'put_25d_vol': put_25d['impliedVolatility'],
                        'call_25d_vol': call_25d['impliedVolatility'],
                        'skew_spread': put_25d['impliedVolatility'] - call_25d['impliedVolatility'],
                        'butterfly_spread': (put_25d['impliedVolatility'] + call_25d['impliedVolatility']) / 2 - atm_vol
                    })

            return pd.DataFrame(skew_data)

        except Exception as e:
            print(f"Warning: Error calculating volatility skew: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def interpolate_by_delta(self, options: pd.DataFrame, target_delta: float) -> Optional[pd.Series]:
        """
        Interpolate option metrics for a specific delta target with improved error handling
        """
        try:
            if len(options) < 2:
                return None

            # Ensure data is clean
            options = options.dropna(subset=['delta', 'impliedVolatility', 'strike'])
            if len(options) < 2:
                return None

            # Sort by delta for interpolation
            options = options.sort_values('delta')

            # Check if target delta is within range
            if not (options['delta'].min() <= target_delta <= options['delta'].max()):
                return None

            # Interpolate implied volatility at target delta
            try:
                vol_interp = interp1d(
                    options['delta'].values,
                    options['impliedVolatility'].values,
                    kind='linear',  # Changed to linear for more stability
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                strike_interp = interp1d(
                    options['delta'].values,
                    options['strike'].values,
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
            except ValueError:
                return None

            interpolated_vol = float(vol_interp(target_delta))
            interpolated_strike = float(strike_interp(target_delta))

            # Validate interpolated values
            if not (np.isfinite(interpolated_vol) and np.isfinite(interpolated_strike)):
                return None

            return pd.Series({
                'strike': interpolated_strike,
                'impliedVolatility': interpolated_vol,
                'delta': target_delta
            })

        except Exception as e:
            print(f"Warning: Error in delta interpolation: {str(e)}")
            return None

    def analyze_volatility_surface(self, options_df: pd.DataFrame) -> Dict:
        """
        Analyze the volatility surface structure with improved error handling
        """
        try:
            if options_df.empty:
                return {
                    'vol_surface_pc1': None,
                    'vol_surface_pc2': None,
                    'term_structure_slope': None,
                    'term_structure_curvature': None
                }

            # Create volatility surface matrix
            surface = options_df.pivot_table(
                values='impliedVolatility',
                index='strike',
                columns='days_to_expiry',
                aggfunc='mean'  # Added explicit aggregation function
            )

            # Handle missing values
            surface = surface.fillna(surface.mean().mean())

            if surface.empty or surface.isna().all().all():
                raise ValueError("No valid data for volatility surface analysis")

            # Calculate principal components using SVD
            U, S, Vt = np.linalg.svd(surface)

            # Calculate explained variance ratios
            explained_var = S ** 2 / np.sum(S ** 2)

            # Fit term structure model
            term_structure = options_df.groupby('days_to_expiry')['impliedVolatility'].mean()
            if len(term_structure) < 3:
                return {
                    'vol_surface_pc1': explained_var[0] if len(explained_var) > 0 else None,
                    'vol_surface_pc2': explained_var[1] if len(explained_var) > 1 else None,
                    'term_structure_slope': None,
                    'term_structure_curvature': None
                }

            term_structure_fit = np.polyfit(
                term_structure.index.astype(float),
                term_structure.values,
                2
            )

            return {
                'vol_surface_pc1': explained_var[0] if len(explained_var) > 0 else None,
                'vol_surface_pc2': explained_var[1] if len(explained_var) > 1 else None,
                'term_structure_slope': term_structure_fit[1],
                'term_structure_curvature': term_structure_fit[0]
            }

        except Exception as e:
            print(f"Warning: Error in volatility surface analysis: {str(e)}")
            return {
                'vol_surface_pc1': None,
                'vol_surface_pc2': None,
                'term_structure_slope': None,
                'term_structure_curvature': None
            }

    def get_top_opportunities(self, options_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Identify top trading opportunities based on multiple criteria with robust error handling

        Parameters:
        -----------
        options_df : pd.DataFrame
            DataFrame containing options data
        n : int
            Number of top opportunities to return for each category

        Returns:
        --------
        pd.DataFrame
            DataFrame containing top opportunities with scores
        """
        try:
            # Define required and optional metrics
            required_metrics = ['impliedVolatility', 'strike', 'option_type', 'lastPrice']


            # Verify required metrics
            missing_required = [col for col in required_metrics if col not in options_df.columns]
            if missing_required:
                raise ValueError(f"Missing required columns: {missing_required}")

            # Initialize scoring DataFrame
            scoring_df = options_df.copy()
            scoring_df['volume_oi_ratio'] = options_df['volume'] / options_df['openInterest']
            scoring_df['dollar_volume'] = options_df['volume'] * options_df['lastPrice']

            # Volatility metrics
            hist_vol = self.fetch_historical_data()['realized_vol'].iloc[-1]
            options_df['iv_percentile'] = options_df.groupby('days_to_expiry')['impliedVolatility'].transform(
                lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')
            )
            options_df['iv_hv_ratio'] = options_df['impliedVolatility'] / hist_vol

            scoring_df['value_score'] = (
                    (10 - options_df['iv_percentile']) * 0.3 +
                    options_df['volume_oi_ratio'] * 0.2 +
                    (options_df['probability_itm'] * 0.5)
            )
            optional_metrics = [
                ('probability_itm', 0.25),
                ('value_score', 0.25),
                ('volume_oi_ratio', 0.2)
            ]
            # Calculate z-scores for available metrics
            metrics_to_score = []

            # Add implied volatility score (always available)
            scoring_df['iv_zscore'] = (
                                        scoring_df['impliedVolatility'] - scoring_df['impliedVolatility'].mean()
                                      ) / scoring_df['impliedVolatility'].std()
            metrics_to_score.append(('iv_zscore', 0.3))

            # Calculate volume/OI ratio if available
            if 'volume' in scoring_df.columns and 'openInterest' in scoring_df.columns:
                scoring_df['volume_oi_ratio'] = scoring_df['volume'] / scoring_df['openInterest'].replace(0, np.nan)
                scoring_df['volume_oi_zscore'] = (
                                                scoring_df['volume_oi_ratio'] - scoring_df['volume_oi_ratio'].mean()
                                                 ) / scoring_df['volume_oi_ratio'].std()
                metrics_to_score.append(('volume_oi_zscore', 0.2))

            # Add probability ITM or its substitute
            for metric, wt in optional_metrics:
                if metric in scoring_df.columns:
                    metric_name = f'{metric}_zscore'
                    print(metric, scoring_df[metric])
                    scoring_df[metric_name] = (
                                            scoring_df[metric] - scoring_df[metric].mean()
                                              ) / scoring_df[metric].std()
                    metrics_to_score.append((metric_name, wt))

            # Add Greeks if available
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                if greek in scoring_df.columns:
                    metric_name = f'{greek}_zscore'
                    scoring_df[metric_name] = (
                                                      scoring_df[greek].abs() - scoring_df[greek].abs().mean()
                                              ) / scoring_df[greek].abs().std()
                    metrics_to_score.append((metric_name, 0.25 / len(['delta', 'gamma', 'theta', 'vega'])))

            # Calculate composite score
            if metrics_to_score:
                # Normalize weights to sum to 1
                total_weight = sum(weight for _, weight in metrics_to_score)
                normalized_weights = [(metric, weight / total_weight) for metric, weight in metrics_to_score]

                scoring_df['opportunity_score'] = sum(
                    scoring_df[metric] * weight for metric, weight in normalized_weights
                )
            else:
                # Fallback to just implied volatility if no other metrics available
                scoring_df['opportunity_score'] = scoring_df['iv_zscore']

            # Get top opportunities for calls and puts separately
            top_calls = scoring_df[scoring_df['option_type'] == 'call'].nlargest(n, 'opportunity_score')
            top_puts = scoring_df[scoring_df['option_type'] == 'put'].nlargest(n, 'opportunity_score')

            # Combine results
            top_opportunities = pd.concat([top_calls, top_puts])

            # Select columns to return
            return_columns = ['strike', 'option_type', 'lastPrice', 'impliedVolatility', 'opportunity_score',
                              'value_score', 'expiry']

            # Add available Greeks
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                if greek in scoring_df.columns:
                    return_columns.append(greek)

            # Add other available metrics
            if 'volume_oi_ratio' in scoring_df.columns:
                return_columns.append('volume_oi_ratio')
            if 'probability_itm' in scoring_df.columns:
                return_columns.append('probability_itm')
            print(top_opportunities[return_columns])
            return top_opportunities[return_columns].copy()

        except Exception as e:
            print(f"Error in get_top_opportunities: {str(e)}")
            # Return minimal DataFrame with basic information
            return options_df[['strike', 'option_type', 'lastPrice', 'impliedVolatility', 'opportunity_score',
                              'value_score', 'expiry', 'probability_itm']].head(2 * n)

    def get_opportunity_metrics(self, options_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate opportunity metrics summary
        """
        try:
            metrics = {
                'total_opportunities': len(options_df),
                'avg_implied_vol': options_df['impliedVolatility'].mean(),
                'max_opportunity_score': options_df.get('opportunity_score', pd.Series()).max()
            }

            # Add Greeks statistics if available
            for greek in ['delta', 'gamma', 'theta', 'vega']:
                if greek in options_df.columns:
                    metrics[f'avg_{greek}'] = options_df[greek].abs().mean()
                    metrics[f'max_{greek}'] = options_df[greek].abs().max()

            return metrics

        except Exception as e:
            print(f"Error in get_opportunity_metrics: {str(e)}")
            return {
                'total_opportunities': len(options_df),
                'avg_implied_vol': options_df['impliedVolatility'].mean()
            }

    def detect_anomalies(self, options_df: pd.DataFrame) -> tuple[DataFrame, DataFrame]:
        """Detect anomalous options activity using machine learning"""
        # Select features for anomaly detection
        features = [
            'impliedVolatility', 'volume_oi_ratio', 'delta', 'gamma',
            'theta', 'vega', 'probability_itm'
        ]

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(options_df[features])

        # Detect anomalies using Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        options_df['anomaly_score'] = iso_forest.fit_predict(X)

        # Calculate feature importance for anomalies
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': np.abs(iso_forest.score_samples(X))
        }).sort_values('importance', ascending=False)

        return options_df, feature_importance

    def prepare_ml_features(self, options_df: pd.DataFrame) -> torch.Tensor:
        """Prepare features for the LSTM model"""
        features = [
            'impliedVolatility', 'delta', 'gamma', 'theta', 'vega',
            'volume_oi_ratio', 'days_to_expiry', 'strike'
        ]

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(options_df[features])

        # Create sequences
        sequence_length = 5
        sequences = []
        for i in range(len(scaled_features) - sequence_length + 1):
            sequences.append(scaled_features[i:i + sequence_length])

        return torch.FloatTensor(sequences).to(self.device)

    def predict_options_movement(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """Predict options price movement using the LSTM model"""
        X = self.prepare_ml_features(options_df)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()

        # Add predictions to dataframe
        options_df['price_movement_pred'] = np.pad(
            predictions.flatten(),
            (4, 0),  # Pad beginning due to sequence length
            'constant',
            constant_values=np.nan
        )

        return options_df


class OptionsDataset(Dataset):
    """PyTorch Dataset for options data"""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def main():
    """Example usage of the options screening system"""
    screener = AdvancedOptionsScreener()

    # Analyze options chain
    analysis = screener.analyze_options_chain()

    # Print summary
    print("\nOptions Analysis Summary:")
    for key, value in analysis['summary'].items():
        if not isinstance(value, pd.DataFrame):
            print(f"{key}: {value}")

    # Display top opportunities
    print("\nTop Trading Opportunities:")
    display_columns = [
        'expiry', 'strike', 'option_type', 'impliedVolatility',
        'probability_itm', 'value_score', 'opportunity_score'
    ]
    print(analysis['summary']['top_opportunities'][display_columns])


if __name__ == "__main__":
    main()
