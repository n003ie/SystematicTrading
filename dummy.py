import pandas as pd
from sklearn.cluster import KMeans

# Step 1: Data Preprocessing
# Load data (assuming you have CSV files with date and price columns for each asset)
assets = ['USDJPY', 'USDEUR', 'Oil', 'Gold', 'GermanBond', 'Nasdaq', 'USTreasuryBond']
data = {}

for asset in assets:
    data[asset] = pd.read_csv(f'{asset}_prices.csv', index_col='Date', parse_dates=True)

# Check for missing values and fill gaps if necessary
for asset in assets:
    data[asset] = data[asset].resample('D').ffill().dropna()

# Normalize data if needed (e.g., dividing by the first value)
# data['AssetName'] = data['AssetName'] / data['AssetName'].iloc[0]

# Step 2: Convert Levels to Returns
for asset in assets:
    data[asset]['Return'] = data[asset]['Price'].pct_change().dropna()

# Step 3: Regime Identification Techniques

# Rolling Correlations
window_size = 30  # Adjust window size as needed
rolling_correlations = pd.DataFrame()

for asset1 in assets:
    for asset2 in assets:
        if asset1 != asset2:
            correlation = data[asset1]['Return'].rolling(window=window_size).corr(data[asset2]['Return'])
            rolling_correlations[f'{asset1}_{asset2}_Corr'] = correlation

# Cluster Analysis (K-Means Clustering)
n_clusters = 3  # Adjust the number of clusters as needed
X = rolling_correlations.dropna().values
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
rolling_correlations['Cluster'] = kmeans.labels_

# Regime Switching Models (e.g., Hidden Markov Model)
# You may need to install the "hmmlearn" library for this part
from hmmlearn import hmm

model = hmm.GaussianHMM(n_components=n_clusters, covariance_type="full", n_iter=1000)
model.fit(X)
states = model.predict(X)
rolling_correlations['HMM_State'] = states

# Print the results
print("Rolling Correlations:")
print(rolling_correlations.head())

print("\nCluster Analysis:")
print(rolling_correlations['Cluster'].value_counts())

print("\nRegime Switching Model States:")
print(rolling_correlations['HMM_State'].value_counts())
