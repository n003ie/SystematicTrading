import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate

# Generate synthetic data (replace this with your actual returns data)
np.random.seed(42)
n_samples = 1000
n_assets = 7
returns_data = np.random.randn(n_samples, n_assets)

# Create a DataFrame from the synthetic data
returns_df = pd.DataFrame(returns_data, columns=['USDJPY', 'USDEUR', 'Oil', 'Gold', 'GermanBond', 'Nasdaq', 'USTreasuryBond'])

# Select the variables of interest (DXY, Nasdaq, US Treasury)
selected_assets = ['USDJPY', 'Nasdaq', 'USTreasuryBond']
selected_data = returns_df[selected_assets]

# Step 1: Fit a Multivariate Copula Model (Gaussian Copula)
copula = GaussianMultivariate()
copula.fit(selected_data)

# Step 2: Generate Synthetic Data from the Copula Model
synthetic_data = copula.sample(n_samples)

# Step 3: Visualize the Copula in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract the synthetic data for each selected asset
x = synthetic_data[selected_assets[0]]
y = synthetic_data[selected_assets[1]]
z = synthetic_data[selected_assets[2]]

ax.scatter(x, y, z, c='b', marker='o', alpha=0.5)
ax.set_xlabel(selected_assets[0])
ax.set_ylabel(selected_assets[1])
ax.set_zlabel(selected_assets[2])

plt.title('3D Copula Visualization')

plt.show()
