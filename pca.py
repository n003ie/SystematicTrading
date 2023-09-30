import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.pca import PCA

from core.analytics.macro_view import get_market_data


def run():
    start_date = '2010-01-01'
    end_date = '2023-09-09'
    returns = get_market_data(start_date, end_date, returnChange=False)
    print(returns.T.cov())
    x = StandardScaler().fit_transform(returns.T.cov())  # normalizing the features
    pca_ = PCA(n_components=2)
    pca_returns = pca_.fit_transform(x)
    pca_Df = pd.DataFrame(data=pca_returns
                          , columns=['principal component 1', 'principal component 2'])
    print(pca_Df)
    print('Explained variation per principal component: {}'.format(pca_.explained_variance_ratio_))
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title("Principal Component Analysis ", fontsize=20)

    indicesToKeep = returns['ZN=F'] > 0
    plt.scatter(pca_Df.loc[indicesToKeep]['principal component 1']
                , pca_Df.loc[indicesToKeep]['principal component 2'], c='b', s=50)

    indicesToKeep = returns['ZN=F'] <= 0
    plt.scatter(pca_Df.loc[indicesToKeep]['principal component 1']
                , pca_Df.loc[indicesToKeep]['principal component 2'], c='r', s=50)

    plt.show()


if __name__ == "__main__":
    run()
