## [ Unsupervised Learning]
#### Summary of Analysis

- **Gaussian Mixture Models (GMM) vs K-Means**
  - Evaluated GMM and K-Means on datasets with clusters having different covariance structures.
  - **Findings:**
    - Spherical covariance: GMM outperformed K-Means in capturing spherical clusters with higher ARI and NMI scores.
    - Diagonal covariance: GMM with diagonal covariance handled elliptical clusters better than K-Means.
    - Fully unrestricted covariance: Fully Gaussian GMM captured correlations in data effectively, outperforming K-Means and diagonal GMM.

- **Spectral Clustering vs K-Means**
  - Implemented Spectral Clustering with various sigma values for the Gaussian kernel.
  - Observed that smaller sigma values focus on local points, while larger values blend clusters.
  - Spectral Clustering effectively handled non-linear distributions (e.g., spiral datasets) that K-Means struggled with.

- **Metrics Used**
  - Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) were used to evaluate clustering performance.
