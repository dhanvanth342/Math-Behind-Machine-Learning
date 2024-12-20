# Math-Behind-Machine-Learning

### Exp 1 file [Dimensional Reduction, Hyper-parameter tuning, Cross validation[with handwritten function( Did not use scipy here :))]]. 
# In this file, my work focuses on:
- Implementing cross-validation techniques for classification models using three datasets: Iris, Breast Cancer, and 20 Newsgroups. The goal is to optimize model hyperparameters and assess their performance.
- Perform Multidimensional Scaling (MDS) on a distance matrix of 24 European cities to create a 2D representation. This helped me understand the effect of dimension reduction from 24 to 2 but preserving
  the information by the original data matrix which led to similar projection of the cities in 2-d plane matching the actual positions in real world.

### Exp 2 file [ Dimensional Reduction]
# In this file, my key findings are: 
- When implemented SVD for Image compression and studied the effect of K[Number of singualar values to be chosen from Diagonal matrix] in clarity of the image compressed.
- When I performed PCA using SVD on log transformed data and normalized z-score data, I have observed a difference between the principal components, projected data points and scatter plot. I can conclude that the 
  choice of transformation plays a vital role in PCA, and should be chosen based on our requirement. In case of treating all the features equally, normalized z-score is preferable while if you want to reduce the 
  number of outliers, log based is preferable.

### Exp 3 file [Semi-Supervised Learning]
# Summary of Findings and Analysis 

- **Effect of Decay Parameter (Alpha):**
  - Observed that increasing alpha significantly improves the accuracy of label predictions.
  - At alpha = 0.8, Label Spreading propagates labels correctly, supporting the role of alpha in controlling label information spread among neighbors.

- **Data Imbalance Issues:**
  - Shuffled dataset led to partial labeling, leaving some classes without predictions.
  - Imbalanced dataset caused biased predictions, heavily favoring certain classes (e.g., class 0).

- **Model Performance with Label Propagation:**
  - Higher labeled samples reduced overfitting and improved test accuracy.
  - Performance metrics revealed that small labeled datasets lead to high variance and overfitting, necessitating larger labeled datasets.

- **Cora Dataset and GCN Insights:**
  - Dataset consists of 2708 research papers classified into seven categories, with features derived from a vocabulary of 1433 words.
  - GCN training highlighted overfitting with small labeled nodes, which was mitigated by increasing labeled samples.

- **Performance Metrics Observations:**
  - Gradual increase in validation accuracy with labeled nodes.
  - At 300 labeled nodes, overfitting was significantly reduced, and test accuracy improved (maximum 81.29%).

### Exp4 file [ Unsupervised Learning]
# Summary of Analysis

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

- **Key Insights**
  - GMM excels in datasets with Gaussian clusters, adapting well to the covariance structure.
  - Spectral Clustering offers a clear advantage for complex, non-linear data distributions due to eigenvector-based transformations.

