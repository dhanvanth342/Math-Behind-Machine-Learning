## Exp 1 file [Dimensional Reduction, Hyper-parameter tuning, Cross validation[with handwritten function( Did not use scipy here :))]]. 
#### In this file, my work focuses on:
- Implementing cross-validation techniques for classification models using three datasets: Iris, Breast Cancer, and 20 Newsgroups. The goal is to optimize model hyperparameters and assess their performance.
- Perform Multidimensional Scaling (MDS) on a distance matrix of 24 European cities to create a 2D representation. This helped me understand the effect of dimension reduction from 24 to 2 but preserving
  the information by the original data matrix which led to similar projection of the cities in 2-d plane matching the actual positions in real world.

## Exp 2 file [ Dimensional Reduction]
#### In this file, my key findings are: 
- When implemented SVD for Image compression and studied the effect of K[Number of singualar values to be chosen from Diagonal matrix] in clarity of the image compressed.
- When I performed PCA using SVD on log transformed data and normalized z-score data, I have observed a difference between the principal components, projected data points and scatter plot. I can conclude that the 
  choice of transformation plays a vital role in PCA, and should be chosen based on our requirement. In case of treating all the features equally, normalized z-score is preferable while if you want to reduce the 
  number of outliers, log based is preferable.
