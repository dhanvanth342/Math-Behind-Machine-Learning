## Exp 3 file [Semi-Supervised Learning]
#### Summary of Findings and Analysis 

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
