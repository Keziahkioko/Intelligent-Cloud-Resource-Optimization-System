# Stage 2A - Regression: Results & Interpretation

## Results Summary

| Model | Train R2 | Val R2 | Test R2 | RMSE | MAE | MAPE | CV R2 (5-fold) |
|---|---|---|---|---|---|---|---|
| Linear Regression | 0.0000 | -0.0001 | -0.0000 | 27.42 | 22.54 | 6.07 | -0.0002 ± 0.000 |
| Ridge (L2) | 0.0000 | -0.0001 | -0.0000 | 27.42 | 22.54 | 6.07 | -0.0002 ± 0.000 |
| Lasso (L1) | 0.0000 | -0.0000 | -0.0000 | 27.42 | 22.54 | 6.07 | -0.0000 ± 0.000 |
| ElasticNet | 0.0000 | -0.0000 | -0.0000 | 27.42 | 22.54 | 6.07 | -0.0000 ± 0.000 |
| Bayesian Ridge | 0.0000 | -0.0000 | -0.0000 | 27.42 | 22.54 | 6.08 | -0.0000 ± 0.000 |
| Decision Tree | 0.8959 | 0.8969 | 0.8969 | 8.80 | 4.59 | 0.14 | 0.8957 ± 0.001 |

Dataset: 2,000,000 rows. Split: 60/20/20 train/val/test. StandardScaler fit on the training set only, then applied to validation and test sets.

## Overfitting Analysis

The five linear-family models (Linear Regression, Ridge, Lasso, ElasticNet, Bayesian Ridge) all scored a Test R2 of approximately 0.0000. This means they predicted about as well as simply guessing the average execution_time for every row, regardless of input features. This matches Stage 1's EDA, where correlation between every feature and execution_time was below |r| = 0.01. Linear models can only capture straight-line relationships, so this result shows there is no linear relationship between the available features and execution_time. This is not a fitting error but a property of the data itself.

The Decision Tree produced a different result: Test R2 of 0.897, with Train R2 of 0.896, Validation R2 of 0.897, and 5-fold CV R2 of 0.896 ± 0.001. These four values are almost identical, so the overfitting gap (Train R2 minus Test R2) is close to zero. This means the model is not memorizing the training data and generalizes well to unseen data.

These results together indicate execution_time depends on non-linear, threshold-based rules rather than a linear combination of the input features - for example, a rule like "if cpu_usage passes a certain value and task_priority equals a certain category, execution_time follows a different pattern." Decision trees can represent this kind of rule directly through their splits, while linear models cannot, which explains the large gap in performance between the two model types.

## Models Not Run

Three model types from the original scaffold were excluded due to the 2,000,000-row dataset size:

- Polynomial Regression (degree=3): expanding around 18 features to degree 3 created a large number of interaction columns and pushed system RAM to 98%, so the process was stopped.
- SVR (RBF and Linear kernels): SVR has roughly O(n^2) to O(n^3) time complexity, which at 1.2M+ training rows would take many hours to days to finish.
- KNN (k=5, k=15): predicting requires comparing each row against 1.2M training rows, and with 5-fold CV added on top this ran for over 2 hours without finishing and was interrupted.

Random Forest and Gradient Boosting were included in the model code but not run due to time constraints. The Decision Tree result already shows a clear, consistent difference between linear and tree-based performance on this dataset, which was the main question this stage needed to answer.