# Stage 1 — EDA Findings

## Bug Fixes Made to the Starter Script
While running the lecturer's original `ingest_and_clean.py`, the script
hung indefinitely partway through EDA plotting. Two separate bugs were
found and fixed:

1. **Non-interactive plotting backend.** The original script never set
   a matplotlib backend and called `plt.show()` after every figure.
   Fixed by adding `matplotlib.use('Agg')` at the top of the file and
   replacing `plt.show()` with `plt.close()` after each `savefig()`
   call, so figures save straight to disk instead of trying to open
   interactive windows.
2. **Unbounded categorical plotting.** The original script generated a
   bar chart for every text/object column with no limit, including
   `vm_id` and `timestamp`, each of which has close to 1.8 million
   unique values. Attempting to draw a bar chart with millions of bars
   is what caused the actual freeze. Fixed by filtering categorical
   columns to only those with 20 or fewer unique values:
   ```python
   cat_cols = [c for c in df_plot.select_dtypes(include=['object', 'category']).columns
               if df_plot[c].nunique() <= 20]
   ```
   This correctly keeps `task_type`, `task_priority`, and `task_status`
   (3 unique values each) while excluding `vm_id` and `timestamp`.

Other changes made along the way (performance and correctness, not
bug-critical): sampling 50,000 rows for plotting instead of the full
2,000,000-row dataset, printing an explicit descriptive statistics
table, explicitly flagging feature pairs with |r| > 0.8, adding a new
box plot figure (not present in the original), and creating the
`data/` output folder automatically if it doesn't exist.

## Dataset Overview
- 2,000,000 rows, 12 original columns (raw cloud telemetry data).
- Every single column had almost exactly 10% missing values (200,000
  rows each). This is far too uniform to be natural — it strongly
  suggests missingness was artificially injected into the dataset
  (a common technique to test that students handle missing data
  correctly), rather than reflecting real-world sensor gaps.
- Missing values were imputed using the median (for numeric columns)
  and the mode / most frequent value (for categorical columns).
- 0 duplicate rows were found.

## Distributions (eda_distributions.png)
- All seven numeric features — cpu_usage, memory_usage,
  network_traffic, power_consumption, num_executed_instructions,
  execution_time, and energy_efficiency — have skewness values
  extremely close to 0.00 (ranging from -0.01 to 0.01).
- Each histogram is essentially flat/rectangular across its full
  range rather than bell-shaped, meaning every value in the range
  is roughly equally likely to occur.
- This is characteristic of a uniform distribution, and combined
  with the exact 10% missingness pattern, further suggests the
  dataset was synthetically generated rather than collected from a
  live production system.

## Correlation Analysis (eda_correlations.png)
- All pairwise correlation coefficients between numeric features are
  effectively zero, ranging only from -0.01 to 0.01.
- No feature pairs exceed the |r| > 0.8 threshold used to flag
  multicollinearity — so no multicollinear features were identified
  in this dataset.
- Correlation with the likely target energy_efficiency is also
  negligible across all features (highest magnitude is 0.009 for
  execution_time).
- Implication for later stages: because no feature meaningfully
  correlates with energy_efficiency (or with execution_time), models
  built for regression/classification on this raw feature set may
  have limited predictive power unless the engineered features
  (ratios, interactions, load index) capture signal the raw columns
  don't.

## Outlier Detection (eda_boxplots.png)
- After the IQR-based capping step in data cleaning, none of the
  seven numeric features show any points beyond the whiskers in
  their box plots.
- This confirms the outlier-capping step in the cleaning pipeline
  worked as intended — no extreme values remain in the cleaned
  dataset.

## Categorical / Class Distributions (eda_categorical.png)
- task_type: network (15,098), io (14,890), compute (14,875) —
  balanced across all 3 categories (sample-based counts).
- task_priority: medium (15,134), high (15,059), low (14,878) —
  balanced across all 3 categories.
- task_status: running (15,085), waiting (15,036), completed
  (14,935) — balanced across all 3 categories.
- No class imbalance issues were found in any categorical column,
  which is favorable for the classification models built in Stage 2b.

## Pairwise Relationships (eda_pairplot.png)
- Scatter plots between key numeric features (cpu_usage,
  memory_usage, power_consumption, execution_time,
  energy_efficiency) show no visible linear or non-linear
  relationships — points appear as unstructured noise regardless of
  task_type.
- This is consistent with the near-zero correlation values found in
  the correlation matrix.

## Feature Engineering
- 14 new features were added, bringing the total from 12 to 26
  columns:
  - Ratio features: cpu_memory_ratio, power_per_instruction,
    throughput
  - Aggregate feature: system_load_index (mean of cpu_usage,
    memory_usage, network_traffic)
  - Binned features: cpu_load_level (Low/Medium/High/Critical),
    exec_time_bucket (Fast/Moderate/Slow/Very Slow)
  - Interaction feature: cpu_power_interaction
  - Label-encoded versions of all categorical columns for use in
    downstream models

## Data Splits
- Regression target (execution_time): 70% train (1,400,000) / 15%
  validation (300,000) / 15% test (300,000).
- Classification target (task_status_encoded): same 70/15/15 split.
- Scaler (StandardScaler) was fit only on the training set and then
  applied to validation/test sets, avoiding data leakage.

## Summary
The dataset is clean, balanced, and free of multicollinearity and
outliers after processing, but shows very weak correlation between
features and the likely prediction targets. This is an important
finding to flag to the team members working on Stage 2 (regression
and classification), since raw feature correlation being near-zero
may limit model performance regardless of which algorithm is used.