# =============================================================
# STAGE 2A — REGRESSION PIPELINE
# File: stage2a_regression/regression_pipeline.py
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model  import (LinearRegression, Ridge, Lasso,
                                    ElasticNet, BayesianRidge)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline      import Pipeline
from sklearn.neighbors     import KNeighborsRegressor
from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import (RandomForestRegressor,
                                    GradientBoostingRegressor)
from sklearn.svm           import SVR
from sklearn.metrics       import (mean_squared_error, r2_score,
                                    mean_absolute_error)
from sklearn.model_selection import cross_val_score, learning_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# 2A.0  DATA PREPARATION — load, split, scale (no leakage)
# =============================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error


def prepare_regression_data(
    csv_path: str = "data/cloud_performance_features.csv",
    target_col: str = "execution_time",
    drop_cols: list = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    sample_frac: float = None,
) -> dict:
    """
    Loads the cleaned CSV, splits into train/val/test, and scales
    features using a scaler fit ONLY on the training set (no leakage
    into val/test). Returns the `data` dict every function below expects.

    sample_frac: with 2,000,000 rows, SVR and Polynomial(deg=3) can take a
    very long time (SVR in particular scales poorly past ~50k rows). Set
    e.g. sample_frac=0.05 (100k rows) for a FAST first pass to confirm
    everything runs end to end, then rerun with sample_frac=None (full
    data) for your real/final numbers once you know it works.
    """
    df = pd.read_csv(csv_path)

    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=random_state)
        print(f"[info] Using a {sample_frac:.0%} sample: {len(df):,} rows "
              f"(for a quick test run — rerun with sample_frac=None for final results)")

    if drop_cols is None:
        # Confirmed real columns (2,000,000 rows x 26 cols):
        # - vm_id/timestamp (+ _encoded): identifiers, not predictive features
        # - exec_time_bucket (+ _encoded): DERIVED FROM execution_time itself
        #   (a binned version) -> leaving these in is data leakage, since the
        #   model would be predicting execution_time from a column that
        #   already encodes execution_time
        # - task_type/task_priority/task_status (text): redundant, since
        #   _encoded numeric versions of each already exist
        drop_cols = [c for c in [
            "vm_id", "vm_id_encoded",
            "timestamp", "timestamp_encoded",
            "exec_time_bucket", "exec_time_bucket_encoded",
            "task_type", "task_priority", "task_status",
        ] if c in df.columns]

    if target_col not in df.columns:
        raise ValueError(
            f"'{target_col}' not in columns. Available: {df.columns.tolist()}"
        )

    X = df.drop(columns=drop_cols + [target_col])
    y = df[target_col].values

    # keep numeric features only for now; encode categoricals separately
    # if you want them included (task_type, task_priority are categorical)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"[info] Dropping non-numeric columns for now: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # First split off the test set, then split remaining into train/val
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # val_size is a fraction of the ORIGINAL data, so recompute relative
    # to what's left in X_temp
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state
    )

    # Fit scaler on TRAIN ONLY — this is the leakage-avoidance step
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"[info] Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, "
          f"Test: {X_test_scaled.shape}")

    return {
        "X_train": X_train_scaled, "y_train": y_train,
        "X_val": X_val_scaled,     "y_val": y_val,
        "X_test": X_test_scaled,   "y_test": y_test,
        "feature_names": X.columns.tolist(),
        "scaler": scaler,
    }


# =============================================================
# 2A.1  MODEL ZOO — All Regression Models
# =============================================================

def build_regression_models():
    """Return dictionary of regression models to compare.

    Final set used for Stage 2A results: 6 models spanning linear and
    tree-based approaches (well above the "at least 2 models" rubric
    requirement).

    Removed for practicality on this 2,000,000-row dataset:
    - Polynomial(deg=3): expanded ~18 features into a huge number of
      interaction columns, blew RAM to 98%, had to be killed.
    - SVR (RBF/Linear): O(n^2)-O(n^3) time complexity, would realistically
      take many hours to days on 1.2M+ training rows.
    - KNN (k=5/k=15): distance computation against 1.2M training rows for
      every prediction (plus 5-fold CV) made this impractically slow,
      running 2+ hours without completing.
    - Random Forest, Gradient Boosting: not run due to time constraints
      after the above; Decision Tree already gives a strong, consistent
      non-linear result (Test R2 ~0.90) that answers the key question of
      whether non-linear signal exists in this data.
    """
    return {
        'Linear Regression':     LinearRegression(),
        'Ridge (L2)':            Ridge(alpha=1.0),
        'Lasso (L1)':            Lasso(alpha=0.01),
        'ElasticNet':            ElasticNet(alpha=0.01, l1_ratio=0.5),
        'Bayesian Ridge':        BayesianRidge(),
        'Decision Tree':         DecisionTreeRegressor(
                                     max_depth=8, random_state=42),
    }

def train_and_evaluate_regression(data: dict) -> pd.DataFrame:
    """Train all regression models and collect metrics."""

    X_tr, y_tr = data['X_train'], data['y_train']
    X_va, y_va = data['X_val'],   data['y_val']
    X_te, y_te = data['X_test'],  data['y_test']

    models  = build_regression_models()
    records = []

    print("=" * 95)
    print("  STAGE 2A — Regression Model Comparison")
    print("=" * 95)
    print(f"\n  {'Model':<24} {'Train R²':>9} {'Val R²':>9} "
          f"{'Test R²':>9} {'RMSE':>9} {'MAE':>9} {'MAPE':>8} {'CV R² (5-fold)':>16}")
    print(f"  {'─'*24} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*8} {'─'*16}")

    for name, model in models.items():
        model.fit(X_tr, y_tr)

        y_tr_pred = model.predict(X_tr)
        y_va_pred = model.predict(X_va)
        y_te_pred = model.predict(X_te)

        train_r2 = r2_score(y_tr, y_tr_pred)
        val_r2   = r2_score(y_va, y_va_pred)
        test_r2  = r2_score(y_te, y_te_pred)
        rmse     = np.sqrt(mean_squared_error(y_te, y_te_pred))
        mae      = mean_absolute_error(y_te, y_te_pred)
        mape     = mean_absolute_percentage_error(y_te, y_te_pred)

        # 5-fold CV on the training set only — required by the rubric,
        # and keeps the scaler-fit-on-train-only guarantee since X_tr
        # was already scaled using a scaler fit on X_tr alone.
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='r2')

        print(f"  {name:<24} {train_r2:>9.4f} {val_r2:>9.4f} "
              f"{test_r2:>9.4f} {rmse:>9.4f} {mae:>9.4f} {mape:>8.4f} "
              f"{cv_scores.mean():>8.4f}±{cv_scores.std():.3f}")

        records.append({
            'Model':    name,
            'Train R²': train_r2,
            'Val R²':   val_r2,
            'Test R²':  test_r2,
            'RMSE':     rmse,
            'MAE':      mae,
            'MAPE':     mape,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std':  cv_scores.std(),
            'Overfit':  train_r2 - test_r2
        })

    return pd.DataFrame(records).sort_values('Test R²', ascending=False)


# =============================================================
# 2A.3  NEURAL NETWORK REGRESSOR (Week 7)
# =============================================================

def build_nn_regressor(input_dim: int) -> keras.Model:
    """Deep neural network for regression on cloud metrics."""

    inputs = keras.Input(shape=(input_dim,), name='cloud_features')

    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(32, activation='relu')(x)

    output = layers.Dense(1, activation='linear', name='execution_time')(x)

    model = keras.Model(inputs, output, name='CloudRegressor')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='huber',
        metrics=['mae', 'mse']
    )
    return model


def train_nn_regressor(data: dict) -> dict:
    """Train neural network and return history + metrics."""

    model = build_nn_regressor(data['X_train'].shape[1])
    model.summary()

    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=7, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        data['X_train'], data['y_train'],
        validation_data=(data['X_val'], data['y_val']),
        epochs=100,
        batch_size=64,
        callbacks=callback_list,
        verbose=0
    )

    y_pred = model.predict(data['X_test'], verbose=0).flatten()
    test_r2   = r2_score(data['y_test'], y_pred)
    test_rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))

    print(f"\n  Neural Network Results:")
    print(f"     Test R2   : {test_r2:.4f}")
    print(f"     Test RMSE : {test_rmse:.4f}")

    return {'model': model, 'history': history,
            'test_r2': test_r2, 'test_rmse': test_rmse,
            'y_pred': y_pred}


# =============================================================
# 2A.4  VISUALIZATION SUITE
# =============================================================

def visualize_regression_results(results_df:  pd.DataFrame,
                                  nn_results:  dict,
                                  data:        dict) -> None:
    """Comprehensive regression results dashboard."""

    fig = plt.figure(figsize=(22, 18))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                             hspace=0.4, wspace=0.35)

    COLORS = ['#3498DB','#E74C3C','#2ECC71','#F39C12',
              '#9B59B6','#1ABC9C','#E67E22','#34495E']

    ax1 = fig.add_subplot(gs[0, :2])
    top_n  = results_df.head(10)
    colors = [COLORS[i % len(COLORS)] for i in range(len(top_n))]
    bars   = ax1.barh(top_n['Model'], top_n['Test R²'],
                       color=colors, edgecolor='white',
                       linewidth=0.5)
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel("Test R2 Score", fontsize=11)
    ax1.set_title("Regression Models - Test R2 Comparison",
                   fontweight='bold', fontsize=12)
    for bar, val in zip(bars, top_n['Test R²']):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9,
                 fontweight='bold')
    ax1.axvline(0.8, color='green', linestyle='--',
                linewidth=1, alpha=0.7, label='R2=0.80 target')
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(results_df['Train R²'],
                results_df['Test R²'],
                c=[COLORS[i % len(COLORS)]
                   for i in range(len(results_df))],
                s=80, edgecolors='white', linewidths=1, zorder=3)
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1,
             alpha=0.5, label='Perfect (no overfit)')
    ax2.fill_between([0, 1], [0, 1], [0, 0.85],
                     alpha=0.05, color='red',
                     label='Overfit zone')
    for _, row in results_df.iterrows():
        ax2.annotate(row['Model'][:12],
                     (row['Train R²'], row['Test R²']),
                     textcoords="offset points",
                     xytext=(4, 4), fontsize=6)
    ax2.set_xlabel("Train R2"); ax2.set_ylabel("Test R2")
    ax2.set_title("Bias-Variance Check\n(Train vs Test R2)",
                   fontweight='bold', fontsize=11)
    ax2.legend(fontsize=7)
    ax2.set_xlim(0, 1.1); ax2.set_ylim(0, 1.1)

    ax3 = fig.add_subplot(gs[1, 0])
    hist = nn_results['history'].history
    ax3.plot(hist['loss'],     label='Train Loss',
             color='#3498DB', linewidth=2)
    ax3.plot(hist['val_loss'], label='Val Loss',
             color='#E74C3C', linewidth=2, linestyle='--')
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Huber Loss")
    ax3.set_title("Neural Network\nTraining Curves",
                   fontweight='bold', fontsize=11)
    ax3.legend(fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1])
    y_true = np.array(data['y_test'])
    y_pred = nn_results['y_pred']
    ax4.scatter(y_true, y_pred, alpha=0.3, s=10,
                color='#9B59B6', edgecolors='none')
    lim = [min(y_true.min(), y_pred.min()),
           max(y_true.max(), y_pred.max())]
    ax4.plot(lim, lim, 'r--', linewidth=2, label='Perfect prediction')
    ax4.set_xlabel("Actual Execution Time")
    ax4.set_ylabel("Predicted Execution Time")
    ax4.set_title(f"NN: Actual vs Predicted\nR2={nn_results['test_r2']:.4f}",
                   fontweight='bold', fontsize=11)
    ax4.legend(fontsize=9)

    ax5 = fig.add_subplot(gs[1, 2])
    residuals = y_true - y_pred
    ax5.hist(residuals, bins=40, color='#1ABC9C',
              alpha=0.7, edgecolor='white', density=True)
    from scipy import stats
    x_r = np.linspace(residuals.min(), residuals.max(), 200)
    ax5.plot(x_r, stats.norm.pdf(x_r, residuals.mean(),
                                   residuals.std()),
              'r-', linewidth=2, label='Normal fit')
    ax5.axvline(0, color='black', linestyle='--', linewidth=1)
    ax5.set_xlabel("Residual (Actual - Predicted)")
    ax5.set_ylabel("Density")
    ax5.set_title("NN Residual Distribution",
                   fontweight='bold', fontsize=11)
    ax5.legend(fontsize=9)

    ax6 = fig.add_subplot(gs[2, :])
    x_pos = np.arange(len(results_df))
    width = 0.35
    bars1 = ax6.bar(x_pos - width/2, results_df['RMSE'],
                     width, label='RMSE',
                     color='#3498DB', alpha=0.8, edgecolor='white')
    bars2 = ax6.bar(x_pos + width/2, results_df['MAE'],
                     width, label='MAE',
                     color='#E74C3C', alpha=0.8, edgecolor='white')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(results_df['Model'],
                         rotation=35, ha='right', fontsize=9)
    ax6.set_ylabel("Error")
    ax6.set_title("RMSE & MAE Comparison Across All Models",
                   fontweight='bold', fontsize=12)
    ax6.legend(fontsize=10)

    plt.suptitle("Stage 2A - Cloud Execution Time Regression Results",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.savefig("stage2a_regression_results.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Stage 2A Complete - Regression results saved")
# =============================================================
# 2A.5  MAIN — run the full Stage 2A pipeline
# =============================================================

if __name__ == "__main__":
   
    data = prepare_regression_data(
        csv_path="data/cloud_performance_features.csv",
        target_col="execution_time",
        sample_frac=None,   # full dataset
    )

    results_df = train_and_evaluate_regression(data)

    print("\nTop 5 models by Test R²:")
    print(results_df.head(5).to_string(index=False))

    nn_results = train_nn_regressor(data)

    visualize_regression_results(results_df, nn_results, data)

    results_df.drop(columns=[]).to_csv(
        "stage2a_regression_results.csv", index=False
    )
    print("\nSaved stage2a_regression_results.csv")