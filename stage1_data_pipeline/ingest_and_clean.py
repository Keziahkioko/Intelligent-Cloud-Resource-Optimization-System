# =============================================================
# STAGE 1 — COMPLETE DATA PIPELINE
# File: stage1_data_pipeline/ingest_and_clean.py
# =============================================================

import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend: prevents freezing and blocking pop-ups

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# ── COLORS & STYLE ───────────────────────────────────────────
PALETTE = {
    'primary':   '#2C3E50',
    'secondary': '#3498DB',
    'accent':    '#E74C3C',
    'success':   '#2ECC71',
    'warning':   '#F39C12',
    'purple':    '#9B59B6',
    'teal':      '#1ABC9C',
}
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#F8F9FA',
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'font.family':      'sans-serif',
})

# =============================================================
# 1.1  LOAD & INITIAL INSPECTION
# =============================================================

def load_and_inspect(filepath: str) -> pd.DataFrame:
    """Load the cloud computing performance dataset."""

    print("=" * 65)
    print("  STAGE 1 — Cloud Computing Performance Data Pipeline")
    print("=" * 65)

    df = pd.read_csv(filepath)

    print(f"\n{'─'*65}")
    print("  📊 DATASET OVERVIEW")
    print(f"{'─'*65}")
    print(f"  Rows          : {df.shape[0]:,}")
    print(f"  Columns       : {df.shape[1]}")
    print(f"  Memory Usage  : {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    print(f"\n{'─'*65}")
    print("  📋 COLUMN INVENTORY")
    print(f"{'─'*65}")
    print(f"  {'Column':<28} {'DType':<12} {'Non-Null':<10} {'Unique'}")
    print(f"  {'─'*28} {'─'*12} {'─'*10} {'─'*10}")
    for col in df.columns:
        print(f"  {col:<28} {str(df[col].dtype):<12} "
              f"{df[col].notna().sum():<10} {df[col].nunique()}")

    return df


# =============================================================
# 1.2  EXPLORATORY DATA ANALYSIS
# =============================================================

def full_eda(df: pd.DataFrame) -> None:
    """Complete EDA with publication-quality visualizations using downsampled data."""

    print(f"\n{'─'*65}")
    print("  📈 GENERATING EDA VISUALIZATIONS (Fast Mode)")
    print(f"{'─'*65}")

    # Sample 50k rows strictly for plotting speed without losing statistical accuracy
    sample_size = min(50000, len(df))
    df_plot = df.sample(n=sample_size, random_state=42).copy()

    numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_plot.select_dtypes(include=['object', 'category']).columns
               if df_plot[c].nunique() <= 20]

    # ── Summary Statistics Table ─────────────────────────────
    stats_df = df[numeric_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
    stats_df['skewness'] = df[numeric_cols].skew()
    print("\n  📝 SUMMARY STATISTICS (Full Dataset):")
    print(stats_df.round(2).to_string())

    # ── Figure 1: Distribution of All Numeric Features ───────
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten()

    colors = list(PALETTE.values())
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        data = df_plot[col].dropna()

        ax.hist(data, bins=40, color=colors[i % len(colors)],
                alpha=0.6, density=True, edgecolor='white', linewidth=0.5)

        if len(data.unique()) > 1:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), color=PALETTE['primary'], linewidth=2)

        ax.axvline(data.mean(), color=PALETTE['accent'],
                   linestyle='--', linewidth=1.5, label=f'Mean={data.mean():.2f}')
        ax.axvline(data.median(), color=PALETTE['success'],
                   linestyle=':', linewidth=1.5, label=f'Median={data.median():.2f}')

        ax.set_title(f"{col}\nSkew={data.skew():.2f}", fontweight='bold', fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Cloud Metrics — Numeric Feature Distributions", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("eda_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✔ Saved: eda_distributions.png")

    # ── Figure 2: Correlation Heatmap & High Correlation Flagging ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    corr = df_plot[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                ax=axes[0], linewidths=0.5, annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
    axes[0].set_title("Feature Correlation Matrix", fontweight='bold', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # Flag pairs with |r| > 0.8
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:
                high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

    if high_corr:
        print("\n  ⚠️ HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.8):")
        for col1, col2, val in high_corr:
            print(f"     • {col1} <--> {col2}: r = {val:.2f}")

    if 'energy_efficiency' in df_plot.columns:
        target_corr = corr['energy_efficiency'].drop('energy_efficiency').sort_values()
        colors_bar = [PALETTE['accent'] if v < 0 else PALETTE['success'] for v in target_corr]
        axes[1].barh(target_corr.index, target_corr.values, color=colors_bar, edgecolor='white', linewidth=0.5)
        axes[1].axvline(0, color='black', linewidth=1)
        axes[1].set_title("Correlation with Energy Efficiency", fontweight='bold', fontsize=12)
        axes[1].set_xlabel("Pearson Correlation Coefficient")
        for i, (val, name) in enumerate(zip(target_corr.values, target_corr.index)):
            axes[1].text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}',
                         va='center', ha='left' if val >= 0 else 'right', fontsize=8, fontweight='bold')

    plt.suptitle("Cloud Metrics — Correlation Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("eda_correlations.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✔ Saved: eda_correlations.png")

    # ── Figure 3: Box Plots for Outlier Inspection ──────────
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        sns.boxplot(x=df_plot[col], ax=ax, color=colors[i % len(colors)], fliersize=2)
        ax.set_title(f"Boxplot: {col}", fontweight='bold', fontsize=10)
        ax.set_xlabel(col, fontsize=8)

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Cloud Metrics — Feature Box Plots (Outlier Identification)", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("eda_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✔ Saved: eda_boxplots.png")

    # ── Figure 4: Categorical Feature Distributions ─────────
    if cat_cols:
        fig, axes = plt.subplots(1, len(cat_cols), figsize=(6 * len(cat_cols), 5))
        if len(cat_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, cat_cols):
            counts = df_plot[col].value_counts()
            bars = ax.bar(counts.index.astype(str), counts.values,
                          color=list(PALETTE.values())[:len(counts)], edgecolor='white', linewidth=0.7)
            ax.set_title(f"Distribution: {col}", fontweight='bold', fontsize=11)
            ax.set_xlabel(col)
            ax.set_ylabel("Count (Sample)")
            ax.tick_params(axis='x', rotation=30)
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{val:,}', ha='center', fontsize=9, fontweight='bold')

        plt.suptitle("Cloud Metrics — Categorical Distributions", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("eda_categorical.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✔ Saved: eda_categorical.png")

    # ── Figure 5: Pairplot ────────────────────────────────────
    key_cols = ['cpu_usage', 'memory_usage', 'power_consumption', 'execution_time', 'energy_efficiency']
    key_cols = [c for c in key_cols if c in df_plot.columns]

    if len(key_cols) >= 3:
        hue_col = 'task_type' if 'task_type' in df_plot.columns else None
        pair_fig = sns.pairplot(
            df_plot[key_cols + ([hue_col] if hue_col else [])].sample(min(1000, len(df_plot)), random_state=42),
            hue=hue_col, diag_kind='hist', plot_kws={'alpha': 0.4, 's': 15}, height=2.0
        )
        pair_fig.fig.suptitle("Cloud Metrics — Key Feature Relationships", y=1.02, fontsize=14, fontweight='bold')
        pair_fig.savefig("eda_pairplot.png", dpi=130, bbox_inches='tight')
        plt.close()
        print("  ✔ Saved: eda_pairplot.png")

    print("\n  ✅ EDA Complete — All 5 visual figures saved")


# =============================================================
# 1.3  DATA CLEANING
# =============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, outliers, and duplicates."""

    print(f"\n{'─'*65}")
    print("  🧹 DATA CLEANING REPORT")
    print(f"{'─'*65}")

    original_shape = df.shape

    # Step 1: Duplicates
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"  Duplicates removed     : {dupes}")

    # Step 2: Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n  Missing values found:")
        for col, n in missing[missing > 0].items():
            pct = n / len(df) * 100
            print(f"    {col:<28}: {n} ({pct:.1f}%)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols     = df.select_dtypes(include=['object']).columns
        if len(numeric_cols) > 0:
            num_imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        print("  → Imputed: median (numeric), mode (categorical)")
    else:
        print("  Missing values         : None ✅")

    # Step 3: Outlier Capping (IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_report = {}

    for col in numeric_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_out > 0:
            outlier_report[col] = n_out
            df[col] = df[col].clip(lower, upper)

    if outlier_report:
        print(f"\n  Outliers capped (IQR method):")
        for col, n in sorted(outlier_report.items(), key=lambda x: -x[1]):
            print(f"    {col:<28}: {n:,} outliers capped")

    print(f"\n  Original shape : {original_shape}")
    print(f"  Cleaned shape  : {df.shape}")
    print(f"  Rows removed   : {original_shape[0] - df.shape[0]}")

    return df


# =============================================================
# 1.4  FEATURE ENGINEERING
# =============================================================

def engineer_features(df: pd.DataFrame):
    """Create domain-informed features for cloud computing."""

    print(f"\n{'─'*65}")
    print("  ⚙️  FEATURE ENGINEERING")
    print(f"{'─'*65}")

    original_cols = df.shape[1]

    if all(c in df.columns for c in ['cpu_usage', 'memory_usage']):
        df['cpu_memory_ratio'] = df['cpu_usage'] / (df['memory_usage'] + 1e-9)
        print("  + cpu_memory_ratio       : cpu_usage / memory_usage")

    if all(c in df.columns for c in ['power_consumption', 'num_executed_instructions']):
        df['power_per_instruction'] = df['power_consumption'] / (df['num_executed_instructions'] + 1e-9)
        print("  + power_per_instruction  : power / instructions")

    if all(c in df.columns for c in ['execution_time', 'num_executed_instructions']):
        df['throughput'] = df['num_executed_instructions'] / (df['execution_time'] + 1e-9)
        print("  + throughput             : instructions / exec_time")

    load_cols = [c for c in ['cpu_usage', 'memory_usage', 'network_traffic'] if c in df.columns]
    if load_cols:
        df['system_load_index'] = df[load_cols].mean(axis=1)
        print(f"  + system_load_index      : mean of {load_cols}")

    if 'cpu_usage' in df.columns:
        df['cpu_load_level'] = pd.cut(df['cpu_usage'], bins=[0, 30, 60, 85, 100], labels=['Low', 'Medium', 'High', 'Critical'])
        print("  + cpu_load_level         : binned cpu_usage")

    if 'execution_time' in df.columns:
        df['exec_time_bucket'] = pd.qcut(df['execution_time'], q=4, labels=['Fast', 'Moderate', 'Slow', 'Very Slow'], duplicates='drop')
        print("  + exec_time_bucket       : quartile-binned exec time")

    if all(c in df.columns for c in ['cpu_usage', 'power_consumption']):
        df['cpu_power_interaction'] = df['cpu_usage'] * df['power_consumption']
        print("  + cpu_power_interaction  : cpu_usage × power_consumption")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    le_dict  = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        print(f"  + {col}_encoded          : LabelEncoded")

    print(f"\n  Features: {original_cols} → {df.shape[1]} (+{df.shape[1]-original_cols})")
    return df, le_dict


# =============================================================
# 1.5  TRAIN / VALIDATION / TEST SPLIT
# =============================================================

def create_splits(df: pd.DataFrame, target: str, test_sz: float = 0.15, val_sz: float = 0.15, seed: int = 42):
    """Create stratified train / validation / test splits."""

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != target and not c.endswith('_encoded')]

    X = df[feature_cols].drop(columns=[target], errors='ignore')
    y = df[target]

    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_sz, random_state=seed)
    val_ratio = val_sz / (1 - test_sz)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=val_ratio, random_state=seed)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n{'─'*65}")
    print("  ✂️  DATA SPLITS")
    print(f"{'─'*65}")
    print(f"  Target         : {target}")
    print(f"  Features       : {X.shape[1]}")
    print(f"  Train          : {len(X_train):,} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  Validation     : {len(X_val):,}  ({len(X_val)/len(X)*100:.0f}%)")
    print(f"  Test           : {len(X_test):,}  ({len(X_test)/len(X)*100:.0f}%)")

    return {
        'X_train': X_train_sc, 'y_train': y_train,
        'X_val':   X_val_sc,   'y_val':   y_val,
        'X_test':  X_test_sc,  'y_test':  y_test,
        'scaler':  scaler,
        'feature_names': X.columns.tolist()
    }


# =============================================================
# MAIN — Run Stage 1
# =============================================================
if __name__ == '__main__':
    df       = load_and_inspect('data/cloud_performance_raw.csv')
    full_eda(df)
    df_clean = clean_data(df)
    df_feat, encoders = engineer_features(df_clean)
    
    # Save processed features CSV for group members
    os.makedirs('data', exist_ok=True)
    df_feat.to_csv('data/cloud_performance_features.csv', index=False)

    reg_data = create_splits(df_feat, target='execution_time')
    clf_data = create_splits(df_feat, target='task_status_encoded')

    print("\n  ✅ Stage 1 Complete — Dataset saved to data/cloud_performance_features.csv")