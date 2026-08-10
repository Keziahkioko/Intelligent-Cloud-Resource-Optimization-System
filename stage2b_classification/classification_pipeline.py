# =============================================================
# STAGE 2B — CLASSIFICATION PIPELINE
# File: stage2b_classification/classification_pipeline.py
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Imports
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# =============================================================
# 2B.1  MODEL ZOO — All Classification Models
# =============================================================

def build_classification_models():
    return {
        # ── Discriminative Models (Week 3, 4) ────────────────
        'Logistic Regression':   LogisticRegression(
                                     max_iter=500, random_state=42, n_jobs=-1),
        'LDA (Generative)':      LinearDiscriminantAnalysis(),
        'QDA (Generative)':      QuadraticDiscriminantAnalysis(reg_param=0.1),
        'Naïve Bayes':           GaussianNB(),

        # ── Tree-Based (Week 6) ───────────────────────────────
        'Decision Tree':         DecisionTreeClassifier(
                                     max_depth=8, random_state=42),
        'Random Forest':         RandomForestClassifier(
                                     n_estimators=100, random_state=42, n_jobs=-1),
        
        # ── Scalable SVM Approximation (Week 8) ───────────────
        # Replaced standard SVC with SGDClassifier for 2M rows scalability
        # 'modified_huber' loss allows for probability estimates (predict_proba)
        'SVM (SGD Approx)':      SGDClassifier(
                                     loss='modified_huber', max_iter=1000, 
                                     random_state=42, n_jobs=-1)
    }

# =============================================================
# 2B.2  TRAIN & EVALUATE ALL CLASSIFIERS
# =============================================================

def train_and_evaluate_classification(data: dict) -> pd.DataFrame:

    X_tr, y_tr = data['X_train'], data['y_train']
    X_va, y_va = data['X_val'],   data['y_val']
    X_te, y_te = data['X_test'],  data['y_test']

    models  = build_classification_models()
    records = []

    print("=" * 145)
    print(" " * 35 + "STAGE 2B — Classification Model Comparison")
    print("=" * 145)

    print(
        f"\n{'Model':<24}"
        f"{'Train':>10}"
        f"{'Val':>10}"
        f"{'Test':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'ROC-AUC':>12}"
        f"{'CV Mean':>14}"
    )

    print("-" * 145)

    for name, model in models.items():

        try:
            model.fit(X_tr, y_tr)

            train_acc = model.score(X_tr, y_tr)
            val_acc   = model.score(X_va, y_va)
            test_acc  = model.score(X_te, y_te)

            y_pred = model.predict(X_te)

            precision = precision_score(
                y_te, y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_te, y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_te,
                y_pred,
                average="weighted"
            )

            try:
                y_prob = model.predict_proba(X_te)
                auc = roc_auc_score(
                    y_te,
                    y_prob,
                    multi_class="ovr",
                    average="weighted"
                )
            except Exception:
                auc = np.nan

            # Added n_jobs=-1 to utilize all 32 CPUs for cross-validation
            cv_scores = cross_val_score(
                model,
                X_tr,
                y_tr,
                cv=5,
                scoring="accuracy",
                n_jobs=-1
            )

            print(
                f"{name:<24}"
                f"{train_acc:>10.4f}"
                f"{val_acc:>10.4f}"
                f"{test_acc:>10.4f}"
                f"{precision:>12.4f}"
                f"{recall:>10.4f}"
                f"{f1:>10.4f}"
                f"{auc:>12.4f}"
                f"{cv_scores.mean():>10.4f} ± {cv_scores.std():.3f}"
            )

            records.append({
                "Model": name,
                "Train Acc": train_acc,
                "Val Acc": val_acc,
                "Test Acc": test_acc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "ROC-AUC": auc,
                "CV Mean": cv_scores.mean(),
                "CV Std": cv_scores.std(),
                "Overfit": train_acc - test_acc,
                "fitted": model
            })

        except Exception as e:
            print(f"{name:<24} FAILED -> {e}")

    return pd.DataFrame(records).sort_values('Test Acc', ascending=False)

# =============================================================
# 2B.3  GENERATIVE vs DISCRIMINATIVE DEEP DIVE (Week 4)
# =============================================================

def generative_vs_discriminative_analysis(data: dict) -> None:
    """Detailed comparison with learning curves."""

    gen_models  = {
        'Naïve Bayes':  GaussianNB(),
        'LDA':          LinearDiscriminantAnalysis(),
        'QDA':          QuadraticDiscriminantAnalysis(),
    }
    
    # Updated discriminative models for 2M row scalability
    disc_models = {
        'Logistic Reg': LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
        'SVM (SGD)':    SGDClassifier(loss='modified_huber', random_state=42, n_jobs=-1),
        'Random Forest':RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    }

    X_all = np.vstack([data['X_train'], data['X_val']])
    y_all = np.concatenate([data['y_train'], data['y_val']])

    train_sizes = np.linspace(0.1, 1.0, 8)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    all_models = {**{'[G] ' + k: v for k, v in gen_models.items()},
                  **{'[D] ' + k: v for k, v in disc_models.items()}}

    for ax, (name, model) in zip(axes.flatten(), all_models.items()):
        # n_jobs=-1 is already set here, which is perfect for 32 CPUs
        train_sz, train_sc, val_sc = learning_curve(
            model, X_all, y_all,
            train_sizes=train_sizes,
            cv=5, scoring='accuracy',
            n_jobs=-1
        )
        train_mean = train_sc.mean(axis=1)
        train_std  = train_sc.std(axis=1)
        val_mean   = val_sc.mean(axis=1)
        val_std    = val_sc.std(axis=1)

        tag   = '🔴 Generative' if name.startswith('[G]') \
                else '🔵 Discriminative'
        color = '#E74C3C' if name.startswith('[G]') else '#3498DB'

        ax.plot(train_sz, train_mean, 'o-', color=color,
                linewidth=2, label='Train', markersize=5)
        ax.fill_between(train_sz,
                         train_mean - train_std,
                         train_mean + train_std,
                         alpha=0.15, color=color)
        ax.plot(train_sz, val_mean, 's--', color=color,
                linewidth=2, label='Val', alpha=0.7, markersize=5)
        ax.fill_between(train_sz,
                         val_mean - val_std,
                         val_mean + val_std,
                         alpha=0.1, color=color)

        final_acc = val_mean[-1]
        ax.set_title(f"{name}\n{tag}  |  Val Acc={final_acc:.3f}",
                      fontweight='bold', fontsize=10)
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=8)

    plt.suptitle(
        "Generative vs Discriminative — Learning Curves on Cloud Data",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("stage2b_gen_vs_disc.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nLearning Curve Interpretation:")
    print("- Small gap between training and validation curves indicates good generalisation.")
    print("- Large gap suggests overfitting.")
    print("- Both curves converging at a low accuracy may indicate underfitting.")

# =============================================================
# 2B.4  CONFUSION MATRIX DASHBOARD
# =============================================================

def plot_confusion_matrices(results_df: pd.DataFrame,
                             data:        dict) -> None:
    """Plot confusion matrices for top 6 models."""

    top6   = results_df.head(6)
    labels = np.unique(data['y_test'])
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for ax, (_, row) in zip(axes.flatten(), top6.iterrows()):
        model = row['fitted']
        y_pred = model.predict(data['X_test'])
        cm     = confusion_matrix(data['y_test'], y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                    ax=ax, linewidths=0.5,
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': '% of True Class'})
        ax.set_title(f"{row['Model']}\nTest Acc={row['Test Acc']:.4f}  "
                     f"F1={row['F1']:.4f}",
                     fontweight='bold', fontsize=10)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.suptitle(
        "Top 6 Models — Confusion Matrices (Task Status Classification)",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("stage2b_confusion_matrices.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  ✅ Stage 2B Complete")
# =============================================================
# MAIN PROGRAM
# =============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("INTELLIGENT CLOUD RESOURCE OPTIMIZATION SYSTEM")
    print("STAGE 2B - CLASSIFICATION PIPELINE")
    print("=" * 80)
    print("\nLoading cleaned dataset...")

    # Updated data path for Lightning AI Studio
    clean_data_path = '/teamspace/studios/this_studio/cloud_performance_features.csv'
    df = pd.read_csv(clean_data_path)

    print(f"Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
   
    feature_columns = [
        "cpu_usage",
        "memory_usage",
        "network_traffic",
        "power_consumption",
        "num_executed_instructions",
        "execution_time",
        "energy_efficiency",
        "cpu_memory_ratio",
        "power_per_instruction",
        "throughput",
        "system_load_index",
        "cpu_power_interaction",
        "task_type_encoded",
        "task_priority_encoded",
        "cpu_load_level_encoded",
        "exec_time_bucket_encoded"
    ]

    X = df[feature_columns]
    print("\nFeatures being used:")
    print(X.columns.tolist())
    print(f"\nNumber of features: {X.shape[1]}")
    y = df["task_status_encoded"]
    
    print("\nOverall Class Distribution")
    print(y.value_counts())
    print(y.value_counts(normalize=True))
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
    
    print("\nTraining Set Class Distribution")
    print(y_train.value_counts(normalize=True))
    
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }
    
    print("\nTraining classification models...")
    results_df = train_and_evaluate_classification(data)

    print("\nModel Ranking")
    print(results_df.drop(columns=["fitted"]))

    plot_confusion_matrices(results_df, data)

    generative_vs_discriminative_analysis(data)

    results_df.drop(columns=["fitted"]).to_csv(
        "stage2b_classification_results.csv",
        index=False
    )

    print("\nResults saved to stage2b_classification_results.csv")