"""
Model Training and Cross-Validation Pipeline
Project: Predicting Delivery Delays in Last-Mile Logistics (ABC Logistics)
Author: Senior Business Analytics Consultant / Data Analyst
Phase: Phase 3 - Machine Learning Model Development
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV

from src.data_preparation import (
    get_linear_preprocessor,
    get_tree_preprocessor,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
TRAIN_DATA_PATH = os.path.join("data", "processed", "train.csv")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
RANDOM_SEED = 42
CV_FOLDS = 5

SCORING_METRICS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ==============================================================================
# STEP 1: LOAD & VERIFY TRAINING PARTITION ONLY
# ==============================================================================
def load_training_data(filepath: str = TRAIN_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads strictly the training partition (train.csv).
    Guarantees the holdout test set is NEVER loaded or touched during Phase 3.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target '{TARGET_COLUMN}' missing from {filepath}")
    
    X_train = df[FEATURE_COLUMNS].copy()
    y_train = df[TARGET_COLUMN].copy()

    # Integrity assertions
    assert TARGET_COLUMN not in X_train.columns, "Leakage Alert: Target in feature matrix!"
    assert len(X_train) == 800, f"Expected 800 training rows, found {len(X_train)}"
    assert len(X_train.columns) == 11, f"Expected 11 features, found {len(X_train.columns)}"
    assert set(y_train.unique()) == {0, 1}, f"Unexpected target values: {set(y_train.unique())}"

    return X_train, y_train


# ==============================================================================
# STEP 2: BUILD CANDIDATE LEAKAGE-SAFE PIPELINES
# ==============================================================================
def build_baseline_pipelines() -> Dict[str, Pipeline]:
    """
    Constructs candidate model pipelines combining preprocessing and classification.
    - Logistic Regression uses StandardScaler on numeric/ordinal & OneHotEncoder (drop='first').
    - Tree models use passthrough on numeric/ordinal & OneHotEncoder (all levels).
    """
    pipelines = {
        "Logistic_Regression": Pipeline([
            ("preprocessor", get_linear_preprocessor(drop_first=True)),
            ("classifier", LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, C=1.0))
        ]),
        "Decision_Tree": Pipeline([
            ("preprocessor", get_tree_preprocessor()),
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=4, min_samples_leaf=5))
        ]),
        "Random_Forest": Pipeline([
            ("preprocessor", get_tree_preprocessor()),
            ("classifier", RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100, max_depth=6, min_samples_leaf=3))
        ]),
        "Gradient_Boosting": Pipeline([
            ("preprocessor", get_tree_preprocessor()),
            ("classifier", GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=100, max_depth=3, learning_rate=0.1))
        ])
    }
    return pipelines


# ==============================================================================
# STEP 3: STRATIFIED 5-FOLD CROSS-VALIDATION EVALUATION
# ==============================================================================
def evaluate_pipelines_cv(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluates candidate pipelines using Stratified 5-Fold Cross-Validation.
    Computes mean and standard deviation for Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    summary_rows = []
    raw_cv_results = {}

    for name, pipe in pipelines.items():
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING_METRICS,
            return_train_score=True,
            n_jobs=1
        )
        raw_cv_results[name] = scores

        summary_rows.append({
            "Model": name,
            "CV_Accuracy_Mean": float(np.mean(scores["test_accuracy"])),
            "CV_Accuracy_Std": float(np.std(scores["test_accuracy"])),
            "CV_Precision_Mean": float(np.mean(scores["test_precision"])),
            "CV_Precision_Std": float(np.std(scores["test_precision"])),
            "CV_Recall_Mean": float(np.mean(scores["test_recall"])),
            "CV_Recall_Std": float(np.std(scores["test_recall"])),
            "CV_F1_Mean": float(np.mean(scores["test_f1"])),
            "CV_F1_Std": float(np.std(scores["test_f1"])),
            "CV_ROC_AUC_Mean": float(np.mean(scores["test_roc_auc"])),
            "CV_ROC_AUC_Std": float(np.std(scores["test_roc_auc"])),
            "Train_F1_Mean": float(np.mean(scores["train_f1"])),
            "Train_F1_Std": float(np.std(scores["train_f1"]))
        })

    comparison_df = pd.DataFrame(summary_rows).sort_values(by="CV_F1_Mean", ascending=False)
    return comparison_df, raw_cv_results


# ==============================================================================
# STEP 4: SYSTEMATIC HYPERPARAMETER TUNING (ON TRAINING DATA ONLY)
# ==============================================================================
def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Runs systematic Grid Search on the training set using Stratified 5-Fold CV.
    Tunes regularization / depth / estimators for each candidate architecture.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    tuned_models = {}

    # 1. Logistic Regression Grid
    lr_pipe = Pipeline([
        ("preprocessor", get_linear_preprocessor(drop_first=True)),
        ("classifier", LogisticRegression(random_state=random_state, max_iter=1000))
    ])
    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0]
    }
    lr_grid = GridSearchCV(lr_pipe, lr_param_grid, cv=cv, scoring="f1", n_jobs=1)
    lr_grid.fit(X_train, y_train)
    tuned_models["Logistic_Regression"] = lr_grid

    # 2. Decision Tree Grid
    dt_pipe = Pipeline([
        ("preprocessor", get_tree_preprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=random_state))
    ])
    dt_param_grid = {
        "classifier__max_depth": [3, 4, 5, 6, 7],
        "classifier__min_samples_leaf": [2, 5, 10, 15],
        "classifier__criterion": ["gini", "entropy"]
    }
    dt_grid = GridSearchCV(dt_pipe, dt_param_grid, cv=cv, scoring="f1", n_jobs=1)
    dt_grid.fit(X_train, y_train)
    tuned_models["Decision_Tree"] = dt_grid

    # 3. Random Forest Grid
    rf_pipe = Pipeline([
        ("preprocessor", get_tree_preprocessor()),
        ("classifier", RandomForestClassifier(random_state=random_state))
    ])
    rf_param_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__max_depth": [4, 6, 8, None],
        "classifier__min_samples_leaf": [2, 4, 8]
    }
    rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=cv, scoring="f1", n_jobs=1)
    rf_grid.fit(X_train, y_train)
    tuned_models["Random_Forest"] = rf_grid

    # 4. Gradient Boosting Grid
    gb_pipe = Pipeline([
        ("preprocessor", get_tree_preprocessor()),
        ("classifier", GradientBoostingClassifier(random_state=random_state))
    ])
    gb_param_grid = {
        "classifier__n_estimators": [50, 100, 150],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 4, 5]
    }
    gb_grid = GridSearchCV(gb_pipe, gb_param_grid, cv=cv, scoring="f1", n_jobs=1)
    gb_grid.fit(X_train, y_train)
    tuned_models["Gradient_Boosting"] = gb_grid

    return tuned_models


# ==============================================================================
# STEP 5: INTERPRETABILITY & FEATURE IMPORTANCE EXTRACTION
# ==============================================================================
def extract_model_interpretability(
    tuned_models: Dict[str, GridSearchCV],
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Extracts:
      1. Logistic Regression standardized coefficients and odds ratios.
      2. Tree-based feature importances (Decision Tree, Random Forest, Gradient Boosting).
      3. Textual decision tree rules representation.
    """
    # 1. Logistic Regression Coefficients
    best_lr = tuned_models["Logistic_Regression"].best_estimator_
    lr_preprocessor = best_lr.named_steps["preprocessor"]
    lr_classifier = best_lr.named_steps["classifier"]
    lr_feature_names = lr_preprocessor.get_feature_names_out()
    lr_coefs = lr_classifier.coef_[0]
    lr_intercept = lr_classifier.intercept_[0]

    lr_coef_df = pd.DataFrame({
        "Feature": lr_feature_names,
        "Coefficient (Log-Odds)": lr_coefs,
        "Odds_Ratio": np.exp(lr_coefs),
        "Abs_Coefficient": np.abs(lr_coefs)
    }).sort_values(by="Abs_Coefficient", ascending=False).drop(columns=["Abs_Coefficient"])
    lr_coef_df["Intercept"] = lr_intercept

    # 2. Tree-Based Feature Importances
    best_dt = tuned_models["Decision_Tree"].best_estimator_
    best_rf = tuned_models["Random_Forest"].best_estimator_
    best_gb = tuned_models["Gradient_Boosting"].best_estimator_

    tree_preprocessor = best_dt.named_steps["preprocessor"]
    tree_feature_names = tree_preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": tree_feature_names,
        "Decision_Tree_Importance": best_dt.named_steps["classifier"].feature_importances_,
        "Random_Forest_Importance": best_rf.named_steps["classifier"].feature_importances_,
        "Gradient_Boosting_Importance": best_gb.named_steps["classifier"].feature_importances_
    })
    importance_df["Mean_Ensemble_Importance"] = (
        importance_df["Random_Forest_Importance"] + importance_df["Gradient_Boosting_Importance"]
    ) / 2
    importance_df = importance_df.sort_values(by="Mean_Ensemble_Importance", ascending=False)

    # 3. Decision Tree Text Export
    dt_classifier = best_dt.named_steps["classifier"]
    tree_rules_text = export_text(
        dt_classifier,
        feature_names=list(tree_feature_names),
        decimals=2
    )

    return lr_coef_df, importance_df, tree_rules_text


# ==============================================================================
# STEP 6: SAVE ARTIFACTS AND WORKFLOW RUNNER
# ==============================================================================
def run_model_development_workflow() -> Dict[str, Any]:
    """
    Executes full Phase 3 model development pipeline on training data only.
    Saves outputs to models/ and outputs/ directories.
    """
    print("=" * 80)
    print("ABC LOGISTICS — PHASE 3 MACHINE LEARNING MODEL DEVELOPMENT")
    print("=" * 80)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Load Data
    print("\n[Step 1] Loading training partition strictly from data/processed/train.csv...")
    X_train, y_train = load_training_data(TRAIN_DATA_PATH)
    print(f"  -> Training records: {len(X_train)} rows x {len(X_train.columns)} features")
    print(f"  -> Target distribution: Delayed (1): {y_train.sum()} ({y_train.mean()*100:.2f}%), On-Time (0): {(y_train==0).sum()} ({(y_train==0).mean()*100:.2f}%)")

    # 2. Baseline Models CV
    print("\n[Step 2] Evaluating candidate baseline pipelines via Stratified 5-Fold CV...")
    base_pipelines = build_baseline_pipelines()
    base_cv_df, _ = evaluate_pipelines_cv(base_pipelines, X_train, y_train)
    print("\n--- Baseline 5-Fold Cross-Validation Performance ---")
    for _, row in base_cv_df.iterrows():
        print(f"  {row['Model']:<22} | Acc: {row['CV_Accuracy_Mean']:.4f} (+/- {row['CV_Accuracy_Std']:.4f}) | Recall: {row['CV_Recall_Mean']:.4f} | F1: {row['CV_F1_Mean']:.4f} | ROC-AUC: {row['CV_ROC_AUC_Mean']:.4f}")

    # 3. Hyperparameter Tuning
    print("\n[Step 3] Running systematic GridSearchCV for each candidate model (5-fold CV)...")
    tuned_models = tune_hyperparameters(X_train, y_train)

    tuned_summary = []
    for name, grid in tuned_models.items():
        best_pipe = grid.best_estimator_
        # Re-evaluate best estimator using 5-fold CV with all metrics
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_validate(best_pipe, X_train, y_train, cv=cv, scoring=SCORING_METRICS, return_train_score=True)
        
        tuned_summary.append({
            "Model": f"{name} (Tuned)",
            "Best_Parameters": str(grid.best_params_),
            "CV_Accuracy_Mean": float(np.mean(scores["test_accuracy"])),
            "CV_Accuracy_Std": float(np.std(scores["test_accuracy"])),
            "CV_Precision_Mean": float(np.mean(scores["test_precision"])),
            "CV_Precision_Std": float(np.std(scores["test_precision"])),
            "CV_Recall_Mean": float(np.mean(scores["test_recall"])),
            "CV_Recall_Std": float(np.std(scores["test_recall"])),
            "CV_F1_Mean": float(np.mean(scores["test_f1"])),
            "CV_F1_Std": float(np.std(scores["test_f1"])),
            "CV_ROC_AUC_Mean": float(np.mean(scores["test_roc_auc"])),
            "CV_ROC_AUC_Std": float(np.std(scores["test_roc_auc"])),
            "Train_F1_Mean": float(np.mean(scores["train_f1"])),
            "Train_F1_Std": float(np.std(scores["train_f1"]))
        })
        print(f"  -> {name:<22}: Best Params: {grid.best_params_} | CV F1 = {np.mean(scores['test_f1']):.4f}")

    tuned_cv_df = pd.DataFrame(tuned_summary).sort_values(by="CV_F1_Mean", ascending=False)
    
    # Save CV comparison
    cv_csv_path = os.path.join(OUTPUTS_DIR, "cv_model_comparison.csv")
    full_comparison_df = pd.concat([base_cv_df, tuned_cv_df], ignore_index=True)
    full_comparison_df.to_csv(cv_csv_path, index=False)
    print(f"\n[Step 4] Saved cross-validation comparison to: {cv_csv_path}")

    # 4. Interpretability & Feature Importances
    print("\n[Step 5] Extracting interpretability artifacts (coefficients, importances, tree rules)...")
    lr_coef_df, importance_df, tree_rules_text = extract_model_interpretability(tuned_models, X_train, y_train)

    coef_csv_path = os.path.join(OUTPUTS_DIR, "logistic_coefficients.csv")
    lr_coef_df.to_csv(coef_csv_path, index=False)
    print(f"  -> Logistic coefficients saved to: {coef_csv_path}")

    imp_csv_path = os.path.join(OUTPUTS_DIR, "feature_importance.csv")
    importance_df.to_csv(imp_csv_path, index=False)
    print(f"  -> Tree feature importances saved to: {imp_csv_path}")

    tree_txt_path = os.path.join(OUTPUTS_DIR, "decision_tree.txt")
    with open(tree_txt_path, "w", encoding="utf-8") as f:
        f.write("=== ABC LOGISTICS: TUNED DECISION TREE RULES (TRAINING SET ONLY) ===\n\n")
        f.write(tree_rules_text)
    print(f"  -> Decision tree rules exported to: {tree_txt_path}")

    # 5. Fit Final Pipelines on Full 800-Row Training Set & Serialize
    print("\n[Step 6] Fitting final tuned pipelines on full training set (N=800) and serializing to models/...")
    serialized_paths = {}
    for name, grid in tuned_models.items():
        best_pipe = grid.best_estimator_
        # Fit on full X_train, y_train
        best_pipe.fit(X_train, y_train)
        model_filename = f"{name.lower()}_pipeline.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(best_pipe, model_path)
        serialized_paths[name] = model_path
        print(f"  -> Saved {name} pipeline ({os.path.getsize(model_path)} bytes) to: {model_path}")

    print("\n" + "=" * 80)
    print("PHASE 3 MODEL DEVELOPMENT COMPLETE — ARTIFACTS READY FOR REVIEW")
    print("=" * 80)

    return {
        "comparison_table": full_comparison_df,
        "best_models": tuned_models,
        "feature_importances": importance_df,
        "logistic_coefficients": lr_coef_df,
        "tree_rules": tree_rules_text
    }


if __name__ == "__main__":
    run_model_development_workflow()
