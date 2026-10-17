"""
Model Evaluation Pipeline on Held-Out Test Set
Project: Predicting Delivery Delays in Last-Mile Logistics (ABC Logistics)
Author: Senior Business Analytics Consultant / Data Analyst
Phase: Phase 4 - Final Model Evaluation on Held-Out Test Set
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.data_preparation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN
)

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
TEST_DATA_PATH = os.path.join("data", "processed", "test.csv")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
CLASSIFICATION_THRESHOLD = 0.50

MODEL_PATHS = {
    "Logistic_Regression": os.path.join(MODELS_DIR, "logistic_regression_pipeline.joblib"),
    "Decision_Tree": os.path.join(MODELS_DIR, "decision_tree_pipeline.joblib"),
    "Random_Forest": os.path.join(MODELS_DIR, "random_forest_pipeline.joblib"),
    "Gradient_Boosting": os.path.join(MODELS_DIR, "gradient_boosting_pipeline.joblib")
}


# ==============================================================================
# STEP 1: VERIFY & LOAD TEST DATA AND SAVED MODELS
# ==============================================================================
def load_and_verify_test_data(filepath: str = TEST_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads and strictly verifies the 200-row held-out test partition.
    Guarantees no records are modified and the target is isolated.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Held-out test set not found at: {filepath}")

    test_df = pd.read_csv(filepath)

    # Verification checks
    assert len(test_df) == 200, f"Expected 200 test rows, found {len(test_df)}"
    assert TARGET_COLUMN in test_df.columns, f"Target column '{TARGET_COLUMN}' missing!"
    assert all(col in test_df.columns for col in FEATURE_COLUMNS), "Missing feature columns in test set!"
    
    counts = test_df[TARGET_COLUMN].value_counts().to_dict()
    assert counts.get(1, 0) == 102, f"Expected 102 delayed records, got {counts.get(1, 0)}"
    assert counts.get(0, 0) == 98, f"Expected 98 on-time records, got {counts.get(0, 0)}"

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    assert TARGET_COLUMN not in X_test.columns, "Leakage Alert: Target remains in feature matrix!"
    return X_test, y_test


def load_and_verify_saved_models(model_paths: Dict[str, str] = MODEL_PATHS) -> Dict[str, Any]:
    """
    Loads and verifies that each serialized model pipeline exists and contains expected steps.
    """
    loaded_models = {}
    for name, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Serialized model file missing: {path}")
        pipe = joblib.load(path)
        step_names = [s[0] for s in pipe.steps]
        assert "preprocessor" in step_names, f"Model {name} missing preprocessor step!"
        assert "classifier" in step_names, f"Model {name} missing classifier step!"
        loaded_models[name] = pipe
    return loaded_models


# ==============================================================================
# STEP 2: EVALUATE CANDIDATE MODELS ON TEST SET
# ==============================================================================
def evaluate_models_on_test_set(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = CLASSIFICATION_THRESHOLD
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes a single-pass evaluation of candidate models on the held-out test set.
    Computes Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, and Confusion Matrix.
    """
    metrics_rows = []
    cm_rows = []

    for name, pipe in models.items():
        # Predict class probabilities / scores
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
            roc_auc = float(roc_auc_score(y_test, y_prob))
        elif hasattr(pipe, "decision_function"):
            decision_scores = pipe.decision_function(X_test)
            y_pred = pipe.predict(X_test)
            roc_auc = float(roc_auc_score(y_test, decision_scores))
        else:
            y_pred = pipe.predict(X_test)
            roc_auc = float(accuracy_score(y_test, y_pred))

        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        metrics_rows.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "ROC_AUC": roc_auc,
            "Specificity": spec,
            "True_Positives": tp,
            "False_Negatives": fn,
            "True_Negatives": tn,
            "False_Positives": fp,
            "Threshold": threshold
        })

        cm_rows.append({
            "Model": name,
            "Actual_OnTime_Predicted_OnTime (TN)": tn,
            "Actual_OnTime_Predicted_Delayed (FP)": fp,
            "Actual_Delayed_Predicted_OnTime (FN)": fn,
            "Actual_Delayed_Predicted_Delayed (TP)": tp,
            "Total_Test_Records": len(y_test)
        })

    metrics_df = pd.DataFrame(metrics_rows).sort_values(by="F1_Score", ascending=False)
    cm_df = pd.DataFrame(cm_rows)
    return metrics_df, cm_df


# ==============================================================================
# STEP 3: EVALUATE THRESHOLD-RULE BENCHMARK SEPARATELY
# ==============================================================================
def evaluate_threshold_benchmark(X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates the Phase 1 operational threshold rules as a descriptive analytical benchmark.
    This benchmark is NOT a trained machine learning model.
    """
    cond1 = X_test["Warehouse_Processing_Time"] >= 91
    cond2 = X_test["Vehicle_Age"] >= 8
    cond3 = (X_test["Traffic_Congestion"] >= 4) & (X_test["Weather_Condition"] >= 2) & (X_test["Delivery_Distance"] > 20.0)

    rule_pred = (cond1 | cond2 | cond3).astype(int)

    acc = float(accuracy_score(y_test, rule_pred))
    prec = float(precision_score(y_test, rule_pred, pos_label=1))
    rec = float(recall_score(y_test, rule_pred, pos_label=1))
    f1 = float(f1_score(y_test, rule_pred, pos_label=1))
    cm = confusion_matrix(y_test, rule_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    return {
        "Benchmark_Name": "Operational_Threshold_Rules_Benchmark",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "True_Positives": tp,
        "False_Negatives": fn,
        "True_Negatives": tn,
        "False_Positives": fp,
        "Mode_1_Warehouse_Count": int(cond1.sum()),
        "Mode_2_Vehicle_Age_Count": int(cond2.sum()),
        "Mode_3_Compound_Transit_Count": int(cond3.sum())
    }


# ==============================================================================
# STEP 4: WORKFLOW RUNNER & ARTIFACT EXPORT
# ==============================================================================
def run_evaluation_workflow() -> Dict[str, Any]:
    """
    Orchestrates the entire Phase 4 evaluation pipeline.
    """
    print("=" * 80)
    print("ABC LOGISTICS — PHASE 4 FINAL MODEL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 80)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Load and Verify Test Set
    print("\n[Step 1] Loading and verifying held-out test partition (data/processed/test.csv)...")
    X_test, y_test = load_and_verify_test_data(TEST_DATA_PATH)
    print(f"  -> Total test records: {len(X_test)}")
    print(f"  -> Delayed (1) records: {int(y_test.sum())} ({y_test.mean()*100:.2f}%)")
    print(f"  -> On-Time (0) records: {int((y_test==0).sum())} ({(y_test==0).mean()*100:.2f}%)")
    print(f"  -> Features verified: {list(X_test.columns)}")

    # 2. Load and Verify Models
    print("\n[Step 2] Loading and verifying pre-specified trained model pipelines...")
    models = load_and_verify_saved_models(MODEL_PATHS)
    for name, pipe in models.items():
        print(f"  -> Loaded {name:<22}: {pipe.named_steps['classifier'].__class__.__name__}")

    # 3. Evaluate Models Once on Test Set
    print("\n[Step 3] Evaluating models once on test set (Threshold = 0.50)...")
    metrics_df, cm_df = evaluate_models_on_test_set(models, X_test, y_test, threshold=CLASSIFICATION_THRESHOLD)

    print("\n--- Final Test Set Model Performance ---")
    for _, row in metrics_df.iterrows():
        print(f"  {row['Model']:<22} | Acc: {row['Accuracy']:.4f} | Prec: {row['Precision']:.4f} | Rec: {row['Recall']:.4f} | F1: {row['F1_Score']:.4f} | ROC-AUC: {row['ROC_AUC']:.4f}")

    print("\n--- Confusion Matrix Breakdown ---")
    for _, row in cm_df.iterrows():
        print(f"  {row['Model']:<22} | TP: {row['Actual_Delayed_Predicted_Delayed (TP)']:>3} | FN: {row['Actual_Delayed_Predicted_OnTime (FN)']:>3} | TN: {row['Actual_OnTime_Predicted_OnTime (TN)']:>3} | FP: {row['Actual_OnTime_Predicted_Delayed (FP)']:>3}")

    # 4. Evaluate Threshold Benchmark
    print("\n[Step 4] Evaluating descriptive threshold-rules benchmark separately...")
    bench = evaluate_threshold_benchmark(X_test, y_test)
    print(f"  -> Benchmark F1: {bench['F1_Score']:.4f} | Acc: {bench['Accuracy']:.4f} | TP: {bench['True_Positives']} | FN: {bench['False_Negatives']} | FP: {bench['False_Positives']} | TN: {bench['True_Negatives']}")

    # 5. Export Deliverables
    metrics_csv_path = os.path.join(OUTPUTS_DIR, "test_model_comparison.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n[Step 5] Saved test metrics comparison to: {metrics_csv_path}")

    cm_csv_path = os.path.join(OUTPUTS_DIR, "test_confusion_matrices.csv")
    cm_df.to_csv(cm_csv_path, index=False)
    print(f"  -> Saved test confusion matrices to: {cm_csv_path}")

    print("\n" + "=" * 80)
    print("PHASE 4 FINAL EVALUATION COMPLETE — RESULTS VERIFIED")
    print("=" * 80)

    return {
        "metrics_df": metrics_df,
        "cm_df": cm_df,
        "benchmark": bench
    }


if __name__ == "__main__":
    run_evaluation_workflow()
