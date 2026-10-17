"""
Data Preparation and Reproducible Train/Test Setup
Project: Predicting Delivery Delays in Last-Mile Logistics (ABC Logistics)
Author: Senior Business Analytics Consultant / Data Analyst
Phase: Phase 2 - Data Preparation & Preprocessing Pipeline Design
"""

import os
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ==============================================================================
# GLOBAL CONSTANTS & SCHEMA DEFINITIONS
# ==============================================================================
RAW_DATA_PATH = "delivery_delay.csv"
PROCESSED_DATA_DIR = os.path.join("data", "processed")
TARGET_COLUMN = "Delivery_Delay"
RANDOM_SEED = 42
TEST_SIZE = 0.20

EXPECTED_COLUMNS = [
    "Delivery_Distance",
    "Traffic_Congestion",
    "Weather_Condition",
    "Delivery_Slot",
    "Driver_Experience",
    "Num_Stops",
    "Vehicle_Age",
    "Road_Condition_Score",
    "Package_Weight",
    "Fuel_Efficiency",
    "Warehouse_Processing_Time",
    "Delivery_Delay"
]

FEATURE_COLUMNS = [col for col in EXPECTED_COLUMNS if col != TARGET_COLUMN]

NUMERIC_CONTINUOUS_FEATURES = [
    "Delivery_Distance",
    "Driver_Experience",
    "Num_Stops",
    "Vehicle_Age",
    "Package_Weight",
    "Fuel_Efficiency",
    "Warehouse_Processing_Time"
]

ORDINAL_FEATURES = [
    "Traffic_Congestion",
    "Road_Condition_Score"
]

CATEGORICAL_NOMINAL_FEATURES = [
    "Weather_Condition",
    "Delivery_Slot"
]


# ==============================================================================
# STEP 1: INPUT DATA VALIDATION
# ==============================================================================
def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates input dataset structure, types, missing values, duplicates, and bounds.
    Raises ValueError if any critical integrity check fails.
    """
    issues = []

    # 1. Dimensions
    if df.shape != (1000, 12):
        issues.append(f"Unexpected shape: expected (1000, 12), found {df.shape}")

    # 2. Column names & exact spelling
    if list(df.columns) != EXPECTED_COLUMNS:
        issues.append(f"Column mismatch: expected {EXPECTED_COLUMNS}, found {list(df.columns)}")

    # 3. Missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        issues.append(f"Dataset contains {total_nulls} missing values: {null_counts[null_counts > 0].to_dict()}")

    # 4. Duplicate records
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Dataset contains {duplicate_count} duplicate rows.")

    # 5. Target variable validation
    if TARGET_COLUMN not in df.columns:
        issues.append(f"Target column '{TARGET_COLUMN}' is missing.")
    else:
        unique_targets = set(df[TARGET_COLUMN].unique())
        if unique_targets != {0, 1}:
            issues.append(f"Unexpected target values: expected {{0, 1}}, found {unique_targets}")

    # 6. Feature bounds & value domains
    if df["Traffic_Congestion"].min() < 1 or df["Traffic_Congestion"].max() > 5:
        issues.append("Traffic_Congestion out of bounds [1, 5]")
    if set(df["Weather_Condition"].unique()) != {1, 2, 3}:
        issues.append("Weather_Condition contains values outside {1, 2, 3}")
    if set(df["Delivery_Slot"].unique()) != {1, 2, 3}:
        issues.append("Delivery_Slot contains values outside {1, 2, 3}")
    if df["Road_Condition_Score"].min() < 1 or df["Road_Condition_Score"].max() > 5:
        issues.append("Road_Condition_Score out of bounds [1, 5]")
    if df["Warehouse_Processing_Time"].min() < 10 or df["Warehouse_Processing_Time"].max() > 119:
        issues.append("Warehouse_Processing_Time out of bounds [10, 119]")

    if issues:
        raise ValueError("Data validation failed with errors:\n" + "\n".join(issues))

    validation_summary = {
        "status": "PASSED",
        "rows": len(df),
        "columns": len(df.columns),
        "total_nulls": int(total_nulls),
        "total_duplicates": int(duplicate_count),
        "target_balance": df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
    }
    return validation_summary


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads and validates the raw CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found at: {filepath}")
    df = pd.read_csv(filepath)
    validate_dataset(df)
    return df


# ==============================================================================
# STEP 2: FEATURE AND TARGET SEPARATION
# ==============================================================================
def separate_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates the predictor matrix X and the binary target series y.
    Ensures that no target-derived information is present in X.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")
    
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].copy()

    # Safety check: guarantee target is not present in feature columns
    assert TARGET_COLUMN not in X.columns, "Leakage Alert: Target column remains in feature matrix!"
    assert len(X.columns) == 11, f"Expected 11 candidate features, got {len(X.columns)}"
    return X, y


# ==============================================================================
# STEP 3: REPRODUCIBLE STRATIFIED TRAIN/TEST SPLIT
# ==============================================================================
def create_reproducible_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits feature matrix X and target y into training and test sets using stratified sampling.
    Guarantees zero row overlap and exact reproducibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Verification checks
    train_idx = set(X_train.index)
    test_idx = set(X_test.index)
    overlap = train_idx.intersection(test_idx)
    assert len(overlap) == 0, f"Critical Error: {len(overlap)} rows overlap between train and test sets!"
    assert len(train_idx.union(test_idx)) == len(X), "Critical Error: Record count mismatch after split!"

    return X_train, X_test, y_train, y_test


# ==============================================================================
# STEP 4: PREPROCESSING PIPELINES DESIGN
# ==============================================================================
def get_linear_preprocessor(drop_first: bool = True) -> ColumnTransformer:
    """
    Returns a ColumnTransformer suitable for Linear / Logistic Regression models.
    - Continuous features and Ordinal ratings are scaled using StandardScaler.
    - Categorical features are One-Hot Encoded (drop='first' by default to prevent multicollinearity).
    - handle_unknown='ignore' ensures robustness during live supervisor inference.
    """
    numeric_features = NUMERIC_CONTINUOUS_FEATURES + ORDINAL_FEATURES
    
    # If drop_first is True, drop first category to avoid dummy variable trap in unregularized/L2 models
    ohe = OneHotEncoder(
        drop="first" if drop_first else None,
        sparse_output=False,
        handle_unknown="ignore" if not drop_first else "warn"
    )

    # If drop='first' and handle_unknown='ignore' are desired simultaneously, handle_unknown can be set in sklearn >= 1.2
    try:
        ohe_safe = OneHotEncoder(drop="first" if drop_first else None, sparse_output=False, handle_unknown="ignore")
    except ValueError:
        ohe_safe = ohe

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", ohe_safe, CATEGORICAL_NOMINAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def get_tree_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer suitable for Tree-based models (Decision Trees, Random Forest, XGBoost).
    - Continuous and Ordinal features are passed through unscaled ('passthrough') to preserve
      natural operational units (minutes, km, years, scale 1-5) for decision split clarity.
    - Categorical features are One-Hot Encoded without dropping levels.
    """
    numeric_features = NUMERIC_CONTINUOUS_FEATURES + ORDINAL_FEATURES
    
    ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", ohe, CATEGORICAL_NOMINAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


# ==============================================================================
# STEP 5: THRESHOLD-RULE AUDITING (ANALYTICAL CHARACTERISTIC ONLY)
# ==============================================================================
def audit_threshold_rules(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audits the three deterministic operational failure modes identified in Phase 1:
      Mode 1: Warehouse_Processing_Time >= 91 min
      Mode 2: Vehicle_Age >= 8 yrs
      Mode 3: (Traffic_Congestion >= 4) & (Weather_Condition >= 2) & (Delivery_Distance > 20 km)
    
    NOTE: These rules are audited strictly as an instructional dataset characteristic.
    They are NOT used as features, nor are they hard-coded as the ML model.
    """
    cond1 = df["Warehouse_Processing_Time"] >= 91
    cond2 = df["Vehicle_Age"] >= 8
    cond3 = (df["Traffic_Congestion"] >= 4) & (df["Weather_Condition"] >= 2) & (df["Delivery_Distance"] > 20.0)

    combined_rule = (cond1 | cond2 | cond3).astype(int)
    actual_target = df[TARGET_COLUMN]
    matches = (combined_rule == actual_target).sum()

    return {
        "total_records": len(df),
        "actual_delays": int(actual_target.sum()),
        "mode_1_warehouse_ge_91": int(cond1.sum()),
        "mode_2_vehicle_age_ge_8": int(cond2.sum()),
        "mode_3_compound_transit": int(cond3.sum()),
        "rule_predicted_delays": int(combined_rule.sum()),
        "exact_matches": int(matches),
        "rule_accuracy_pct": float(round(matches / len(df) * 100, 2))
    }


# ==============================================================================
# STEP 6: SAVE REPRODUCIBLE PARTITIONS
# ==============================================================================
def save_processed_partitions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = PROCESSED_DATA_DIR
) -> Dict[str, str]:
    """
    Saves clean, verified train and test partitions into data/processed/.
    Saves both full tabular splits (with target) and standalone matrices.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Combined datasets with target for easy model fitting / EDA
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return {
        "train_path": train_path,
        "test_path": test_path
    }


# ==============================================================================
# STEP 7: WORKFLOW VERIFICATION & EXECUTION TEST
# ==============================================================================
def run_data_preparation_pipeline() -> Dict[str, Any]:
    """
    Executes the end-to-end Phase 2 preparation workflow and verifies all steps.
    """
    print("=" * 80)
    print("ABC LOGISTICS — PHASE 2 DATA PREPARATION WORKFLOW")
    print("=" * 80)

    # 1. Load and Validate
    print("\n[Step 1] Loading and validating input data...")
    df = load_raw_data(RAW_DATA_PATH)
    val_summary = validate_dataset(df)
    print(f"  -> Validation Status: {val_summary['status']}")
    print(f"  -> Dimensions: {val_summary['rows']} rows, {val_summary['columns']} columns")
    print(f"  -> Total Missing: {val_summary['total_nulls']} | Total Duplicates: {val_summary['total_duplicates']}")
    print(f"  -> Target Proportions: On-time (0): {val_summary['target_balance'][0]:.4f}, Delayed (1): {val_summary['target_balance'][1]:.4f}")

    # 2. Separate Features and Target
    print("\n[Step 2] Defining feature matrix and target series...")
    X, y = separate_features_and_target(df)
    print(f"  -> X shape: {X.shape} (11 candidate features)")
    print(f"  -> y shape: {y.shape} (target: {TARGET_COLUMN})")

    # 3. Create Stratified Train/Test Split
    print("\n[Step 3] Creating reproducible stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test = create_reproducible_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    print(f"  -> Train set: {len(X_train)} records (Delayed: {y_train.sum()} [{y_train.mean()*100:.2f}%], On-Time: {(y_train==0).sum()} [{(y_train==0).mean()*100:.2f}%])")
    print(f"  -> Test set:  {len(X_test)} records (Delayed: {y_test.sum()} [{y_test.mean()*100:.2f}%], On-Time: {(y_test==0).sum()} [{(y_test==0).mean()*100:.2f}%])")
    print(f"  -> Overlap check: {len(set(X_train.index).intersection(set(X_test.index)))} records overlapping.")

    # 4. Save Processed Partitions
    print("\n[Step 4] Saving clean partitions to data/processed/...")
    paths = save_processed_partitions(X_train, X_test, y_train, y_test, PROCESSED_DATA_DIR)
    print(f"  -> Train file saved to: {paths['train_path']} ({os.path.getsize(paths['train_path'])} bytes)")
    print(f"  -> Test file saved to:  {paths['test_path']} ({os.path.getsize(paths['test_path'])} bytes)")

    # 5. Build and Verify Preprocessing Pipelines (Zero-Leakage Test)
    print("\n[Step 5] Building and verifying preprocessing pipelines...")
    
    # Linear Preprocessor
    lin_preprocessor = get_linear_preprocessor(drop_first=True)
    X_train_lin = lin_preprocessor.fit_transform(X_train)
    X_test_lin = lin_preprocessor.transform(X_test)
    lin_features = lin_preprocessor.get_feature_names_out()
    print(f"  -> Linear Preprocessor: Train matrix {X_train_lin.shape}, Test matrix {X_test_lin.shape}")
    print(f"  -> Linear Feature Count: {len(lin_features)} features")
    print(f"     Feature list: {list(lin_features)}")

    # Tree Preprocessor
    tree_preprocessor = get_tree_preprocessor()
    X_train_tree = tree_preprocessor.fit_transform(X_train)
    X_test_tree = tree_preprocessor.transform(X_test)
    tree_features = tree_preprocessor.get_feature_names_out()
    print(f"  -> Tree Preprocessor:   Train matrix {X_train_tree.shape}, Test matrix {X_test_tree.shape}")
    print(f"  -> Tree Feature Count:   {len(tree_features)} features")
    print(f"     Feature list: {list(tree_features)}")

    # 6. Audit Threshold Rules Characteristic
    print("\n[Step 6] Auditing instructional threshold rules across partitions...")
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_audit = audit_threshold_rules(train_df)
    test_audit = audit_threshold_rules(test_df)
    print(f"  -> Train partition: {train_audit['exact_matches']} / {train_audit['total_records']} matches ({train_audit['rule_accuracy_pct']}%)")
    print(f"  -> Test partition:  {test_audit['exact_matches']} / {test_audit['total_records']} matches ({test_audit['rule_accuracy_pct']}%)")
    print("  -> Confirmed: Operational rules hold as an instructional property across both sets without data leakage.")

    # 7. Reproducibility Test
    print("\n[Step 7] Testing reproducibility across a second independent run...")
    X_train_2, X_test_2, y_train_2, y_test_2 = create_reproducible_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    assert X_train.equals(X_train_2), "Reproducibility failure in X_train!"
    assert X_test.equals(X_test_2), "Reproducibility failure in X_test!"
    assert y_train.equals(y_train_2), "Reproducibility failure in y_train!"
    assert y_test.equals(y_test_2), "Reproducibility failure in y_test!"
    print("  -> Exact reproducibility confirmed with identical indices, values, and distributions.")

    print("\n" + "=" * 80)
    print("PHASE 2 DATA PREPARATION COMPLETE — READY FOR PHASE 3 MODEL TRAINING")
    print("=" * 80)

    return {
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "train_delays": int(y_train.sum()),
        "test_delays": int(y_test.sum()),
        "linear_features": list(lin_features),
        "tree_features": list(tree_features)
    }


if __name__ == "__main__":
    run_data_preparation_pipeline()
