"""
Application Utilities, Data Loaders, and Caching Helpers
Project: Predicting Delivery Delays in Last-Mile Logistics — ABC Logistics Pvt. Ltd.
Phase: Phase 5 - Interactive Streamlit Application
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional, Iterable, List
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preparation import FEATURE_COLUMNS, TARGET_COLUMN

# File path constants
DATA_RAW_PATH = os.path.join(PROJECT_ROOT, "delivery_delay.csv")
DATA_TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
CV_COMPARISON_PATH = os.path.join(PROJECT_ROOT, "outputs", "cv_model_comparison.csv")
TEST_COMPARISON_PATH = os.path.join(PROJECT_ROOT, "outputs", "test_model_comparison.csv")
TEST_CONFUSION_PATH = os.path.join(PROJECT_ROOT, "outputs", "test_confusion_matrices.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(PROJECT_ROOT, "outputs", "feature_importance.csv")
LOGISTIC_COEFFS_PATH = os.path.join(PROJECT_ROOT, "outputs", "logistic_coefficients.csv")
DECISION_TREE_RULES_PATH = os.path.join(PROJECT_ROOT, "outputs", "decision_tree.txt")

MODEL_PATHS = {
    "Decision Tree": os.path.join(PROJECT_ROOT, "models", "decision_tree_pipeline.joblib"),
    "Gradient Boosting": os.path.join(PROJECT_ROOT, "models", "gradient_boosting_pipeline.joblib"),
    "Random Forest": os.path.join(PROJECT_ROOT, "models", "random_forest_pipeline.joblib"),
    "Logistic Regression": os.path.join(PROJECT_ROOT, "models", "logistic_regression_pipeline.joblib")
}

# Value mappings for readable display
WEATHER_OPTIONS = {
    1: "1 — Clear Weather (Normal Transit)",
    2: "2 — Rainy Conditions (Wet Roads, Moderate Delays)",
    3: "3 — Stormy Weather (Heavy Rain / Strong Winds)"
}

SLOT_OPTIONS = {
    1: "1 — Morning Slot (08:00 - 12:00)",
    2: "2 — Afternoon Slot (12:00 - 16:00)",
    3: "3 — Evening Slot (16:00 - 20:00)"
}

TRAFFIC_OPTIONS = {
    1: "1 — Very Low / Free Flow",
    2: "2 — Low Congestion",
    3: "3 — Moderate Road Traffic",
    4: "4 — Heavy Congestion",
    5: "5 — Severe Bottleneck / Standstill"
}

ROAD_OPTIONS = {
    1: "1 — Poor (Potholes, Road Digging)",
    2: "2 — Sub-par Surface Quality",
    3: "3 — Average Urban Roads",
    4: "4 — Good Condition",
    5: "5 — Excellent Expressways"
}

# Dispatch Scenarios for quick simulation
PRESET_SCENARIOS = {
    "Custom User Input": None,
    "Scenario A: Warehouse Stagnation (>90 min wait)": {
        "Delivery_Distance": 18.0,
        "Traffic_Congestion": 2,
        "Weather_Condition": 1,
        "Delivery_Slot": 1,
        "Driver_Experience": 12,
        "Num_Stops": 4,
        "Vehicle_Age": 3,
        "Road_Condition_Score": 4,
        "Package_Weight": 22.0,
        "Fuel_Efficiency": 14.5,
        "Warehouse_Processing_Time": 98
    },
    "Scenario B: Aging Vehicle Breakdown (8+ yrs fleet)": {
        "Delivery_Distance": 14.0,
        "Traffic_Congestion": 2,
        "Weather_Condition": 1,
        "Delivery_Slot": 2,
        "Driver_Experience": 15,
        "Num_Stops": 3,
        "Vehicle_Age": 9,
        "Road_Condition_Score": 3,
        "Package_Weight": 15.0,
        "Fuel_Efficiency": 12.0,
        "Warehouse_Processing_Time": 42
    },
    "Scenario C: Compound Route Stress (Traffic + Rain + Distance)": {
        "Delivery_Distance": 36.0,
        "Traffic_Congestion": 5,
        "Weather_Condition": 3,
        "Delivery_Slot": 3,
        "Driver_Experience": 10,
        "Num_Stops": 6,
        "Vehicle_Age": 4,
        "Road_Condition_Score": 2,
        "Package_Weight": 32.0,
        "Fuel_Efficiency": 10.5,
        "Warehouse_Processing_Time": 50
    },
    "Scenario D: Clean Dispatch (Low Risk Profile)": {
        "Delivery_Distance": 12.5,
        "Traffic_Congestion": 2,
        "Weather_Condition": 1,
        "Delivery_Slot": 1,
        "Driver_Experience": 8,
        "Num_Stops": 3,
        "Vehicle_Age": 3,
        "Road_Condition_Score": 4,
        "Package_Weight": 18.0,
        "Fuel_Efficiency": 15.0,
        "Warehouse_Processing_Time": 35
    },
    "Scenario E: Debunking the Driver Myth (Rookie Driver, Great Van)": {
        "Delivery_Distance": 15.0,
        "Traffic_Congestion": 2,
        "Weather_Condition": 1,
        "Delivery_Slot": 2,
        "Driver_Experience": 1,  # Inexperienced driver (Rohit's blame)
        "Num_Stops": 8,         # High stops (Rohit's blame)
        "Vehicle_Age": 2,         # Young van
        "Road_Condition_Score": 1,# Terrible road (Rohit's blame)
        "Package_Weight": 20.0,
        "Fuel_Efficiency": 13.0,
        "Warehouse_Processing_Time": 38 # Good warehouse speed
    }
}


# ==============================================================================
# CACHED DATA & MODEL LOADERS
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_historical_data() -> pd.DataFrame:
    """Loads historical dataset for exploratory analytics."""
    if not os.path.exists(DATA_RAW_PATH):
        st.error(f"Dataset file not found at: {DATA_RAW_PATH}")
        return pd.DataFrame()
    return pd.read_csv(DATA_RAW_PATH)


@st.cache_data(show_spinner=False)
def load_test_data() -> pd.DataFrame:
    """Loads held-out test partition for evaluation display."""
    if not os.path.exists(DATA_TEST_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATA_TEST_PATH)


@st.cache_data(show_spinner=False)
def load_evaluation_metrics() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads Phase 3 and Phase 4 model evaluation artifacts."""
    cv_df = pd.read_csv(CV_COMPARISON_PATH) if os.path.exists(CV_COMPARISON_PATH) else pd.DataFrame()
    test_df = pd.read_csv(TEST_COMPARISON_PATH) if os.path.exists(TEST_COMPARISON_PATH) else pd.DataFrame()
    cm_df = pd.read_csv(TEST_CONFUSION_PATH) if os.path.exists(TEST_CONFUSION_PATH) else pd.DataFrame()
    imp_df = pd.read_csv(FEATURE_IMPORTANCE_PATH) if os.path.exists(FEATURE_IMPORTANCE_PATH) else pd.DataFrame()
    return cv_df, test_df, cm_df, imp_df


@st.cache_data(show_spinner=False)
def load_logistic_coefficients() -> pd.DataFrame:
    """Loads logistic regression standardized coefficients and odds ratios."""
    if os.path.exists(LOGISTIC_COEFFS_PATH):
        return pd.read_csv(LOGISTIC_COEFFS_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_tree_rules_text() -> str:
    """Loads decision tree text rules export."""
    if os.path.exists(DECISION_TREE_RULES_PATH):
        with open(DECISION_TREE_RULES_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "Decision tree text rules file not found."


@st.cache_resource(show_spinner=False)
def load_model_pipeline(model_name: str) -> Optional[Any]:
    """Loads a serialized scikit-learn pipeline by model name."""
    path = MODEL_PATHS.get(model_name)
    if not path or not os.path.exists(path):
        st.warning(f"Model file missing for '{model_name}' at: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load model pipeline '{model_name}': {e}")
        return None


def validate_input_features(features_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates user inputs prior to model inference."""
    for col in FEATURE_COLUMNS:
        if col not in features_dict:
            return False, f"Missing required feature: {col}"
        val = features_dict[col]
        if pd.isna(val):
            return False, f"Null or invalid value for: {col}"

    if not (1.0 <= features_dict["Delivery_Distance"] <= 60.0):
        return False, "Delivery Distance must be between 1.0 km and 60.0 km"
    if not (10 <= features_dict["Warehouse_Processing_Time"] <= 180):
        return False, "Warehouse Processing Time must be between 10 and 180 minutes"
    if not (1 <= features_dict["Vehicle_Age"] <= 15):
        return False, "Vehicle Age must be between 1 and 15 years"
    
    return True, "Valid"


def aggregate_feature_importance_by_variable(imp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates one-hot encoded dummy feature importances back to their original 11 input features.
    """
    if imp_df.empty:
        return pd.DataFrame()
    
    mapping = {
        "num__Warehouse_Processing_Time": "Warehouse_Processing_Time",
        "num__Vehicle_Age": "Vehicle_Age",
        "num__Traffic_Congestion": "Traffic_Congestion",
        "num__Delivery_Distance": "Delivery_Distance",
        "cat__Weather_Condition_1": "Weather_Condition",
        "cat__Weather_Condition_2": "Weather_Condition",
        "cat__Weather_Condition_3": "Weather_Condition",
        "num__Package_Weight": "Package_Weight",
        "num__Fuel_Efficiency": "Fuel_Efficiency",
        "num__Driver_Experience": "Driver_Experience",
        "num__Num_Stops": "Num_Stops",
        "num__Road_Condition_Score": "Road_Condition_Score",
        "cat__Delivery_Slot_1": "Delivery_Slot",
        "cat__Delivery_Slot_2": "Delivery_Slot",
        "cat__Delivery_Slot_3": "Delivery_Slot",
    }
    df = imp_df.copy()
    df["Original_Variable"] = df["Feature"].map(lambda f: mapping.get(f, f.replace("num__", "").replace("cat__", "")))
    num_cols = ["Decision_Tree_Importance", "Random_Forest_Importance", "Gradient_Boosting_Importance", "Mean_Ensemble_Importance"]
    agg_df = df.groupby("Original_Variable", as_index=False)[num_cols].sum()
    agg_df = agg_df.rename(columns={"Original_Variable": "Feature"})
    agg_df = agg_df.sort_values(by="Mean_Ensemble_Importance", ascending=False).reset_index(drop=True)
    return agg_df


def filter_historical_data(
    df: pd.DataFrame,
    weather_filter: Optional[Iterable[Any]] = None,
    slot_filter: Optional[Iterable[Any]] = None,
    max_traffic: Optional[Any] = None
) -> pd.DataFrame:
    """
    Applies interactive operational filters to the historical deliveries dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Historical deliveries dataset containing 'Weather_Condition', 'Delivery_Slot', 'Traffic_Congestion'.
    weather_filter : Iterable or None
        Selected weather category codes (e.g. [1, 2, 3] or subset).
        An empty iterable yields 0 rows (representing an empty filter selection).
        If None, no filtering is applied on Weather_Condition.
    slot_filter : Iterable or None
        Selected delivery slot codes (e.g. [1, 2, 3] or subset).
        An empty iterable yields 0 rows (representing an empty filter selection).
        If None, no filtering is applied on Delivery_Slot.
    max_traffic : int, float, or None
        Upper bound (inclusive) for Traffic_Congestion level (1 to 5).
        If None, no upper-bound filtering is applied on Traffic_Congestion.

    Returns:
    --------
    pd.DataFrame:
        Filtered copy of the dataset respecting the logical AND intersection of all criteria.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()

    # Weather filter with safe integer type coercion
    if weather_filter is not None:
        try:
            weather_codes = [int(w) for w in weather_filter]
        except (ValueError, TypeError):
            weather_codes = list(weather_filter)
        filtered = filtered[filtered["Weather_Condition"].isin(weather_codes)]

    # Slot filter with safe integer type coercion
    if slot_filter is not None:
        try:
            slot_codes = [int(s) for s in slot_filter]
        except (ValueError, TypeError):
            slot_codes = list(slot_filter)
        filtered = filtered[filtered["Delivery_Slot"].isin(slot_codes)]

    # Max traffic congestion cap
    if max_traffic is not None:
        try:
            traffic_cap = int(max_traffic)
            filtered = filtered[filtered["Traffic_Congestion"] <= traffic_cap]
        except (ValueError, TypeError):
            pass

    return filtered

