"""
Prediction Service and Model Explanation Engine
Project: Predicting Delivery Delays in Last-Mile Logistics — ABC Logistics Pvt. Ltd.
Phase: Phase 5 - Interactive Streamlit Application
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.data_preparation import FEATURE_COLUMNS

def predict_single_delivery(
    pipeline: Pipeline,
    input_df: pd.DataFrame,
    model_name: str
) -> Dict[str, Any]:
    """
    Executes inference for a single delivery record using the loaded scikit-learn pipeline.
    Returns predicted class, label, model risk score, and confidence level.
    """
    # Ensure correct columns and order
    clean_input = input_df[FEATURE_COLUMNS].copy()

    # Raw class prediction
    pred_class = int(pipeline.predict(clean_input)[0])
    pred_label = "Delayed" if pred_class == 1 else "On-Time"

    # Probability / Risk score extraction
    risk_score = 0.50
    has_proba = False
    
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(clean_input)[0]
        # probas[1] corresponds to positive class (Delivery_Delay = 1)
        risk_score = float(probas[1])
        has_proba = True
    elif hasattr(pipeline, "decision_function"):
        score = float(pipeline.decision_function(clean_input)[0])
        # Sigmoid transform for relative score
        risk_score = 1.0 / (1.0 + np.exp(-score))
        has_proba = True

    # Risk level classification
    if risk_score >= 0.70 or pred_class == 1:
        risk_level = "High Delay Risk"
        risk_color = "#ef4444" # Crimson
        status_badge = "🚨 DELAY PREDICTED"
    elif risk_score >= 0.35:
        risk_level = "Moderate Transit Risk"
        risk_color = "#f59e0b" # Amber
        status_badge = "⚠️ MODERATE RISK"
    else:
        risk_level = "Low Risk (On-Time Dispatch)"
        risk_color = "#10b981" # Emerald
        status_badge = "✅ ON-TIME DISPATCH"

    return {
        "model_name": model_name,
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "status_badge": status_badge,
        "has_proba": has_proba
    }


def explain_decision_tree_prediction(
    tree_pipeline: Pipeline,
    input_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Extracts the exact decision path traversed by the sample inside the Decision Tree classifier.
    Translates splits into human-readable managerial explanations.
    """
    clean_input = input_df[FEATURE_COLUMNS].copy()
    preprocessor = tree_pipeline.named_steps["preprocessor"]
    clf = tree_pipeline.named_steps["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())

    # Transform input sample
    X_trans = preprocessor.transform(clean_input)
    dp = clf.decision_path(X_trans)
    node_indices = dp.indices[dp.indptr[0]:dp.indptr[1]]

    path_steps = []
    primary_causes = []

    for node_id in node_indices:
        feat_idx = clf.tree_.feature[node_id]
        if feat_idx >= 0:
            feat_name = feature_names[feat_idx]
            threshold = clf.tree_.threshold[node_id]
            val = float(X_trans[0, feat_idx])
            went_right = val > threshold
            
            # Translate feature name and human explanation
            if "Warehouse_Processing_Time" in feat_name:
                op_val = input_df["Warehouse_Processing_Time"].iloc[0]
                if went_right:
                    desc = f"Warehouse Staging Delay: Parcel waited {op_val} min inside warehouse (> {threshold:.1f} min operational limit)."
                    primary_causes.append("Warehouse Stagnation (Excessive Wait Time)")
                else:
                    desc = f"Warehouse Speed: Parcel processed in {op_val} min (within safe limit <= {threshold:.1f} min)."
            
            elif "Vehicle_Age" in feat_name:
                op_val = input_df["Vehicle_Age"].iloc[0]
                if went_right:
                    desc = f"Aging Vehicle Alert: Assigned vehicle is {op_val} years old (> {threshold:.1f} years reliability threshold)."
                    primary_causes.append("Fleet Mechanical Age (Vehicle >= 8 years old)")
                else:
                    desc = f"Fleet Condition: Vehicle age is {op_val} years (within reliable limit <= {threshold:.1f} years)."

            elif "Traffic_Congestion" in feat_name:
                op_val = input_df["Traffic_Congestion"].iloc[0]
                if went_right:
                    desc = f"High Road Congestion: Route traffic level is {op_val} / 5 (> {threshold:.1f} congestion threshold)."
                    primary_causes.append("Severe Route Traffic (Level 4–5)")
                else:
                    desc = f"Road Traffic: Traffic level is {op_val} / 5 (manageable traffic <= {threshold:.1f})."

            elif "Delivery_Distance" in feat_name:
                op_val = input_df["Delivery_Distance"].iloc[0]
                if went_right:
                    desc = f"Long Distance: Route distance is {op_val:.1f} km (> {threshold:.1f} km compound risk cutoff)."
                    primary_causes.append("Long Transit Distance (>20 km)")
                else:
                    desc = f"Route Proximity: Distance is {op_val:.1f} km (short-to-medium radius <= {threshold:.1f} km)."

            elif "Weather_Condition_1" in feat_name: # Weather_Condition_1 is Clear
                weather_code = input_df["Weather_Condition"].iloc[0]
                weather_label = "Clear" if weather_code == 1 else ("Rainy" if weather_code == 2 else "Stormy")
                if not went_right: # <= 0.50 means NOT Clear -> Rainy or Stormy
                    desc = f"Adverse Weather: Weather is {weather_label} (wet/stormy conditions compound road delays)."
                    primary_causes.append(f"Adverse Weather ({weather_label})")
                else:
                    desc = "Clear Skies: Weather is Clear (no environmental travel impediments)."

            else:
                comp_sym = ">" if went_right else "<="
                desc = f"Feature split: {feat_name} ({val:.2f}) {comp_sym} {threshold:.2f}"

            path_steps.append({
                "node_id": int(node_id),
                "feature": feat_name,
                "value": val,
                "threshold": threshold,
                "direction": "Right (>)" if went_right else "Left (<=)",
                "explanation": desc
            })

    pred_class = int(clf.predict(X_trans)[0])
    
    if pred_class == 1:
        summary_conclusion = (
            f"The delivery was flagged as DELAYED primarily due to: {', '.join(primary_causes)}."
            if primary_causes else "Flagged as DELAYED by tree split criteria."
        )
    else:
        summary_conclusion = (
            "The delivery was cleared as ON-TIME: warehouse wait is below 91 min, vehicle is under 8 years old, "
            "and route conditions do not trigger compound traffic-weather bottlenecks."
        )

    return {
        "path_steps": path_steps,
        "primary_causes": primary_causes,
        "summary_conclusion": summary_conclusion,
        "tree_depth_reached": len(path_steps)
    }


def generate_supervisor_action_plan(
    input_df: pd.DataFrame,
    prediction: Dict[str, Any],
    explanation: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates actionable operational intervention recommendations for the dispatch supervisor.
    """
    wh_time = input_df["Warehouse_Processing_Time"].iloc[0]
    veh_age = input_df["Vehicle_Age"].iloc[0]
    traffic = input_df["Traffic_Congestion"].iloc[0]
    weather = input_df["Weather_Condition"].iloc[0]
    distance = input_df["Delivery_Distance"].iloc[0]

    actions = []
    priority = "NORMAL"
    badge_style = "info"

    if prediction["predicted_class"] == 1:
        priority = "URGENT INTERVENTION REQUIRED"
        badge_style = "danger"

        # Check warehouse stagnation
        if wh_time >= 91:
            actions.append({
                "trigger": f"Warehouse Processing Time ({wh_time} min >= 91 min limit)",
                "action": "Immediate Yard Dispatch & Floor Escalation",
                "department": "Warehouse Staging / Sorting Lead",
                "details": (
                    "This parcel has already exhausted its delivery window buffer inside the hub. "
                    "Expedite manifest sign-off and immediately load onto the next ready departure wave. "
                    "Alert customer service of potential narrow delivery window."
                )
            })
        elif wh_time >= 75:
            actions.append({
                "trigger": f"Elevated Processing Time ({wh_time} min in staging)",
                "action": "Fast-Track Manifesting",
                "department": "Warehouse Shift Supervisor",
                "details": "Dispatch within 15 minutes to prevent parcel breaching the 90-minute delay threshold."
            })

        # Check vehicle age
        if veh_age >= 8:
            actions.append({
                "trigger": f"Aging Vehicle Assignment (Vehicle Age: {veh_age} yrs >= 8 yrs)",
                "action": "Fleet Reassignment / Vehicle Swap",
                "department": "Fleet & Dispatch Controller",
                "details": (
                    "Vehicles aged 8+ years suffer 100% historical delivery failure rates on quick-commerce routes. "
                    "Reassign this package to a fleet unit aged under 7 years or a reserve two-wheeler."
                )
            })

        # Check compound route risk
        if traffic >= 4 and weather >= 2 and distance > 20.0:
            weather_text = "Rainy" if weather == 2 else "Stormy"
            actions.append({
                "trigger": f"Compound Route Bottleneck (Traffic Level {traffic} + {weather_text} + {distance:.1f} km)",
                "action": "Route Re-sequencing & Proactive SLA Window Extension",
                "department": "Logistics Route Planning Lead",
                "details": (
                    "Long distance combined with heavy congestion and rain makes SLA compliance physically impossible on standard routing. "
                    "Re-sequence stops to prioritize this drop first, or trigger an automated ShopNow customer notification "
                    "offering a revised 30-minute delivery slot window before customer escalates."
                )
            })

        if not actions:
            actions.append({
                "trigger": "Elevated Model Risk Score",
                "action": "Supervisor Route Audit",
                "department": "Dispatch Supervisor",
                "details": "Audit driver departure timing and monitor GPS transit telemetry."
            })

    else:
        priority = "ROUTINE DISPATCH CLEAR"
        badge_style = "success"
        actions.append({
            "trigger": "All Operational Parameters Within Safe SLA Thresholds",
            "action": "Standard Yard Release",
            "department": "Yard Master",
            "details": (
                f"Warehouse wait is acceptable ({wh_time} min), vehicle age ({veh_age} yrs) is reliable, "
                "and route transit stress is within nominal operating tolerances. Cleared for standard departure."
            )
        })

    return {
        "priority": priority,
        "badge_style": badge_style,
        "actions": actions
    }
