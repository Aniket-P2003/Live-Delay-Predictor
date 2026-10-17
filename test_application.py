"""
Automated Test Suite for ABC Logistics Delivery Intelligence Dashboard
Verifies data loading, model loading, predictions, decision explanations, and charts.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preparation import FEATURE_COLUMNS, TARGET_COLUMN
from src.app_utils import (
    load_historical_data,
    load_test_data,
    load_evaluation_metrics,
    load_model_pipeline,
    validate_input_features,
    aggregate_feature_importance_by_variable,
    filter_historical_data,
    PRESET_SCENARIOS,
    MODEL_PATHS
)
from src.prediction_service import (
    predict_single_delivery,
    explain_decision_tree_prediction,
    generate_supervisor_action_plan
)
from src.dashboard_components import (
    plot_delay_donut,
    plot_delay_rate_by_feature,
    plot_traffic_weather_heatmap,
    plot_confusion_matrix_heatmap,
    plot_feature_importance_bar
)

class TestABCLogisticsApp(unittest.TestCase):

    def test_01_historical_data_loading(self):
        """Verifies that delivery_delay.csv loads correctly with 1,000 rows and 12 columns."""
        df = load_historical_data()
        self.assertFalse(df.empty, "Historical data is empty!")
        self.assertEqual(len(df), 1000, "Expected 1,000 rows in delivery_delay.csv")
        self.assertEqual(len(df.columns), 12, "Expected 12 columns in delivery_delay.csv")
        self.assertIn(TARGET_COLUMN, df.columns)

    def test_02_model_loading_and_steps(self):
        """Verifies that all 4 candidate model pipelines load and contain preprocessor and classifier."""
        for name in MODEL_PATHS.keys():
            pipe = load_model_pipeline(name)
            self.assertIsNotNone(pipe, f"Model {name} failed to load!")
            step_names = [s[0] for s in pipe.steps]
            self.assertIn("preprocessor", step_names, f"{name} missing preprocessor!")
            self.assertIn("classifier", step_names, f"{name} missing classifier!")

    def test_03a_prediction_service_decision_tree_scenarios(self):
        """Verifies that Decision Tree accurately classifies all preset scenarios."""
        tree_pipe = load_model_pipeline("Decision Tree")
        self.assertIsNotNone(tree_pipe)

        # Scenario A: Warehouse Stagnation -> Delayed (1)
        scen_a = pd.DataFrame([PRESET_SCENARIOS["Scenario A: Warehouse Stagnation (>90 min wait)"]])
        pred_a = predict_single_delivery(tree_pipe, scen_a, "Decision Tree")
        self.assertEqual(pred_a["predicted_class"], 1, "Scenario A should predict Delayed!")
        self.assertEqual(pred_a["predicted_label"], "Delayed")

        # Scenario B: Aging Vehicle -> Delayed (1)
        scen_b = pd.DataFrame([PRESET_SCENARIOS["Scenario B: Aging Vehicle Breakdown (8+ yrs fleet)"]])
        pred_b = predict_single_delivery(tree_pipe, scen_b, "Decision Tree")
        self.assertEqual(pred_b["predicted_class"], 1, "Scenario B should predict Delayed!")

        # Scenario C: Compound Transit Stress -> Delayed (1)
        scen_c = pd.DataFrame([PRESET_SCENARIOS["Scenario C: Compound Route Stress (Traffic + Rain + Distance)"]])
        pred_c = predict_single_delivery(tree_pipe, scen_c, "Decision Tree")
        self.assertEqual(pred_c["predicted_class"], 1, "Scenario C should predict Delayed!")

        # Scenario D: Clean Dispatch -> On-Time (0)
        scen_d = pd.DataFrame([PRESET_SCENARIOS["Scenario D: Clean Dispatch (Low Risk Profile)"]])
        pred_d = predict_single_delivery(tree_pipe, scen_d, "Decision Tree")
        self.assertEqual(pred_d["predicted_class"], 0, "Scenario D should predict On-Time!")
        self.assertEqual(pred_d["predicted_label"], "On-Time")

        # Scenario E: Rookie Driver Myth -> On-Time (0)
        scen_e = pd.DataFrame([PRESET_SCENARIOS["Scenario E: Debunking the Driver Myth (Rookie Driver, Great Van)"]])
        pred_e = predict_single_delivery(tree_pipe, scen_e, "Decision Tree")
        self.assertEqual(pred_e["predicted_class"], 0, "Scenario E should predict On-Time!")

    def test_03b_prediction_service_gradient_boosting_scenarios(self):
        """Verifies that Gradient Boosting accurately classifies all 5 scenarios."""
        gb_pipe = load_model_pipeline("Gradient Boosting")
        self.assertIsNotNone(gb_pipe)

        scen_a = pd.DataFrame([PRESET_SCENARIOS["Scenario A: Warehouse Stagnation (>90 min wait)"]])
        self.assertEqual(predict_single_delivery(gb_pipe, scen_a, "Gradient Boosting")["predicted_class"], 1)

        scen_b = pd.DataFrame([PRESET_SCENARIOS["Scenario B: Aging Vehicle Breakdown (8+ yrs fleet)"]])
        self.assertEqual(predict_single_delivery(gb_pipe, scen_b, "Gradient Boosting")["predicted_class"], 1)

        scen_c = pd.DataFrame([PRESET_SCENARIOS["Scenario C: Compound Route Stress (Traffic + Rain + Distance)"]])
        self.assertEqual(predict_single_delivery(gb_pipe, scen_c, "Gradient Boosting")["predicted_class"], 1)

        scen_d = pd.DataFrame([PRESET_SCENARIOS["Scenario D: Clean Dispatch (Low Risk Profile)"]])
        self.assertEqual(predict_single_delivery(gb_pipe, scen_d, "Gradient Boosting")["predicted_class"], 0)

        scen_e = pd.DataFrame([PRESET_SCENARIOS["Scenario E: Debunking the Driver Myth (Rookie Driver, Great Van)"]])
        self.assertEqual(predict_single_delivery(gb_pipe, scen_e, "Gradient Boosting")["predicted_class"], 0)

    def test_03c_prediction_service_random_forest_scenarios(self):
        """Verifies that Random Forest accurately classifies all 5 scenarios."""
        rf_pipe = load_model_pipeline("Random Forest")
        self.assertIsNotNone(rf_pipe)

        scen_a = pd.DataFrame([PRESET_SCENARIOS["Scenario A: Warehouse Stagnation (>90 min wait)"]])
        self.assertEqual(predict_single_delivery(rf_pipe, scen_a, "Random Forest")["predicted_class"], 1)

        scen_b = pd.DataFrame([PRESET_SCENARIOS["Scenario B: Aging Vehicle Breakdown (8+ yrs fleet)"]])
        self.assertEqual(predict_single_delivery(rf_pipe, scen_b, "Random Forest")["predicted_class"], 1)

        scen_c = pd.DataFrame([PRESET_SCENARIOS["Scenario C: Compound Route Stress (Traffic + Rain + Distance)"]])
        self.assertEqual(predict_single_delivery(rf_pipe, scen_c, "Random Forest")["predicted_class"], 1)

        scen_d = pd.DataFrame([PRESET_SCENARIOS["Scenario D: Clean Dispatch (Low Risk Profile)"]])
        self.assertEqual(predict_single_delivery(rf_pipe, scen_d, "Random Forest")["predicted_class"], 0)

        scen_e = pd.DataFrame([PRESET_SCENARIOS["Scenario E: Debunking the Driver Myth (Rookie Driver, Great Van)"]])
        self.assertEqual(predict_single_delivery(rf_pipe, scen_e, "Random Forest")["predicted_class"], 0)

    def test_03d_prediction_service_logistic_regression_scenarios(self):
        """Verifies Logistic Regression predictions and documents its linear smoothing behavior."""
        lr_pipe = load_model_pipeline("Logistic Regression")
        self.assertIsNotNone(lr_pipe)

        # Scenarios A and B: Favorable distance, traffic and weather linearly compensate for high wait/age -> predicts On-Time (0)
        scen_a = pd.DataFrame([PRESET_SCENARIOS["Scenario A: Warehouse Stagnation (>90 min wait)"]])
        pred_a = predict_single_delivery(lr_pipe, scen_a, "Logistic Regression")
        self.assertEqual(pred_a["predicted_class"], 0, "Logistic Regression predicts 0 due to linear compensatory effect")
        self.assertTrue(0.20 <= pred_a["risk_score"] <= 0.45)

        scen_b = pd.DataFrame([PRESET_SCENARIOS["Scenario B: Aging Vehicle Breakdown (8+ yrs fleet)"]])
        pred_b = predict_single_delivery(lr_pipe, scen_b, "Logistic Regression")
        self.assertEqual(pred_b["predicted_class"], 0, "Logistic Regression predicts 0 due to linear compensatory effect")
        self.assertTrue(0.20 <= pred_b["risk_score"] <= 0.45)

        # Scenario C: Severe compound stress -> predicts Delayed (1)
        scen_c = pd.DataFrame([PRESET_SCENARIOS["Scenario C: Compound Route Stress (Traffic + Rain + Distance)"]])
        pred_c = predict_single_delivery(lr_pipe, scen_c, "Logistic Regression")
        self.assertEqual(pred_c["predicted_class"], 1)
        self.assertTrue(pred_c["risk_score"] > 0.50)

        # Scenarios D and E: Clean conditions -> predicts On-Time (0)
        scen_d = pd.DataFrame([PRESET_SCENARIOS["Scenario D: Clean Dispatch (Low Risk Profile)"]])
        self.assertEqual(predict_single_delivery(lr_pipe, scen_d, "Logistic Regression")["predicted_class"], 0)

        scen_e = pd.DataFrame([PRESET_SCENARIOS["Scenario E: Debunking the Driver Myth (Rookie Driver, Great Van)"]])
        self.assertEqual(predict_single_delivery(lr_pipe, scen_e, "Logistic Regression")["predicted_class"], 0)

    def test_03e_preset_scenarios_schema_integrity(self):
        """Verifies that all 5 preset scenarios contain exactly the 11 FEATURE_COLUMNS."""
        for name, scen in PRESET_SCENARIOS.items():
            if scen is None:
                continue
            self.assertEqual(len(scen), 11, f"Scenario '{name}' does not have 11 features!")
            for feat in FEATURE_COLUMNS:
                self.assertIn(feat, scen, f"Scenario '{name}' is missing feature: {feat}")

    def test_04_decision_path_explanation(self):
        """Verifies that the decision tree explanation engine parses splits correctly."""
        tree_pipe = load_model_pipeline("Decision Tree")
        scen_a = pd.DataFrame([PRESET_SCENARIOS["Scenario A: Warehouse Stagnation (>90 min wait)"]])
        expl_a = explain_decision_tree_prediction(tree_pipe, scen_a)
        
        self.assertTrue(len(expl_a["path_steps"]) > 0, "No path steps extracted!")
        self.assertIn("Warehouse Stagnation", expl_a["summary_conclusion"])

    def test_05_supervisor_action_plan(self):
        """Verifies that supervisor action plans generate concrete recommendations."""
        scen_b = pd.DataFrame([PRESET_SCENARIOS["Scenario B: Aging Vehicle Breakdown (8+ yrs fleet)"]])
        pred = {"predicted_class": 1, "predicted_label": "Delayed", "risk_score": 1.0}
        plan = generate_supervisor_action_plan(scen_b, pred)
        
        self.assertEqual(plan["priority"], "URGENT INTERVENTION REQUIRED")
        self.assertTrue(any("Vehicle Swap" in act["action"] for act in plan["actions"]))

    def test_06_dashboard_charts_generation(self):
        """Verifies that Plotly chart functions return valid Figure objects without errors."""
        df = load_historical_data()
        fig_donut = plot_delay_donut(df)
        self.assertIsNotNone(fig_donut)

        fig_wh = plot_delay_rate_by_feature(df, "Warehouse_Processing_Time", "Test WH", "X", is_continuous=True)
        self.assertIsNotNone(fig_wh)

        fig_heat = plot_traffic_weather_heatmap(df)
        self.assertIsNotNone(fig_heat)

        fig_cm = plot_confusion_matrix_heatmap(98, 0, 0, 102, "Decision Tree")
        self.assertIsNotNone(fig_cm)

    def test_07_input_validation(self):
        """Verifies bounds and type validation for user input features."""
        valid_sample = PRESET_SCENARIOS["Scenario D: Clean Dispatch (Low Risk Profile)"].copy()
        is_val, msg = validate_input_features(valid_sample)
        self.assertTrue(is_val)

        # Invalid distance
        invalid_sample = valid_sample.copy()
        invalid_sample["Delivery_Distance"] = 150.0 # Out of bounds
        is_val, msg = validate_input_features(invalid_sample)
        self.assertFalse(is_val)

    def test_08_feature_importance_aggregation(self):
        """Verifies feature importance calculation across transformed columns and original variables."""
        _, _, _, imp_df = load_evaluation_metrics()
        self.assertFalse(imp_df.empty, "feature_importance.csv is empty!")
        
        # 15 transformed columns from ColumnTransformer
        self.assertEqual(len(imp_df), 15, "Expected 15 transformed features from one-hot encoding!")
        
        # Verify top 5 transformed features account for 93.31%
        top5_trans_sum = imp_df.head(5)["Mean_Ensemble_Importance"].sum()
        self.assertAlmostEqual(top5_trans_sum, 0.93307, places=3, msg="Top 5 transformed columns should sum to ~93.31%")

        # Aggregate by original variable (11 variables)
        agg_df = aggregate_feature_importance_by_variable(imp_df)
        self.assertEqual(len(agg_df), 11, "Expected 11 original input variables!")
        
        # Total sum must equal 100%
        total_sum = agg_df["Mean_Ensemble_Importance"].sum()
        self.assertAlmostEqual(total_sum, 1.0, places=4, msg="Aggregated feature importance must sum to 100%!")

        # Top 5 original variables account for 94.65%
        top5_orig_sum = agg_df.head(5)["Mean_Ensemble_Importance"].sum()
        self.assertAlmostEqual(top5_orig_sum, 0.94654, places=3, msg="Top 5 original variables should sum to ~94.65%")

        # Verify Weather_Condition aggregates all 3 dummy categories
        weather_row = agg_df[agg_df["Feature"] == "Weather_Condition"].iloc[0]
        self.assertAlmostEqual(weather_row["Mean_Ensemble_Importance"], 0.07093, places=3)

    def test_09_independent_filters(self):
        """Verifies that weather and delivery slot filters independently filter the dataset correctly."""
        df = load_historical_data()
        self.assertEqual(len(df), 1000)

        # Single weather category
        df_w1 = filter_historical_data(df, weather_filter=[1])
        self.assertEqual(len(df_w1), 346)
        self.assertTrue((df_w1["Weather_Condition"] == 1).all())

        df_w2 = filter_historical_data(df, weather_filter=[2])
        self.assertEqual(len(df_w2), 321)
        self.assertTrue((df_w2["Weather_Condition"] == 2).all())

        df_w3 = filter_historical_data(df, weather_filter=[3])
        self.assertEqual(len(df_w3), 333)
        self.assertTrue((df_w3["Weather_Condition"] == 3).all())

        # Multiple weather categories
        df_w12 = filter_historical_data(df, weather_filter=[1, 2])
        self.assertEqual(len(df_w12), 667)
        self.assertTrue(df_w12["Weather_Condition"].isin([1, 2]).all())

        # String coercion safety for weather
        df_w_str = filter_historical_data(df, weather_filter=["1", "3"])
        self.assertEqual(len(df_w_str), 679)
        self.assertTrue(df_w_str["Weather_Condition"].isin([1, 3]).all())

        # Single scheduled slot
        df_s1 = filter_historical_data(df, slot_filter=[1])
        self.assertEqual(len(df_s1), 350)
        self.assertTrue((df_s1["Delivery_Slot"] == 1).all())

        df_s2 = filter_historical_data(df, slot_filter=[2])
        self.assertEqual(len(df_s2), 317)
        self.assertTrue((df_s2["Delivery_Slot"] == 2).all())

        df_s3 = filter_historical_data(df, slot_filter=[3])
        self.assertEqual(len(df_s3), 333)
        self.assertTrue((df_s3["Delivery_Slot"] == 3).all())

        # Multiple scheduled slots
        df_s13 = filter_historical_data(df, slot_filter=[1, 3])
        self.assertEqual(len(df_s13), 683)
        self.assertTrue(df_s13["Delivery_Slot"].isin([1, 3]).all())

        # String coercion safety for slots
        df_s_str = filter_historical_data(df, slot_filter=["2"])
        self.assertEqual(len(df_s_str), 317)
        self.assertTrue((df_s_str["Delivery_Slot"] == 2).all())

    def test_10_traffic_congestion_filter(self):
        """Verifies that traffic congestion filter respects selected maximum thresholds."""
        df = load_historical_data()

        # Traffic <= 1
        df_t1 = filter_historical_data(df, max_traffic=1)
        self.assertEqual(len(df_t1), 193)
        self.assertTrue((df_t1["Traffic_Congestion"] <= 1).all())
        self.assertFalse((df_t1["Traffic_Congestion"] > 1).any())

        # Traffic <= 3
        df_t3 = filter_historical_data(df, max_traffic=3)
        self.assertEqual(len(df_t3), 578)
        self.assertTrue((df_t3["Traffic_Congestion"] <= 3).all())
        self.assertFalse((df_t3["Traffic_Congestion"] > 3).any())

        # Traffic <= 5 (full dataset)
        df_t5 = filter_historical_data(df, max_traffic=5)
        self.assertEqual(len(df_t5), 1000)

        # String coercion safety for traffic cap
        df_t_str = filter_historical_data(df, max_traffic="2")
        self.assertTrue((df_t_str["Traffic_Congestion"] <= 2).all())

    def test_11_combined_filters_intersection(self):
        """Verifies that combining filters produces the exact logical AND intersection."""
        df = load_historical_data()

        # Combination 1: Weather=1, Slot=1, Traffic<=2
        combo1 = filter_historical_data(df, weather_filter=[1], slot_filter=[1], max_traffic=2)
        self.assertEqual(len(combo1), 52)
        self.assertEqual(int(combo1["Delivery_Delay"].sum()), 29)
        self.assertTrue((combo1["Weather_Condition"] == 1).all())
        self.assertTrue((combo1["Delivery_Slot"] == 1).all())
        self.assertTrue((combo1["Traffic_Congestion"] <= 2).all())

        # Combination 2: Weather in [2, 3], Slot=3, Traffic<=4
        combo2 = filter_historical_data(df, weather_filter=[2, 3], slot_filter=[3], max_traffic=4)
        self.assertTrue(combo2["Weather_Condition"].isin([2, 3]).all())
        self.assertTrue((combo2["Delivery_Slot"] == 3).all())
        self.assertTrue((combo2["Traffic_Congestion"] <= 4).all())
        expected_len = len(df[df["Weather_Condition"].isin([2, 3]) & (df["Delivery_Slot"] == 3) & (df["Traffic_Congestion"] <= 4)])
        self.assertEqual(len(combo2), expected_len)

        # Reset condition: All weathers, all slots, max traffic 5 restores full 1,000 records
        df_reset = filter_historical_data(df, weather_filter=[1, 2, 3], slot_filter=[1, 2, 3], max_traffic=5)
        self.assertEqual(len(df_reset), 1000)
        self.assertEqual(int(df_reset["Delivery_Delay"].sum()), 511)

    def test_12_empty_filters_and_zero_rows_handling(self):
        """Verifies that empty selections yield 0 rows and charts handle empty DataFrames gracefully."""
        df = load_historical_data()

        # Deselecting all weather categories yields 0 rows
        df_empty_weather = filter_historical_data(df, weather_filter=[])
        self.assertEqual(len(df_empty_weather), 0)
        self.assertEqual(list(df_empty_weather.columns), list(df.columns))

        # Deselecting all slots yields 0 rows
        df_empty_slot = filter_historical_data(df, slot_filter=[])
        self.assertEqual(len(df_empty_slot), 0)

        # Empty DataFrame input to charts returns valid Plotly Figure without crashing
        fig_donut_empty = plot_delay_donut(df_empty_weather)
        self.assertIsNotNone(fig_donut_empty)

        fig_wh_empty = plot_delay_rate_by_feature(
            df_empty_weather,
            col="Warehouse_Processing_Time",
            title="Empty Test",
            xaxis_title="Min",
            is_continuous=True
        )
        self.assertIsNotNone(fig_wh_empty)

        fig_heat_empty = plot_traffic_weather_heatmap(df_empty_weather)
        self.assertIsNotNone(fig_heat_empty)

    def test_13_prediction_engine_switching_and_isolation(self):
        """Verifies that changing active prediction engine updates the model and does not reuse stale outputs."""
        # Use Scenario A (Warehouse Stagnation: 98 min wait, vehicle age 3, distance 18, traffic 2, weather 1, slot 1)
        sample = pd.DataFrame([PRESET_SCENARIOS["Scenario A: Warehouse Stagnation (>90 min wait)"]])

        # 1. Decision Tree prediction
        dt_pipe = load_model_pipeline("Decision Tree")
        res_dt = predict_single_delivery(dt_pipe, sample, "Decision Tree")
        self.assertEqual(res_dt["model_name"], "Decision Tree")
        self.assertEqual(res_dt["predicted_class"], 1)
        self.assertEqual(res_dt["predicted_label"], "Delayed")
        self.assertEqual(res_dt["risk_score"], 1.0)

        # Explanation for Decision Tree contains rule path
        expl_dt = explain_decision_tree_prediction(dt_pipe, sample)
        self.assertTrue(len(expl_dt["path_steps"]) > 0)
        self.assertIn("Warehouse_Processing_Time", expl_dt["path_steps"][0]["feature"])

        # 2. Switch to Logistic Regression
        lr_pipe = load_model_pipeline("Logistic Regression")
        res_lr = predict_single_delivery(lr_pipe, sample, "Logistic Regression")
        self.assertEqual(res_lr["model_name"], "Logistic Regression")
        # Due to linear compensatory effect, LR predicts On-Time (0) for this sample
        self.assertEqual(res_lr["predicted_class"], 0)
        self.assertEqual(res_lr["predicted_label"], "On-Time")
        self.assertTrue(res_lr["risk_score"] < 0.50)
        # Verify LR prediction is NOT contaminated by DT's previous prediction
        self.assertNotEqual(res_lr["predicted_class"], res_dt["predicted_class"])
        self.assertNotEqual(res_lr["risk_score"], res_dt["risk_score"])

        # 3. Switch to Random Forest
        rf_pipe = load_model_pipeline("Random Forest")
        res_rf = predict_single_delivery(rf_pipe, sample, "Random Forest")
        self.assertEqual(res_rf["model_name"], "Random Forest")
        self.assertEqual(res_rf["predicted_class"], 1)
        self.assertTrue(0.70 <= res_rf["risk_score"] <= 0.95)
        # Verify RF risk score is distinct from DT's 1.0 and LR's score
        self.assertNotEqual(res_rf["risk_score"], res_lr["risk_score"])

        # 4. Switch to Gradient Boosting
        gb_pipe = load_model_pipeline("Gradient Boosting")
        res_gb = predict_single_delivery(gb_pipe, sample, "Gradient Boosting")
        self.assertEqual(res_gb["model_name"], "Gradient Boosting")
        self.assertEqual(res_gb["predicted_class"], 1)
        self.assertTrue(res_gb["risk_score"] > 0.95)

if __name__ == "__main__":
    unittest.main()

