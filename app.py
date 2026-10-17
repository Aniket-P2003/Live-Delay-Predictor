"""
ABC Logistics | Delivery Intelligence
Interactive Streamlit Decision-Support Dashboard
Project: Predicting Delivery Delays in Last-Mile Logistics — ABC Logistics Pvt. Ltd.
Author: Senior Business Analytics Consultant / Data Analyst
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preparation import FEATURE_COLUMNS, TARGET_COLUMN
from src.app_utils import (
    load_historical_data,
    load_test_data,
    load_evaluation_metrics,
    load_logistic_coefficients,
    load_tree_rules_text,
    load_model_pipeline,
    validate_input_features,
    aggregate_feature_importance_by_variable,
    filter_historical_data,
    WEATHER_OPTIONS,
    SLOT_OPTIONS,
    TRAFFIC_OPTIONS,
    ROAD_OPTIONS,
    PRESET_SCENARIOS,
    MODEL_PATHS
)
from src.prediction_service import (
    predict_single_delivery,
    explain_decision_tree_prediction,
    generate_supervisor_action_plan
)
from src.dashboard_components import (
    inject_custom_css,
    render_brand_header,
    render_kpi_card,
    plot_delay_donut,
    plot_delay_rate_by_feature,
    plot_traffic_weather_heatmap,
    plot_confusion_matrix_heatmap,
    plot_model_comparison_bar,
    plot_feature_importance_bar,
    plot_myth_buster_comparison
)

# Page Configuration
st.set_page_config(
    page_title="ABC Logistics | Delivery Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom modern styling
inject_custom_css()

# ==============================================================================
# SIDEBAR NAVIGATION & CONTROLS
# ==============================================================================
st.sidebar.markdown("""
    <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #334155; margin-bottom: 1rem;">
        <h2 style="margin: 0; font-size: 1.25rem; color: #f8fafc; font-weight: 700;">🚚 ABC Logistics</h2>
        <span style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Dispatch Intelligence Hub</span>
    </div>
""", unsafe_allow_html=True)

nav_page = st.sidebar.radio(
    "NAVIGATION MENU",
    [
        "🏢 Executive Overview",
        "🧪 Model Performance Lab",
        "⚡ Live Delay Predictor",
        "📊 Operations Insights",
        "🎯 Manager Action Centre"
    ],
    index=0
)

st.sidebar.markdown("---")

# Global Model Selection
st.sidebar.markdown("### 🤖 Decision Model")
if "active_model_name" not in st.session_state:
    st.session_state["active_model_name"] = "Decision Tree"

model_options_list = list(MODEL_PATHS.keys())
default_model_idx = model_options_list.index(st.session_state["active_model_name"]) if st.session_state["active_model_name"] in model_options_list else 0

selected_model_name = st.sidebar.selectbox(
    "Active Prediction Engine",
    model_options_list,
    index=default_model_idx,
    key="active_prediction_engine_select",
    help="Select the machine learning model pipeline to use for Live Delay Predictions and Model Performance inspection."
)
st.session_state["active_model_name"] = selected_model_name
st.sidebar.caption(f"Currently Active: **{selected_model_name}**")

st.sidebar.markdown("---")

# Client Escalation Alert Card
st.sidebar.markdown("""
    <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; padding: 0.85rem; margin-top: 0.5rem;">
        <span style="font-size: 0.72rem; font-weight: 700; color: #ef4444; text-transform: uppercase;">⚡ Urgent Client SLA Alert</span>
        <div style="font-size: 0.88rem; color: #f8fafc; font-weight: 600; margin-top: 0.2rem;">ShopNow 90-Day Ultimatum</div>
        <div style="font-size: 0.76rem; color: #cbd5e1; margin-top: 0.25rem;">
            41 daily escalation complaints. SLA turnaround required within quarter to prevent contract re-tendering.
        </div>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div style="font-size: 0.72rem; color: #64748b; margin-top: 1.5rem; text-align: center;">
        ABC Logistics Pvt. Ltd. &copy; 2026<br>Pune Dispatch Analytics Operations
    </div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if nav_page == "🏢 Executive Overview":
    render_brand_header(subtitle="Executive Operations Overview & Historical Delivery SLA Profile")

    df_hist = load_historical_data()
    if df_hist.empty:
        st.error("Historical dataset `delivery_delay.csv` could not be loaded.")
        st.stop()

    # Session state defaults for Executive Overview filters
    if "filter_weather" not in st.session_state:
        st.session_state["filter_weather"] = [1, 2, 3]
    if "filter_slot" not in st.session_state:
        st.session_state["filter_slot"] = [1, 2, 3]
    if "filter_traffic" not in st.session_state:
        st.session_state["filter_traffic"] = 5

    def reset_executive_filters():
        st.session_state["filter_weather"] = [1, 2, 3]
        st.session_state["filter_slot"] = [1, 2, 3]
        st.session_state["filter_traffic"] = 5

    st.markdown('<div class="section-header">Operational Delivery SLA Overview & Interactive Filters</div>', unsafe_allow_html=True)

    # Filter Bar with 3 filter controls and a Reset button
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1.2, 1.2, 1.1, 0.5])
    with filter_col1:
        weather_filter = st.multiselect(
            "Filter by Weather Status",
            options=[1, 2, 3],
            key="filter_weather",
            format_func=lambda x: WEATHER_OPTIONS[x],
            help="Select one or more weather categories. Clear all to test zero-selection."
        )
    with filter_col2:
        slot_filter = st.multiselect(
            "Filter by Scheduled Slot",
            options=[1, 2, 3],
            key="filter_slot",
            format_func=lambda x: SLOT_OPTIONS[x],
            help="Select one or more scheduled dispatch slots. Clear all to test zero-selection."
        )
    with filter_col3:
        traffic_filter = st.slider(
            "Max Traffic Congestion Level",
            min_value=1,
            max_value=5,
            key="filter_traffic",
            help="Excludes dispatches whose traffic congestion exceeds this maximum."
        )
    with filter_col4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.button(
            "🔄 Reset",
            on_click=reset_executive_filters,
            use_container_width=True,
            help="Reset all filters to default (full 1,000 dispatches)"
        )

    # Apply filters via centralized utility
    filtered_df = filter_historical_data(
        df_hist,
        weather_filter=weather_filter,
        slot_filter=slot_filter,
        max_traffic=traffic_filter
    )

    total_deliveries = len(filtered_df)
    if total_deliveries > 0:
        delayed_count = int(filtered_df["Delivery_Delay"].sum())
        delayed_pct = (delayed_count / total_deliveries) * 100
        ontime_count = total_deliveries - delayed_count
        ontime_pct = 100 - delayed_pct
    else:
        delayed_count = 0
        delayed_pct = 0.0
        ontime_count = 0
        ontime_pct = 0.0

    # Status Caption
    if total_deliveries == len(df_hist):
        st.caption(f"Showing all **{total_deliveries:,}** historical dispatches (unfiltered dataset).")
    elif total_deliveries > 0:
        pct_active = (total_deliveries / len(df_hist)) * 100
        st.caption(f"Showing **{total_deliveries:,}** of {len(df_hist):,} dispatches ({pct_active:.1f}%) based on active filters.")
    else:
        st.caption("Showing **0** dispatches. No records match the active filter criteria.")

    # KPI Summary Row - dynamically calculated from filtered_df
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        pct_of_total = (total_deliveries / len(df_hist)) * 100 if len(df_hist) > 0 else 0.0
        render_kpi_card("Filtered Deliveries", f"{total_deliveries:,}", f"{pct_of_total:.1f}% of Fleet Volume ({len(df_hist):,} Total)", "#6366f1")
    with kpi2:
        render_kpi_card("Delivery Delays", f"{delayed_count:,} ({delayed_pct:.1f}%)", "Missed Promised Delivery Window", "#ef4444")
    with kpi3:
        render_kpi_card("On-Time Dispatches", f"{ontime_count:,} ({ontime_pct:.1f}%)", "Met Guaranteed Delivery Window", "#10b981")
    with kpi4:
        render_kpi_card("ShopNow Escalations", "41 Complaints", "Daily Peak / Contract Re-Tender Risk", "#f59e0b")

    # Visualizations or Graceful Empty State
    if filtered_df.empty:
        st.warning(
            "⚠️ **No delivery records match the active filter criteria.** "
            "Please adjust your filter selections or click **🔄 Reset** above to restore all 1,000 dispatches."
        )
    else:
        # Charts Row 1
        col_chart1, col_chart2 = st.columns([1, 1.3])
        with col_chart1:
            fig_donut = plot_delay_donut(filtered_df)
            st.plotly_chart(fig_donut, use_container_width=True)
        with col_chart2:
            fig_wh = plot_delay_rate_by_feature(
                filtered_df,
                col="Warehouse_Processing_Time",
                title="Delay Rate by Warehouse Staging Time (Quintiles)",
                xaxis_title="Warehouse Staging Minutes",
                is_continuous=True,
                bins=5
            )
            st.plotly_chart(fig_wh, use_container_width=True)

        # Charts Row 2
        col_chart3, col_chart4 = st.columns([1.1, 1.2])
        with col_chart3:
            fig_age = plot_delay_rate_by_feature(
                filtered_df,
                col="Vehicle_Age",
                title="Delay Rate by Vehicle Age (Years in Fleet)",
                xaxis_title="Vehicle Age (Years)",
                is_continuous=False
            )
            st.plotly_chart(fig_age, use_container_width=True)
        with col_chart4:
            fig_heatmap = plot_traffic_weather_heatmap(filtered_df)
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Business Interpretation Callout
        st.markdown("""
            <div class="callout-box">
                <h4 style="margin: 0 0 0.5rem 0; color: #0f172a;">Executive Analytical Findings:</h4>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem; color: #334155; line-height: 1.6;">
                    <li><strong>Ananya Rao's Warehouse Bottleneck Observed:</strong> In this dataset, all observed parcels delayed inside the warehouse beyond 90 minutes experienced delivery delays (268 of 268 records).</li>
                    <li><strong>Observed Fleet Age Pattern (8+ Years Old):</strong> In this dataset, all observed deliveries using vehicles aged 8 and 9 years were delayed (202 records), whereas vehicles aged 1 to 7 years showed an observed delay rate of ~38.7%. This observed pattern suggests prioritizing an operational evaluation of older vehicles before committing to across-the-board fleet purchases, subject to controlled pilot testing.</li>
                    <li><strong>Compound Route Risk:</strong> Adverse weather (rain/storm) combined with high congestion (levels 4–5) and distances >20 km coincides with severe transit bottlenecks (157 of 157 delayed).</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # CSV Download Button
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"📥 Download Filtered Historical Data ({len(filtered_df):,} Rows CSV)",
            data=csv_data,
            file_name="abc_logistics_filtered_data.csv",
            mime="text/csv"
        )


# ==============================================================================
# PAGE 2: MODEL PERFORMANCE LAB
# ==============================================================================
elif nav_page == "🧪 Model Performance Lab":
    render_brand_header(subtitle="Machine Learning Benchmark, Cross-Validation & Generalization Lab")

    cv_df, test_df, cm_df, imp_df = load_evaluation_metrics()
    
    if test_df.empty:
        st.error("Evaluation metrics files missing. Please run Phase 3 and Phase 4 scripts first.")
        st.stop()

    tab_test, tab_cv, tab_gap = st.tabs([
        "🎯 Held-Out Test Set Benchmark (N=200)",
        "🔄 5-Fold Stratified Cross-Validation (N=800)",
        "⚖️ Generalization Gap Audit"
    ])

    with tab_test:
        st.markdown('<div class="section-header">Held-Out Test Set Performance Benchmark (N=200 Unseen Deliveries)</div>', unsafe_allow_html=True)

        # Comparative Metrics Table
        display_test_df = test_df.copy()
        display_test_df["Accuracy"] = display_test_df["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
        display_test_df["Precision"] = display_test_df["Precision"].apply(lambda x: f"{x*100:.2f}%")
        display_test_df["Recall"] = display_test_df["Recall"].apply(lambda x: f"{x*100:.2f}%")
        display_test_df["F1_Score"] = display_test_df["F1_Score"].apply(lambda x: f"{x:.4f}")
        display_test_df["ROC_AUC"] = display_test_df["ROC_AUC"].apply(lambda x: f"{x:.4f}")

        st.dataframe(
            display_test_df[["Model", "Accuracy", "Recall", "Precision", "F1_Score", "ROC_AUC", "True_Positives", "False_Negatives", "True_Negatives", "False_Positives"]],
            use_container_width=True,
            hide_index=True
        )

        # Comparative Bar Chart
        fig_comp = plot_model_comparison_bar(test_df)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-header">Interactive Model Diagnostic & Error Analysis</div>', unsafe_allow_html=True)

        model_inspect_options = test_df["Model"].tolist()
        active_mapped_model = selected_model_name.replace(" ", "_")
        default_inspect_idx = model_inspect_options.index(active_mapped_model) if active_mapped_model in model_inspect_options else 0

        model_inspect = st.selectbox(
            "Inspect Model Confusion Matrix & Error Profile:",
            model_inspect_options,
            index=default_inspect_idx,
            help=f"Pre-selected to match the Active Prediction Engine ({selected_model_name}). You can also inspect any other model directly."
        )

        row_metrics = test_df[test_df["Model"] == model_inspect].iloc[0]
        tn = int(row_metrics["True_Negatives"])
        fp = int(row_metrics["False_Positives"])
        fn = int(row_metrics["False_Negatives"])
        tp = int(row_metrics["True_Positives"])

        col_cm_plot, col_cm_desc = st.columns([1.2, 1])

        with col_cm_plot:
            fig_cm = plot_confusion_matrix_heatmap(tn, fp, fn, tp, model_inspect)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_cm_desc:
            st.markdown(f"#### Strategic Error Diagnosis: {model_inspect}")
            if fn > 0:
                st.markdown(f"""
                    <div class="callout-box callout-danger">
                        <strong>Critical SLA Risk ({fn} False Negatives):</strong><br>
                        {fn} delayed parcels were predicted to be on time. These parcels will miss their delivery window without supervisor intervention, directly violating ShopNow's contract SLAs.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="callout-box callout-success">
                        <strong>Zero Missed Delays (0 False Negatives):</strong><br>
                        Every single delayed parcel in the unseen test set was correctly flagged prior to dispatch.
                    </div>
                """, unsafe_allow_html=True)

            if fp > 0:
                st.markdown(f"""
                    <div class="callout-box callout-warning">
                        <strong>Operational Friction ({fp} False Positives):</strong><br>
                        {fp} on-time parcels were falsely flagged as high-risk, triggering unnecessary vehicle swaps and supervisor overhead.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="callout-box callout-success">
                        <strong>Zero False Alarms (0 False Positives):</strong><br>
                        Supervisors only intervene when true operational breakdowns exist.
                    </div>
                """, unsafe_allow_html=True)

    with tab_cv:
        st.markdown('<div class="section-header">5-Fold Stratified Cross-Validation on Training Partition (N=800)</div>', unsafe_allow_html=True)
        st.write("To prevent data leakage, hyperparameter tuning and model architecture selection were evaluated exclusively using Stratified 5-Fold Cross-Validation on the 800 training records.")
        if not cv_df.empty:
            display_cv = cv_df.copy()
            display_cv["CV_Accuracy"] = display_cv.apply(lambda r: f"{r['CV_Accuracy_Mean']*100:.2f}% ± {r['CV_Accuracy_Std']*100:.2f}%", axis=1)
            display_cv["CV_Recall"] = display_cv.apply(lambda r: f"{r['CV_Recall_Mean']*100:.2f}% ± {r['CV_Recall_Std']*100:.2f}%", axis=1)
            display_cv["CV_Precision"] = display_cv.apply(lambda r: f"{r['CV_Precision_Mean']*100:.2f}% ± {r['CV_Precision_Std']*100:.2f}%", axis=1)
            display_cv["CV_F1"] = display_cv.apply(lambda r: f"{r['CV_F1_Mean']:.4f} ± {r['CV_F1_Std']:.4f}", axis=1)
            display_cv["CV_ROC_AUC"] = display_cv.apply(lambda r: f"{r['CV_ROC_AUC_Mean']:.4f} ± {r['CV_ROC_AUC_Std']:.4f}", axis=1)
            
            st.dataframe(
                display_cv[["Model", "CV_Accuracy", "CV_Recall", "CV_Precision", "CV_F1", "CV_ROC_AUC", "Best_Parameters"]],
                use_container_width=True,
                hide_index=True
            )
            st.caption("Cross-validation standard deviations indicate performance stability across folds during training.")

    with tab_gap:
        st.markdown('<div class="section-header">Generalization Gap & Overfitting Audit</div>', unsafe_allow_html=True)
        st.write("Comparison of validation performance during training (CV) versus confirmatory performance on the held-out test partition (Unseen Test Data):")
        
        gap_rows = [
            {"Model": "Decision Tree (Tuned)", "CV F1 (Training CV)": "1.0000", "Test F1 (Held-Out Test)": "1.0000", "Absolute Change (Percentage Points)": "0.00 pp", "Relative Change (%)": "0.00%", "Analytical Assessment": "Exact fit on synthetic deterministic boundaries"},
            {"Model": "Gradient Boosting (Tuned)", "CV F1 (Training CV)": "1.0000", "Test F1 (Held-Out Test)": "1.0000", "Absolute Change (Percentage Points)": "0.00 pp", "Relative Change (%)": "0.00%", "Analytical Assessment": "Exact fit on synthetic deterministic boundaries"},
            {"Model": "Random Forest (Tuned)", "CV F1 (Training CV)": "0.9964", "Test F1 (Held-Out Test)": "1.0000", "Absolute Change (Percentage Points)": "+0.36 pp", "Relative Change (%)": "+0.36%", "Analytical Assessment": "Marginal variation across partitions"},
            {"Model": "Logistic Regression (Tuned)", "CV F1 (Training CV)": "0.7931", "Test F1 (Held-Out Test)": "0.7415", "Absolute Change (Percentage Points)": "-5.16 pp (decline of 5.16 percentage points)", "Relative Change (%)": "-6.51% (relative decline of 6.51%)", "Analytical Assessment": "Linear boundary degrades on unseen test partition"}
        ]
        st.dataframe(pd.DataFrame(gap_rows), use_container_width=True, hide_index=True)
        st.markdown("""
            <div class="callout-box">
                <strong>Methodological Finding:</strong> In this evaluated synthetic dataset, tree-based models achieve 100.00% accuracy and F1 on this particular held-out test partition (N=200) because the underlying data generation mechanism followed deterministic step-function rules. This 100% metric is strictly specific to this instructional dataset and held-out partition, and cannot be assumed to guarantee equivalent performance under real-world logistics noise.<br><br>
                In contrast, Logistic Regression exhibits an absolute decline of <strong>5.16 percentage points</strong> in F1-score (from 0.7931 to 0.7415), corresponding to a <strong>6.51% relative decline</strong>. This degradation reflects the inability of a linear hyperplane to separate non-linear logical interactions (e.g. Traffic &ge; 4 AND Weather in [Rain, Storm] AND Distance &gt; 20 km) across varying sample splits.
            </div>
        """, unsafe_allow_html=True)

    # Feature Importance Section
    if not imp_df.empty:
        st.markdown("---")
        st.markdown('<div class="section-header">Global Feature Importance & Learned Operational Associations</div>', unsafe_allow_html=True)
        
        st.markdown("""
            Feature importance is evaluated using **Mean Decrease in Impurity (MDI / Gini Importance)**, computed as the average of the fitted 
            <strong>Random Forest</strong> and <strong>Gradient Boosting</strong> ensemble pipelines (saved in <code>outputs/feature_importance.csv</code>).
        """)

        tab_imp_trans, tab_imp_agg = st.tabs([
            "📊 Transformed Model Columns (15 Encoded Features)",
            "📁 Aggregated by Original Variable (11 Raw Features)"
        ])

        with tab_imp_trans:
            col_imp_chart1, col_imp_desc1 = st.columns([1.3, 1])
            with col_imp_chart1:
                fig_imp_trans = plot_feature_importance_bar(
                    imp_df, 
                    title="<b>Transformed Feature Importance (MDI across Ensemble)</b>",
                    xaxis_title="Mean Decrease in Impurity (%)"
                )
                st.plotly_chart(fig_imp_trans, use_container_width=True)
            with col_imp_desc1:
                st.markdown("##### Transformed Columns Audit (Top 5 = 93.31%)")
                st.markdown("""
                    The top 5 transformed columns account for **93.31%** of total ensemble impurity reduction:
                    - `Warehouse_Processing_Time`: **37.70%**
                    - `Vehicle_Age`: **30.78%**
                    - `Traffic_Congestion`: **10.17%**
                    - `Delivery_Distance`: **8.89%**
                    - `Weather_Condition: 1 (Clear)`: **5.75%**
                    
                    <div class="callout-box" style="font-size: 0.85rem; padding: 0.75rem;">
                        <strong>Categorical Encoding Audit:</strong> <code>Weather_Condition: 1 (Clear)</code> is an individual one-hot dummy column, 
                        not the entire weather variable. Stormy and Rainy conditions form separate dummy columns (0.81% and 0.53% MDI respectively).
                    </div>
                """, unsafe_allow_html=True)

        with tab_imp_agg:
            agg_df = aggregate_feature_importance_by_variable(imp_df)
            col_imp_chart2, col_imp_desc2 = st.columns([1.3, 1])
            with col_imp_chart2:
                fig_imp_agg = plot_feature_importance_bar(
                    agg_df, 
                    title="<b>Aggregated Feature Importance by Original Variable (Sum of Dummies)</b>",
                    xaxis_title="Aggregated Mean Decrease in Impurity (%)"
                )
                st.plotly_chart(fig_imp_agg, use_container_width=True)
            with col_imp_desc2:
                st.markdown("##### Original Variables Audit (Top 5 = 94.65%)")
                st.markdown("""
                    Aggregating one-hot encoded categories back to their original 11 variables (summing to exactly 100.00%):
                    - `Warehouse_Processing_Time`: **37.70%**
                    - `Vehicle_Age`: **30.78%**
                    - `Traffic_Congestion`: **10.17%**
                    - `Delivery_Distance`: **8.89%**
                    - `Weather_Condition` (all 3 dummy categories combined): **7.09%**
                    
                    Together, these top 5 original input variables account for **94.65%** of ensemble impurity reduction.
                """)

        # Decision Tree hierarchy rules & causality notice
        st.markdown("---")
        col_tree_rules, col_imp_notice = st.columns([1.2, 1])
        with col_tree_rules:
            st.markdown("#### Learned Decision Tree Hierarchy")
            st.markdown("The Decision Tree independently identified the operational failure boundaries:")
            rules_text = load_tree_rules_text()
            with st.expander("📄 View Textual Decision Tree Hierarchy", expanded=False):
                st.code(rules_text, language="text")

        with col_imp_notice:
            st.markdown("#### ⚠️ Interpretability & Causality Notice")
            st.markdown("""
                <div class="callout-box" style="font-size: 0.86rem; line-height: 1.5;">
                    Tree feature importance (MDI) measures associative split utility during model training—specifically, how often 
                    a variable was chosen to reduce node impurity. It <strong>does not</strong> represent causal impact or guarantee 
                    that intervening on a feature in real operations will produce an equivalent reduction in delivery failures.
                </div>
            """, unsafe_allow_html=True)

    # Instructional Dataset Caveat
    st.info("ℹ️ **Pedagogical Note:** The 100% test accuracy achieved by tree-based models reflects the deterministic structure of this instructional synthetic dataset and its specific test partition. In real-world urban logistics, stochastic events (traffic incidents, customer unresponsiveness, weather variability) will introduce noise.")


# ==============================================================================
# PAGE 3: LIVE DELIVERY DELAY PREDICTOR
# ==============================================================================
elif nav_page == "⚡ Live Delay Predictor":
    render_brand_header(subtitle="Interactive Dispatch Risk Scoring & Operational Decision Support")

    # Load active model pipeline
    active_pipe = load_model_pipeline(selected_model_name)
    if active_pipe is None:
        st.error(f"Selected model '{selected_model_name}' could not be loaded from disk.")
        st.stop()

    # Active Prediction Engine Banner
    st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid #6366f1; border-radius: 8px; padding: 0.65rem 1.1rem; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 0.72rem; text-transform: uppercase; color: #6366f1; font-weight: 700; letter-spacing: 0.05em;">⚡ Active Prediction Engine</span><br>
                <strong style="color: #0f172a; font-size: 1.05rem;">🤖 {selected_model_name}</strong>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; text-align: right;">
                Controlled by sidebar selector<br>
                Pipeline: <code>models/{selected_model_name.lower().replace(' ', '_')}_pipeline.joblib</code>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Pre-Dispatch Delivery Parameter Entry</div>', unsafe_allow_html=True)

    # Preset Scenario Selector
    scenario_choice = st.selectbox(
        "⚡ Quick-Fill Operational Scenarios (Classroom & Testing Demos):",
        list(PRESET_SCENARIOS.keys()),
        index=0,
        help="Select a pre-configured operational dispatch scenario to test specific failure modes."
    )

    preset_vals = PRESET_SCENARIOS.get(scenario_choice) or {}

    # Synchronize scenario selection with session state
    if "last_scenario_selected" not in st.session_state:
        st.session_state["last_scenario_selected"] = scenario_choice

    if scenario_choice != st.session_state["last_scenario_selected"]:
        st.session_state["last_scenario_selected"] = scenario_choice
        if scenario_choice != "Custom User Input" and preset_vals:
            st.session_state["live_predictor_input"] = pd.DataFrame([preset_vals])[FEATURE_COLUMNS]

    # Form with 2 balanced columns
    with st.form("prediction_input_form"):
        col_form1, col_form2 = st.columns(2)

        with col_form1:
            st.markdown("##### 📦 Warehouse, Fleet & Driver Profile")
            
            wh_proc_time = st.slider(
                "Warehouse Processing / Staging Time (minutes)",
                min_value=10,
                max_value=120,
                value=int(preset_vals.get("Warehouse_Processing_Time", 45)),
                step=1,
                help="Total elapsed time since package arrival at hub before dispatch gate release."
            )

            vehicle_age = st.slider(
                "Delivery Vehicle Age (years)",
                min_value=1,
                max_value=9,
                value=int(preset_vals.get("Vehicle_Age", 4)),
                step=1,
                help="Age of assigned delivery van or two-wheeler."
            )

            driver_exp = st.slider(
                "Driver Delivery Experience (years)",
                min_value=1,
                max_value=19,
                value=int(preset_vals.get("Driver_Experience", 10)),
                step=1,
                help="Tenure of assigned delivery partner."
            )

            num_stops = st.slider(
                "Planned Number of Route Drops / Stops",
                min_value=1,
                max_value=9,
                value=int(preset_vals.get("Num_Stops", 4)),
                step=1,
                help="Intermediate parcel drop-offs scheduled prior to this delivery."
            )

            pkg_weight = st.number_input(
                "Package Weight (kg)",
                min_value=0.5,
                max_value=50.0,
                value=float(preset_vals.get("Package_Weight", 22.5)),
                step=0.5
            )

            fuel_eff = st.number_input(
                "Vehicle Fuel Efficiency (km/l)",
                min_value=5.0,
                max_value=20.0,
                value=float(preset_vals.get("Fuel_Efficiency", 12.5)),
                step=0.5
            )

        with col_form2:
            st.markdown("##### 🗺️ Route, Environment & Scheduling")

            delivery_dist = st.slider(
                "Delivery Distance from Hub (km)",
                min_value=1.0,
                max_value=50.0,
                value=float(preset_vals.get("Delivery_Distance", 22.0)),
                step=0.5,
                help="One-way transit distance to customer doorstep."
            )

            traffic_congestion = st.selectbox(
                "Forecasted Route Traffic Congestion",
                options=[1, 2, 3, 4, 5],
                index=int(preset_vals.get("Traffic_Congestion", 3)) - 1,
                format_func=lambda x: TRAFFIC_OPTIONS[x]
            )

            weather_cond = st.selectbox(
                "Forecasted Weather Conditions",
                options=[1, 2, 3],
                index=int(preset_vals.get("Weather_Condition", 1)) - 1,
                format_func=lambda x: WEATHER_OPTIONS[x]
            )

            delivery_slot = st.selectbox(
                "Promised Delivery Time Slot",
                options=[1, 2, 3],
                index=int(preset_vals.get("Delivery_Slot", 1)) - 1,
                format_func=lambda x: SLOT_OPTIONS[x]
            )

            road_cond = st.selectbox(
                "Route Road Surface Quality Score",
                options=[1, 2, 3, 4, 5],
                index=int(preset_vals.get("Road_Condition_Score", 3)) - 1,
                format_func=lambda x: ROAD_OPTIONS[x]
            )

        submit_btn = st.form_submit_button("⚡ Predict Delivery Risk & Generate Action Plan", use_container_width=True)

    # Handle Form Submission
    if submit_btn:
        form_data = pd.DataFrame([{
            "Delivery_Distance": delivery_dist,
            "Traffic_Congestion": traffic_congestion,
            "Weather_Condition": weather_cond,
            "Delivery_Slot": delivery_slot,
            "Driver_Experience": driver_exp,
            "Num_Stops": num_stops,
            "Vehicle_Age": vehicle_age,
            "Road_Condition_Score": road_cond,
            "Package_Weight": pkg_weight,
            "Fuel_Efficiency": fuel_eff,
            "Warehouse_Processing_Time": wh_proc_time
        }])[FEATURE_COLUMNS]

        is_valid, err_msg = validate_input_features(form_data.iloc[0].to_dict())
        if not is_valid:
            st.error(f"Input Validation Error: {err_msg}")
        else:
            st.session_state["live_predictor_input"] = form_data

    # Display prediction results dynamically whenever live_predictor_input is present
    if "live_predictor_input" in st.session_state and st.session_state["live_predictor_input"] is not None:
        active_input = st.session_state["live_predictor_input"]

        # Run model prediction dynamically with current active pipeline
        pred_result = predict_single_delivery(active_pipe, active_input, selected_model_name)

        st.markdown('<div class="section-header">Prediction Result & Operational Risk Assessment</div>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            if pred_result["predicted_class"] == 1:
                st.markdown(f"""
                    <div class="callout-box callout-danger" style="text-align: center; padding: 1.5rem;">
                        <span style="font-size: 2.2rem;">🚨</span>
                        <h2 style="margin: 0.5rem 0 0.2rem 0; color: #991b1b;">PREDICTED: DELIVERY DELAY</h2>
                        <p style="margin: 0; font-size: 0.95rem; font-weight: 600;">High SLA Breach Risk</p>
                        <hr style="margin: 1rem 0; border: none; border-top: 1px solid rgba(239, 68, 68, 0.3);">
                        <div style="font-size: 0.85rem; color: #7f1d1d;">
                            Engine: <strong>{selected_model_name}</strong><br>
                            Relative Model Score (Uncalibrated): <strong>{pred_result['risk_score']:.2f}</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="callout-box callout-success" style="text-align: center; padding: 1.5rem;">
                        <span style="font-size: 2.2rem;">✅</span>
                        <h2 style="margin: 0.5rem 0 0.2rem 0; color: #166534;">PREDICTED: ON-TIME DISPATCH</h2>
                        <p style="margin: 0; font-size: 0.95rem; font-weight: 600;">SLA Window Safe</p>
                        <hr style="margin: 1rem 0; border: none; border-top: 1px solid rgba(16, 185, 129, 0.3);">
                        <div style="font-size: 0.85rem; color: #14532d;">
                            Engine: <strong>{selected_model_name}</strong><br>
                            Relative Model Score (Uncalibrated): <strong>{pred_result['risk_score']:.2f}</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Operational Root-Cause Diagnosis")
            
            # If Decision Tree is active, extract exact decision path
            if selected_model_name == "Decision Tree":
                explanation = explain_decision_tree_prediction(active_pipe, active_input)
                st.write(explanation["summary_conclusion"])
                
                with st.expander("🔍 Step-by-Step Decision Tree Rule Path", expanded=True):
                    for step in explanation["path_steps"]:
                        st.markdown(f"• **{step['explanation']}**")
            elif selected_model_name == "Logistic Regression":
                st.info(f"Model **Logistic Regression** computed an uncalibrated sigmoid score of **{pred_result['risk_score']:.2f}** based on linear log-odds.")
                wh_time = active_input["Warehouse_Processing_Time"].iloc[0]
                veh_age = active_input["Vehicle_Age"].iloc[0]
                if (wh_time >= 91 or veh_age >= 8) and pred_result["predicted_class"] == 0:
                    st.warning(
                        "⚠️ **Linear Smoothing Discrepancy:** This delivery violates an empirical failure threshold "
                        f"(Warehouse Time = {wh_time} min or Vehicle Age = {veh_age} yrs), yet Logistic Regression predicts On-Time. "
                        "Because logistic regression adds linear weighted terms, highly favorable road parameters (low traffic, "
                        "sunny weather, short distance) mathematically counteract the penalty from warehouse wait or vehicle age. "
                        "This compensatory effect directly explains Logistic Regression's 26 test false negatives and demonstrates "
                        "why non-linear tree models are required for dispatch operations."
                    )
                explanation = None
            else:
                st.info(f"Model **{selected_model_name}** generated an uncalibrated relative score of **{pred_result['risk_score']:.2f}** based on ensemble leaf proportions. Select **Decision Tree** in the sidebar for full step-by-step tree path explainability.")
                explanation = None

        # Supervisor Action Plan
        st.markdown('<div class="section-header">Supervisor Action Plan & Remediation Checklist</div>', unsafe_allow_html=True)
        action_plan = generate_supervisor_action_plan(active_input, pred_result, explanation)

        st.markdown(f"**Action Priority:** `{action_plan['priority']}`")
        for act in action_plan["actions"]:
            st.markdown(f"""
                <div class="callout-box" style="margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong style="color: #0f172a; font-size: 1rem;">Action: {act['action']}</strong>
                        <span style="background: #e2e8f0; color: #334155; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">Owner: {act['department']}</span>
                    </div>
                    <div style="font-size: 0.84rem; color: #64748b; margin: 0.3rem 0;">Trigger: <em>{act['trigger']}</em></div>
                    <div style="font-size: 0.88rem; color: #1e293b; margin-top: 0.4rem;">{act['details']}</div>
                </div>
            """, unsafe_allow_html=True)

        # Download Prediction Receipt
        receipt_df = active_input.copy()
        receipt_df["Predicted_Outcome"] = pred_result["predicted_label"]
        receipt_df["Model_Risk_Score"] = pred_result["risk_score"]
        receipt_df["Model_Used"] = selected_model_name
        
        st.download_button(
            label="📥 Download Dispatch Prediction Record (CSV)",
            data=receipt_df.to_csv(index=False).encode("utf-8"),
            file_name=f"dispatch_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("💡 Select an operational scenario above or configure parameters and click **⚡ Predict Delivery Risk & Generate Action Plan** to evaluate dispatch risk.")


# ==============================================================================
# PAGE 4: OPERATIONS INSIGHTS
# ==============================================================================
elif nav_page == "📊 Operations Insights":
    render_brand_header(subtitle="Deep Root-Cause Analytics & Operational Failure Mode Investigation")

    df_hist = load_historical_data()
    if df_hist.empty:
        st.error("Historical dataset not loaded.")
        st.stop()

    st.markdown('<div class="section-header">The Three Operational Failure Modes Discovered by Machine Learning</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏭 Failure Mode 1: Warehouse Wait",
        "🚛 Failure Mode 2: Aging Fleet",
        "🌧️ Failure Mode 3: Compound Route",
        "🔍 Driver, Stops & Road Analysis"
    ])

    with tab1:
        st.markdown("#### Failure Mode 1: Warehouse Staging Bottleneck (>90 min)")
        st.write("Ananya Rao suspected that unmeasured wait times before dispatch consume the delivery window. In this synthetic dataset, all 268 observed parcels with warehouse staging times of 91 or more minutes were delayed (100% observed delay rate).")
        
        col_wh1, col_wh2 = st.columns([1.3, 1])
        with col_wh1:
            fig_wh_deep = plot_delay_rate_by_feature(
                df_hist,
                col="Warehouse_Processing_Time",
                title="Delay Rate by Warehouse Wait Duration (10 to 119 min)",
                xaxis_title="Warehouse Processing Minutes (Quintiles)",
                is_continuous=True,
                bins=5
            )
            st.plotly_chart(fig_wh_deep, use_container_width=True)
        with col_wh2:
            st.markdown("""
                <div class="callout-box">
                    <h5 style="margin: 0 0 0.5rem 0;">Key Takeaway for Warehouse Operations:</h5>
                    <p style="font-size: 0.88rem; line-height: 1.5; color: #334155;">
                        • <strong>10–75 min wait:</strong> Moderate observed delay rate (~34%).<br>
                        • <strong>76–90 min wait:</strong> Elevated observed delay rate (30.5% in this band; ~33% overall below 91 min).<br>
                        • <strong>>= 91 min wait:</strong> <strong>100.0% Observed Delay Rate (268 of 268 parcels).</strong><br><br>
                        <em>Managerial Policy:</em> Propose a 75-minute warehouse staging target buffer with an 85-minute escalation alert to evaluate in a controlled pilot.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Failure Mode 2: Aging Fleet Unreliability (Vehicles >= 8 Years Old)")
        st.write("Fleet VP Rohit Deshpande requested capital to purchase more vehicles. In this dataset, delay rates are concentrated in vehicles aged 8 and 9 years (100% delayed, 202 records), whereas vehicles aged 1–7 years show moderate observed delay rates (~38.7%). All observed deliveries involving vehicles aged eight or more years were delayed in this dataset. A fleet restriction is a proposed policy to evaluate, not proof that removing those vehicles will eliminate the same number of failures. We recommend a controlled operational pilot and maintenance telemetry audit before making irreversible fleet decisions.")
        
        col_veh1, col_veh2 = st.columns([1.3, 1])
        with col_veh1:
            fig_veh_deep = plot_delay_rate_by_feature(
                df_hist,
                col="Vehicle_Age",
                title="Delay Rate Across Vehicle Age Cohorts (Years)",
                xaxis_title="Vehicle Age in Fleet",
                is_continuous=False
            )
            st.plotly_chart(fig_veh_deep, use_container_width=True)
        with col_veh2:
            st.markdown("""
                <div class="callout-box">
                    <h5 style="margin: 0 0 0.5rem 0;">Fleet Management Finding:</h5>
                    <p style="font-size: 0.88rem; line-height: 1.5; color: #334155;">
                        • <strong>Ages 1 to 7 years:</strong> Moderate observed delay rate (~38.7%, 309 of 798 deliveries).<br>
                        • <strong>Ages 8 and 9 years:</strong> <strong>100.0% Observed Delay Rate (202 of 202 parcels).</strong><br><br>
                        <em>Proposed Policy:</em> Pilot a targeted restriction on 8+ year vehicles on time-critical ShopNow runs, validating maintenance logs before making capital disposal decisions.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Failure Mode 3: Compound Route Transit Stress")
        st.write("When severe traffic (levels 4–5), adverse weather (rain/storm), and long distances (>20 km) coincide, delays become mathematically certain on standard schedules.")
        
        fig_heat_deep = plot_traffic_weather_heatmap(df_hist)
        st.plotly_chart(fig_heat_deep, use_container_width=True)

    with tab4:
        st.markdown("#### Observational Analysis: Driver Tenure, Route Stops, and Road Quality")
        st.write("During Monday operations reviews, operational leaders often attribute delivery failures to driver inexperience, route density, or poor road conditions. In this synthetic dataset, driver experience, number of stops, and road-condition score show little observed association with the delay target. This finding does not establish that these factors have no causal effect in real logistics operations.")

        fig_myth = plot_myth_buster_comparison(df_hist)
        st.plotly_chart(fig_myth, use_container_width=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            fig_stops = plot_delay_rate_by_feature(df_hist, "Num_Stops", "Delay Rate by Number of Route Stops", "Number of Stops")
            st.plotly_chart(fig_stops, use_container_width=True)
        with col_m2:
            fig_road = plot_delay_rate_by_feature(df_hist, "Road_Condition_Score", "Delay Rate by Road Quality Score", "Road Condition (1=Poor to 5=Good)")
            st.plotly_chart(fig_road, use_container_width=True)

        st.caption("Notice that across driver experience cohorts, stop counts, and road quality scores in this dataset, observed delay rates remain near ~50%. Controlled operational pilots and validation against live telemetry are essential before making irreversible recruitment or capital-allocation decisions.")


# ==============================================================================
# PAGE 5: MANAGER ACTION CENTRE
# ==============================================================================
elif nav_page == "🎯 Manager Action Centre":
    render_brand_header(subtitle="Operational Policies, Capital Allocation & ShopNow Turnaround Plan")

    st.markdown('<div class="section-header">ABC Logistics 90-Day Executive Turnaround Plan</div>', unsafe_allow_html=True)

    st.markdown("""
        To resolve ShopNow's 41 daily escalation complaints and protect the contract from re-tendering within the 90-day deadline, 
        Ananya Rao, Meera Iyer, and Rohit Deshpande should align on these four **proposed operational policies**. 
        These policies are derived from empirical dataset analytics to create proactive safety buffers before critical failure thresholds are breached:
    """)

    st.info(
        "💡 **Managerial Safeguard:** All interventions proposed below represent operational hypotheses derived from associative patterns "
        "in this instructional dataset. Prior to making irreversible capital allocations, fleet retirements, or major policy changes, "
        "executive leadership (Rohit Deshpande, Meera Iyer, Ananya Rao) should execute controlled 30-day operational pilots and validate assumptions against live telemetry."
    )

    col_pol1, col_pol2 = st.columns(2)

    with col_pol1:
        st.markdown("""
            <div class="callout-box" style="border-left: 5px solid #6366f1;">
                <h4 style="margin: 0; color: #1e293b;">Proposed Policy 1: Warehouse Staging SLA Protocol</h4>
                <p style="font-size: 0.85rem; color: #64748b; margin: 0.2rem 0 0.6rem 0;">Owner: Warehouse Operations & Sorting Leads</p>
                <ul style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                    <li><strong>Proposed Target Buffer:</strong> Target all parcels to clear intake, sorting, and manifesting within <strong>75 minutes</strong> of arrival (providing a 15-minute margin before the 91-min failure cliff).</li>
                    <li><strong>Proposed Escalation Trigger:</strong> Any parcel approaching <strong>85 minutes</strong> is auto-flagged for priority floor expediting.</li>
                    <li><strong>Observed Dataset Basis:</strong> In the dataset, parcels staged &ge; 91 min suffered a 100% delay rate (268 of 511 total delays; 52.4%).</li>
                    <li><strong>Pilot Recommendation:</strong> Test this buffer in a 30-day controlled trial at the Pune hub before network-wide rollout.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="callout-box" style="border-left: 5px solid #10b981;">
                <h4 style="margin: 0; color: #1e293b;">Proposed Policy 2: Fleet Modernization & Tiering</h4>
                <p style="font-size: 0.85rem; color: #64748b; margin: 0.2rem 0 0.6rem 0;">Owner: Fleet Management & Maintenance Lead</p>
                <ul style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                    <li><strong>Proposed Fleet Policy:</strong> All observed deliveries involving vehicles aged eight or more years were delayed in this dataset. A fleet restriction is a proposed policy to evaluate, not proof that removing those vehicles will eliminate the same number of failures.</li>
                    <li><strong>Capital Allocation Alignment:</strong> Prioritize an operational pilot and maintenance telemetry audit on 8+ year units before committing capital to across-the-board fleet replacement sprees.</li>
                    <li><strong>Observed Dataset Basis:</strong> In the dataset, vehicles aged 8–9 years experienced a 100% delay rate (202 of 511 total delays; 39.5%).</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col_pol2:
        st.markdown("""
            <div class="callout-box" style="border-left: 5px solid #f59e0b;">
                <h4 style="margin: 0; color: #1e293b;">Proposed Policy 3: Dynamic Weather-Traffic Dispatch Matrix</h4>
                <p style="font-size: 0.85rem; color: #64748b; margin: 0.2rem 0 0.6rem 0;">Owner: Central Route Planning & Customer Support</p>
                <ul style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                    <li><strong>Proposed Compound Risk Alert:</strong> When Traffic &ge; 4 and Weather is Rain/Storm on routes &gt; 20 km, trigger proactive route re-sequencing.</li>
                    <li><strong>Proactive Customer Communication:</strong> Issue automated ShopNow customer notifications providing revised 30-minute delivery slot windows <em>before</em> customer escalation occurs.</li>
                    <li><strong>Observed Dataset Basis:</strong> This compound condition accounts for 157 delays (17.2% of dataset; 30.7% of total delays) that occur even with healthy warehouse staging and younger vehicles.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="callout-box" style="border-left: 5px solid #ef4444;">
                <h4 style="margin: 0; color: #1e293b;">Proposed Policy 4: Workforce Strategy Realignment</h4>
                <p style="font-size: 0.85rem; color: #64748b; margin: 0.2rem 0 0.6rem 0;">Owner: HR & Operations Review Board</p>
                <ul style="font-size: 0.88rem; color: #334155; line-height: 1.6; margin: 0; padding-left: 1.2rem;">
                    <li><strong>Driver Tenure Observation:</strong> In this synthetic dataset, driver experience shows negligible correlation with delivery delays ($r = +0.028$). While this does not prove tenure is irrelevant in all real-world scenarios, it suggests that hiring experienced drivers alone is unlikely to resolve the current crisis.</li>
                    <li><strong>Proposed Budget Pilot:</strong> Recommend piloting investments in warehouse intake and sorting automation rather than large-scale senior driver recruitment sprees, validating results against real-world operations.</li>
                    <li><strong>Protect Client Goodwill:</strong> Re-establish ShopNow account trust through data-backed operational transparency.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Shift Supervisor Pre-Departure Checklist</div>', unsafe_allow_html=True)

    check1 = st.checkbox("✅ 1. Confirm package warehouse staging time is under 75 minutes prior to manifesting.")
    check2 = st.checkbox("✅ 2. Confirm assigned delivery van or two-wheeler is under 8 years of age.")
    check3 = st.checkbox("✅ 3. Check route traffic & weather forecast; re-sequence long routes (>20 km) in rainy/stormy weather.")
    check4 = st.checkbox("✅ 4. Verify ShopNow package tag and ensure priority loading in first departure wave.")

    if check1 and check2 and check3 and check4:
        st.success("🎉 All operational dispatch protocols satisfied. Vehicle cleared for on-time yard departure!")
    else:
        st.warning("⚠️ Pending checklist items. Verify all conditions before authorizing vehicle release.")

    st.markdown("""
        <div style="font-size: 0.78rem; color: #64748b; margin-top: 2rem; border-top: 1px solid #e2e8f0; padding-top: 0.5rem;">
            <strong>Disclaimer:</strong> This application provides decision-support risk predictions based on historical patterns to assist managerial judgment. 
            It does not replace on-ground supervisory discretion during unforeseen transit events.
        </div>
    """, unsafe_allow_html=True)
