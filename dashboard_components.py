"""
Reusable Dashboard Components, Plotly Charts, and Styling
Project: Predicting Delivery Delays in Last-Mile Logistics — ABC Logistics Pvt. Ltd.
Phase: Phase 5 - Interactive Streamlit Application
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Color Palette Constants
COLOR_PRIMARY = "#1e293b"      # Deep Slate
COLOR_ON_TIME = "#10b981"      # Emerald Green
COLOR_DELAYED = "#ef4444"      # Crimson Red
COLOR_WARNING = "#f59e0b"      # Amber
COLOR_ACCENT = "#6366f1"       # Indigo
COLOR_MUTED = "#64748b"        # Slate Gray
COLOR_BG_CARD = "#f8fafc"      # Soft White/Gray

def inject_custom_css():
    """Injects modern, executive-grade CSS styling for the Streamlit dashboard."""
    st.markdown("""
        <style>
        /* Base typography & layout */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main title header */
        .brand-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #ffffff;
            padding: 1.5rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
        }
        .brand-header h1 {
            color: #ffffff;
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .brand-header p {
            color: #94a3b8;
            margin: 0.35rem 0 0 0;
            font-size: 0.95rem;
        }
        
        /* KPI Cards */
        .kpi-container {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.2rem;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
            border-left: 5px solid #6366f1;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        }
        .kpi-title {
            color: #64748b;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.3rem;
        }
        .kpi-value {
            color: #0f172a;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .kpi-subtitle {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }

        /* Result Callouts */
        .callout-box {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 1.25rem;
            margin: 1rem 0;
        }
        .callout-danger {
            background: #fef2f2;
            border-left: 5px solid #ef4444;
            color: #991b1b;
        }
        .callout-success {
            background: #f0fdf4;
            border-left: 5px solid #10b981;
            color: #166534;
        }
        .callout-warning {
            background: #fffbeb;
            border-left: 5px solid #f59e0b;
            color: #92400e;
        }

        /* Section divider */
        .section-header {
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.4rem;
            margin: 1.5rem 0 1rem 0;
            font-size: 1.25rem;
            font-weight: 600;
            color: #0f172a;
        }
        </style>
    """, unsafe_allow_html=True)


def render_brand_header(subtitle: str = "Last-Mile Delivery Decision Support & Delay Intelligence"):
    """Renders the top executive header banner."""
    st.markdown(f"""
        <div class="brand-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>ABC Logistics | Delivery Intelligence</h1>
                    <p>{subtitle}</p>
                </div>
                <div style="text-align: right; background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 8px;">
                    <span style="font-size: 0.75rem; text-transform: uppercase; color: #cbd5e1; letter-spacing: 0.05em;">Client Focus</span><br>
                    <strong style="color: #f8fafc; font-size: 0.95rem;">ShopNow SLA Account (90-Day Plan)</strong>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, subtitle: str, border_color: str = "#6366f1"):
    """Renders a single styled KPI card."""
    st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {border_color};">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PLOTLY CHART BUILDERS
# ==============================================================================
def plot_delay_donut(df: pd.DataFrame) -> go.Figure:
    """Renders an interactive Donut chart showing overall delay distribution."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No delivery records match active filters",
            showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(
            title=dict(text="<b>Overall Delivery Outcome Distribution</b>", font=dict(size=15)),
            height=320,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        return fig

    counts = df["Delivery_Delay"].value_counts()
    labels = ["On-Time (Class 0)", "Delayed (Class 1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    colors = [COLOR_ON_TIME, COLOR_DELAYED]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        textfont=dict(size=13, family="Inter"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
    )])

    fig.update_layout(
        title=dict(text="<b>Overall Delivery Outcome Distribution</b>", font=dict(size=15)),
        margin=dict(t=40, b=20, l=20, r=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    return fig


def plot_delay_rate_by_feature(
    df: pd.DataFrame,
    col: str,
    title: str,
    xaxis_title: str,
    is_continuous: bool = False,
    bins: int = 5
) -> go.Figure:
    """Renders a grouped bar chart displaying delay rates alongside record counts."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No delivery records match active filters",
            showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=14)),
            height=340,
            margin=dict(t=40, b=40, l=40, r=20)
        )
        return fig

    data = df.copy()
    if is_continuous:
        num_unique = data[col].nunique()
        if num_unique < 2:
            data["bin"] = data[col].astype(str)
            grouped = data.groupby("bin", observed=False)["Delivery_Delay"].agg(["count", "mean"]).reset_index()
            grouped["label"] = grouped["bin"].astype(str)
        else:
            actual_bins = min(bins, num_unique)
            try:
                data["bin"] = pd.qcut(data[col], q=actual_bins, duplicates="drop")
            except (ValueError, IndexError):
                data["bin"] = pd.cut(data[col], bins=actual_bins)
            grouped = data.groupby("bin", observed=False)["Delivery_Delay"].agg(["count", "mean"]).reset_index()
            grouped["label"] = grouped["bin"].astype(str)
    else:
        grouped = data.groupby(col)["Delivery_Delay"].agg(["count", "mean"]).reset_index()
        grouped["label"] = grouped[col].astype(str)

    grouped["delay_rate_pct"] = grouped["mean"] * 100
    grouped["on_time_pct"] = 100 - grouped["delay_rate_pct"]

    # Color code bars based on severity
    bar_colors = [COLOR_DELAYED if r >= 70 else (COLOR_WARNING if r >= 45 else COLOR_ON_TIME) for r in grouped["delay_rate_pct"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["label"],
        y=grouped["delay_rate_pct"],
        name="Delay Rate (%)",
        marker=dict(color=bar_colors),
        text=[f"{r:.1f}%<br>(n={c})" for r, c in zip(grouped["delay_rate_pct"], grouped["count"])],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Delay Rate: %{y:.1f}%<br>Sample Count: %{customdata}<extra></extra>",
        customdata=grouped["count"]
    ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14)),
        xaxis_title=xaxis_title,
        yaxis_title="Delay Rate (%)",
        yaxis=dict(range=[0, 110]),
        margin=dict(t=40, b=40, l=40, r=20),
        height=340
    )
    return fig


def plot_traffic_weather_heatmap(df: pd.DataFrame) -> go.Figure:
    """Renders a 2D Heatmap cross-tabulating Traffic vs Weather Delay Rates."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No delivery records match active filters",
            showarrow=False,
            font=dict(size=14, color="#64748b")
        )
        fig.update_layout(
            title=dict(text="<b>Compound Transit Risk: Traffic Congestion × Weather Status</b>", font=dict(size=14)),
            height=350,
            margin=dict(t=40, b=40, l=40, r=20)
        )
        return fig

    ct = pd.crosstab(
        df["Traffic_Congestion"],
        df["Weather_Condition"],
        values=df["Delivery_Delay"],
        aggfunc="mean"
    ) * 100

    # Ensure all 5 traffic levels (rows) and 3 weather categories (columns) are represented
    ct = ct.reindex(index=[1, 2, 3, 4, 5], columns=[1, 2, 3])

    weather_labels = ["1 - Clear", "2 - Rainy", "3 - Stormy"]
    traffic_labels = ["1 - Very Low", "2 - Low", "3 - Moderate", "4 - Heavy", "5 - Severe"]

    # Format cell text matrix
    text_matrix = []
    for row in ct.values:
        row_text = []
        for val in row:
            if pd.isna(val):
                row_text.append("—")
            else:
                row_text.append(f"{val:.1f}%")
        text_matrix.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=ct.values,
        x=weather_labels,
        y=traffic_labels,
        colorscale="RdYlGn_r",
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=12, family="Inter", color="#000000"),
        colorbar=dict(title="Delay %"),
        hovertemplate="Weather: %{x}<br>Traffic: %{y}<br>Delay Rate: %{text}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Compound Transit Risk: Traffic Congestion × Weather Status</b>", font=dict(size=14)),
        xaxis_title="Weather Condition",
        yaxis_title="Traffic Congestion Level",
        margin=dict(t=40, b=40, l=40, r=20),
        height=350
    )
    return fig


def plot_confusion_matrix_heatmap(tn: int, fp: int, fn: int, tp: int, model_name: str) -> go.Figure:
    """Renders an annotated Confusion Matrix heatmap."""
    z = [[tn, fp], [fn, tp]]
    x_labels = ["Predicted: On-Time (0)", "Predicted: Delayed (1)"]
    y_labels = ["Actual: On-Time (0)", "Actual: Delayed (1)"]

    annotations = [
        [f"True Negative (TN)<br><b>{tn}</b><br>(Clean Clear)", f"False Positive (FP)<br><b>{fp}</b><br>(False Alarm)"],
        [f"False Negative (FN)<br><b>{fn}</b><br>(Missed Delay / Breach)", f"True Positive (TP)<br><b>{tp}</b><br>(Delay Caught)"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, "#f8fafc"], [0.5, "#93c5fd"], [1, "#3b82f6"]],
        showscale=False,
        text=annotations,
        texttemplate="%{text}",
        textfont=dict(size=13, family="Inter")
    ))

    fig.update_layout(
        title=dict(text=f"<b>Held-Out Test Confusion Matrix: {model_name} (N=200)</b>", font=dict(size=14)),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        margin=dict(t=50, b=40, l=40, r=40),
        height=320
    )
    return fig


def plot_model_comparison_bar(test_df: pd.DataFrame) -> go.Figure:
    """Renders a comparative bar chart of model test performance across key metrics."""
    metrics = ["Accuracy", "Recall", "Precision", "F1_Score", "ROC_AUC"]
    models = test_df["Model"].tolist()

    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=[test_df.loc[test_df["Model"] == m, metric].values[0] for m in models],
            text=[f"{test_df.loc[test_df['Model'] == m, metric].values[0]:.2f}" for m in models],
            textposition="auto"
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Held-Out Test Performance Across Candidate Models</b>", font=dict(size=14)),
        yaxis=dict(range=[0, 1.15], title="Score (0.0 to 1.0)"),
        margin=dict(t=40, b=40, l=40, r=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig


def plot_feature_importance_bar(
    imp_df: pd.DataFrame,
    title: Optional[str] = None,
    xaxis_title: str = "Mean Decrease in Impurity (% Split Contribution)"
) -> go.Figure:
    """Renders a horizontal bar chart of tree-based feature importances."""
    clean_df = imp_df.sort_values(by="Mean_Ensemble_Importance", ascending=True).copy()
    
    # Format readable labels distinguishing dummy categories from raw variables
    label_map = {
        "cat__Weather_Condition_1": "Weather_Condition: 1 (Clear)",
        "cat__Weather_Condition_2": "Weather_Condition: 2 (Rainy)",
        "cat__Weather_Condition_3": "Weather_Condition: 3 (Stormy)",
        "cat__Delivery_Slot_1": "Delivery_Slot: 1 (Morning)",
        "cat__Delivery_Slot_2": "Delivery_Slot: 2 (Afternoon)",
        "cat__Delivery_Slot_3": "Delivery_Slot: 3 (Evening)",
    }
    clean_df["Readable_Feature"] = clean_df["Feature"].apply(
        lambda x: label_map.get(x, x.replace("num__", "").replace("cat__", ""))
    )
    
    fig = go.Figure(go.Bar(
        x=clean_df["Mean_Ensemble_Importance"] * 100,
        y=clean_df["Readable_Feature"],
        orientation="h",
        marker=dict(color=clean_df["Mean_Ensemble_Importance"], colorscale="Blues"),
        text=[f"{v*100:.1f}%" for v in clean_df["Mean_Ensemble_Importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>"
    ))

    max_val = clean_df["Mean_Ensemble_Importance"].max() * 100
    x_max = min(100.0, max(45.0, max_val + 8.0))
    chart_title = title if title is not None else "<b>Ensemble Feature Importance (Mean Decrease in Impurity)</b>"

    fig.update_layout(
        title=dict(text=chart_title, font=dict(size=14)),
        xaxis_title=xaxis_title,
        yaxis_title="",
        xaxis=dict(range=[0, x_max]),
        margin=dict(t=40, b=40, l=190, r=40),
        height=max(360, len(clean_df) * 26)
    )
    return fig


def plot_myth_buster_comparison(df: pd.DataFrame) -> go.Figure:
    """Renders a comparative chart showing delay rates across driver experience cohorts in the dataset."""
    # Binned driver experience
    df_copy = df.copy()
    df_copy["drv_tier"] = pd.cut(df_copy["Driver_Experience"], bins=[0, 5, 10, 15, 20], labels=["1-5 yrs (Junior)", "6-10 yrs (Mid)", "11-15 yrs (Senior)", "16-19 yrs (Veteran)"])
    drv_rates = df_copy.groupby("drv_tier", observed=False)["Delivery_Delay"].mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=drv_rates.index.astype(str),
        y=drv_rates.values,
        name="Delay Rate by Driver Tenure",
        marker=dict(color="#64748b"),
        text=[f"{v:.1f}%" for v in drv_rates.values],
        textposition="auto"
    ))

    fig.add_hline(y=51.1, line_dash="dash", line_color=COLOR_DELAYED, annotation_text="Dataset Fleet Avg Delay (51.1%)", annotation_position="top right")

    fig.update_layout(
        title=dict(text="<b>Observational Analysis: Delay Rate Across Driver Experience Tiers</b>", font=dict(size=14)),
        yaxis_title="Delay Rate (%)",
        xaxis_title="Driver Experience Cohort",
        yaxis=dict(range=[0, 75]),
        margin=dict(t=40, b=40, l=40, r=20),
        height=300
    )
    return fig
