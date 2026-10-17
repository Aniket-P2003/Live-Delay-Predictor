# ABC Logistics | Delivery Intelligence
## Predicting Delivery Delays in Last-Mile Logistics

An interactive, executive-grade business analytics decision-support application built for **ABC Logistics Pvt. Ltd.** to predict, diagnose, and mitigate last-mile delivery delays on the anchor quick-commerce **ShopNow** account.

---

## 1. Business Context & Strategic Challenge
- **Company Profile:** ABC Logistics operates ~3,500 delivery partners across 7 Indian metropolitan areas, moving ~90,000 parcels daily with promised same-day or next-slot delivery windows.
- **The Crisis:** Nearly 50% of sampled deliveries were missing their promised windows. Anchor client **ShopNow** (accounting for nearly a third of total volume) lodged 41 escalation complaints in a single day and issued a **90-day ultimatum**: achieve decisive SLA turnaround or the contract will be re-tendered.
- **Executive Deadlock:**
  - *Rohit Deshpande (VP Fleet & Operations):* Blamed inexperienced drivers, excessive stops, and poor roads; requested large capital expenditure to hire senior riders and buy more vehicles.
  - *Meera Iyer (Finance Director):* Skeptical of continuous capex; demanded empirical proof of root causes before signing off on funds.
  - *Ananya Rao (VP Service Excellence):* Suspected pre-dispatch warehouse bottlenecks (parcels sitting before gate departure).
- **The Data Analytics Solution:** An empirical machine learning application evaluating 11 pre-dispatch features to predict delay probability, explain root causes, and provide immediate remediation checklists for dispatch supervisors.

---

## 2. Technology Stack & Architecture
- **Language & Core:** Python 3.13, Pandas, NumPy
- **Machine Learning:** Scikit-learn (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting)
- **Web Interface:** Streamlit (Multi-page dashboard with custom modern CSS)
- **Data Visualizations:** Interactive Plotly charts (heatmaps, donuts, binned rate comparisons, confusion matrices)
- **Serialization:** Joblib

### Directory Structure
```
ABC Delivery/
├── app.py                         # Main Streamlit application entry point
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation and user guide
├── delivery_delay.csv             # Original historical delivery dataset (1,000 records)
├── ABC_Delivery_Delay_CASE.docx   # Case study distribution document
├── ABC_Delivery_Delay_CASE.pdf    # Case study distribution document
├── inspect_data.py                # Phase 1 data audit & profiling script
├── PHASE_1_DATA_AUDIT.md          # Phase 1 Source inspection report
├── PHASE_2_DATA_PREPARATION.md    # Phase 2 Reproducible data preparation report
├── PHASE_3_MODEL_DEVELOPMENT.md   # Phase 3 ML model development & CV report
├── PHASE_4_MODEL_EVALUATION.md    # Phase 4 Final held-out test evaluation report
│
├── src/
│   ├── __init__.py                # Package marker
│   ├── app_utils.py               # Data loaders, caching, validation & scenario definitions
│   ├── dashboard_components.py    # Plotly figures, KPI cards, custom CSS styling
│   ├── data_preparation.py        # Stratified train/test splitting & ColumnTransformer pipelines
│   ├── model_evaluation.py        # Confirmatory test evaluation script
│   ├── model_training.py          # Cross-validation & hyperparameter tuning pipeline
│   └── prediction_service.py      # Inference engine, tree path explainer & action plans
│
├── data/
│   └── processed/
│       ├── train.csv              # Stratified training split (800 records, 80%)
│       └── test.csv               # Held-out test split (200 records, 20%)
│
├── models/
│   ├── decision_tree_pipeline.joblib       # Tuned Decision Tree pipeline (Champion)
│   ├── gradient_boosting_pipeline.joblib   # Tuned Gradient Boosting pipeline
│   ├── random_forest_pipeline.joblib       # Tuned Random Forest pipeline
│   └── logistic_regression_pipeline.joblib # Regularized Logistic Regression pipeline
│
├── outputs/
│   ├── cv_model_comparison.csv    # Stratified 5-fold cross-validation metrics
│   ├── decision_tree.txt          # Textual export of learned decision tree rules
│   ├── feature_importance.csv     # Normalized tree-based feature importances
│   ├── logistic_coefficients.csv  # Standardized coefficients & odds ratios
│   ├── test_confusion_matrices.csv# Confusion matrices on held-out test partition
│   └── test_model_comparison.csv  # Final confirmatory metrics on held-out test set
│
└── tests/
    └── test_application.py        # Automated test suite (data, models, charts, predictions)
```

---

## 3. Installation & Launch Instructions

### Prerequisites
Ensure Python 3.10+ is installed on your system.

### Step 1: Install Dependencies
Open PowerShell or your terminal in the project directory:
```bash
pip install -r requirements.txt
```

### Step 2: Run Automated Test Suite
Verify that all data loaders, model pipelines, and prediction services pass unit testing:
```bash
python tests/test_application.py
```

### Step 3: Launch the Streamlit Dashboard
Launch the application with the following command:
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 4. Application Guide: The 5 Functional Modules

### Module 1: 🏢 Executive Overview
- **Purpose:** Immediate situational awareness of last-mile delivery SLA compliance.
- **Features:**
  - Executive KPI Cards: Monitored dispatches (1,000), Historical Delays (511, 51.1%), On-Time Deliveries (489, 48.9%), and ShopNow Escalation Alert.
  - Interactive SLA Distribution Donut chart.
  - Multi-dimensional filters (Weather, Scheduled Slot, Traffic level).
  - Delay rate across warehouse staging times and vehicle age tiers.
  - Downloadable filtered historical dataset (CSV).

### Module 2: 🧪 Model Performance Lab
- **Purpose:** Transparent evaluation of the candidate machine learning models.
- **Features:**
  - Confirmatory test results on the unseen 200-record holdout partition across Accuracy, Precision, Recall, F1, and ROC-AUC.
  - Interactive model selector with annotated Confusion Matrix heatmaps (True Positives, False Positives, True Negatives, False Negatives).
  - Clear explanation of business costs: False Negatives (missed delays $\rightarrow$ ShopNow contract cancellation) vs False Positives (wasted interventions).
  - Global Feature Importance bar chart and raw Decision Tree textual rule hierarchy.

### Module 3: ⚡ Live Delivery Delay Predictor
- **Purpose:** Primary interactive decision-support tool for dispatch supervisors.
- **Features:**
  - Full parameter inputs across all 11 pre-dispatch features.
  - **Quick-Fill Scenarios:** Pre-loaded one-click test cases (e.g. *Warehouse Stagnation*, *Aging Vehicle Breakdown*, *Compound Route Stress*, *Clean Dispatch*, *Rookie Driver Myth*).
  - Instant delay prediction with relative model risk scores.
  - **White-Box Decision Tree Explainability:** Automatically extracts the exact split conditions from the fitted scikit-learn decision tree and displays human-readable root causes.
  - **Supervisor Action Plan:** Specific, actionable recommendations (e.g., "Reassign to vehicle <8 yrs", "Expedite warehouse staging <90 min", "Route resequencing & customer notification").
  - Downloadable prediction receipt (CSV).

### Module 4: 📊 Operations Insights
- **Purpose:** In-depth investigation of the underlying operational dynamics.
- **Features:**
  - Dedicated tabs exploring the Three Operational Failure Modes:
    1. *Warehouse Wait:* The 91-minute threshold cliff-edge.
    2. *Aging Fleet:* The 8+ year vehicle unreliability cliff-edge.
    3. *Compound Transit:* Severe traffic $\times$ rain/storm $\times$ long distance.
  - **Observational Analysis Panel:** Demonstrates that in this synthetic dataset, driver experience ($r = +0.028$), number of stops ($r = -0.018$), and road condition scores ($r = -0.038$) show little observed association with delivery delays. Explicitly notes that this finding does not establish that these factors have no causal effect in real logistics operations, and recommends operational pilots before altering recruitment strategy.

### Module 5: 🎯 Manager Action Centre
- **Purpose:** Strategic turnaround policies and operational SOPs for executive leadership.
- **Features:**
  - The 90-Day ShopNow Turnaround Plan (4 proposed operational policies).
  - Capital allocation justification aligning Ananya Rao, Meera Iyer, and Rohit Deshpande with pilot testing safeguards.
  - Interactive Shift Supervisor Pre-Departure Checklist for vehicle yard clearance.

---

## 5. Machine Learning Summary & Operational Findings

| Model | Cross-Validation F1 (Phase 3) | Test Accuracy (Phase 4)* | Test Recall (Delayed) | Test Precision | Test F1-Score | Operational Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Decision Tree (Tuned)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **Champion:** 100% white-box explainability for floor supervisors |
| **Gradient Boosting (Tuned)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **Champion:** Smooth continuous risk scoring for automated queues |
| **Random Forest (Tuned)** | **0.9964** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | Strong ensemble benchmark |
| **Logistic Regression (Tuned)** | **0.7931** | **0.7350** | **0.7451** | **0.7379** | **0.7415** | Deprecated: Misses 25.5% of delays due to linear hyperplane smoothing |

*\*Note: 100% test performance is strictly specific to this evaluated synthetic dataset and its particular held-out test partition ($N=200$).*

### The Three Operational Failure Modes Discovered by Models
$$\text{Delivery Delay} = 1 \iff (\text{Warehouse Wait} \ge 91\text{ min}) \lor (\text{Vehicle Age} \ge 8\text{ yrs}) \lor (\text{Traffic} \ge 4 \land \text{Weather} \in \{\text{Rain, Storm}\} \land \text{Distance} > 20\text{ km})$$
- **Warehouse Bottleneck:** Observed in **52.4% of all delays** (268 parcels).
- **Aging Fleet Breakdown:** Observed in **39.5% of all delays** (202 parcels).
- **Compound Transit Stress:** Observed in **17.2% of all delays** (157 parcels).

---

## 6. Disclaimers, Pilot Recommendations & Limitations
- **Instructional Synthetic Dataset:** As noted in the case study documentation, the dataset was synthetically compiled for business analytics education. While tree models achieve 100% test accuracy on this curated dataset, live urban logistics environments contain stochastic noise (flat tires, gate access delays, customer unavailability).
- **Proposed Policies vs Causal Proof:** All proposed interventions (e.g. 75-min warehouse target, 8+ year vehicle restrictions) are proposed operational hypotheses derived from associative patterns in this dataset, not causal proof that removing older vehicles or setting warehouse targets will automatically eliminate failures.
- **Pilot Recommendation:** Prior to making irreversible capital expenditures or fleet retirements, executive management should conduct controlled 30-day operational pilots and validate assumptions against live telemetry.
- **Decision Support:** Predictions generated by this system are designed to assist and inform supervisory decision-making, not replace on-ground human judgment.
