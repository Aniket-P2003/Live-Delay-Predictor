# Phase 4: Final Model Evaluation on the Held-Out Test Set Report
**Project:** Predicting Delivery Delays in Last-Mile Logistics  
**Client / Case Subject:** ABC Logistics Pvt. Ltd.  
**Author:** Senior Business Analytics Consultant / Data Analyst  
**Date:** October 2026  
**Status:** Phase 4 Complete — Ready for Review & Approval  

---

## Executive Summary

Phase 4 concludes the formal machine learning evaluation cycle by assessing the pre-specified, frozen candidate model pipelines against the **unseen held-out test partition** ([`data/processed/test.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/test.csv)).

In strict compliance with evaluation integrity standards:
- The held-out test partition was loaded **strictly once** for confirmatory scoring.
- **No hyperparameters were tuned, no classification thresholds were adjusted, no features were re-selected, and no models were retrained** based on test performance.
- All four candidate pipelines developed in Phase 3 were evaluated on the identical 200 test records using a documented decision threshold of **0.50** (with positive class defined as `Delivery_Delay = 1`).
- The operational threshold rules identified in Phase 1 are reported separately as a **descriptive analytical benchmark**, not as a machine learning model.
- A fully reproducible evaluation script has been established at [`src/model_evaluation.py`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/src/model_evaluation.py), and outputs have been saved to [`outputs/test_model_comparison.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/test_model_comparison.csv) and [`outputs/test_confusion_matrices.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/test_confusion_matrices.csv).

### Key Test Results at a Glance

| Candidate Model | Test Accuracy | Test Precision (Delayed) | Test Recall (Delayed) | Test F1-Score | Test ROC-AUC | True Positives (TP) | False Negatives (FN) | True Negatives (TN) | False Positives (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Tuned)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **102** | **0** | **98** | **0** |
| **Gradient Boosting (Tuned)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **102** | **0** | **98** | **0** |
| **Random Forest (Tuned)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **102** | **0** | **98** | **0** |
| **Logistic Regression (Tuned)** | **0.7350** | **0.7379** | **0.7451** | **0.7415** | **0.8285** | **76** | **26** | **71** | **27** |

---

## Step 1: Verification of Saved Models and Held-Out Test Data

### 1.1 Model Artifact Verification
Prior to evaluation, all serialized scikit-learn pipelines saved during Phase 3 were verified for structural integrity:
- [`models/logistic_regression_pipeline.joblib`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/models/logistic_regression_pipeline.joblib) (4,353 bytes): Verified fitted on training set; contains `StandardScaler`, `OneHotEncoder(drop='first')`, and `LogisticRegression(C=1.0)`.
- [`models/decision_tree_pipeline.joblib`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/models/decision_tree_pipeline.joblib) (4,875 bytes): Verified fitted on training set; contains numeric passthrough, `OneHotEncoder`, and `DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)`.
- [`models/random_forest_pipeline.joblib`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/models/random_forest_pipeline.joblib) (634,194 bytes): Verified fitted on training set; contains `RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=2)`.
- [`models/gradient_boosting_pipeline.joblib`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/models/gradient_boosting_pipeline.joblib) (179,170 bytes): Verified fitted on training set; contains `GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4)`.

### 1.2 Held-Out Test Set Verification
The test partition [`data/processed/test.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/test.csv) was loaded and confirmed:
- **Total Records:** Exactly **200 rows** (20% holdout split created in Phase 2).
- **Target Column:** `Delivery_Delay` strictly isolated from input feature matrix $X_{test}$.
- **Target Distribution:**
  - Delayed (`1`): **102 records (51.00%)**
  - On-Time (`0`): **98 records (49.00%)**
- **Feature Space:** Exactly 11 input features matching Exhibit 1 (`Delivery_Distance`, `Traffic_Congestion`, `Weather_Condition`, `Delivery_Slot`, `Driver_Experience`, `Num_Stops`, `Vehicle_Age`, `Road_Condition_Score`, `Package_Weight`, `Fuel_Efficiency`, `Warehouse_Processing_Time`).

---

## Step 2: Test Set Evaluation & Confusion Matrix Analysis

The four candidate models were evaluated on the 200 holdout test cases at a standard decision threshold of $\tau = 0.50$.

```
Test Set Ground Truth:
  Total Deliveries = 200  │  Delayed (Actual = 1) = 102  │  On-Time (Actual = 0) = 98

Confusion Matrix Summary:
  Model                 TP (Caught)   FN (Missed)   TN (Correct)   FP (False Alarm)
  ─────────────────────────────────────────────────────────────────────────────────
  Decision Tree            102             0             98               0
  Random Forest            102             0             98               0
  Gradient Boosting        102             0             98               0
  Logistic Regression       76            26             71              27
```

### Detailed Breakdown by Candidate Model

#### 1. Decision Tree (Tuned: `max_depth = 5, min_samples_leaf = 2`)
- **Accuracy:** **100.0%** (200 / 200 correct)
- **Precision:** **100.0%** (102 / 102 flagged delayed were actually delayed)
- **Recall:** **100.0%** (102 / 102 actual delays successfully flagged)
- **F1-Score:** **1.0000**
- **ROC-AUC:** **1.0000**
- **Confusion Matrix:** $\begin{bmatrix} TN=98 & FP=0 \\ FN=0 & TP=102 \end{bmatrix}$
- **Operational Impact:** Catches 100% of delayed parcels with zero false alarms.

#### 2. Gradient Boosting (Tuned: `n_estimators = 100, lr = 0.05, max_depth = 4`)
- **Accuracy:** **100.0%**
- **Precision:** **100.0%**
- **Recall:** **100.0%**
- **F1-Score:** **1.0000**
- **ROC-AUC:** **1.0000**
- **Confusion Matrix:** $\begin{bmatrix} TN=98 & FP=0 \\ FN=0 & TP=102 \end{bmatrix}$
- **Operational Impact:** Identical discrete classification to the Decision Tree, while providing smooth, monotonic scoring across deliveries.

#### 3. Random Forest (Tuned: `n_estimators = 150, max_depth = 6, min_samples_leaf = 2`)
- **Accuracy:** **100.0%**
- **Precision:** **100.0%**
- **Recall:** **100.0%**
- **F1-Score:** **1.0000**
- **ROC-AUC:** **1.0000**
- **Confusion Matrix:** $\begin{bmatrix} TN=98 & FP=0 \\ FN=0 & TP=102 \end{bmatrix}$
- **Operational Impact:** Ensemble averaging perfectly separates all 200 test cases with zero classification errors.

#### 4. Logistic Regression (Tuned: $C = 1.0$, L2 Penalty)
- **Accuracy:** **73.50%** (147 / 200 correct)
- **Precision:** **73.79%** (76 / 103 flagged delayed were delayed; 27 false alarms)
- **Recall:** **74.51%** (76 / 102 actual delays caught; **26 missed delays**)
- **F1-Score:** **0.7415**
- **ROC-AUC:** **0.8285**
- **Confusion Matrix:** $\begin{bmatrix} TN=71 & FP=27 \\ FN=26 & TP=76 \end{bmatrix}$
- **Operational Impact:** Suffers from both critical error types: misses 26 late deliveries (breaching client SLAs) and triggers 27 false alarms (wasting supervisor time).

---

## Step 3: Business Implications for ABC Logistics

In last-mile operations, statistical error types carry direct operational and strategic consequences:

```
                                 ACTUAL OUTCOME
                         Delayed (1)          On-Time (0)
                   ┌─────────────────────┬─────────────────────┐
      Flagged      │ True Positive (TP)  │ False Positive (FP) │
      Delayed (1)  │  Proactive Action   │  Wasted Dispatch    │
PREDICTED          │  Saves the SLA      │    Intervention     │
                   ├─────────────────────┼─────────────────────┤
      Flagged      │ False Negative (FN) │ True Negative (TN)  │
      On-Time (0)  │  CRITICAL BREACH:   │   Clean Dispatch:   │
                   │  Late Parcel / Churn│   Normal Transit    │
                   └─────────────────────┴─────────────────────┘
```

### 3.1 The Impact of False Negatives (Missed Delays)
- **Operational Reality:** A parcel predicted to arrive on time is dispatched under standard operating procedures without intervention. When it subsequently misses its delivery window, the client experiences a service failure.
- **Strategic Consequence:** For ABC Logistics, ShopNow has lodged 41 complaints in a single day and threatened contract termination within 90 days. Because ShopNow represents nearly one-third of total volume, missed delays (False Negatives) directly jeopardize contract retention and trigger contractual penalty clauses.
- **Model Comparison:**
  - **Decision Tree, Random Forest, Gradient Boosting:** **0 False Negatives**. Zero late parcels slip through undetected.
  - **Logistic Regression:** **26 False Negatives** out of 102 delayed parcels (**25.5% missed failure rate**). Relying on Logistic Regression would cause ABC to miss more than one out of every four delayed parcels.

### 3.2 The Impact of False Positives (False Alarms)
- **Operational Reality:** An on-time parcel is mistakenly predicted as delayed. The dispatch supervisor intervenes unnecessarily (e.g., reassigning drivers, expediting packages, splitting loads, or sending premature warning notifications to customers).
- **Strategic Consequence:** While false positives do not breach customer SLAs, they consume supervisor bandwidth, increase dispatch friction, and erode team trust in the predictive system.
- **Model Comparison:**
  - **Tree Models:** **0 False Positives**. Dispatchers only intervene when genuine failure conditions exist.
  - **Logistic Regression:** **27 False Positives** out of 98 on-time parcels (**27.6% false alarm rate**), causing more than one in four on-time dispatches to trigger unnecessary interventions.

### 3.3 The Operational Trade-Off
In last-mile logistics, the operational objective is asymmetric: **prioritize Recall (minimizing False Negatives)** to protect the ShopNow account, while maintaining acceptable Precision (minimizing False Positives) to avoid operational exhaustion. Tree-based models achieve Pareto optimality on this dataset by maximizing both dimensions simultaneously ($\text{Precision} = 1.0, \text{Recall} = 1.0$).

---

## Step 4: Comparison Between Cross-Validation and Test Performance

| Candidate Model | Cross-Validation Accuracy (Phase 3) | Test Accuracy (Phase 4) | Cross-Validation F1 (Phase 3) | Test F1 (Phase 4) | Cross-Validation ROC-AUC | Test ROC-AUC | Generalization Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Decision Tree** | 1.0000 $\pm 0.000$ | **1.0000** | 1.0000 $\pm 0.000$ | **1.0000** | 1.0000 $\pm 0.000$ | **1.0000** | **Perfect Generalization** (Zero gap) |
| **Gradient Boosting** | 1.0000 $\pm 0.000$ | **1.0000** | 1.0000 $\pm 0.000$ | **1.0000** | 1.0000 $\pm 0.000$ | **1.0000** | **Perfect Generalization** (Zero gap) |
| **Random Forest** | 0.9963 $\pm 0.005$ | **1.0000** | 0.9964 $\pm 0.005$ | **1.0000** | 0.9997 $\pm 0.000$ | **1.0000** | **Perfect Generalization** (+0.0036 gain) |
| **Logistic Regression** | 0.7863 $\pm 0.040$ | **0.7350** | 0.7931 $\pm 0.035$ | **0.7415** | 0.8740 $\pm 0.018$ | **0.8285** | **Generalization Drop** (-0.0513 F1) |

### 4.1 Why Tree Models Generalized Without Overfitting
- The cross-validation performance of Decision Tree and Gradient Boosting on training data ($N = 800$) was not an artifact of overfitting. Both models achieved identical, perfect classification on the unseen test set ($N = 200$).
- Because the true data generation process in this instructional dataset is governed by axis-aligned operational threshold boundaries (`Warehouse_Processing_Time >= 91`, `Vehicle_Age >= 8`, and compound transit stress), tree-based architectures with orthogonal decision splits match the underlying data structure with structural fidelity.

### 4.2 Why Logistic Regression Exhibited a Generalization Drop
- Logistic Regression degraded from **78.6% CV accuracy** to **73.5% test accuracy** (a ~5.1% drop).
- As a linear classifier, Logistic Regression attempts to separate classes with a continuous hyper-plane. In regions of the feature space where sharp threshold cut-offs exist, linear models produce probability estimates near 0.50. When applied to unseen test points in these boundary zones, the rigid planar boundary misclassifies observations, resulting in 26 missed delays and 27 false alarms.

### 4.3 Instructional Synthetic Dataset Caveat
- **Crucial Caution for Business Leaders:** While tree models achieve 100% accuracy on this dataset, **this does not imply that ABC Logistics can achieve 100% delay prediction in live urban operations**.
- In live operations, unmeasured stochastic events occur: flat tires, sudden downpours, traffic accidents, gate access delays, or customer unresponsiveness at the doorstep. The perfect separation here reflects the curated instructional nature of the case dataset (as stated in the case study footnote).
- In production, models will face noise, boundary drift, and missing signals. Therefore, ongoing monitoring, probability scoring, and supervisor discretion remain vital.

---

## Step 5: Separate Evaluation of the Threshold-Rule Benchmark

In Phase 1, analysis revealed that the instructional dataset follows three deterministic operational failure modes. This benchmark was audited separately on the held-out test set:

$$\text{Rule Delay} = (\text{Warehouse\_Time} \ge 91) \lor (\text{Vehicle\_Age} \ge 8) \lor (\text{Traffic} \ge 4 \land \text{Weather} \ge 2 \land \text{Distance} > 20)$$

### Benchmark Results on Test Partition ($N = 200$)
- **Accuracy:** **100.0%** (200 / 200)
- **Precision:** **100.0%** (102 / 102)
- **Recall:** **100.0%** (102 / 102)
- **F1-Score:** **1.0000**
- **Failure Mode 1 (Warehouse $\ge 91$ min):** 57 test deliveries triggered (all delayed).
- **Failure Mode 2 (Vehicle Age $\ge 8$ years):** 43 test deliveries triggered (all delayed).
- **Failure Mode 3 (Compound Transit Stress):** 25 test deliveries triggered (all delayed).
- **Union of Modes:** Exactly 102 test deliveries flagged, matching the 102 actual delayed deliveries with zero false positives.

### Methodological Distinction
- This benchmark represents an **analytical discovery** of how the classroom dataset was formulated.
- It was **not** used to train, tune, or alter any candidate machine learning model.
- Its value is pedagogical and diagnostic: it confirms that the Decision Tree algorithm successfully reverse-engineered the exact operational rules through automated learning from training data alone.

---

## Step 6: Predicted Probability & Score Interpretation

### 6.1 Understanding Model Probability Outputs
Each model produces different types of continuous scoring outputs:
1. **Decision Tree:** Outputs leaf purity fractions. Because the tuned tree leaves are pure, predicted probabilities are discrete ($0.0$ or $1.0$).
2. **Random Forest:** Outputs the proportion of 150 decision trees voting for delay (e.g., $0.78$ indicates 117 of 150 trees voted delayed).
3. **Gradient Boosting:** Computes probabilities via the logistic transformation of additive tree ensembles: $\hat{p} = \frac{1}{1 + e^{-F(x)}}$.
4. **Logistic Regression:** Computes probabilities directly via the standardized sigmoid link function.

### 6.2 Calibration Limitations
- **Documentation of Limitation:** Output values between $0.0$ and $1.0$ represent **model-specific confidence scores**, not guaranteed real-world frequencies.
- Formal calibration (e.g., Platt Scaling or Isotonic Regression via Brier score / reliability diagrams) was not explicitly fitted in this phase.
- In operational use, a score of $0.85$ from Gradient Boosting indicates elevated composite transit risk, but supervisors should interpret it as a prioritized operational risk ranking rather than an exact actuarial probability.

---

## Step 7: Final Model Recommendation for Phase 5 (Application Development)

Based on empirical test performance, executive interpretability, and operational feasibility, the recommended deployment architecture for Phase 5 is:

### **Primary Recommendation: Dual-Engine Model Strategy**

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ABC LOGISTICS DISPATCH DECISION SYSTEM               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
┌────────────────────────────────────┐    ┌───────────────────────────────────┐
│     OPERATIONAL EXPLAINABILITY     │    │     CONTINUOUS RISK PROBABILITY   │
│         ENGINE: DECISION TREE      │    │     ENGINE: GRADIENT BOOSTING     │
├────────────────────────────────────┤    ├───────────────────────────────────┤
│ • 100% Test Accuracy & F1          │    │ • 100% Test Accuracy & F1         │
│ • 0 False Negatives (100% Recall)  │    │ • Smooth Continuous Scoring (0-1) │
│ • 100% Human-Readable If-Then Rules│    │ • Fine-Grained Risk Prioritization│
│ • Clear Supervisor Interventions:  │    │ • Dynamic Dispatch Thresholding   │
│   - "Hold: Warehouse > 90 min"     │    │ • Automated Route Triage          │
│   - "Swap: Vehicle Age >= 8 yrs"   │    │                                   │
└────────────────────────────────────┘    └───────────────────────────────────┘
```

1. **Why the Tuned Decision Tree is Recommended for the UI:**
   - **Full Explainability for Dispatch Floor Supervisors:** The decision tree generates transparent operational diagnoses (e.g., *"Flagged because Warehouse Wait = 98 min > 90.5 min threshold"* or *"Flagged because Van Age = 9 yrs >= 8 yrs threshold"*). A supervisor cannot act on an abstract black-box number; they need to know *what* to fix (e.g., expedite packaging vs. swap the vehicle).
   - **Executive Defensibility:** Ananya Rao can project the tree diagram directly onto a boardroom screen to prove to Rohit Deshpande that driver tenure is irrelevant, while vehicle retirement (8+ years) and warehouse bottlenecks (>90 min) solve 85%+ of delays.
2. **Why Gradient Boosting is Recommended as the Scoring Engine:**
   - In automated backend environments where parcels must be ranked by continuous risk from highest to lowest, Gradient Boosting provides smooth, continuous probability estimates suitable for queue prioritization.
3. **Why Logistic Regression is Deprecated for Deployment:**
   - With 26 missed delays out of 102 test cases, deploying Logistic Regression would fail to solve ShopNow's delivery crisis and leave ABC exposed to contract re-tendering.

---

## Step 8: Inventory of Deliverables & Project Files

| Artifact | File Path | Status | Purpose |
| :--- | :--- | :---: | :--- |
| **Evaluation Script** | [`src/model_evaluation.py`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/src/model_evaluation.py) | **Created** | Reproducible script loading test data and evaluating models. |
| **Test Metrics Table** | [`outputs/test_model_comparison.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/test_model_comparison.csv) | **Created** | Confirmatory test metrics across all four models. |
| **Confusion Matrices** | [`outputs/test_confusion_matrices.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/test_confusion_matrices.csv) | **Created** | Actual vs. predicted confusion matrices on the test set. |
| **Phase 4 Report** | [`PHASE_4_MODEL_EVALUATION.md`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/PHASE_4_MODEL_EVALUATION.md) | **Created** | Complete evaluation report and business synthesis. |
| **Held-Out Test Data** | [`data/processed/test.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/test.csv) | **Untouched** | Evaluated once; original holdout data preserved. |
| **Training Data** | [`data/processed/train.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/train.csv) | **Untouched** | Preserved from Phase 2. |
| **Model Artifacts** | `models/*.joblib` | **Untouched** | Preserved from Phase 3. |
| **Source Documents** | `delivery_delay.csv`, `*.docx`, `*.pdf` | **Untouched** | Preserved from Phase 1. |

---

**Phase 4 is now complete. I will stop here and await your review and approval before beginning Phase 5: Interactive Streamlit Application Development.**
