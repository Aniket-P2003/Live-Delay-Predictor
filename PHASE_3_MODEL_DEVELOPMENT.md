# Phase 3: Machine Learning Model Development Report
**Project:** Predicting Delivery Delays in Last-Mile Logistics  
**Client / Case Subject:** ABC Logistics Pvt. Ltd.  
**Author:** Senior Business Analytics Consultant / Data Analyst  
**Date:** October 2026  
**Status:** Phase 3 Complete — Ready for Review & Approval  

---

## Executive Summary

Phase 3 transitions the ABC Logistics predictive delivery delay initiative from data preparation to **machine learning model development, validation, and interpretability analysis**. In strict compliance with the project guidelines:
- All model training, hyperparameter tuning, and cross-validation were conducted **exclusively on the 800-row training partition** ([`data/processed/train.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/train.csv)).
- The 200-row held-out test partition ([`data/processed/test.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/test.csv)) was **strictly isolated and untouched**. No test-set predictions, metrics, or tuning adjustments were performed.
- No threshold rules were manually injected into the feature matrices or hard-coded into the model algorithms.
- **Four candidate model architectures** were developed, tuned, and compared:
  1. **Regularized Logistic Regression** (Interpretable linear baseline)
  2. **Decision Tree Classifier** (Constrained, interpretable tree)
  3. **Random Forest Classifier** (Non-linear bagging ensemble)
  4. **Gradient Boosting Classifier** (Non-linear sequential boosting ensemble)
- All workflows were encapsulated within leak-free scikit-learn `Pipeline` objects and saved to [`models/`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/models/) and [`outputs/`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/).

### Key Development Takeaways
- **The Non-Linear Reality of Delivery Logistics:** Tuned tree-based models (**Decision Tree** and **Gradient Boosting**) achieved **100.0% cross-validated F1-score and Accuracy**, successfully discovering the exact operational threshold boundaries from data alone.
- **Why Logistic Regression Fell Short (~79% Accuracy / F1):** Logistic Regression assumes a linear relationship between input features and the log-odds of delay. Because the physical logistics problem is governed by sharp operational cut-offs (e.g., packages waiting $>90$ minutes in warehouses or delivery vans $\ge 8$ years old having near-certain breakdown rates), a linear hyper-plane produces probabilistic smoothing that misclassifies deliveries near the threshold boundaries.
- **Explainability for Leadership:** The tuned Decision Tree produced a simple, transparent 5-level decision tree that translates directly into actionable dispatch rules for Ananya Rao to present to Rohit Deshpande and Meera Iyer.

---

## Step 1: Modeling Setup Verification

The training environment was initialized and audited prior to model fitting:
- **Target Variable:** `Delivery_Delay` (Binary: `0 = On-Time`, `1 = Delayed`).
- **Target Distribution in Training Set ($N = 800$):**
  - Delayed (`1`): **409 records (51.12%)**
  - On-Time (`0`): **391 records (48.88%)**
- **Feature Matrix ($X$):** Exactly 11 candidate predictors from Exhibit 1, strictly excluding the target variable.
- **Feature Preservation:** All 11 features were retained in the primary benchmark to allow empirical algorithms to learn feature relevance without human confirmation bias.

---

## Step 2 & 3: Model Architectures & Leakage-Safe Pipelines

To prevent data leakage during cross-validation, preprocessing transformers were bundled directly with classifiers inside scikit-learn `Pipeline` objects. In each fold of cross-validation, preprocessing parameters (means, standard deviations, categorical levels) were fitted strictly on the 4-fold training split and applied to the 1-fold validation split.

```
Candidate Model Architecture Overview:

1. Logistic Regression Pipeline:
   Raw Input (11 Features) ──► StandardScaler (9 Numeric/Ordinal) ──► LogisticRegression(C=1.0)
                             OneHotEncoder (drop='first')          (Output: 13 Features)

2. Decision Tree Pipeline:
   Raw Input (11 Features) ──► Passthrough (9 Numeric/Ordinal)    ──► DecisionTreeClassifier(depth=5)
                             OneHotEncoder (all levels)            (Output: 15 Features)

3. Random Forest Pipeline:
   Raw Input (11 Features) ──► Passthrough (9 Numeric/Ordinal)    ──► RandomForestClassifier(150 trees)
                             OneHotEncoder (all levels)            (Output: 15 Features)

4. Gradient Boosting Pipeline:
   Raw Input (11 Features) ──► Passthrough (9 Numeric/Ordinal)    ──► GradientBoostingClassifier(100 trees)
                             OneHotEncoder (all levels)            (Output: 15 Features)
```

### Detailed Pipeline Configurations
1. **Logistic Regression Pipeline:**
   - Preprocessing: `StandardScaler` on continuous/ordinal features; `OneHotEncoder(drop='first', handle_unknown='ignore')` on categorical variables.
   - Regularization: L2 Ridge penalty.
   - Grid Search Space: $C \in [0.01, 0.1, 1.0, 5.0, 10.0]$.
   - Optimal Parameter: $C = 1.0$.
2. **Decision Tree Pipeline:**
   - Preprocessing: `'passthrough'` for continuous/ordinal variables (preserving natural physical units: minutes, kilometers, years); `OneHotEncoder` for categoricals.
   - Grid Search Space: `max_depth` $\in [3, 4, 5, 6, 7]$, `min_samples_leaf` $\in [2, 5, 10, 15]$, `criterion` $\in [\text{'gini'}, \text{'entropy'}]$.
   - Optimal Parameters: `max_depth = 5`, `min_samples_leaf = 2`, `criterion = 'gini'`.
3. **Random Forest Pipeline:**
   - Preprocessing: Passthrough numeric; OneHotEncoder categoricals.
   - Grid Search Space: `n_estimators` $\in [50, 100, 150]$, `max_depth` $\in [4, 6, 8, \text{None}]$, `min_samples_leaf` $\in [2, 4, 8]$.
   - Optimal Parameters: `n_estimators = 150`, `max_depth = 6`, `min_samples_leaf = 2`.
4. **Gradient Boosting Pipeline:**
   - Preprocessing: Passthrough numeric; OneHotEncoder categoricals.
   - Grid Search Space: `n_estimators` $\in [50, 100, 150]$, `learning_rate` $\in [0.05, 0.1, 0.2]$, `max_depth` $\in [3, 4, 5]$.
   - Optimal Parameters: `n_estimators = 100`, `learning_rate = 0.05`, `max_depth = 4`.

---

## Step 4: Empirical Cross-Validation Performance (Training Partition Only)

All models were evaluated using **Stratified 5-Fold Cross-Validation** on [`data/processed/train.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/train.csv). The positive class was defined as `Delivery_Delay = 1`.

The full comparison table generated at [`outputs/cv_model_comparison.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/cv_model_comparison.csv) is summarized below:

| Model Architecture | CV Accuracy | CV Precision | CV Recall (Sensitivity) | CV F1-Score | CV ROC-AUC | Train F1-Score | Tuning Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree (Tuned)** | **1.0000** $\pm 0.000$ | **1.0000** $\pm 0.000$ | **1.0000** $\pm 0.000$ | **1.0000** $\pm 0.000$ | **1.0000** $\pm 0.000$ | 1.0000 | Depth=5, MinLeaf=2 |
| **Gradient Boosting (Tuned)** | **1.0000** $\pm 0.000$ | **1.0000** $\pm 0.000$ | **1.0000** $\pm 0.000$ | **1.0000** $\pm 0.000$ | **1.0000** $\pm 0.000$ | 1.0000 | Est=100, LR=0.05, Depth=4 |
| **Gradient Boosting (Baseline)** | 0.9975 $\pm 0.003$ | 0.9976 $\pm 0.005$ | 0.9976 $\pm 0.005$ | 0.9976 $\pm 0.003$ | 1.0000 $\pm 0.000$ | 1.0000 | Default (Depth=3) |
| **Random Forest (Tuned)** | 0.9963 $\pm 0.005$ | 0.9928 $\pm 0.010$ | 1.0000 $\pm 0.000$ | 0.9964 $\pm 0.005$ | 0.9997 $\pm 0.000$ | 1.0000 | Est=150, Depth=6, MinLeaf=2 |
| **Random Forest (Baseline)** | 0.9950 $\pm 0.005$ | 0.9904 $\pm 0.009$ | 1.0000 $\pm 0.000$ | 0.9952 $\pm 0.005$ | 0.9997 $\pm 0.001$ | 0.9997 | Default (Depth=6) |
| **Decision Tree (Baseline)** | 0.9413 $\pm 0.023$ | 0.8983 $\pm 0.034$ | 1.0000 $\pm 0.000$ | 0.9461 $\pm 0.019$ | 0.9895 $\pm 0.003$ | 0.9457 | Depth=4, MinLeaf=5 |
| **Logistic Regression (Tuned)** | 0.7863 $\pm 0.040$ | 0.7881 $\pm 0.046$ | 0.7995 $\pm 0.035$ | 0.7931 $\pm 0.035$ | 0.8740 $\pm 0.018$ | 0.8068 | $C=1.0$ (L2 Penalty) |
| **Logistic Regression (Baseline)** | 0.7863 $\pm 0.040$ | 0.7881 $\pm 0.046$ | 0.7995 $\pm 0.035$ | 0.7931 $\pm 0.035$ | 0.8740 $\pm 0.018$ | 0.8068 | Default ($C=1.0$) |

### Performance Analysis
1. **Decision Tree & Gradient Boosting Domination:** Both tuned Decision Tree and Gradient Boosting achieved **100% cross-validated metrics** across all 5 folds. The standard deviation across folds was $\pm 0.0000$, indicating complete stability.
2. **Zero False Negatives in Tree Ensembles:** Random Forest and Decision Tree achieved **100% Recall** ($1.0000$), meaning they did not miss a single delayed delivery in cross-validation. For ABC Logistics, where missed windows trigger contractual penalties and client attrition from ShopNow, high recall is the primary business objective.
3. **Logistic Regression Ceiling:** Logistic Regression topped out at an F1-score of **0.7931** and an Accuracy of **78.63%**, failing to separate ~21% of deliveries.

---

## Step 5: Model Interpretability & Feature Insights

### 5.1 Logistic Regression Coefficients & Odds Ratios
Extracted from [`outputs/logistic_coefficients.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/logistic_coefficients.csv) (Model Intercept $\beta_0 = -0.8167$):

| Rank | Feature | Standardized Coef ($\beta$) | Odds Ratio ($e^\beta$) | Practical Business Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `num__Warehouse_Processing_Time` | **+1.5190** | **4.57** | Each 1 standard deviation increase in warehouse staging time multiplies the odds of delivery delay by **4.57x**. |
| 2 | `cat__Weather_Condition_3` (Stormy) | **+1.2948** | **3.65** | Deliveries dispatched during stormy conditions face **3.65x higher odds** of delay compared to clear weather. |
| 3 | `num__Vehicle_Age` | **+1.1882** | **3.28** | Each 1 standard deviation increase in vehicle age multiplies delay odds by **3.28x**. |
| 4 | `cat__Weather_Condition_2` (Rainy) | **+1.1638** | **3.20** | Deliveries dispatched during rainy weather face **3.20x higher odds** of delay compared to clear weather. |
| 5 | `num__Traffic_Congestion` | **+0.7093** | **2.03** | Each 1 standard deviation increase in traffic level doubles the odds of delay (**2.03x**). |
| 6 | `num__Delivery_Distance` | **+0.3933** | **1.48** | Longer distance moderately increases delay odds (**1.48x** per std dev). |
| 7 | `cat__Delivery_Slot_2` (Afternoon) | **+0.1350** | **1.14** | Marginal increase in delay odds relative to morning slot. |
| 8 | `cat__Delivery_Slot_3` (Evening) | **+0.0810** | **1.08** | Negligible change in delay odds relative to morning slot. |
| 9 | `num__Driver_Experience` | **+0.0748** | **1.08** | Driver tenure has virtually no effect on delivery delay odds. |
| 10 | `num__Package_Weight` | **-0.0261** | **0.97** | Package weight has zero practical predictive power. |
| 11 | `num__Fuel_Efficiency` | **-0.0240** | **0.98** | Vehicle fuel efficiency has zero practical predictive power. |
| 12 | `num__Road_Condition_Score` | **+0.0056** | **1.01** | Road quality rating has zero practical predictive power. |
| 13 | `num__Num_Stops` | **-0.0028** | **1.00** | Number of route drops has zero practical predictive power. |

### 5.2 Tree-Based Feature Importance Ranking
Extracted from [`outputs/feature_importance.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/feature_importance.csv):

```
Ensemble Feature Importance Breakdown:
  Warehouse_Processing_Time:   37.70%  ███████████████████████
  Vehicle_Age:                 30.78%  ███████████████████
  Traffic_Congestion:          10.17%  ██████
  Delivery_Distance:            8.91%  █████
  Weather_Condition (Clear):    5.75%  ███
  Package_Weight:               1.59%  █
  Fuel_Efficiency:              1.35%  █
  Weather_Condition (Stormy):   0.81%  
  Driver_Experience:            0.78%  
  Num_Stops:                    0.70%  
  Road_Condition_Score:         0.56%  
  Weather_Condition (Rainy):    0.53%  
  Delivery_Slot:                0.36%  
```

In both Random Forest and Gradient Boosting, **over 93% of the total model importance** is concentrated in five operational variables:
1. `Warehouse_Processing_Time`
2. `Vehicle_Age`
3. `Traffic_Congestion`
4. `Delivery_Distance`
5. `Weather_Condition`

The remaining six variables (`Driver_Experience`, `Num_Stops`, `Road_Condition_Score`, `Package_Weight`, `Fuel_Efficiency`, `Delivery_Slot`) contribute virtually zero predictive power.

---

## Step 6: Alignment with Operational Threshold Rules & Learned Logic

### 6.1 The Learned Decision Tree Rules
The tuned Decision Tree classifier (`max_depth = 5, min_samples_leaf = 2`), exported to [`outputs/decision_tree.txt`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/decision_tree.txt), produced the following transparent rule hierarchy purely from training data:

```
|--- num__Warehouse_Processing_Time <= 90.50 min
|   |--- num__Vehicle_Age <= 7.50 yrs
|   |   |--- num__Traffic_Congestion <= 3.50
|   |   |   |--- class: 0 (On-Time)
|   |   |--- num__Traffic_Congestion >  3.50
|   |   |   |--- num__Delivery_Distance <= 19.92 km
|   |   |   |   |--- class: 0 (On-Time)
|   |   |   |--- num__Delivery_Distance >  19.92 km
|   |   |   |   |--- cat__Weather_Condition_1 <= 0.50 (Rainy or Stormy)
|   |   |   |   |   |--- class: 1 (Delayed)
|   |   |   |   |--- cat__Weather_Condition_1 >  0.50 (Clear Weather)
|   |   |   |   |   |--- class: 0 (On-Time)
|   |--- num__Vehicle_Age >  7.50 yrs (Vehicles 8–9 Years Old)
|   |   |--- class: 1 (Delayed)
|--- num__Warehouse_Processing_Time >  90.50 min (Warehouse Staging >= 91 min)
|   |--- class: 1 (Delayed)
```

### 6.2 Analytical Interpretation
1. **Zero Hard-Coding Required:** The decision tree was never given the Phase 1 rules. It was provided raw training data. Purely by minimizing Gini impurity, the tree identified:
   - Split 1: `Warehouse_Processing_Time > 90.50` ($\ge 91$ min) $\rightarrow$ Instant Delay.
   - Split 2: `Vehicle_Age > 7.50` ($\ge 8$ years) $\rightarrow$ Instant Delay.
   - Split 3: Compound risk: `Traffic > 3.50` ($\ge 4$) $\land$ `Distance > 19.92` ($> 20$ km) $\land$ `Weather != Clear` $\rightarrow$ Delay.
2. **Why Logistic Regression Underperformed:**
   - Logistic Regression models probability as an S-curve: $P(Y=1) = \frac{1}{1 + e^{-z}}$.
   - A step-function (where probability jumps discontinuously from ~35% to 100% when warehouse wait hits 91 minutes or vehicle age hits 8 years) cannot be represented by a linear combination without extreme steepness, which distorts predictions across the rest of the feature space.
3. **Association vs. Causation Caution:**
   - While `Vehicle_Age > 7.5` strongly predicts delay, age itself is a proxy for mechanical wear, lower transit speed, or lack of GPS/telematics in older fleet units.
   - Similarly, `Driver_Experience` showing near-zero importance does not mean driver skill is irrelevant; rather, under ABC's current operating regime, driver experience cannot overcome the severe physical handicaps of a 90-minute warehouse delay or a broken-down 9-year-old vehicle.

---

## Step 7: Model Selection & Recommendation for Phase 4

| Candidate Model | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **Tuned Decision Tree** | 100% cross-validation accuracy; perfect recall; 100% human-interpretable rules; easy to encode into operational checklists. | Can be sensitive to small sample changes in non-synthetic datasets. | **Primary Candidate for Operational Dashboard (Managerial Interpretability)** |
| **Tuned Gradient Boosting** | 100% cross-validation accuracy; perfect recall; outputs well-calibrated continuous risk probabilities ($0.0$ to $1.0$). | Less directly inspectable than a single tree. | **Primary Candidate for Automated Backend Scoring (Probability Calibration)** |
| **Tuned Random Forest** | 99.6% F1-score; robust ensemble averaging; zero false negatives. | Larger artifact size (~634 KB) compared to single tree (~5 KB). | **Strong Alternative Ensemble** |
| **Logistic Regression** | Classical econometric baseline; transparent odds ratios. | Inadequate accuracy (~78.6%) due to inability to model sharp step thresholds. | **Retain as Educational / Econometric Baseline** |

### Recommendation for Phase 4 Test Evaluation:
Benchmark all four models against the held-out test partition (`test.csv`) to verify if the 100% cross-validated performance of Decision Tree and Gradient Boosting generalizes to unseen data without overfitting.

---

## Step 8: Inventory of Created Artifacts

All files were generated strictly from training data and are preserved in the project directory:

| Artifact Path | Format | Size | Description |
| :--- | :---: | :---: | :--- |
| [`src/model_training.py`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/src/model_training.py) | Python Script | ~11.5 KB | Modular, leak-free training, CV evaluation, tuning, and serialization pipeline. |
| [`outputs/cv_model_comparison.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/cv_model_comparison.csv) | CSV | ~2.0 KB | Stratified 5-fold cross-validation metrics across all candidate models and configurations. |
| [`outputs/logistic_coefficients.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/logistic_coefficients.csv) | CSV | ~1.1 KB | Standardized coefficients and odds ratios for Logistic Regression baseline. |
| [`outputs/feature_importance.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/feature_importance.csv) | CSV | ~1.4 KB | Normalized feature importance scores across Decision Tree, Random Forest, and Gradient Boosting. |
| [`outputs/decision_tree.txt`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/outputs/decision_tree.txt) | Text | 706 bytes | Full textual hierarchical rule representation of the tuned Decision Tree. |
| [`models/logistic_regression_pipeline.joblib`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/models/logistic_regression_pipeline.joblib) | Binary | 4.3 KB | Fitted, serialized Logistic Regression scikit-learn pipeline. |
| [`models/decision_tree_pipeline.joblib`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/models/decision_tree_pipeline.joblib) | Binary | 4.8 KB | Fitted, serialized Decision Tree scikit-learn pipeline. |
| [`models/random_forest_pipeline.joblib`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/models/random_forest_pipeline.joblib) | Binary | 634 KB | Fitted, serialized Random Forest scikit-learn pipeline. |
| [`models/gradient_boosting_pipeline.joblib`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/models/gradient_boosting_pipeline.joblib) | Binary | 179 KB | Fitted, serialized Gradient Boosting scikit-learn pipeline. |
| [`PHASE_3_MODEL_DEVELOPMENT.md`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/PHASE_3_MODEL_DEVELOPMENT.md) | Markdown | ~16 KB | Comprehensive Phase 3 development and interpretability report. |

---

## Next Steps: Preparation for Phase 4

In Phase 4, the serialized models will be evaluated against the held-out test partition ([`data/processed/test.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/test.csv)). Phase 4 will produce:
1. Final test-set performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
2. Confusion matrices and ROC / Precision-Recall curves.
3. Cost-benefit financial evaluation modeling the business cost of missed SLA penalties vs. proactive intervention costs.
4. Final model selection for the Phase 5 Streamlit application.

---

**Phase 3 is now complete. I will stop here and await your review and approval before beginning Phase 4: Final Model Evaluation on the Held-Out Test Set.**
