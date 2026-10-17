# Phase 2: Data Preparation and Reproducible Train/Test Setup Report
**Project:** Predicting Delivery Delays in Last-Mile Logistics  
**Client / Case Subject:** ABC Logistics Pvt. Ltd.  
**Author:** Senior Business Analytics Consultant / Data Analyst  
**Date:** October 2026  
**Status:** Phase 2 Complete — Ready for Review & Approval  

---

## Executive Summary

Following the formal approval of Phase 1 (Source Inspection and Data Understanding), **Phase 2: Data Preparation and Reproducible Train/Test Setup** establishes a rigorous, leak-free, and reproducible data pipeline for ABC Logistics' delivery delay prediction project.

In strict adherence to the project instructions:
- **No machine learning models were trained** in this phase.
- **No performance metrics were calculated or simulated**.
- **No original source files** (`delivery_delay.csv`, `ABC_Delivery_Delay_CASE.docx`, `ABC_Delivery_Delay_CASE.pdf`) were altered, cleaned in-place, or overwritten.
- A fully reproducible data preparation module has been created at [`src/data_preparation.py`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/src/data_preparation.py).
- Clean, stratified training (800 records) and testing (200 records) partitions have been saved to [`data/processed/train.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/train.csv) and [`data/processed/test.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/test.csv).

---

## Step 1: Input Data Validation & Verification

The raw data file [`delivery_delay.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/delivery_delay.csv) was reloaded and audited against the schema established in Phase 1:

| Integrity Check | Expected Specification | Actual Finding | Verification Status |
| :--- | :--- | :--- | :---: |
| **Row Count** | Exactly 1,000 records | 1,000 records | **PASSED** |
| **Column Count** | Exactly 12 columns | 12 columns | **PASSED** |
| **Column Naming** | Preserving original case spelling | Exact match to schema | **PASSED** |
| **Target Variable** | `Delivery_Delay` $\in \{0, 1\}$ | Values $\{0, 1\}$ only | **PASSED** |
| **Missing Values** | 0 missing values across all columns | 0 missing (0.00%) | **PASSED** |
| **Duplicate Rows** | 0 duplicate records | 0 duplicates (0.00%) | **PASSED** |
| **Feature Ranges** | Within operational physical bounds | All values valid | **PASSED** |

### Verified Target Variable Distribution
- **On-Time (`0`):** 489 deliveries (**48.90%**)
- **Delayed (`1`):** 511 deliveries (**51.10%**)
- **Assessment:** Naturally balanced target classes. No artificial sampling (SMOTE or under-sampling) is necessary.

---

## Step 2: Feature Matrix and Target Definition

All 11 candidate features described in Case Study Exhibit 1 have been retained in the initial feature matrix $X$. In alignment with MBA analytics best practices, **features were not discarded simply due to low bivariate correlation with the target**, preserving full information for multivariate modeling and non-linear interactions in Phase 3.

### Feature Inventory & Modeling Roles

| Feature Name | Operational Meaning | Data Type | Modeling Role | Preprocessing Assumption / Rationale |
| :--- | :--- | :---: | :---: | :--- |
| `Delivery_Distance` | Distance from warehouse to destination (km) | `float64` | Continuous Predictor | Scaled for linear models; preserved in km for tree models. |
| `Traffic_Congestion` | Road traffic level (1=Low to 5=High) | `int64` | Ordinal Predictor | Natural order preserved; scaled alongside continuous features in linear models. |
| `Weather_Condition` | Weather status (1=Clear, 2=Rainy, 3=Stormy) | `int64` | Nominal / Categorical | One-hot encoded to avoid imposing arbitrary linear distance between weather states. |
| `Delivery_Slot` | Scheduled time window (1=Morn, 2=Aft, 3=Eve) | `int64` | Nominal Categorical | Unordered operational time blocks; strictly one-hot encoded. |
| `Driver_Experience` | Driver tenure in logistics (years) | `int64` | Continuous Predictor | Retained to empirically test Rohit Deshpande's claims during model evaluation. |
| `Num_Stops` | Planned drops prior to final delivery | `int64` | Count Predictor | Preserved as integer count; standardized for linear models. |
| `Vehicle_Age` | Age of assigned delivery vehicle (years) | `int64` | Continuous / Discrete | Continuous age; key driver of mechanical unreliability at older ages. |
| `Road_Condition_Score` | Route road quality (1=Poor to 5=Excellent) | `int64` | Ordinal Predictor | Natural order preserved (1 to 5); standardized for linear models. |
| `Package_Weight` | Parcel gross weight (kg) | `float64` | Continuous Predictor | Standardized for linear models; preserved in kg for tree models. |
| `Fuel_Efficiency` | Vehicle fuel mileage (km/l) | `float64` | Continuous Predictor | Continuous fleet efficiency metric; standardized for linear models. |
| `Warehouse_Processing_Time` | Elapsed staging/dispatch duration (min) | `int64` | Continuous Predictor | Strongest single predictor; standardized for linear models; kept in minutes for trees. |
| `Delivery_Delay` | **Target Variable** (0=On-time, 1=Delayed) | `int64` | **Target ($y$)** | **Strictly excluded from feature matrix $X$.** |

### Data Leakage Safeguards
- **Pre-Dispatch Temporality:** All 11 features are captured at or before dispatch (package weight at weigh-in, driver/vehicle from roster, weather/traffic forecasts, warehouse wait time at gate release).
- **Zero Post-Dispatch Telemetry:** No mid-transit variables (e.g., breakdown timestamps, customer unavailability, or actual transit duration) are present.
- **No Target Rule Injection:** Target-derived deterministic rules were **not** engineered into $X$.

---

## Step 3: Reproducible Train / Test Split Design

To ensure unbiased model evaluation, the dataset was split into an **80% Training Set** ($N = 800$) and a **20% Test Set** ($N = 200$) using **Stratified Sampling** on `Delivery_Delay` with a fixed random seed (`RANDOM_SEED = 42`).

### Partition Breakdown & Class Balance

| Partition | Total Records | Delayed (`1`) Count | Delayed (`1`) % | On-Time (`0`) Count | On-Time (`0`) % | Proportion of Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Dataset** | 1,000 | 511 | 51.10% | 489 | 48.90% | 100.0% |
| **Training Set (`train.csv`)** | 800 | 409 | 51.125% | 391 | 48.875% | 80.0% |
| **Test Set (`test.csv`)** | 200 | 102 | 51.000% | 98 | 49.000% | 20.0% |

### Split Verification Checks
1. **Overlap Check:** Set intersection of row indices between training and test sets yielded **exactly 0 overlapping rows**.
2. **Completeness Check:** Union of row indices between training and test sets yielded **exactly 1,000 unique rows**.
3. **Partition Isolation:** The test partition is held out and saved separately to serve strictly as an unseen holdout benchmark in Phase 3.

---

## Step 4: Preprocessing Pipeline Architectures

Different machine learning model families have distinct mathematical requirements. In [`src/data_preparation.py`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/src/data_preparation.py), two specialized, scikit-learn compatible `ColumnTransformer` pipelines were constructed.

```
Raw Training Data (X_train: 800 x 11)
       │
       ├──► Linear / Logistic Pipeline ──► StandardScaler (Numeric/Ordinal) ──► 13 Features
       │                                 OneHotEncoder (drop='first')
       │
       └──► Tree-Based Pipeline ─────────► Passthrough (Numeric/Ordinal)    ──► 15 Features
                                         OneHotEncoder (all levels)
```

### 4.1 Linear / Logistic Regression Preprocessor (`get_linear_preprocessor`)
- **Mathematical Need:** Logistic Regression uses gradient-based optimization and penalization ($L_1 / L_2$) that are sensitive to variable scale. Furthermore, linear models cannot naturally handle nominal integer codes without assuming artificial distances.
- **Numerical & Ordinal Transformer:** `StandardScaler()` applied to 9 features (`Delivery_Distance`, `Driver_Experience`, `Num_Stops`, `Vehicle_Age`, `Package_Weight`, `Fuel_Efficiency`, `Warehouse_Processing_Time`, `Traffic_Congestion`, `Road_Condition_Score`).
- **Categorical Transformer:** `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')` applied to `Weather_Condition` and `Delivery_Slot`.
  - Dropping the first category prevents exact multicollinearity (the "dummy variable trap") in linear models.
- **Output Dimensionality:** **13 features** ($9 \text{ numeric} + 2 \text{ weather dummies} + 2 \text{ slot dummies}$).
- **Feature Names:**
  1. `num__Delivery_Distance`
  2. `num__Driver_Experience`
  3. `num__Num_Stops`
  4. `num__Vehicle_Age`
  5. `num__Package_Weight`
  6. `num__Fuel_Efficiency`
  7. `num__Warehouse_Processing_Time`
  8. `num__Traffic_Congestion`
  9. `num__Road_Condition_Score`
  10. `cat__Weather_Condition_2` (Rainy vs Clear baseline)
  11. `cat__Weather_Condition_3` (Stormy vs Clear baseline)
  12. `cat__Delivery_Slot_2` (Afternoon vs Morning baseline)
  13. `cat__Delivery_Slot_3` (Evening vs Morning baseline)

### 4.2 Tree-Based Preprocessor (`get_tree_preprocessor`)
- **Mathematical Need:** Decision Trees, Random Forests, and Gradient Boosters are scale-invariant. Standardizing continuous features obscures natural business units (e.g., standardizing `Warehouse_Processing_Time` into z-scores makes decision trees output splits like `z > 0.82` instead of the intuitive managerial rule `wait > 90 minutes`).
- **Numerical & Ordinal Transformer:** `'passthrough'` preserves raw units (minutes, kilometers, years, 1–5 ratings).
- **Categorical Transformer:** `OneHotEncoder(sparse_output=False, handle_unknown='ignore')` preserves all levels (`Weather_Condition_1, 2, 3` and `Delivery_Slot_1, 2, 3`).
- **Output Dimensionality:** **15 features** ($9 \text{ raw numeric} + 3 \text{ weather flags} + 3 \text{ slot flags}$).

### 4.3 Data Leakage Prevention Guarantee
- All scalers and encoders are fitted **strictly on `X_train`** (`fit_transform`).
- The test set is transformed **only using the parameters (means, standard deviations, categorical encodings) learned from the training set** (`transform`).
- No test-set statistics were computed or exposed during preprocessor fitting.

---

## Step 5: Investigation of the Threshold-Rule Finding

### 5.1 Rechecking the Finding Across Partitions
In Phase 1, an exact boolean failure logic was identified in the instructional dataset. In Phase 2, this characteristic was retested independently across the train and test splits:

$$\text{Predicted Delay} = (\text{Warehouse\_Time} \ge 91) \lor (\text{Vehicle\_Age} \ge 8) \lor (\text{Traffic} \ge 4 \land \text{Weather} \ge 2 \land \text{Distance} > 20)$$

| Partition | Total Records | Mode 1 (Warehouse $\ge 91$) | Mode 2 (Vehicle Age $\ge 8$) | Mode 3 (Compound Route) | Union of Rules | Actual Delays | Rule Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Set** | 800 | 211 | 159 | 132 | **409** | **409** | **100.0%** |
| **Test Set** | 200 | 57 | 43 | 25 | **102** | **102** | **100.0%** |
| **Combined** | 1,000 | 268 | 202 | 157 | **511** | **511** | **100.0%** |

### 5.2 Methodological & Pedagogical Interpretation
1. **Instructional Characteristic:** As noted in the case study footnote, the data was synthetically generated for classroom analytics. The clean separation confirms that the case designers built the simulation around three plausible operational failure modes.
2. **Modeling Integrity Maintained:**
   - Target labels were **never used** to engineer synthetic predictor features.
   - These rules are **not hard-coded** into the machine learning models.
   - The feature matrix presented to ML models contains only the 11 raw operational factors.
3. **Value for Phase 3 Model Benchmarking:**
   - In Phase 3, learned machine learning models (Logistic Regression, Decision Trees, Random Forests, XGBoost) will be evaluated on how accurately and smoothly they learn these underlying operational dynamics from data alone.
   - The threshold rules provide a transparent analytical benchmark to compare learned model boundaries against an ideal rule-based heuristic.

---

## Step 6: Verification Test Results

Execution of [`src/data_preparation.py`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/src/data_preparation.py) was tested directly in the project environment:
- **Execution Return Code:** `0` (Success, no errors or warnings).
- **Reproducibility Test:** Two independent, consecutive runs using `RANDOM_SEED = 42` yielded identical dataframes (`X_train.equals(X_train_2) == True`, `X_test.equals(X_test_2) == True`).
- **File Integrity:** The original source file [`delivery_delay.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/delivery_delay.csv) maintained its exact original file size (56,632 bytes), confirming zero accidental mutation.

---

## Step 7: Summary of Files Created & Modified

| File Path | Status | Purpose / Description |
| :--- | :---: | :--- |
| [`src/data_preparation.py`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/src/data_preparation.py) | **Created** | Modular, fully documented data loading, validation, stratified splitting, and preprocessing pipeline module. |
| [`src/__init__.py`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/src/__init__.py) | **Created** | Package initialization marker for `src`. |
| [`data/processed/train.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/train.csv) | **Created** | Stratified training partition (800 rows, 11 features + `Delivery_Delay`). |
| [`data/processed/test.csv`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/data/processed/test.csv) | **Created** | Held-out stratified testing partition (200 rows, 11 features + `Delivery_Delay`). |
| [`PHASE_2_DATA_PREPARATION.md`](file:///d:/Aniket/Aniket/College/SEM%203/BA/ABC%20Delivery/PHASE_2_DATA_PREPARATION.md) | **Created** | Complete Phase 2 audit and architectural design report. |
| `delivery_delay.csv` | **Untouched** | Original source dataset preserved in its original location. |
| `ABC_Delivery_Delay_CASE.docx` | **Untouched** | Original case document preserved. |
| `ABC_Delivery_Delay_CASE.pdf` | **Untouched** | Original case document preserved. |

---

## Items for User Alignment Ahead of Phase 3

Before initiating Phase 3 (Machine Learning Model Development), here are the key modeling decisions prepared for your review:

1. **Feature Set Strategy (Full vs. Pruned):**
   - *Option A (Recommended):* Train initial models on all 11 features, then train a pruned variant excluding uninformative features (`Driver_Experience`, `Num_Stops`, `Road_Condition_Score`, `Package_Weight`, `Fuel_Efficiency`) to demonstrate parsimony to Meera Iyer.
   - *Option B:* Retain all 11 features across all models without pruning.
2. **Model Evaluation Metrics:**
   - Recommend tracking **Accuracy**, **Precision**, **Recall (Sensitivity)**, **F1-Score**, and **ROC-AUC**, with a special focus on Recall on Delays (minimizing missed late deliveries to protect the ShopNow contract).
3. **Cost-Sensitive Decision Threshold Tuning:**
   - In addition to the default $0.50$ probability threshold, test a cost-sensitive threshold reflecting the contractual penalty of a missed delivery versus the minor operational cost of an early warning/driver reassignment.

---

**Phase 2 is now complete. I will pause here and await your review and approval before starting Phase 3: Machine Learning Model Development.**
