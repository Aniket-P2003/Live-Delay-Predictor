# Phase 1 Data Audit & Source Inspection Report
**Project:** Predicting Delivery Delays in Last-Mile Logistics  
**Client / Case Subject:** ABC Logistics Pvt. Ltd.  
**Auditor:** Senior Business Analytics Consultant / Data Analyst  
**Date:** October 2026  
**Status:** Phase 1 Complete — Ready for Review & Approval  

---

## Executive Summary

ABC Logistics operates a mid-sized last-mile delivery network across seven Indian metros, processing ~90,000 parcels daily. The company's value proposition relies on guaranteed same-day or next-slot delivery windows, with missed deliveries incurring contractual financial penalties and client churn. Following a critical escalation of 41 delay complaints in a single day from anchor client ShopNow (~30% of total volume), the executive team faces a 90-day deadline to dramatically cut delivery delays.

Prior to deploying machine learning models, this **Phase 1 Data Audit** was conducted to:
1. Verify the business problem and reconcile differences between case study documents (`ABC_Delivery_Delay_CASE.docx` and `ABC_Delivery_Delay_CASE.pdf`).
2. Conduct an empirical audit of the actual 1,000-delivery dataset (`delivery_delay.csv`).
3. Evaluate the proposed modelling strategy, test leadership hypotheses against data reality, and map operational failure modes.

### Key Audit Findings at a Glance
- **Target Balance:** The target variable `Delivery_Delay` is nearly perfectly balanced (51.1% Delayed, 48.9% On-Time). No synthetic oversampling (e.g., SMOTE) is needed.
- **Data Integrity:** The CSV contains exactly 1,000 rows and 12 columns. There are **0 missing values**, **0 duplicate records**, and **0 out-of-bounds values**.
- **Case Summary Verification:** Summary statistics published in Exhibit 3 of the case study match the actual CSV values down to the first decimal across all 11 features.
- **Rohit Deshpande's Hypotheses Debunked:** Fleet VP Rohit Deshpande argued that delays stem from inexperienced drivers, excessive stops, and poor road conditions. The empirical audit shows **near-zero correlation** with delays for Driver Experience ($r = +0.028$), Number of Stops ($r = -0.018$), and Road Condition Score ($r = -0.038$). Spending capital on hiring senior drivers or blaming road conditions is not supported by the data.
- **Ananya Rao's Hypothesis Validated:** VP of Service Excellence Ananya Rao suspected warehouse dispatch bottlenecks. The data shows `Warehouse_Processing_Time` is the single strongest predictor of delays ($r = +0.448$).
- **Deterministic Failure Modes Uncovered:** The dataset contains three distinct operational breakdown triggers that explain **100.0% of all delayed parcels** (511 out of 511) with zero errors:
  1. **Warehouse Stagnation:** Parcels held in the warehouse for $\ge 91$ minutes are **100% delayed** (268 deliveries).
  2. **Aging Fleet Breakdown:** Delivery vehicles aged $\ge 8$ years suffer a **100% delay rate** (202 deliveries).
  3. **Compounding Route Risk:** Deliveries exceeding 20 km facing severe traffic ($\ge 4$) during adverse weather (rainy/stormy, $\ge 2$) are **100% delayed** (157 deliveries).

---

## Step 1: Case Study Understanding & Document Reconciliation

### 1.1 Business Context & Problem Statement
- **Company Profile:** ABC Logistics Pvt. Ltd., founded in 2016, operates ~3,500 delivery partners and a mixed fleet of vans and two-wheelers across seven Indian metropolitan areas.
- **The Core Business Problem:** Delivery delays have reached ~50%, leading to contract-threatening client dissatisfaction. ShopNow has threatened to re-tender its account unless SLA compliance improves within 90 days.
- **The Management Debate:**
  - **Rohit Deshpande (VP Fleet & Operations):** Believes new drivers lack route knowledge, make too many stops, and road quality is poor. He advocates for increased capital expenditure to hire experienced riders and purchase more vehicles.
  - **Meera Iyer (Finance Director):** Skeptical of continuous capex demands; requires empirical proof of root causes before releasing funds.
  - **Ananya Rao (VP Service Excellence):** Suspects that unmeasured pre-dispatch wait times in warehouses (manifesting, sorting, load delays) consume delivery buffer time.
- **The Core Questions to Answer:**
  1. Build a predictive model to estimate the probability of a delivery being delayed using only pre-dispatch information.
  2. Rigorously evaluate model performance.
  3. Design an interactive decision application for dispatch supervisors and managers to flag high-risk deliveries and trigger proactive interventions.

### 1.2 Comparison: DOCX vs. PDF
Both source documents (`ABC_Delivery_Delay_CASE.docx` and `ABC_Delivery_Delay_CASE.pdf`) were extracted, parsed, and cross-compared line-by-line.

| Aspect | `ABC_Delivery_Delay_CASE.docx` | `ABC_Delivery_Delay_CASE.pdf` | Reconciliation Status |
| :--- | :--- | :--- | :--- |
| **Narrative Body** | Full case text across 36 paragraphs | Full case text across 5 pages | **Identical** narrative content, character quotes, and background figures. |
| **Questions Section** | Unnumbered bullet/paragraph format: <br>• *Build a model to predict...*<br>• *Evaluate the model?*<br>• *Make an application...* | Explicitly numbered questions: <br>• *Q1. Build a model to predict...*<br>• *Q2. Evaluate the model ?*<br>• *Q3. Make an application...* | **Minor formatting divergence.** The analytical questions are substantively identical. |
| **Exhibits** | Exhibit 1 (Data Dictionary), Exhibit 2 (Sample 12 rows), Exhibit 3 (Summary Stats) formatted as Word tables. | Exhibits 1, 2, and 3 formatted as structured text tables. | **Identical** values, column labels, and scales across both formats. |
| **Headers / Footers** | None present in document body. | Running header on pages 1–5: `ABC Logistics – Predicting Delivery Delays – Page X`. | Formatting artifact of PDF export. |
| **Punctuation Nuance** | Questions 2 and 3 end with `?` preceded by a space (` ?`). | Identical spacing before question mark (` ?`). | Maintained identically across documents. |

---

## Step 2: Empirical Dataset Audit (`delivery_delay.csv`)

### 2.1 File Characteristics & Data Integrity
- **File Name:** `delivery_delay.csv`
- **File Size:** 56,632 bytes (~55.3 KB)
- **Total Records (Rows):** 1,000
- **Total Fields (Columns):** 12
- **Missing / Null Values:** **0** (0.00% across all columns)
- **Duplicate Records:** **0** (0.00%)
- **Data Leakage Risk:** **None detected**. All 11 predictor variables represent operational, fleet, environmental, or package characteristics measurable **at or before dispatch**. No post-dispatch telemetry (e.g., actual transit time, en-route breakdown, customer handover timestamp) is present.
- **Identifier Columns:** No arbitrary ID columns (such as `Order_ID` or `Driver_ID`) exist. All columns are valid features or targets.

### 2.2 Column Inventory & Data Dictionary Alignment

The exact column names, data types, value ranges, and alignment with Case Study Exhibit 1 are documented below:

| Column Name (Exact Case) | Data Type | Non-Null Count | Unique Values | Min Value | Max Value | Example (Row 0) | Case Exhibit 1 Scale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Delivery_Distance` | `float64` | 1,000 | 1,000 | 1.23 km | 49.99 km | 19.352 | Continuous (km) |
| `Traffic_Congestion` | `int64` | 1,000 | 5 | 1 | 5 | 4 | Ordinal 1 (low) to 5 (high) |
| `Weather_Condition` | `int64` | 1,000 | 3 | 1 | 3 | 3 | Categorical: 1=Clear, 2=Rainy, 3=Stormy |
| `Delivery_Slot` | `int64` | 1,000 | 3 | 1 | 3 | 2 | Categorical: 1=Morning, 2=Afternoon, 3=Evening |
| `Driver_Experience` | `int64` | 1,000 | 19 | 1 yr | 19 yrs | 16 | Continuous (years) |
| `Num_Stops` | `int64` | 1,000 | 9 | 1 | 9 | 6 | Count (number of intermediate stops) |
| `Vehicle_Age` | `int64` | 1,000 | 9 | 1 yr | 9 yrs | 9 | Continuous / Discrete (years) |
| `Road_Condition_Score` | `int64` | 1,000 | 5 | 1 | 5 | 3 | Ordinal 1 (poor) to 5 (excellent) |
| `Package_Weight` | `float64` | 1,000 | 1,000 | 0.51 kg | 49.96 kg | 33.623 | Continuous (kg) |
| `Fuel_Efficiency` | `float64` | 1,000 | 1,000 | 5.00 km/l | 19.96 km/l | 13.018 | Continuous (km/l) |
| `Warehouse_Processing_Time` | `int64` | 1,000 | 110 | 10 min | 119 min | 58 | Continuous (minutes) |
| `Delivery_Delay` | `int64` | 1,000 | 2 | 0 | 1 | 1 | Binary Target: 0=On-time, 1=Delayed |

### 2.3 Target Variable Analysis (`Delivery_Delay`)
- **Target Definition:** Binary outcome indicating whether the parcel arrived after the promised delivery window (`1 = Delayed`, `0 = On-Time`).
- **Class Distribution:**
  - **Delayed (`1`):** 511 records (**51.10%**)
  - **On-Time (`0`):** 489 records (**48.90%**)
- **Imbalance Assessment:** The target classes are naturally balanced (ratio ~ 1.045 : 1). No class weighting, downsampling, or synthetic minority oversampling (SMOTE) is necessary or advisable.

### 2.4 Verification of Case Exhibit 3 (Summary Statistics)
Each summary statistic presented to Ananya Rao in the case study was compared against the empirical calculations from `delivery_delay.csv`:

| Feature | Case Mean | Actual Mean | Case Std Dev | Actual Std Dev | Case Min | Actual Min | Case Max | Actual Max | Verification Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Delivery_Distance (km)` | 25.0 | 25.0 | 14.3 | 14.3 | 1.2 | 1.2 | 50.0 | 50.0 | **Exact Match** |
| `Traffic_Congestion (1–5)` | 3.0 | 3.0 | 1.4 | 1.4 | 1 | 1 | 5 | 5 | **Exact Match** |
| `Weather_Condition (1–3)` | 2.0 | 2.0 | 0.8 | 0.8 | 1 | 1 | 3 | 3 | **Exact Match** |
| `Delivery_Slot (1–3)` | 2.0 | 2.0 | 0.8 | 0.8 | 1 | 1 | 3 | 3 | **Exact Match** |
| `Driver_Experience (yrs)` | 10.4 | 10.4 | 5.3 | 5.3 | 1 | 1 | 19 | 19 | **Exact Match** |
| `Num_Stops` | 5.0 | 5.0 | 2.6 | 2.6 | 1 | 1 | 9 | 9 | **Exact Match** |
| `Vehicle_Age (yrs)` | 4.9 | 4.9 | 2.5 | 2.5 | 1 | 1 | 9 | 9 | **Exact Match** |
| `Road_Condition_Score (1–5)` | 3.0 | 3.0 | 1.4 | 1.4 | 1 | 1 | 5 | 5 | **Exact Match** |
| `Package_Weight (kg)` | 24.7 | 24.7 | 14.1 | 14.1 | 0.5 | 0.5 | 50.0 | 50.0 | **Exact Match** |
| `Fuel_Efficiency (km/l)` | 12.3 | 12.3 | 4.3 | 4.3 | 5.0 | 5.0 | 20.0 | 20.0 | **Exact Match** |
| `Warehouse_Processing_Time (min)` | 64.4 | 64.4 | 32.2 | 32.2 | 10 | 10 | 119 | 119 | **Exact Match** |

*Conclusion:* The dataset provided in `delivery_delay.csv` is the exact underlying source used to compile the case study exhibits.

### 2.5 Detailed Numerical Distribution Audit

| Feature | Mean | Std Dev | Min | 25% | Median | 75% | Max | IQR | Skewness | Outliers ($1.5 \times \text{IQR}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Delivery_Distance` | 25.023 | 14.315 | 1.227 | 12.563 | 25.344 | 37.472 | 49.986 | 24.909 | +0.039 | 0 |
| `Traffic_Congestion` | 3.038 | 1.431 | 1.000 | 2.000 | 3.000 | 4.000 | 5.000 | 2.000 | -0.024 | 0 |
| `Weather_Condition` | 1.987 | 0.824 | 1.000 | 1.000 | 2.000 | 3.000 | 3.000 | 2.000 | +0.024 | 0 |
| `Delivery_Slot` | 1.983 | 0.827 | 1.000 | 1.000 | 2.000 | 3.000 | 3.000 | 2.000 | +0.032 | 0 |
| `Driver_Experience` | 10.417 | 5.341 | 1.000 | 6.000 | 11.000 | 15.000 | 19.000 | 9.000 | -0.116 | 0 |
| `Num_Stops` | 5.032 | 2.571 | 1.000 | 3.000 | 5.000 | 7.000 | 9.000 | 4.000 | -0.029 | 0 |
| `Vehicle_Age` | 4.930 | 2.540 | 1.000 | 3.000 | 5.000 | 7.000 | 9.000 | 4.000 | +0.028 | 0 |
| `Road_Condition_Score` | 2.960 | 1.404 | 1.000 | 2.000 | 3.000 | 4.000 | 5.000 | 2.000 | +0.056 | 0 |
| `Package_Weight` | 24.690 | 14.077 | 0.512 | 12.577 | 24.356 | 36.699 | 49.961 | 24.122 | +0.078 | 0 |
| `Fuel_Efficiency` | 12.275 | 4.309 | 5.004 | 8.534 | 12.028 | 15.936 | 19.961 | 7.402 | +0.060 | 0 |
| `Warehouse_Processing_Time` | 64.430 | 32.160 | 10.000 | 35.000 | 64.000 | 94.000 | 119.000 | 59.000 | +0.020 | 0 |

**Observations on Distributions:**
1. All continuous and discrete variables exhibit near-zero skewness ($|\text{skew}| < 0.12$).
2. Continuous variables (`Delivery_Distance`, `Package_Weight`, `Fuel_Efficiency`, `Warehouse_Processing_Time`) follow approximately continuous uniform distributions within their stated bounds.
3. Categorical and discrete variables (`Traffic_Congestion`, `Weather_Condition`, `Delivery_Slot`, `Road_Condition_Score`, `Num_Stops`, `Vehicle_Age`) follow uniform discrete frequency distributions across their respective valid values.
4. Using the Tukey $1.5 \times \text{IQR}$ standard, **there are 0 statistical outliers** in any column.

---

## Step 3: Analytical Investigation & Stakeholder Hypothesis Testing

### 3.1 Linear Correlation with `Delivery_Delay`
Evaluating Pearson correlation coefficients between each candidate feature and the target variable:

| Rank | Feature | Pearson Correlation ($r$) | Direction / Strength | Direct Managerial Meaning |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `Warehouse_Processing_Time` | **+0.4478** | Moderate Positive | Long internal processing times strongly increase delay risk. |
| 2 | `Vehicle_Age` | **+0.3442** | Moderate Positive | Older vehicles are significantly more prone to delivery delays. |
| 3 | `Traffic_Congestion` | **+0.1546** | Weak-Moderate Positive | Higher road traffic increases delivery duration and delays. |
| 4 | `Weather_Condition` | **+0.1497** | Weak-Moderate Positive | Rain/storms increase delivery delays relative to clear skies. |
| 5 | `Delivery_Distance` | **+0.0941** | Weak Positive | Longer routes have slightly elevated delay risk. |
| 6 | `Driver_Experience` | **+0.0281** | Negligible | Driver tenure has virtually no correlation with delays. |
| 7 | `Package_Weight` | **+0.0040** | Negligible | Heavier packages do not delay delivery. |
| 8 | `Fuel_Efficiency` | **-0.0128** | Negligible | Vehicle mileage does not impact on-time performance. |
| 9 | `Num_Stops` | **-0.0182** | Negligible | Number of route drops does not meaningfully predict delays. |
| 10 | `Road_Condition_Score` | **-0.0379** | Negligible | Road quality rating does not explain delays. |
| 11 | `Delivery_Slot` | **-0.0395** | Negligible | Scheduled time slot (morning/afternoon/evening) is uninformative. |

### 3.2 Testing the Operations Review Hypotheses

#### A. Rohit Deshpande's Claims (Refuted by Data)
1. **"The new hires don't know the routes..." (Driver Experience):**
   - *Data Evidence:* Binned quintiles show:
     - 1 to 5 yrs experience: **49.1%** delay rate.
     - 6 to 9 yrs experience: **49.5%** delay rate.
     - 10 to 13 yrs experience: **50.9%** delay rate.
     - 14 to 16 yrs experience: **57.1%** delay rate.
     - 17 to 19 yrs experience: **49.7%** delay rate.
   - *Verdict:* **Refuted.** Highly experienced drivers have the exact same ~50% delay rate as new hires. Allocating budget to hire "veteran" riders will waste capital.
2. **"They take too many stops..." (Num_Stops):**
   - *Data Evidence:* 1 stop = 53.3% delay; 5 stops = 53.9% delay; 9 stops = 52.2% delay.
   - *Verdict:* **Refuted.** The number of stops does not drive delivery delay.
3. **"Half our streets are dug up..." (Road Condition Score):**
   - *Data Evidence:* Score 1 (poor) = 52.3% delay; Score 5 (excellent) = 51.0% delay.
   - *Verdict:* **Refuted.** Potholes and road degradation affect all deliveries equally and do not explain why half miss their SLA while half do not.

#### B. Ananya Rao's Hunch (Validated by Data)
- **"How long parcels sometimes sat between arrival and dispatch..." (`Warehouse_Processing_Time`):**
  - *Data Evidence:*
    - 10 to 30 min wait: **34.5%** delay rate.
    - 31 to 53 min wait: **31.5%** delay rate.
    - 54 to 75 min wait: **34.7%** delay rate.
    - 76 to 90 min wait: **56.2%** delay rate.
    - **$\ge 91$ min wait: 100.0% delay rate (268 out of 268 parcels delayed!)**
  - *Verdict:* **Strongly Confirmed.** Pre-dispatch warehouse stagnation directly consumes the delivery window buffer, making on-time completion physically impossible once the wait passes ~90 minutes.

#### C. Rohit's Vehicle Point (Nuanced Truth Uncovered)
- **"Give me budget... to add vehicles" (`Vehicle_Age`):**
  - *Data Evidence:*
    - Vehicles aged 1 to 5 years: ~34%–40% delay rate.
    - Vehicles aged 6 to 7 years: ~34%–50% delay rate.
    - **Vehicles aged 8 and 9 years: 100.0% delay rate (202 out of 202 parcels delayed!)**
  - *Verdict:* **Partially Valid, but Actionably Specific.** Rohit is wrong that ABC needs a general fleet expansion. Rather, vehicles aged 8+ years suffer persistent mechanical unreliability or speed deficits. Retiring or overhauling the 8+ year fleet segment directly eliminates ~39% of all delays.

---

## Step 4: Deep Data Dynamics — Uncovering the Ground Truth Mechanism

Because this dataset was curated for instructional analytics, deep segmentation revealed that the target variable `Delivery_Delay` is generated through **three exact operational failure modes**:

$$\text{Delivery\_Delay} = 1 \iff (\text{Condition 1}) \lor (\text{Condition 2}) \lor (\text{Condition 3})$$

### The Three Failure Modes
1. **Failure Mode 1 (Warehouse Bottleneck):**
   $$\text{Warehouse\_Processing\_Time} \ge 91 \text{ minutes}$$
   - *Volume:* 268 parcels.
   - *Empirical Delay Rate:* **100.0%** (268 delayed, 0 on time).
2. **Failure Mode 2 (Aging Vehicle Breakdown):**
   $$\text{Vehicle\_Age} \ge 8 \text{ years}$$
   - *Volume:* 202 parcels.
   - *Empirical Delay Rate:* **100.0%** (202 delayed, 0 on time).
3. **Failure Mode 3 (Compound Transit Stress):**
   $$(\text{Traffic\_Congestion} \ge 4) \land (\text{Weather\_Condition} \ge 2) \land (\text{Delivery\_Distance} > 20.0 \text{ km})$$
   - *Volume:* 157 parcels.
   - *Empirical Delay Rate:* **100.0%** (157 delayed, 0 on time).

### Perfect Explanatory Power
When these three logical conditions are combined, they account for:
- **Total Positive Matches:** Exactly 511 records.
- **Empirical Delayed Records in Dataset:** Exactly 511 records.
- **Overall Rule Accuracy:** **100.0%** (1,000 out of 1,000 records matched perfectly; 0 false positives, 0 false negatives).
- **Inter-feature Correlations:** All correlations between the 11 candidate features are between $-0.066$ and $+0.057$. Features were generated orthogonally, making multi-collinearity completely non-existent ($VIF \approx 1.0$).

---

## Step 5: Proposed Modelling Strategy Evaluation

### 5.1 Problem Classification
- **Type of Machine Learning Problem:** **Supervised Binary Classification**.
- **Objective:** Predict the probability $P(Y = 1 | X)$ where $Y \in \{0, 1\}$ represents delivery delay, given feature vector $X$ known at dispatch.

### 5.2 Suitability of Logistic Regression as a Baseline
- **Strengths:**
  1. *Direct Interpretability:* Produces odds ratios ($e^{\beta_j}$) allowing business leaders (Meera and Rohit) to understand the percentage increase in delay odds per unit change in warehouse minutes or vehicle age.
  2. *Probability Output:* Provides well-calibrated continuous risk scores ($0.0 \le \hat{p} \le 1.0$) needed by dispatch supervisors to prioritize interventions.
  3. *Regulatory & Stakeholder Buy-in:* Transparent, defensible, and fast to train.
- **Critical Limitations / Risks:**
  1. *Linearity in Log-Odds vs. Step-Function Thresholds:* The true data generation features strict threshold cut-offs ($\ge 91$ min, $\ge 8$ yrs). Standard Logistic Regression fits a smooth sigmoidal transition, which may underestimate delay risk immediately past the threshold.
  2. *Higher-Order Interactions:* Failure Mode 3 is a 3-way non-linear interaction ($\text{Traffic} \ge 4 \times \text{Weather} \ge 2 \times \text{Distance} > 20$). Standard Logistic Regression will fail to capture this compound stress unless explicit interaction terms are engineered.

### 5.3 Recommended Champion & Alternative Models
To address these dynamics in Phase 2, a multi-model benchmarking framework is recommended:
1. **Baseline Model:** Standard Logistic Regression (with standardized continuous features).
2. **Feature-Engineered Logistic Regression:** Logistic Regression with domain threshold indicators (`wh_gt_90`, `veh_age_gt_7`) and route interaction terms.
3. **Tree-Based Models (Decision Tree, Random Forest, XGBoost / LightGBM):** Tree structures naturally partition orthogonal features along axis-aligned thresholds and capture 3-way logical splits effortlessly. A shallow Decision Tree or Random Forest is expected to achieve near-optimal classification performance.

### 5.4 Feature Selection & Preprocessing Strategy for Phase 2

| Feature | Suitability for Model | Recommended Preprocessing | Analytical Rationale |
| :--- | :---: | :--- | :--- |
| `Warehouse_Processing_Time` | **Essential** | Standardize / MinMax scale; evaluate threshold dummy ($\ge 91$) | Strongest predictor ($r = 0.448$). |
| `Vehicle_Age` | **Essential** | Standardize; evaluate threshold dummy ($\ge 8$) | Major failure mode ($r = 0.344$). |
| `Traffic_Congestion` | **Essential** | Keep as ordinal (1–5) or one-hot encode high traffic ($\ge 4$) | Key factor in compound transit delay. |
| `Weather_Condition` | **Essential** | One-hot encode (Clear vs Rainy vs Stormy) | Severe weather interacts with distance and traffic. |
| `Delivery_Distance` | **Essential** | Standardize; evaluate interaction with traffic/weather | Required for compound risk detection. |
| `Delivery_Slot` | Low Value | Nominal One-Hot Encoding (Drop 1st) | Zero direct correlation; test for subtle residual effects. |
| `Driver_Experience` | Inactive / Noise | Standardize if retained; candidate for feature pruning | Zero predictive correlation ($r = 0.028$). |
| `Num_Stops` | Inactive / Noise | Standardize if retained; candidate for feature pruning | Zero predictive correlation ($r = -0.018$). |
| `Road_Condition_Score` | Inactive / Noise | Standardize if retained; candidate for feature pruning | Zero predictive correlation ($r = -0.038$). |
| `Package_Weight` | Inactive / Noise | Standardize if retained; candidate for feature pruning | Zero predictive correlation ($r = 0.004$). |
| `Fuel_Efficiency` | Inactive / Noise | Standardize if retained; candidate for feature pruning | Zero predictive correlation ($r = -0.013$). |

---

## Step 6: Data-Quality Concerns, Risks, and Limitations

1. **Instructional / Synthetic Nature of the Data:**
   - As stated in the case footnote, this dataset was synthetically engineered for educational purposes.
   - Features are completely orthogonal (uncorrelated with each other), which rarely occurs in real-world supply chains (where heavier packages typically reduce fuel efficiency or rainy weather correlates with traffic).
   - Models trained here will perform exceptionally well, but supervisors must recognize that real-world operational environments will exhibit more stochastic noise.
2. **Absence of Post-Dispatch Dynamic Updates:**
   - All factors are frozen at dispatch. If a flat tire occurs mid-route or a customer is unavailable, the model cannot update its prediction.
3. **Sample Size:**
   - $N = 1,000$ completed deliveries provides sufficient statistical power for 11 features (~90 observations per feature), but test partitions will contain ~200 rows, requiring stratified splitting to maintain class ratios.
4. **Encoding Considerations:**
   - `Weather_Condition` and `Delivery_Slot` are stored as numbers (1, 2, 3). While `Weather_Condition` has an inherent severity order (1=Clear < 2=Rainy < 3=Stormy), `Delivery_Slot` is nominal (Morning vs Afternoon vs Evening) and must not be treated as a continuous scale in linear models.

---

## Step 7: Proposed Next-Step Plan for Phase 2

Upon user review and approval of this Phase 1 audit, Phase 2 will execute the following structured roadmap:

1. **Phase 2.1 — Data Preparation & Partitioning:**
   - Formulate clean train/validation/test splits using Stratified Sampling (80% Train, 20% Test, random seed locked for reproducibility).
   - Construct robust preprocessing pipelines: One-Hot Encoding for categorical slots, ordinal scaling for congestion/weather, and standard scaling for continuous features.
2. **Phase 2.2 — Model Development & Benchmarking:**
   - **Model 1 (Baseline):** Regularized Logistic Regression.
   - **Model 2 (Interpretable Tree):** Decision Tree Classifier (pruned to depth 3–4 for managerial transparency).
   - **Model 3 (Ensemble):** Random Forest Classifier.
   - **Model 4 (Gradient Boosting):** LightGBM / XGBoost.
3. **Phase 2.3 — Comprehensive Model Evaluation:**
   - Measure Accuracy, Precision, Recall (Sensitivity), Specificity, F1-Score, and ROC-AUC.
   - Cost-Benefit Matrix: Align decision thresholds with the business penalty structure (penalizing false negatives where a delay is missed vs false positives where unnecessary intervention occurs).
4. **Phase 2.4 — Managerial Decision App Formulation (Phase 3 Preparation):**
   - Translate model rules into clear operational recommendations for dispatchers:
     - Flag Red: Warehouse wait $> 75$ min or Vehicle age $\ge 8$ yrs $\rightarrow$ Reassign to newer vehicle / expedite.
     - Flag Orange: High traffic + rain + distance $> 20$ km $\rightarrow$ Resequence route or warn customer.
     - Flag Green: Low risk $\rightarrow$ Dispatch normally.

---

## Files Created or Modified in Phase 1

1. `inspect_data.py` *(Created)*: Read-only Python inspection script verifying case documents, computing descriptive statistics, auditing data integrity, checking correlations, and validating deterministic failure modes.
2. `PHASE_1_DATA_AUDIT.md` *(Created)*: This complete Phase 1 Source Inspection and Data Audit markdown report.
3. `ABC_Delivery_Delay_CASE.docx` *(Untouched / Unaltered)*.
4. `ABC_Delivery_Delay_CASE.pdf` *(Untouched / Unaltered)*.
5. `delivery_delay.csv` *(Untouched / Unaltered)*.
