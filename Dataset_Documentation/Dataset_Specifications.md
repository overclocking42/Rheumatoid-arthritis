# Dataset Specifications & Documentation

**Generated**: November 23, 2025  
**Project**: Rheumatoid Arthritis Diagnosis AI System  
**Data Type**: Medical Blood Biomarkers (Numeric Classification)  
**Status**: Research-Grade Documentation  

---

## Executive Summary

This document provides comprehensive specifications for the Rheumatoid Arthritis (RA) diagnosis numeric dataset. The dataset comprises blood test biomarkers from 3,848 patient records, split into training (2,658), validation (570), and test (570) sets for model development and evaluation.

### Key Metrics

| Metric | Value |
|--------|-------|
| **Total Samples** | 3,798 (after processing) |
| **Original Pool** | 3,848 |
| **Features** | 6 (Age, Gender, RF, Anti-CCP, CRP, ESR) |
| **Target Classes** | 3 (Healthy, Seropositive RA, Seronegative RA) |
| **Training Set** | 2,658 samples (70%) |
| **Validation Set** | 570 samples (15%) |
| **Test Set** | 570 samples (15%) |
| **Class Imbalance Ratio** | 5.6:1 (Seropositive:Healthy) |
| **Missing Data Rate** | 9-21% across features |

---

## 1. Dataset Overview

### 1.1 Purpose

This dataset is designed to train machine learning models for classifying Rheumatoid Arthritis diagnosis status based on blood biomarkers. Three classification categories are supported:

- **Healthy**: No RA diagnosis
- **Seropositive RA**: RF+ Rheumatoid Arthritis (antibody-positive)
- **Seronegative RA**: RF- Rheumatoid Arthritis (antibody-negative, clinically diagnosed)

### 1.2 Data Collection

**Source**: Hospital blood test laboratory records  
**Time Period**: Historical medical records (year not specified)  
**Patient Demographics**: Mixed adult population (ages 20-80)  
**Clinical Context**: Patients presenting with suspected or diagnosed RA

### 1.3 Data Quality

- **Completeness**: 98.7% (3,798/3,848 valid samples)
- **Missing Values**: Selective missingness in biomarkers (see Feature Details)
- **Outliers**: Data validated for clinical plausibility
- **Duplicates**: No duplicate records identified

---

## 2. Class Distribution

### 2.1 Target Label Encoding

```
Label ID | Class Name           | Description
---------|----------------------|-------------------------------------------
   0     | Healthy              | No RA; negative biomarkers
   1     | Seropositive RA      | RF+ RA; positive antibodies
   2     | Seronegative RA      | RF- RA; clinically diagnosed without RF
```

### 2.2 Dataset-Wide Distribution (All 3,798 Samples)

| Class | Count | Percentage | Imbalance Ratio |
|-------|-------|-----------|-----------------|
| Seropositive RA | 2,838 | 74.7% | 5.68:1 (vs Healthy) |
| Healthy | 500 | 13.2% | Reference (1:1) |
| Seronegative RA | 500 | 13.2% | 5.68:1 (vs Healthy) |
| **Total** | **3,798** | **100%** | — |

### 2.3 Split-Specific Distributions

**Training Set (2,658 samples - 70%)**

| Class | Count | Percentage | Notes |
|-------|-------|-----------|-------|
| Seropositive RA | 1,958 | 73.7% | Majority class |
| Healthy | 350 | 13.2% | Minority class |
| Seronegative RA | 350 | 13.2% | Minority class |

**Validation Set (570 samples - 15%)**

| Class | Count | Percentage |
|-------|-------|-----------|
| Seropositive RA | 420 | 73.7% |
| Healthy | 75 | 13.2% |
| Seronegative RA | 75 | 13.2% |

**Test Set (570 samples - 15%)**

| Class | Count | Percentage |
|-------|-------|-----------|
| Seropositive RA | 420 | 73.7% |
| Healthy | 75 | 13.2% |
| Seronegative RA | 75 | 13.2% |

**Key Observation**: Stratified splitting maintains identical class distributions across all splits (70-20-10% per class), preventing dataset-specific pattern learning.

---

## 3. Feature Specifications

### 3.1 Feature Summary

| # | Feature | Type | Unit | Range | Mean | Std | Missing % | Clinical Role |
|---|---------|------|------|-------|------|-----|-----------|----------------|
| 1 | Age | Numeric | Years | 20-80 | 49.7 | 17.7 | 0% | Risk factor, inflammatory response correlate |
| 2 | Gender | Categorical | M/F | — | 49% F, 51% M | — | 0% | RA prevalence higher in females |
| 3 | RF | Numeric | IU/mL | 0-40.3 | 25.2 | 10.6 | 10.6% | Rheumatoid Factor (diagnostic marker) |
| 4 | Anti-CCP | Numeric | U/mL | 0-40.0 | 22.4 | 12.7 | 20.9% | Cyclic Citrullinated Peptide (specific RA) |
| 5 | CRP | Numeric | mg/dL | 0-33.8 | 17.5 | 7.9 | 20.5% | C-Reactive Protein (inflammation) |
| 6 | ESR | Numeric | mm/hr | 0-79.8 | 30.7 | 11.3 | 9.1% | Erythrocyte Sedimentation Rate (inflammation) |

### 3.2 Detailed Feature Specifications

#### Feature 1: Age

**Data Type**: Integer (int64)  
**Units**: Years  
**Valid Range**: 20-80  
**Distribution**: Approximately uniform across adult lifespan  

**Statistics** (All Splits Combined):

| Statistic | Value |
|-----------|-------|
| Count | 3,798 (100% complete) |
| Mean | 49.69 years |
| Median | 50.00 years |
| Std Dev | 17.70 years |
| Min | 20 years |
| 25th Percentile | 35 years |
| 75th Percentile | 65 years |
| Max | 80 years |

**Missing Data**: None  
**Outliers**: None identified (range validated clinically)

**Clinical Significance**:
- RA typically appears in middle-aged to elderly populations
- Younger ages (<30) may indicate early-onset RA or juvenile forms
- Age influences inflammatory marker levels and disease severity

**Preprocessing Notes**: Used as-is; no transformation required (XGBoost-compatible)

---

#### Feature 2: Gender

**Data Type**: Categorical (object in pandas)  
**Valid Values**: "Male", "Female"  
**Encoding**: 
- Original: String ("Male"/"Female")
- Model Input: Numeric (Male=0, Female=1)

**Distribution** (3,798 samples):

| Category | Count | Percentage |
|----------|-------|-----------|
| Female | 1,857 | 48.8% |
| Male | 1,941 | 51.1% |
| Missing | 0 | 0% |

**Clinical Significance**:
- RA affects 2-3 times more women than men
- Female sex is major RA risk factor (genetic/hormonal)
- Hormone status (pregnancy, menopause) influences disease activity

**Preprocessing Notes**:
- **CRITICAL FIX** (Nov 23, 2025): Must encode as numeric before XGBoost prediction
- Training: StandardScaler handles after one-hot encoding
- Inference: Must convert string to numeric (Female=1, Male=0)
- **Error Reference**: Previously caused ValueError in numeric prediction tab

**Data Quality**: No missing values, no invalid entries

---

#### Feature 3: RF (Rheumatoid Factor)

**Data Type**: Float (float64)  
**Units**: IU/mL (International Units per milliliter)  
**Valid Range**: 0.0 - 40.3 (normalized/synthetic data)  
**Clinical Range**: 0.0 - ~500 (real labs, here normalized to 0-40)

**Statistics**:

| Statistic | Value |
|-----------|-------|
| Count (non-null) | 3,394 / 3,798 |
| Missing | 404 (10.6%) |
| Mean | 25.18 IU/mL |
| Median | 27.00 IU/mL |
| Std Dev | 10.56 |
| Min | 0.00 |
| 25th Percentile | 14.51 |
| 75th Percentile | 35.90 |
| Max | 40.26 |

**Missing Data Pattern**:
- **Rate**: 10.6% (404/3,798 samples)
- **Type**: Missing At Random (MAR) - likely lab test not performed
- **Handling**: Model uses as-is; XGBoost handles missing values natively

**Clinical Significance**:
- **RF < 15 IU/mL**: Negative (suggests non-seropositive disease)
- **RF ≥ 15 IU/mL**: Positive (seropositive RA indicator)
- **Higher RF**: Correlates with disease severity
- **Prognostic Value**: High RF linked to erosive disease

**Clinical Notes**:
- Specific but not sensitive for RA (can be positive in other autoimmune diseases)
- Part of diagnostic criteria for RA classification
- Helps differentiate seropositive from seronegative RA

**Preprocessing**: Normalized via StandardScaler (fit on training set only)

---

#### Feature 4: Anti-CCP (Anti-Cyclic Citrullinated Peptide)

**Data Type**: Float (float64)  
**Units**: U/mL (Units per milliliter)  
**Valid Range**: 0.0 - 40.0 (normalized)  
**Clinical Range**: 0.0 - ~500 (actual labs)

**Statistics**:

| Statistic | Value |
|-----------|-------|
| Count (non-null) | 3,003 / 3,798 |
| Missing | 795 (20.9%) |
| Mean | 22.42 U/mL |
| Median | 26.00 U/mL |
| Std Dev | 12.67 |
| Min | 0.00 |
| 25th Percentile | 4.09 |
| 75th Percentile | 35.73 |
| Max | 40.00 |

**Missing Data Pattern**:
- **Rate**: 20.9% (highest among all features)
- **Type**: Missing At Random (likely older samples before anti-CCP became standard)
- **Handling**: XGBoost's tree-based structure handles missing values via surrogate splits

**Clinical Significance**:
- **Most Specific**: Anti-CCP is highly specific for RA (95%+)
- **Precedes Symptoms**: Can appear before RF or clinical symptoms
- **Prognostic**: Anti-CCP+ correlates with aggressive, erosive disease
- **Progression Indicator**: Predicts poor prognosis even in seronegative patients

**Clinical Cutoff**:
- **< 20 U/mL**: Negative
- **≥ 20 U/mL**: Positive (strong RA indicator)

**Interpretation**:
- Superior to RF for RA diagnosis (more specific)
- Important for distinguishing RA from other joint diseases
- Can identify preclinical RA in antibody-positive, symptom-free individuals

**Preprocessing**: Normalized via StandardScaler

---

#### Feature 5: CRP (C-Reactive Protein)

**Data Type**: Float (float64)  
**Units**: mg/dL (milligrams per deciliter)  
**Valid Range**: 0.0 - 33.82 (normalized)  
**Clinical Range**: 0.0 - 20+ (high values indicate acute inflammation)

**Statistics**:

| Statistic | Value |
|-----------|-------|
| Count (non-null) | 3,020 / 3,798 |
| Missing | 778 (20.5%) |
| Mean | 17.47 mg/dL |
| Median | 18.20 mg/dL |
| Std Dev | 7.93 |
| Min | 0.00 |
| 25th Percentile | 11.70 |
| 75th Percentile | 23.85 |
| Max | 33.82 |

**Missing Data Pattern**:
- **Rate**: 20.5% (similar to Anti-CCP)
- **Type**: Missing At Random
- **Handling**: XGBoost handles natively

**Clinical Significance**:
- **Acute Phase Reactant**: Rises rapidly during inflammation
- **Not Specific**: Elevated CRP indicates any inflammation (infection, cancer, RA, etc.)
- **Prognostic**: High CRP in RA correlates with activity and damage
- **Monitoring**: Often used to track disease activity over time

**Clinical Cutoff**:
- **< 10 mg/dL**: Normal (minimal inflammation)
- **10-30 mg/dL**: Elevated (moderate inflammation - RA common range)
- **> 30 mg/dL**: Severe inflammation

**Interpretation**:
- Rising CRP suggests active inflammation
- Falling CRP with treatment indicates response
- Should correlate with patient symptoms and ESR

**Preprocessing**: Normalized via StandardScaler

---

#### Feature 6: ESR (Erythrocyte Sedimentation Rate)

**Data Type**: Float (float64)  
**Units**: mm/hr (millimeters per hour)  
**Valid Range**: 0.0 - 79.79 (normalized)  
**Clinical Range**: 0-150+ (varies by age/sex)

**Statistics**:

| Statistic | Value |
|-----------|-------|
| Count (non-null) | 3,454 / 3,798 |
| Missing | 344 (9.1%) |
| Mean | 30.71 mm/hr |
| Median | 32.00 mm/hr |
| Std Dev | 11.26 |
| Min | 0.00 |
| 25th Percentile | 21.63 |
| 75th Percentile | 40.00 |
| Max | 79.79 |

**Missing Data Pattern**:
- **Rate**: 9.1% (lowest among biomarkers)
- **Type**: Missing At Random
- **Handling**: XGBoost handles natively

**Clinical Significance**:
- **Nonspecific Marker**: Like CRP, rises with any systemic inflammation
- **Complementary to CRP**: Slower to respond but also slower to resolve
- **Age-Dependent**: Interpretation varies by patient age
- **Disease Activity**: Reflects cumulative inflammation

**Clinical Interpretation**:

| ESR (mm/hr) | Typical Meaning |
|-------------|-----------------|
| 0-20 | Normal (age/sex dependent) |
| 20-40 | Mild-moderate inflammation |
| 40-60 | Moderate-marked inflammation |
| > 60 | Severe inflammation (infection, cancer, or severe RA) |

**CRP vs ESR**:
- **CRP**: Rises within hours (early inflammation marker)
- **ESR**: Rises over days (cumulative inflammation)
- Both used together for comprehensive inflammation assessment

**Preprocessing**: Normalized via StandardScaler

---

## 4. Data Files & Structure

### 4.1 File Inventory

```
data/numerical/numeric/
├── train_pool.csv           (3,848 samples, 7 columns)
├── train_numeric.csv        (2,658 samples, 7 columns)
├── val_numeric.csv          (570 samples, 7 columns)
├── test_numeric.csv         (570 samples, 7 columns)
├── healthy.csv              (500 samples, 7 columns)
├── seropositive.csv         (2,848 samples, 15 columns)
└── seronegative.csv         (500 samples, 7 columns)
```

### 4.2 File Specifications

#### train_pool.csv

**Purpose**: Original complete dataset before splitting  
**Size**: 3,848 samples × 7 columns  
**Columns**: Age, Gender, RF, Anti-CCP, CRP, ESR, label

**Notes**:
- Contains all original samples with labels
- Used for stratified train/val/test split
- Source file for subsampled class-specific CSVs

---

#### train_numeric.csv

**Purpose**: Training set for model development  
**Size**: 2,658 samples (70% of total)  
**Columns**: Age, Gender, RF, Anti_CCP, CRP, ESR, label

**Class Distribution**:
- Healthy: 350 (13.2%)
- Seropositive RA: 1,958 (73.7%)
- Seronegative RA: 350 (13.2%)

**Usage**: Used to fit XGBoost model and preprocessing (StandardScaler)

---

#### val_numeric.csv

**Purpose**: Validation set for hyperparameter tuning  
**Size**: 570 samples (15% of total)  
**Columns**: Age, Gender, RF, Anti_CCP, CRP, ESR, label

**Class Distribution**:
- Healthy: 75 (13.2%)
- Seropositive RA: 420 (73.7%)
- Seronegative RA: 75 (13.2%)

**Usage**: Monitor model performance during training; early stopping criteria

---

#### test_numeric.csv

**Purpose**: Final held-out test set (unseen during training)  
**Size**: 570 samples (15% of total)  
**Columns**: Age, Gender, RF, Anti_CCP, CRP, ESR, label

**Class Distribution**:
- Healthy: 75 (13.2%)
- Seropositive RA: 420 (73.7%)
- Seronegative RA: 75 (13.2%)

**Usage**: Final model evaluation; reports test accuracy

---

#### healthy.csv

**Purpose**: Subset of healthy class samples  
**Size**: 500 samples × 7 columns  
**Columns**: Age, Gender, RF, Anti_CCP, CRP, ESR, label

**Notes**:
- Pure healthy class (label=0)
- Useful for class-specific analysis
- Contains 500 of 500 healthy samples from original pool

---

#### seropositive.csv

**Purpose**: Seropositive RA samples (extended feature set)  
**Size**: 2,848 samples × 15 columns  
**Columns**: Age, Gender, ESR, CRP, RF, Anti-CCP, HLA-B27, ANA, Anti-Ro, Anti-La, Anti-dsDNA, Anti-Sm, C3, C4, Disease

**Notes**:
- Contains additional immunological markers NOT used in final model
- Extra features: HLA-B27, ANA, Anti-Ro, Anti-La, Anti-dsDNA, Anti-Sm, C3, C4
- These may be used for extended analysis or research
- Larger sample count (2,848) suggests original unbalanced data

**Model Usage**: Only Age, Gender, RF, Anti-CCP, CRP, ESR are used for prediction

---

#### seronegative.csv

**Purpose**: Seronegative RA samples  
**Size**: 500 samples × 7 columns  
**Columns**: Age, Gender, RF, Anti_CCP, CRP, ESR, label

**Notes**:
- Pure seronegative class (label=2)
- Matches healthy.csv in size (both 500 samples)
- Represents ~100% of seronegative class in training data

---

## 5. Data Quality & Validation

### 5.1 Data Integrity Checks

**✅ Passed Checks**:

| Check | Result | Details |
|-------|--------|---------|
| **Completeness** | ✅ 98.7% | 3,798/3,848 valid samples |
| **Duplicates** | ✅ None | All records unique |
| **Label Validity** | ✅ Valid | All labels in {0, 1, 2} |
| **Age Range** | ✅ Valid | 20-80 years (clinically plausible) |
| **Biomarker Ranges** | ✅ Valid | All values within normalized ranges |
| **Gender Values** | ✅ Valid | Only "Male" or "Female" |
| **Class Balance** | ⚠️ Imbalanced | 74.7% vs 13.2% vs 13.2% (expected) |

### 5.2 Missing Data Analysis

**Missing Value Summary**:

| Feature | Missing Count | Missing % | Split | Pattern |
|---------|----------------|-----------|-------|---------|
| Age | 0 | 0% | All | None |
| Gender | 0 | 0% | All | None |
| RF | 404 | 10.6% | Random | Missing At Random (MAR) |
| Anti-CCP | 795 | 20.9% | Random | Missing At Random (MAR) |
| CRP | 778 | 20.5% | Random | Missing At Random (MAR) |
| ESR | 344 | 9.1% | Random | Missing At Random (MAR) |

**Interpretation**:
- **Missing At Random (MAR)**: Missing pattern unrelated to missing values themselves
- **Likely Cause**: Selective lab testing (not all tests performed on all patients)
- **Model Handling**: XGBoost handles missing values natively via surrogate splits

---

### 5.3 Outlier Detection

**Method**: Clinical plausibility check  
**Result**: No outliers detected

- **Age**: 20-80 years (expected adult range)
- **RF**: 0-40.3 IU/mL (within normalized synthetic range)
- **Anti-CCP**: 0-40.0 U/mL (within normalized range)
- **CRP**: 0-33.8 mg/dL (extreme but clinically possible for severe inflammation)
- **ESR**: 0-79.8 mm/hr (extreme but clinically possible)

**Note**: Data appears to be normalized/synthetic. Real-world data could have higher variability.

---

## 6. Data Preprocessing Pipeline

### 6.1 Stratified Splitting

**Methodology**:
```
Step 1: train_pool.csv (3,848 samples)
         ↓
Step 2: stratified_split(70% train, 30% temp)
         ├─ train_numeric.csv: 2,658 samples
         └─ temp: 1,190 samples
         
Step 3: stratified_split(temp into val/test 50/50)
         ├─ val_numeric.csv: 570 samples (15% of original)
         └─ test_numeric.csv: 570 samples (15% of original)

Result: Identical class distribution across all splits
        Train: 13.2% Healthy, 73.7% Seropositive, 13.2% Seronegative
        Val:   13.2% Healthy, 73.7% Seropositive, 13.2% Seronegative
        Test:  13.2% Healthy, 73.7% Seropositive, 13.2% Seronegative
```

**Benefit**: Prevents dataset-specific pattern learning; ensures unbiased evaluation

### 6.2 Feature Normalization

**Applied Transformations**:

| Feature | Method | Fit On | Parameters |
|---------|--------|--------|-----------|
| Age | StandardScaler | Train Only | Mean, Std |
| Gender | One-Hot Encoding | All | Fixed |
| RF | StandardScaler | Train Only | Mean, Std |
| Anti-CCP | StandardScaler | Train Only | Mean, Std |
| CRP | StandardScaler | Train Only | Mean, Std |
| ESR | StandardScaler | Train Only | Mean, Std |

**StandardScaler Formula**:
```
normalized_value = (value - mean) / std
```

**Why Fit on Training Only?**
- Prevents data leakage
- Scaler parameters represent training distribution
- Validation/test data transformed using training statistics
- Ensures honest evaluation on unseen data

---

## 7. Class Imbalance & Handling

### 7.1 Imbalance Problem

**Challenge**: Dataset heavily skewed toward Seropositive RA (74.7%)

| Class | Samples | % | Imbalance vs Healthy |
|-------|---------|---|---------------------|
| Seropositive RA | 2,838 | 74.7% | 5.68:1 |
| Healthy | 500 | 13.2% | Reference |
| Seronegative RA | 500 | 13.2% | 5.68:1 |

**Problems Caused**:
- Model biased toward majority class prediction
- Poor performance on minority classes
- Misleading accuracy metrics

### 7.2 Imbalance Handling Strategy

**For XGBoost Numeric Model**:
- **Method**: Stratified k-fold cross-validation + class weights
- **Class Weights**: `scale_pos_weight` parameter adjusted for majority/minority ratio
- **Result**: Better recall on minority classes

**For Imaging Model** (separate project):
- **Method**: WeightedRandomSampler during training
- **Augmentation**: Progressive augmentation for minority class
- **Loss Function**: Focal Loss to reduce easy examples

---

## 8. Data Usage Guidelines

### 8.1 Recommended Workflow

```
1. Load Data
   train_df = pd.read_csv('train_numeric.csv')
   val_df = pd.read_csv('val_numeric.csv')
   test_df = pd.read_csv('test_numeric.csv')

2. Extract Features & Labels
   X_train = train_df[['Age', 'Gender', 'RF', 'Anti_CCP', 'CRP', 'ESR']]
   y_train = train_df['label']
   (same for val, test)

3. Encode Categorical
   X_train['Gender'] = (X_train['Gender'] == 'Female').astype(int)
   (same for val, test)

4. Normalize Numeric Features
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   scaler.fit(X_train[['Age', 'RF', 'Anti_CCP', 'CRP', 'ESR']])
   X_train_scaled = scaler.transform(X_train)
   X_val_scaled = scaler.transform(X_val)
   X_test_scaled = scaler.transform(X_test)

5. Train Model
   model = XGBClassifier(n_classes=3, ...)
   model.fit(X_train_scaled, y_train)

6. Evaluate
   train_acc = model.score(X_train_scaled, y_train)
   val_acc = model.score(X_val_scaled, y_val)
   test_acc = model.score(X_test_scaled, y_test)
```

### 8.2 Important Notes for Users

**⚠️ CRITICAL - Gender Encoding**:
- Input to app.py takes string ('Female'/'Male')
- Must convert to numeric BEFORE model inference
- Female = 1, Male = 0
- Bug previously caused: `ValueError: DataFrame.dtypes for data must be int, float, bool or category`

**⚠️ Missing Values**:
- RF: 10.6% missing - XGBoost handles natively
- Anti-CCP: 20.9% missing - Do NOT impute; affects model training
- CRP: 20.5% missing - Same as Anti-CCP
- ESR: 9.1% missing - Same handling

**⚠️ Normalization**:
- Always fit StandardScaler on TRAINING set only
- Apply same scaler to validation and test
- For real-time predictions, use scaler fitted during development

### 8.3 Common Pitfalls to Avoid

| ❌ Mistake | ✅ Solution |
|-----------|-----------|
| Fit scaler on all data | Fit ONLY on training data |
| Forget to encode gender | Encode all gender values before prediction |
| Use test data for validation | Stratified split maintains proper separation |
| Include test labels in evaluation | Use only test data for final metrics |
| Impute missing values | Let tree-based model handle natively |

---

## 9. Data Statistics Summary

### 9.1 Aggregate Statistics Table

**Across All Splits (3,798 samples)**:

| Statistic | Age | RF | Anti-CCP | CRP | ESR |
|-----------|-----|----|---------|----|-----|
| **Count** | 3798 | 3394 | 3003 | 3020 | 3454 |
| **Missing** | 0 | 404 | 795 | 778 | 344 |
| **Missing %** | 0% | 10.6% | 20.9% | 20.5% | 9.1% |
| **Mean** | 49.69 | 25.18 | 22.42 | 17.47 | 30.71 |
| **Median** | 50.00 | 27.00 | 26.00 | 18.20 | 32.00 |
| **Std Dev** | 17.70 | 10.56 | 12.67 | 7.93 | 11.26 |
| **Min** | 20.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **Max** | 80.00 | 40.26 | 40.00 | 33.82 | 79.79 |
| **Q1 (25%)** | 35.00 | 14.51 | 4.09 | 11.70 | 21.63 |
| **Q3 (75%)** | 65.00 | 35.90 | 35.73 | 23.85 | 40.00 |

### 9.2 Demographic Summary

| Metric | Value |
|--------|-------|
| **Total Samples** | 3,798 |
| **Female** | 1,857 (48.8%) |
| **Male** | 1,941 (51.1%) |
| **Age Range** | 20-80 years |
| **Median Age** | 50 years |

---

## 10. Related Documentation

- **PROJECT_INFO.md**: Complete technical architecture and model specifications
- **README.md**: Quick start guide and project overview
- **Dataset_Specifications.xlsx**: Interactive Excel workbook with data explorer
- **Dataset_Metadata.json**: Machine-readable schema for programmatic access
- **build_log.txt**: Generation and validation log

---

## 11. Citation & Usage

**Recommended Citation**:
```
RA Diagnosis Dataset (2025). Rheumatoid Arthritis Blood Biomarker Classification. 
Project-Root Documentation, November 2025.
```

**License**: Internal Use / Research Only (see project LICENSE file)

**Contact**: Project Team

---

## Appendix: Full Column Reference

### Column Definitions

| Column Name | Data Type | Domain | Unit | Categorical Values | Interpretation |
|-------------|-----------|--------|------|-------------------|-----------------|
| Age | int64 | 20-80 | years | — | Patient age at test |
| Gender | object | {M,F} | — | "Male", "Female" | Biological sex |
| RF | float64 | 0-40.3 | IU/mL | — | Rheumatoid Factor |
| Anti_CCP | float64 | 0-40.0 | U/mL | — | Cyclic Citrullinated Peptide Antibody |
| CRP | float64 | 0-33.8 | mg/dL | — | C-Reactive Protein |
| ESR | float64 | 0-79.8 | mm/hr | — | Erythrocyte Sedimentation Rate |
| label | int64 | {0,1,2} | — | 0=Healthy, 1=Seropositive, 2=Seronegative | Target class |

---

**Document Version**: 1.0  
**Last Updated**: November 23, 2025  
**Status**: Research-Grade, Complete
