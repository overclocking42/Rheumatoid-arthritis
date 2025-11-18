# Rheumatoid Arthritis Diagnosis System - Quick Start


> **For comprehensive technical documentation**, see **[PROJECT_INFO.md](PROJECT_INFO.md)**

---

For data access, the full data folder is available at [google-drive](https://drive.google.com/drive/folders/1vP4q1CzZiUh1e1OyM84okWWBBQDayGhj?usp=sharing) add that to the project root folder if needed.

## 🚀 Quick Start

### Run the Dashboard
```bash
# From project root directory 
streamlit run src/app/app_medical_dashboard.py
```


**Three tabs**:
1. **Lab Assessment**: Input 6 biomarkers → Get RA diagnosis
2. **X-ray Analysis**: Upload hand X-ray → Get erosion classification  
3. **Combined Results**: See both predictions together
4. **Model Performance**: View model accuracy and selection rationale

**Important**: Models are already in `models/` folder. No additional setup needed!

---

## 📊 What This Does

**Input**: Blood tests (6 biomarkers) + Hand X-ray image  
**Output**: RA diagnosis (Healthy / Seropositive / Seronegative) + Erosion status  
**Accuracy**: 89% (blood tests) + 84.17% (X-ray with optimized threshold)

---

## 🔄 Data Flow: Input → Model → Output

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION (UI)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Tab 1: Lab Assessment              Tab 2: X-ray Analysis               │
│  ┌─────────────────────────┐       ┌─────────────────────────-─┐        │
│  │ Input 6 Biomarkers:     │       │ Upload Hand X-ray Image:  │        │
│  │ • Age (years)           │       │ • JPG/PNG/BMP format      │        │
│  │ • Gender (M/F)          │       │ • 224×224 or larger       │        │
│  │ • RF (IU/mL)            │       │                           │        │
│  │ • Anti-CCP (IU/mL)      │       │ Click: "Analyze X-ray"    │        │
│  │ • CRP (mg/L)            │       │                           │        │
│  │ • ESR (mm/hr)           │       │                           │        │
│  │                         │       │                           │        │
│  │ Click: "Get Diagnosis"  │       │                           │        │
│  └─────────────────────────┘       └──────────────────────────-┘        │
│            ↓                                 ↓                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA PREPROCESSING (Backend)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  NUMERIC DATA (Blood Tests)       IMAGE DATA (X-ray)                    │
│  ─────────────────────────       ─────────────────────                  │
│  Input: [Age, Gender, RF, ...]   Input: Image pixels                    │
│         ↓                                ↓                              │
│  1. StandardScaler normalization  1. Resize to 224×224                  │
│     (subtract mean, divide by std) 2. Convert to 3-channel RGB          │
│         ↓                           3. Apply ImageNet normalization     │
│  Normalized values ready           ↓                                    │
│  for model input                   Preprocessed image ready             │
│                                    for model input                      │
│                                                                         │
│  See PROJECT_INFO.md "Data Preprocessing" section for details           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     MODEL INFERENCE (Prediction)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PATH 1: Numeric Model              PATH 2: Imaging Model               │
│  ─────────────────────────          ──────────────────────              │
│  Preprocessed biomarkers             Preprocessed image                 │
│           ↓                                  ↓                          │
│    ┌──────────────┐              ┌──────────────────┐                   │
│    │   XGBoost    │              │ EfficientNet-B3  │                   │
│    │   Classifier │              │     CNN          │                   │
│    │ (100 trees)  │              │  (10.3M params)  │                   │    
│    └──────────────┘              └──────────────────┘                   │    
│           ↓                                  ↓                          │
│   Multiclass Output:              Binary Output:                        │
│   P(Healthy) = 0.15               P(Erosive) = 0.72                     │
│   P(Seroneg) = 0.25               (72% confident)                       │
│   P(Seropos) = 0.60 ← Max                 ↓                             │
│           ↓                       Decision Threshold = 0.35             │
│           ↓                       Since 0.72 > 0.35:                    │
│   Prediction:                     Predict: "EROSIVE"                    │
│   "SEROPOSITIVE"                          ↓                             │
│   (60% confident)                 Confidence = 0.72                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     USER OUTPUT (UI Display)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Lab Assessment Tab Shows:       X-ray Analysis Tab Shows:              │
│  ┌──────────────────────┐       ┌──────────────────────┐                │
│  │ Diagnosis Result:    │       │ X-ray Classification:│                │
│  │ ✓ SEROPOSITIVE RA    │       │ ✓ EROSIVE            │                │
│  │                      │       │                      │                │
│  │ Confidence: 60%      │       │ Confidence: 72%      │                │
│  │                      │       │ Decision: Threshold  │                │
│  │ Breakdown:           │       │          = 0.35      │                │
│  │ • P(Healthy) = 15%   │       │                      │                │
│  │ • P(Seroneg) = 25%   │       │ Interpretation:      │                │
│  │ • P(Seropos) = 60%   │       │ "Joint erosions      │                │
│  │                      │       │  are present"        │                │
│  │ Clinical Action:     │       │                      │                │
│  │ → Start DMARD        │       │ Clinical Action:     │                │
│  │   therapy            │       │ → Confirm with       │                │
│  │ → Monitor closely    │       │   radiologist        │                │
│  │ → Follow-up in 6 wks │       │ → Adjust treatment   │                │
│  └──────────────────────┘       └──────────────────────┘                │
│                                                                         │
│  Combined Results Tab Shows:                                            │
│  ┌──────────────────────────────────────────┐                           │
│  │ OVERALL RA DIAGNOSIS SUMMARY             │                           │
│  │                                          │                           │
│  │ Blood Tests: SEROPOSITIVE (60%)          │                           │
│  │ Hand X-rays: EROSIVE (72%)               │                           │
│  │                                          │                           │
│  │ Combined Assessment:                     │                           │
│  │ ✓ HIGH RA LIKELIHOOD                     │                           │
│  │   - Positive autoimmune markers          │                           │
│  │   - Visible joint erosions               │                           │
│  │                                          │                           │
│  │ Recommendation:                          │                           │
│  │ → Advanced RA suspected                  │                           │
│  │ → Aggressive treatment indicated         │                           │
│  │ → Consider rheumatology referral         │                           │
│  └──────────────────────────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Stage | Input | Processing | Output |
|-------|-------|-----------|--------|
| **User Input** | Biomarkers or X-ray image | Enter via UI | Raw data |
| **Preprocessing** | Raw values/pixels | Normalize, resize, format | Ready for model |
| **Model Inference** | Preprocessed data | Neural net / Tree ensemble | Probability scores |
| **Decision** | Probabilities | Apply threshold | Class prediction |
| **UI Display** | Prediction + confidence | Format for display | Clinical summary |

---

## 📁 Project Structure

```
data/raw_data/
├── numeric/
│   ├── train_pool.csv         (3,848 original samples)
│   ├── train_numeric.csv      (2,658 training samples)
│   ├── val_numeric.csv        (570 validation)
│   ├── test_numeric.csv       (570 test)
│   ├── healthy.csv            (synthetic)
│   └── seronegative.csv       (synthetic)
│
└── imaging/RAM-W600/
    ├── JointLocationDetection/images/  (800 X-ray images)
    ├── splits/
    │   ├── train.csv          (560 training)
    │   ├── val.csv            (120 validation)
    │   └── test.csv           (120 test)
    └── SvdHBEScoreClassification/
        └── JointBE_SvdH_GT.json       (erosion labels)

models/
├── xgb_model.joblib           (1.1 MB - blood test classifier)
└── EfficientNet-B3_best.pth   (43.3 MB - X-ray classifier)

src/
├── app/
│   ├── app_medical_dashboard.py    (Main app)
│   └── demo_predict.py             (Test predictions)
└── data/
    └── synth_and_numeric.py        (Data preprocessing)
```

---

## 🤖 Models

### 1. Numeric Model: XGBoost
- **Input**: 6 blood test biomarkers
- **Output**: Healthy / Seropositive RA / Seronegative RA
- **Accuracy**: 89.28%
- **F1-Score**: 85.77%
- **ROC-AUC**: 93.21%
- **Speed**: 15-50 ms
- **Why this model**: Best for tabular data, fast, interpretable, handles mixed feature types

### 2. Imaging Model: EfficientNet-B3
- **Input**: 224×224 hand X-ray image
- **Output**: Erosive / Non-erosive
- **Accuracy**: 84.17% ✅ (improved from 77.94% after optimising threshold)
- **ROC-AUC**: 89.18%
- **F1-Score**: 72.06%
- **Speed**: 200-500 ms
- **Why selected**: Best balance between detecting both classes (vs ViT: 91.39% ROC but only 53% F1)
- **Optimization**: Threshold tuned to 0.35 for improved accuracy

---

## 🎓 Understanding Train/Validation/Test Splits

This is **critical** for understanding why our models are trustworthy:

| Set | Size | Purpose | Model Learns? | Accuracy |
|-----|------|---------|---------------|----------|
| **Training** | 2,658 | Model learns patterns | ✅ Yes | 90-95% |
| **Validation** | 570 | Detect overfitting | ❌ No | 87-89% |
| **Test** | 570 | Final honest score | ❌ No | 85-89% |

**Why this matters for patients**: 
- Without proper split: Model claims 95% but only 40% on new patients = wrong diagnosis ✗
- With proper splits: Model says 89% on unseen data = doctor can trust it ✓

**train_pool.csv (3,848 samples)**: Original raw data before splitting. We split this 70/15/15 to create train/val/test. Kept for reproducibility.

---

## 💡 Key Concepts

**Blood Test Features**:
- **Age**: Patient age
- **Gender**: Male/Female
- **RF**: Rheumatoid factor (autoimmune antibody)
- **Anti-CCP**: Anti-cyclic citrullinated peptide antibody (RA-specific)
- **CRP**: C-reactive protein (inflammation marker)
- **ESR**: Erythrocyte sedimentation rate (inflammation indicator)

**X-ray Analysis**:
- Detects hand bone erosions (joint damage)
- Uses SvdH (Sharp Van Der Heide) scoring
- Binary: Erosive (damage present) or Non-erosive (no damage)

**Data Processing**:
Each data type goes through specific preprocessing before model input:

**Numeric Data**:
- Normalization: StandardScaler (subtract mean, divide by std)
- Handles missing values with forward-fill + mean imputation
- Stratified split maintains class proportions

**Image Data**:
- Resize to 224×224 pixels
- Convert grayscale to 3-channel RGB (model requirement)
- ImageNet normalization (mean/std from pre-training)
- Data augmentation during training (rotations, flips, scaling)

*→ See [PROJECT_INFO.md - Data Preprocessing](PROJECT_INFO.md#data-preprocessing) for complete technical details*

---

## 📖 For Full Technical Details

**[PROJECT_INFO.md](PROJECT_INFO.md)** covers:
- ✅ Complete train/validation/test split explanation (with clinical implications)
- ✅ How train_pool.csv relates to train/val/test
- ✅ Full project architecture and technical specifications
- ✅ How each model works (XGBoost, EfficientNet-B3)
- ✅ Performance metrics (accuracy, F1, ROC-AUC)
- ✅ Preprocessing steps with code examples
- ✅ Training details and hyperparameters
- ✅ How to make predictions programmatically
- ✅ Exactly where training data comes from
- ✅ Why data is organized this way
- ✅ Data flow diagrams
- ✅ File verification commands

---

## 🔧 Installation & Setup

### 1. Prerequisites
- Python 3.8+
- pip or conda

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Dashboard
```bash
# Make sure you're in the project root directory
streamlit run src/app/app_medical_dashboard.py
```

Opens at `http://localhost:8501`

---

## 🌍 Portability - Running Anywhere

**This project is fully portable!** You can run it on any system (Windows, Mac, Linux) because:

✅ **All paths are relative** - No hardcoded machine-specific paths
✅ **Auto-detects project structure** - `ROOT = os.path.dirname(...)` finds models anywhere
✅ **Works from any directory** - Just `cd` to project root and run
✅ **All dependencies in requirements.txt** - One command to install everything
✅ **Models included** - `models/xgb_model.joblib` and `models/EfficientNet-B3_best.pth` already in repo

**To clone and run on another machine:**
```bash
# 1. Clone repository
git clone https://github.com/maxQterminal/Rheumatoid-arthritis.git
cd Rheumatoid-arthritis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run app (works immediately, no configuration needed!)
streamlit run src/app/app_medical_dashboard.py
```


---

## ✅ Status

**Production Ready**: All models trained, optimized, and tested  
**Documentation**: Complete and comprehensive  
**Performance**: 89.28% accuracy (numeric/blood tests) + 84.17% accuracy (imaging/X-rays)  
**Data Processing**: Comprehensive preprocessing pipeline (see PROJECT_INFO.md)

---

**Version**: 1.0 | **Last Updated**: November 18, 2025
