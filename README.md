# Rheumatoid Arthritis Diagnosis System - Quick Start

> **For comprehensive technical documentation**, see **[PROJECT_INFO.md](PROJECT_INFO.md)**

-------
Imaging Data:

tags:
- X-ray
- Wrist
- Segmentation
- Classification
license: bigscience-openrail-m

# Dataset Card for RAM-W600
Benchmark code is available in <https://github.com/maxQterminal/Rheumatoid-arthritis>.

## Download
Please run the following command to download RAM-W600 image data:

```bash
git clone https://huggingface.co/datasets/TokyoTechMagicYang/RAM-W600
```

Numerical Data: [data](https://github.com/maxQterminal/Rheumatoid-arthritis/data/numerical/numeric)

---



**Four tabs**:
1. **Lab Assessment**: Input 6 biomarkers → Get RA diagnosis
2. **X-ray Analysis**: Upload hand X-ray → Get erosion classification  
3. **Combined Results**: See both predictions together
4. **Model Performance**: View model accuracy, comparison, and augmentation strategy

**Important**: Models are already in `models/` folder (EfficientNet-B3, XGBoost). No additional setup needed!

---

## 📊 What This Does

**Input**: Blood tests (6 biomarkers) + Hand X-ray image  
**Output**: RA diagnosis (Healthy / Seropositive / Seronegative) + Erosion status  
**Accuracy**: 89% (blood tests) + 85.83% (X-ray with augmentation strategy)

---

## 🔄 Data Flow: Input → Model → Output

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION (UI)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Tab 1: Lab Assessment              Tab 2: X-ray Analysis               │
│  ┌─────────────────────────┐       ┌──────────────────────────┐         │
│  │ Input 6 Biomarkers:     │       │ Upload Hand X-ray Image: │         │
│  │ • Age (years)           │       │ • JPG/PNG/BMP format     │         │
│  │ • Gender (M/F)          │       │ • 224×224 or larger      │         │
│  │ • RF (IU/mL)            │       │                          │         │
│  │ • Anti-CCP (IU/mL)      │       │ Click: "Analyze X-ray"   │         │
│  │ • CRP (mg/L)            │       │                          │         │
│  │ • ESR (mm/hr)           │       │                          │         │
│  │                         │       │                          │         │
│  │ Click: "Get Diagnosis"  │       │                          │         │
│  └─────────────────────────┘       └──────────────────────────┘         │
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
│  1. StandardScaler normalization   1. Resize to 224×224                 │
│    (subtract mean, divide by std)  2. Convert to 3-channel RGB          │
│         ↓                          3. Apply ImageNet normalization      │
│  Normalized values ready                  ↓                             │
│  for model input                   Preprocessed image ready             │
│                                    for model input                      │
│                                                                         │
│  See PROJECT_INFO.md "Data Preprocessing" section for details           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────-──────────────────┐
│                     MODEL INFERENCE (Prediction)                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PATH 1: Numeric Model              PATH 2: Imaging Model            │
│  ─────────────────────────          ──────────────────────           │
│  Preprocessed biomarkers             Preprocessed image              │
│           ↓                                  ↓                       │
│    ┌──────────────┐              ┌──────────────────┐                │
│    │   XGBoost    │              │ EfficientNet-B3  │                │
│    │   Classifier │              │     CNN          │                │
│    │ (100 trees)  │              │  (10.3M params)  │                │
│    └──────────────┘              └──────────────────┘                │
│           ↓                                  ↓                       │
│   Multiclass Output:              Binary Output:                     │
│   P(Healthy) = 0.15               P(Erosive) = 0.72                  │
│   P(Seroneg) = 0.25               (72% confident)                    │
│   P(Seropos) = 0.60 ← Max         Threshold: 0.5 (default)           │
│           ↓                       Since 0.72 > 0.5:                  │
│           ↓                       Predict: "EROSIVE"                 │
│   Prediction:                                                        │
│   "SEROPOSITIVE"                  Confidence = 0.72                  │
│   (60% confident)                          ↓                         │
└──────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│                     USER OUTPUT (UI Display)                          │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Lab Assessment Tab Shows:       X-ray Analysis Tab Shows:            │
│  ┌──────────────────────┐       ┌──────────────────────┐              │
│  │ Diagnosis Result:    │       │ X-ray Classification:│              │
│  │ ✓ SEROPOSITIVE RA    │       │ ✓ EROSIVE            │              │
│  │                      │       │                      │              │
│  │ Confidence: 60%      │       │ Confidence: 72%      │              │
│  │                      │       │ Decision: Threshold  │              │
│  │ Breakdown:           │       │          = 0.35      │              │
│  │ • P(Healthy) = 15%   │       │                      │              │
│  │ • P(Seroneg) = 25%   │       │ Interpretation:      │              │
│  │ • P(Seropos) = 60%   │       │ "Joint erosions      │              │
│  │                      │       │  are present"        │              │
│  │ Clinical Action:     │       │                      │              │
│  │ → Start DMARD        │       │ Clinical Action:     │              │
│  │   therapy            │       │ → Confirm with       │              │
│  │ → Monitor closely    │       │   radiologist        │              │
│  │ → Follow-up in 6 wks │       │ → Adjust treatment   │              │
│  └──────────────────────┘       └──────────────────────┘              │
│                                                                       │
│  Combined Results Tab Shows:                                          │
│  ┌──────────────────────────────────────────┐                         │
│  │ OVERALL RA DIAGNOSIS SUMMARY             │                         │
│  │                                          │                         │
│  │ Blood Tests: SEROPOSITIVE (60%)          │                         │ 
│  │ Hand X-rays: EROSIVE (72%)               │                         │
│  │                                          │                         │
│  │ Combined Assessment:                     │                         │
│  │ ✓ HIGH RA LIKELIHOOD                     │                         │
│  │   - Positive autoimmune markers          │                         │
│  │   - Visible joint erosions               │                         │
│  │                                          │                         │
│  │ Recommendation:                          │                         │
│  │ → Advanced RA suspected                  │                         │
│  │ → Aggressive treatment indicated         │                         │
│  │ → Consider rheumatology referral         │                         │
│  └──────────────────────────────────────────┘                         │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
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
├── efficientnet.pth           (41.3 MB - X-ray classifier - PRIMARY MODEL)
├── resnet50.pth               (41.3 MB - X-ray classifier - alternative)
└── vit.pth                    (328 MB - X-ray classifier - alternative)

src/
├── app/
│   ├── app_medical_dashboard.py    (Main app)
│   └── demo_predict.py             (Test predictions)
└── data/
    └── synth_and_numeric.py        (Data preprocessing)
```

---

## 🤖 Models

### 2. Imaging Models: Three CNNs with Augmentation Strategy

**Problem Solved**: Severe class imbalance (4.59:1 - 82% Erosive vs 18% Non-Erosive)

**Solution Applied**:
- **WeightedRandomSampler**: Balances batch-level sampling to 1:1 ratio
- **Focal Loss** (γ=2.0): Focuses training on hard-to-learn minority class examples
- **Progressive Augmentation**: Flips, rotations ±15°, color jitter, Gaussian blur
- **F1-based Early Stopping**: Monitors erosive class F1 (not validation loss)
- **Optimized for M4 Metal GPU**: Float32 dtype, batch size 16

**Model Comparison** (all trained with identical augmentation pipeline):

| Model | Accuracy | F1 Erosive | F1 Non-Erosive | Status |
|-------|----------|-----------|----------------|--------|
| **EfficientNet-B3** | **85.83%** | **91.63%** | **54.05%** | ✅ **PRIMARY** |
| ResNet50 | 82.50% | 89.45% | 48.78% | Alternative |
| ViT-B/16 | 80.00% | 87.23% | 53.85% | Alternative |

**Selected Model: EfficientNet-B3**
- Highest overall accuracy (85.83%, +5.83pp vs ViT)
- Best minority class F1 (54.05%, handles early RA detection)
- Optimal erosive recall (95.04%, catches most erosion cases)
- Fast inference (200-500 ms) vs ViT (slower, larger memory)
- See `reports/image/model_comparison_all_models.png` for visualizations

---

### 1. Numeric Model: XGBoost
- **Input**: 6 blood test biomarkers
- **Output**: Healthy / Seropositive RA / Seronegative RA
- **Accuracy**: 89.28%
- **F1-Score**: 85.77%
- **ROC-AUC**: 93.21%
- **Speed**: 15-50 ms
- **Why this model**: Best for tabular data, fast, interpretable, handles mixed feature types

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
streamlit run src/app/app.py
```

**That's it!** No paths to update, no files to move. The app finds everything automatically.

---

## ✅ Status

**Production Ready**: All models trained, optimized, and tested  
**Documentation**: Complete and comprehensive  
**Performance**: 89.28% accuracy (numeric/blood tests) + 84.17% accuracy (imaging/X-rays)  
**Data Processing**: Comprehensive preprocessing pipeline (see PROJECT_INFO.md)

---

**Version**: 1.0 | **Last Updated**: November 18, 2025
