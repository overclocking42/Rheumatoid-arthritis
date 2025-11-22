import os
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from PIL import Image
import cv2
import torch
from torchvision import models, transforms

st.set_page_config(
    page_title="Clinical RA Assessment System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for medical dashboard
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .healthy-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 10px;
        color: black;
        margin: 10px 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 20px;
        border-radius: 10px;
        color: black;
        margin: 10px 0;
    }
    .erosive-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .non-erosive-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Get the project root (2 levels up from this file: src/app/app_medical_dashboard.py)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@st.cache_resource
def load_numeric_model():
    """Load XGBoost numeric model for blood test classification."""
    models_dir = os.path.join(ROOT, 'models')
    model_path = os.path.join(models_dir, 'xgb_model.joblib')
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Model is already a pipeline with preprocessing built-in
        return model, None
    return None, None

@st.cache_resource
def load_image_model():
    models_dir = os.path.join(ROOT, 'models')
    # Try new augmented model first, fallback to old name
    ckpt = os.path.join(models_dir, 'efficientnet.pth')
    if not os.path.exists(ckpt):
        ckpt = os.path.join(models_dir, 'EfficientNet-B3_best.pth')
    if not os.path.exists(ckpt):
        return None
    
    device = torch.device('cpu')
    model = models.efficientnet_b3(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, 1)
    
    checkpoint = torch.load(ckpt, map_location=device)
    if isinstance(checkpoint, dict) and 'classifier.1.weight' in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        if hasattr(checkpoint, 'state_dict'):
            model.load_state_dict(checkpoint.state_dict())
        else:
            model = checkpoint
    
    model.eval()
    return model

def preprocess_image(img: Image.Image):
    tf = transforms.Compose([
        transforms.Lambda(lambda im: im.convert('L')),
        transforms.Resize((224,224)),
        transforms.Lambda(lambda im: torch.from_numpy(np.array(im).astype(np.float32))[None, ...]/255.0),
        transforms.Normalize(mean=[0.5], std=[0.25]),
        transforms.Lambda(lambda t: t.repeat(3,1,1)),
    ])
    return tf(img)

def predict_image(model, img_tensor):
    """Predict erosion classification with optimized threshold."""
    with torch.no_grad():
        logit = model(img_tensor.unsqueeze(0)).squeeze(1)
        prob = torch.sigmoid(logit).item()
    
    # Use optimized threshold of 0.35 instead of default 0.5
    # This improves accuracy from 77.94% to 84.17%
    optimal_threshold = 0.35
    label = 'Erosive' if prob >= optimal_threshold else 'Non-Erosive'
    confidence = prob if label == 'Erosive' else (1 - prob)
    return label, confidence

# Header
st.markdown("# 🏥 Clinical RA Assessment System", unsafe_allow_html=True)
st.markdown("**Evidence-Based Diagnosis Support** | Numerical + Imaging Analysis")
st.divider()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Lab Assessment", "🖼️ X-Ray Analysis", "📊 Combined Results", "📈 Model Performance"])

# ============================================================================
# TAB 1: NUMERIC (LAB TEST) ASSESSMENT
# ============================================================================
with tab1:
    st.subheader("Patient Lab Assessment")
    st.markdown("*Enter patient clinical data for automatic classification*")
    
    model, pre = load_numeric_model()
    if model is None:
        st.error("❌ Numeric model not available")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age (years)', min_value=0, max_value=120, value=55, key='lab_age')
            gender = st.selectbox('Gender', ['Female', 'Male'], key='lab_gender')
            rf = st.number_input('Rheumatoid Factor (RF) [IU/mL]', value=25.0, key='lab_rf')
        
        with col2:
            anti_ccp = st.number_input('Anti-CCP [U/mL]', value=15.0, key='lab_anticcp')
            crp = st.number_input('C-Reactive Protein (CRP) [mg/dL]', value=8.0, key='lab_crp')
            esr = st.number_input('Erythrocyte Sedimentation Rate (ESR) [mm/h]', value=30.0, key='lab_esr')
        
        if st.button("🔬 Analyze Lab Results", key='lab_btn', use_container_width=True):
            # Prepare data
            df = pd.DataFrame([{
                'Age': age,
                'Gender': 1 if gender == 'Female' else 0,  # Encode: Female=1, Male=0
                'RF': rf,
                'Anti_CCP': anti_ccp,
                'CRP': crp,
                'ESR': esr
            }])
            
            # Predict using the model
            try:
                proba = model.predict_proba(df)[0]
                prediction = int(np.argmax(proba))
                confidence = max(proba) * 100
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.stop()
            
            # Display results
            st.divider()
            class_names = ['Healthy', 'Seropositive RA', 'Seronegative RA']
            result_label = class_names[prediction]
            
            if prediction == 0:
                st.markdown(f"""
                <div class="healthy-card">
                    <h3>✓ RESULT: {result_label}</h3>
                    <h4>Confidence: {confidence:.1f}%</h4>
                    <p><strong>Clinical Interpretation:</strong> No RA indicators detected in blood tests.</p>
                    <p><strong>Recommendation:</strong> ❌ NO X-RAY NEEDED - Patient appears healthy</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-card">
                    <h3>⚠️ RESULT: {result_label}</h3>
                    <h4>Confidence: {confidence:.1f}%</h4>
                    <p><strong>Clinical Interpretation:</strong> RA indicators detected. {'Positive serology (RF/Anti-CCP)' if prediction == 1 else 'Negative serology but elevated inflammation'}</p>
                    <p><strong>Recommendation:</strong> ✓ X-RAY RECOMMENDED - Proceed to X-ray analysis to assess bone damage</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Store result in session
                st.session_state.lab_result = {
                    'class': prediction,
                    'label': result_label,
                    'confidence': confidence
                }
            
            # Show confidence breakdown
            st.write("### Class Probabilities")
            prob_df = pd.DataFrame({
                'Diagnosis': class_names,
                'Probability': [f"{p*100:.1f}%" for p in proba]
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 2: IMAGING (X-RAY) ASSESSMENT
# ============================================================================
with tab2:
    st.subheader("Joint X-Ray Analysis")
    st.markdown("*Upload hand/wrist X-ray for erosion detection*")
    
    img_model = load_image_model()
    if img_model is None:
        st.error("❌ Imaging model not available")
    else:
        uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=['bmp','png','jpg','jpeg'], key='xray_upload')
        
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
            except:
                img_bytes = uploaded_file.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                cv_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if cv_img is not None:
                    img = Image.fromarray(cv_img)
                else:
                    st.error("Could not load image")
                    st.stop()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(img, caption="Uploaded X-ray", use_container_width=True)
            
            with col2:
                if st.button("🔍 Analyze X-Ray", key='xray_btn', use_container_width=True):
                    # Predict
                    x_tensor = preprocess_image(img)
                    label, confidence = predict_image(img_model, x_tensor)
                    
                    st.divider()
                    
                    # Display result
                    if label == 'Erosive':
                        st.markdown(f"""
                        <div class="erosive-card">
                            <h3>❌ FINDING: {label}</h3>
                            <h4>Confidence: {confidence*100:.1f}%</h4>
                            <p><strong>Clinical Significance:</strong> Bone erosions detected - indicates advanced RA with structural damage.</p>
                            <p><strong>Recommendation:</strong> 🔴 AGGRESSIVE TREATMENT REQUIRED</p>
                            <ul>
                                <li>Consider biologic DMARD therapy</li>
                                <li>Early intervention to prevent progression</li>
                                <li>Rheumatology referral recommended</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="non-erosive-card">
                            <h3>✓ FINDING: {label}</h3>
                            <h4>Confidence: {confidence*100:.1f}%</h4>
                            <p><strong>Clinical Significance:</strong> No bone erosions detected - early stage or well-controlled disease.</p>
                            <p><strong>Recommendation:</strong> 🟢 CONVENTIONAL THERAPY APPROPRIATE</p>
                            <ul>
                                <li>Start with conventional DMARD (methotrexate)</li>
                                <li>Monitor for progression with repeat imaging</li>
                                <li>Consider biologic if inadequate response</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Store result
                    st.session_state.xray_result = {
                        'label': label,
                        'confidence': confidence
                    }
                    
                    # Model info
                    st.caption("**Model**: EfficientNet-B3 (Augmented) | **Accuracy**: 85.83% | **F1 Erosive**: 91.63% | **F1 Non-Erosive**: 54.05%")

# ============================================================================
# TAB 3: COMBINED CLINICAL SUMMARY
# ============================================================================
with tab3:
    st.subheader("Clinical Summary & Recommendations")
    st.markdown("*Combined assessment from both lab tests and X-ray analysis*")

    if 'lab_result' in st.session_state:
        lab_result = st.session_state.lab_result

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Lab Test Result")
            st.markdown(f"""
            **Diagnosis:** {lab_result['label']}  
            **Confidence:** {lab_result['confidence']:.1f}%
            """)

        with col2:
            st.write("### X-Ray Finding")
            if 'xray_result' in st.session_state:
                xray_result = st.session_state.xray_result
                st.markdown(f"""
                **Finding:** {xray_result['label']}  
                **Confidence:** {xray_result['confidence']*100:.1f}%
                """)
            else:
                st.info("⏳ Upload and analyze X-ray in Tab 2")

        st.divider()

        # Show clinical assessment
        if lab_result['class'] == 0:
            st.markdown("""
            <div class="healthy-card">
                <h3>✓ OVERALL ASSESSMENT: HEALTHY</h3>
                <p>Blood tests show no RA indicators. No further imaging needed at this time.</p>
                <p><strong>Monitoring:</strong> Annual follow-up recommended for early detection if symptoms develop.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if 'xray_result' in st.session_state:
                xray_result = st.session_state.xray_result
                if xray_result['label'] == 'Erosive':
                    st.markdown("""
                    <div class="erosive-card">
                        <h3>🔴 OVERALL ASSESSMENT: ADVANCED RA WITH BONE DAMAGE</h3>
                        <p><strong>RA Confirmed</strong> + <strong>Structural Damage Present</strong></p>
                        <p><strong>Treatment Strategy:</strong></p>
                        <ul>
                            <li>Initiate biologic DMARD (TNF inhibitor, JAK inhibitor)</li>
                            <li>Combination therapy if monotherapy inadequate</li>
                            <li>Regular monitoring: CRP, ESR, ACR/EULAR response</li>
                            <li>Repeat imaging in 1 year to assess response</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-card">
                        <h3>🟡 OVERALL ASSESSMENT: RA WITHOUT CURRENT BONE DAMAGE</h3>
                        <p><strong>RA Confirmed</strong> but <strong>No Structural Damage Yet</strong></p>
                        <p><strong>Treatment Strategy:</strong></p>
                        <ul>
                            <li>Early DMARD initiation (window of opportunity)</li>
                            <li>Start conventional therapy (methotrexate-based)</li>
                            <li>Escalate to biologic if poor response</li>
                            <li>Prevent progression to erosive disease</li>
                            <li>Repeat imaging in 6 months</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("⏳ Complete X-ray analysis above for full assessment")
    else:
        st.info("👈 Complete **Lab Assessment** first to generate clinical summary")

# ============================================================================
# TAB 4: MODEL PERFORMANCE & VALIDATION
# ============================================================================
with tab4:
    st.subheader("Model Performance & Selection Justification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📊 Numeric Model (XGBoost)")
        st.markdown("""
        - **Task**: 3-class classification (Healthy/Seropositive/Seronegative)
        - **Validation Macro-F1**: 85.77%
        - **Features**: Age, Gender, RF, Anti-CCP, CRP, ESR
        """)
    
    with col2:
        st.write("### 🎯 Imaging Model (EfficientNet-B3)")
        st.markdown("""
        - **Task**: 2-class classification (Erosive/Non-Erosive)
        - **Test Accuracy**: 85.83% ✅ 
        - **Test ROC-AUC**: ~91%
        - **Test Macro-F1**: 72.84% ⭐
        - **Architecture**: EfficientNet-B3 (CNN-based)
        """)
    
    st.divider()
    
    st.write("### 📈 Why EfficientNet-B3 Selected? (Trained with Augmentation Strategy)")
    
    comparison_df = pd.DataFrame({
        'Model': ['ResNet50', 'EfficientNet-B3 ★', 'ViT-B/16'],
        'Accuracy': ['82.50%', '85.83%', '80.00%'],
        'F1 Erosive': ['89.45%', '91.63%', '87.23%'],
        'F1 Non-Erosive': ['48.78%', '54.05%', '53.85%'],
        'Erosive Recall': ['94.12%', '95.04%', '92.86%'],
        'Non-Erosive Recall': ['45.00%', '45.00%', '47.50%'],
        'Model Size': ['90 MB', '41 MB', '327 MB'],
        'Selection': ['❌', '✅', '❌']
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Augmentation Strategy Applied to All Models**:
    - WeightedRandomSampler: Balances 4.59:1 imbalance to ~1:1 per batch
    - Focal Loss (γ=2.0): Focuses training on hard-to-learn minority class examples
    - Progressive Augmentation: Horizontal flip, rotation ±15°, color jitter, Gaussian blur
    - F1-based Early Stopping: Monitors erosive class F1 (not validation loss)
    - Result: Significant improvement in minority class detection (~20pp F1 increase)
    
    **Selection Criteria:**
    1. **Accuracy**: EfficientNet-B3 highest (85.83%)
    2. **Minority Class F1**: EfficientNet-B3 best (54.05%, crucial for early RA detection)
    3. **Production Ready**: Most efficient (41MB vs 327MB for ViT)
    4. **Erosive Recall**: All models >92% (catches most erosion cases)
    
    See `reports/image/model_comparison_all_models.png` for detailed comparison visualizations
    - EfficientNet-B3 provides best balance for real-world deployment
    """)
    
    # Display evaluation visualizations if available
    eval_dir = os.path.join(ROOT, 'reports/evaluation_dashboard')
    if os.path.exists(eval_dir):
        st.divider()
        st.write("### 📊 Detailed Evaluation Charts")
        
        viz_files = sorted([f for f in os.listdir(eval_dir) if f.endswith('.png')])
        for viz_file in viz_files[:3]:  # Show first 3
            viz_path = os.path.join(eval_dir, viz_file)
            img = Image.open(viz_path)
            st.image(img, use_container_width=True)

# Footer
st.divider()
st.caption("🏥 Clinical RA Assessment System | EfficientNet-B3 Imaging Model | XGBoost Numeric Model | For Research Use Only")
