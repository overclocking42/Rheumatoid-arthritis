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
from model_utils import (
    ROOT,
    load_image_model_artifact,
    load_numeric_model_artifacts,
    normalize_numeric_features,
    predict_xray_image,
    preprocess_xray_image,
)

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

@st.cache_resource
def load_numeric_model():
    try:
        return load_numeric_model_artifacts()
    except Exception as e:
        st.error(f"❌ Error loading ANN model: {str(e)}")
        return None, None

@st.cache_resource
def load_image_model():
    try:
        return load_image_model_artifact()
    except Exception as e:
        st.error(f"❌ Error loading Swin model: {str(e)}")
        return None

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
    
    model, scaler = load_numeric_model()
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
            # Prepare data for ANN
            device = torch.device('cpu')

            # Raw feature vector in original units (matching scaler fit)
            raw_features = [
                float(age),
                1.0 if gender == 'Female' else 0.0,
                float(rf),
                float(anti_ccp),
                float(crp),
                float(esr)
            ]

            # Apply fitted scaler if available (RobustScaler/StandardScaler, etc.)
            try:
                normalized = normalize_numeric_features(raw_features, scaler)
            except Exception as e:
                st.error(f"❌ Scaling error: {e}")
                st.stop()

            data = torch.tensor(normalized, dtype=torch.float32).to(device)

            # Predict using the ANN model
            try:
                with torch.no_grad():
                    logits = model(data.unsqueeze(0))
                    proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
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
                    x_tensor = preprocess_xray_image(img)
                    label, confidence = predict_xray_image(img_model, x_tensor)
                    
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
                    st.caption("**Model**: Swin Transformer (Fold 4) | **Accuracy**: 85.83% | **F1 Erosive**: 91.71% | **F1 Non-Erosive**: 88.79%")

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
        st.write("### 📊 Numeric Model (ANN)")
        st.markdown("""
        - **Task**: 3-class classification (Healthy/Seropositive/Seronegative)
        - **CV Accuracy**: 88.92% ✅
        - **CV Std Dev**: ±0.22% (most stable)
        - **F1 Seropositive**: 96.59% (excellent disease detection)
        - **Features**: Age, Gender, RF, Anti-CCP, CRP, ESR
        - **Architecture**: 6→128→64→3 (ANN with ReLU)
        """)
    
    with col2:
        st.write("### 🎯 Imaging Model (Swin Transformer)")
        st.markdown("""
        - **Task**: 2-class classification (Erosive/Non-Erosive)
        - **Test Accuracy**: 85.83% ✅ 
        - **Test ROC-AUC**: ~0.95
        - **F1 Erosive**: 91.71%
        - **F1 Non-Erosive**: 88.79%
        - **Recall**: 94.95% (catches 95% of erosion cases)
        - **Architecture**: Swin-Base (Vision Transformer)
        """)
    
    st.divider()
    
    st.write("### 📈 Numeric Models Comparison")
    
    numeric_comparison = pd.DataFrame({
        'Model': ['XGBoost', 'CatBoost', 'ANN ⭐'],
        'CV Accuracy': ['88.07%', '88.74%', '88.92%'],
        'CV Std Dev': ['±0.82%', '±1.41%', '±0.22%'],
        'F1 Seropositive': ['94.90%', '95.56%', '96.59%'],
        'F1 Healthy': ['72.92%', '69.24%', '68.18%'],
        'Generalization': ['Good', 'Moderate', 'Excellent'],
        'Selection': ['❌', '❌', '✅']
    })
    
    st.dataframe(numeric_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **ANN Selection Reason**: 
    - Lowest variance (0.22% vs 0.82% XGBoost) → Most stable predictions
    - Highest Seropositive detection (96.59% F1) → Best disease detection
    - Best overall score (0.9164) across weighted metrics
    - Smallest model size → Easy deployment
    """)
    
    st.divider()
    
    st.write("### 📈 Imaging Models Comparison")
    
    imaging_comparison = pd.DataFrame({
        'Model': ['DenseNet121', 'Swin Transformer ⭐'],
        'Mean Accuracy': ['77.00%', '83.50%'],
        'Best Fold Accuracy': ['81.67%', '85.83%'],
        'Mean F1': ['77.34%', '90.38%'],
        'Recall': ['77.00%', '94.75%'],
        'Precision': ['77.85%', '87.09%'],
        'Selection': ['❌', '✅']
    })
    
    st.dataframe(imaging_comparison, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Swin Transformer Selection Reason**:
    - Exceeds 85% accuracy target (85.83% vs 77% DenseNet)
    - Vision Transformer captures long-range dependencies in X-ray images
    - Excellent recall (94.95%) ensures disease detection
    - High precision (88.79%) minimizes false alarms
    - Consistent performance across folds (std=1.78%)
    - ImageNet-21k pretraining provides better generalization
    
    **Fold 4 Selected** for deployment: Best balance of all metrics
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
