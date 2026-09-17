"""
Streamlit Web Application for 4-Class Brain MRI Tumor Classification
Serving ResNet50, DenseNet121, EfficientNetV2B0, and Baseline Attention CNN
"""
import os
import glob
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

from src import config
from src.preprocessing import get_preprocessing_function
from src.models import apply_cbam_attention

# Page Configuration
st.set_page_config(
    page_title="Brain MRI Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Medical Web App Interface
st.markdown("""
<style>
    /* Dark Theme Custom Styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    .main-header {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .result-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .badge-glioma { background-color: #ef4444; color: white; }
    .badge-meningioma { background-color: #f59e0b; color: white; }
    .badge-notumor { background-color: #10b981; color: white; }
    .badge-pituitary { background-color: #6366f1; color: white; }
    
    .stSelectbox label { color: #cbd5e1 !important; font-weight: 600; }
    .stFileUploader label { color: #cbd5e1 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Cache Model Loading to prevent reload overhead
@st.cache_resource
def load_trained_model(model_key):
    """Load specified Keras model with custom layers."""
    custom_objects = {"apply_cbam_attention": apply_cbam_attention}
    
    model_paths = {
        "ResNet50": os.path.join(config.MODELS_DIR, "ResNet50_best.keras"),
        "DenseNet121": os.path.join(config.MODELS_DIR, "DenseNet121_best.keras"),
        "EfficientNetV2B0": os.path.join(config.MODELS_DIR, "EfficientNetV2B0_best.keras"),
        "Baseline Attention CNN": os.path.join(config.MODELS_DIR, "Baseline_CNN.keras"),
    }
    
    # Fallback paths
    fallbacks = {
        "ResNet50": os.path.join(config.MODELS_DIR, "ResNet50.keras"),
        "DenseNet121": os.path.join(config.MODELS_DIR, "DenseNet121.keras"),
        "EfficientNetV2B0": os.path.join(config.MODELS_DIR, "EfficientNetV2B0.keras"),
        "Baseline Attention CNN": os.path.join(config.MODELS_DIR, "Baseline_CNN_stage1.keras"),
    }
    
    target_path = model_paths.get(model_key)
    if not os.path.exists(target_path):
        target_path = fallbacks.get(model_key)
        
    if not os.path.exists(target_path):
        st.error(f"Model file not found for {model_key} at {target_path}")
        return None
        
    model = tf.keras.models.load_model(target_path, custom_objects=custom_objects)
    return model

def preprocess_image(pil_img, model_key, target_size=(224, 224)):
    """Preprocess PIL image tensor for target architecture."""
    img_rgb = pil_img.convert("RGB")
    img_resized = img_rgb.resize(target_size)
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    pfn = get_preprocessing_function(model_key)
    preprocessed_batch = pfn(img_batch)
    return preprocessed_batch, img_rgb

# Model Metadata Dictionary
MODEL_METADATA = {
    "ResNet50": {
        "accuracy": "94.87%",
        "precision": "94.87%",
        "recall": "94.85%",
        "macro_f1": "94.79%",
        "roc_auc": "0.9931",
        "params": "24.1 M",
        "inference_time": "47.5 ms",
        "badge_color": "#10b981",
        "desc": "Deep residual learning architecture with modified adaptation head & BN stabilization. Recommended model."
    },
    "DenseNet121": {
        "accuracy": "92.94%",
        "precision": "92.89%",
        "recall": "92.84%",
        "macro_f1": "92.80%",
        "roc_auc": "0.9896",
        "params": "7.3 M",
        "inference_time": "43.3 ms",
        "badge_color": "#3b82f6",
        "desc": "Densely connected convolutional network maximizing feature reuse across dense blocks."
    },
    "EfficientNetV2B0": {
        "accuracy": "90.43%",
        "precision": "90.31%",
        "recall": "90.26%",
        "macro_f1": "90.13%",
        "roc_auc": "0.9896",
        "params": "6.3 M",
        "inference_time": "18.2 ms",
        "badge_color": "#8b5cf6",
        "desc": "Ultra-lightweight architecture with fused MBConv layers. Fastest inference speed."
    },
    "Baseline Attention CNN": {
        "accuracy": "86.40%",
        "precision": "86.40%",
        "recall": "86.20%",
        "macro_f1": "86.30%",
        "roc_auc": "0.9650",
        "params": "1.2 M",
        "inference_time": "28.4 ms",
        "badge_color": "#f59e0b",
        "desc": "Custom 5-block CNN with Convolutional Block Attention Module (CBAM) channel & spatial attention."
    }
}

TUMOR_DESCRIPTIONS = {
    "glioma": {
        "title": "Glioma Tumor",
        "badge_class": "badge-glioma",
        "info": "Gliomas originate in the glial cells of the brain and spinal cord. They represent ~30% of all brain tumors and often require surgical resection, radiation, or chemotherapy."
    },
    "meningioma": {
        "title": "Meningioma Tumor",
        "badge_class": "badge-meningioma",
        "info": "Meningiomas arise from the meninges surrounding the brain and spinal cord. Most meningiomas are non-cancerous (benign) and grow slowly."
    },
    "notumor": {
        "title": "No Tumor Detected",
        "badge_class": "badge-notumor",
        "info": "The scan displays healthy brain tissue with no detectable tumor mass in the analyzed axial region."
    },
    "pituitary": {
        "title": "Pituitary Tumor",
        "badge_class": "badge-pituitary",
        "info": "Pituitary tumors develop in the pituitary gland at the base of the brain. Most are benign adenomas that may alter endocrine hormone levels."
    }
}

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("## ⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Deep Learning Model",
    options=["ResNet50", "DenseNet121", "EfficientNetV2B0", "Baseline Attention CNN"],
    index=0,
    help="Choose the model architecture for MRI diagnosis."
)

meta = MODEL_METADATA[model_choice]

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📊 Model Info: **{model_choice}**")
st.sidebar.markdown(f"*{meta['desc']}*")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    st.metric("Test Accuracy", meta["accuracy"])
    st.metric("Macro F1", meta["macro_f1"])
with col_sb2:
    st.metric("Parameters", meta["params"])
    st.metric("Inference Time", meta["inference_time"])

st.sidebar.markdown("---")

# Sample Image Picker Option
st.sidebar.markdown("### 🖼️ Sample MRI Library")
sample_files = []
for c in config.CLASSES:
    cdir = os.path.join(config.DATA_DIR, c)
    if os.path.exists(cdir):
        imgs = [f for f in os.listdir(cdir) if f.endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.')]
        if imgs:
            sample_files.append((c, os.path.join(cdir, imgs[0])))

use_sample = st.sidebar.checkbox("Use a sample MRI scan", value=False)
selected_sample_path = None
if use_sample and sample_files:
    sample_labels = [f"{c.upper()}: {os.path.basename(p)}" for c, p in sample_files]
    sample_idx = st.sidebar.selectbox("Choose sample MRI", range(len(sample_labels)), format_func=lambda i: sample_labels[i])
    selected_sample_path = sample_files[sample_idx][1]

# --- MAIN APP LAYOUT ---
st.markdown('<div class="main-header">🧠 Brain MRI Tumor Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated 4-Class Brain MRI Diagnosis & Multi-Model Comparative Serving</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_predict, tab_comparison, tab_analysis = st.tabs(["🔬 Predict & Diagnose", "📊 Model Comparison", "📈 Model Artifacts"])

with tab_predict:
    col_left, col_right = st.columns([1, 1], gap="large")

    uploaded_file = None
    input_img = None

    with col_left:
        st.markdown("### 📤 Upload Brain MRI Scan")
        uploaded_file = st.file_uploader("Upload a Brain MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            input_img = Image.open(uploaded_file)
        elif use_sample and selected_sample_path:
            input_img = Image.open(selected_sample_path)
            st.info(f"Loaded sample image: `{os.path.basename(selected_sample_path)}`")
            
        if input_img is not None:
            st.image(input_img, caption="Uploaded Brain MRI Scan", use_column_width=True)
        else:
            st.warning("Please upload a JPG or PNG image, or select a sample image from the sidebar.")

    with col_right:
        st.markdown("### 🩺 Diagnostic Prediction")
        
        if input_img is not None:
            # Load selected model
            with st.spinner(f"Loading {model_choice} model and analyzing scan..."):
                model = load_trained_model(model_choice)
                
                if model is not None:
                    # Preprocess and predict
                    input_shape = model.input_shape[1:3] if model.input_shape[1] is not None else (224, 224)
                    preprocessed_x, img_rgb = preprocess_image(input_img, model_choice, target_size=input_shape)
                    
                    preds = model.predict(preprocessed_x, verbose=0)[0]
                    pred_class_idx = int(np.argmax(preds))
                    pred_class_name = config.CLASSES[pred_class_idx]
                    confidence = float(preds[pred_class_idx]) * 100.0
                    
                    tumor_meta = TUMOR_DESCRIPTIONS[pred_class_name]
                    
                    # Output Badge & Prediction Summary
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.9); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 1rem;">
                        <span class="result-badge {tumor_meta['badge_class']}">{tumor_meta['title']}</span>
                        <h2 style="margin: 0.5rem 0; font-size: 2rem; color: #f8fafc;">Confidence: {confidence:.2f}%</h2>
                        <p style="color: #cbd5e1; font-size: 1rem; margin-top: 0.5rem;">{tumor_meta['info']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability Bar Chart (Plotly)
                    st.markdown("#### 📊 Prediction Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Class": [c.capitalize() for c in config.CLASSES],
                        "Probability": [float(p) * 100.0 for p in preds],
                        "Color": ["#ef4444" if c == pred_class_name else "#3b82f6" for c in config.CLASSES]
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x="Probability",
                        y="Class",
                        orientation="h",
                        text=prob_df["Probability"].apply(lambda val: f"{val:.2f}%"),
                        color="Class",
                        color_discrete_map={
                            "Glioma": "#ef4444",
                            "Meningioma": "#f59e0b",
                            "Notumor": "#10b981",
                            "Pituitary": "#6366f1"
                        }
                    )
                    fig.update_layout(
                        xaxis_title="Probability (%)",
                        yaxis_title="Tumor Class",
                        xaxis=dict(range=[0, 105]),
                        showlegend=False,
                        height=280,
                        margin=dict(l=0, r=20, t=10, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e2e8f0")
                    )
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)

with tab_comparison:
    st.markdown("### 🏆 Comprehensive Model Comparison")
    
    comp_csv_path = os.path.join(config.RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_csv_path):
        df_comp = pd.read_csv(comp_csv_path)
        
        # Display styled dataframe
        st.dataframe(
            df_comp.style.highlight_max(axis=0, subset=["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1", "ROC-AUC"], color="#1e3a8a"),
            use_container_width=True
        )
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Accuracy Comparison")
            fig_bar = px.bar(
                df_comp,
                x="Model",
                y="Accuracy",
                color="Model",
                text=df_comp["Accuracy"].apply(lambda v: f"{v*100:.2f}%"),
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig_bar.update_layout(
                yaxis=dict(range=[0, 1.08]),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_c2:
            st.markdown("#### Inference Speed vs Accuracy")
            fig_scatter = px.scatter(
                df_comp,
                x="Inference Time (ms)",
                y="Accuracy",
                size="Parameters",
                color="Model",
                text="Model",
                size_max=30
            )
            fig_scatter.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Model comparison data not found.")

with tab_analysis:
    st.markdown(f"### 📈 Performance Artifacts for **{model_choice}**")
    
    col_a1, col_a2 = st.columns(2)
    
    # 1. Learning Curve
    lc_path = os.path.join(config.PLOTS_DIR, f"{model_choice}_learning_curves_combined.png")
    if not os.path.exists(lc_path):
        lc_path = os.path.join(config.PLOTS_DIR, f"{model_choice}_learning_curves.png")
        
    with col_a1:
        st.markdown("#### Training & Validation Learning Curves")
        if os.path.exists(lc_path):
            st.image(lc_path, use_column_width=True)
        else:
            st.info("Learning curve plot not found.")
            
    # 2. Confusion Matrix
    cm_path = os.path.join(config.CONFUSION_DIR, f"{model_choice}_normalized_confusion_matrix.png")
    if not os.path.exists(cm_path):
        cm_path = os.path.join(config.CONFUSION_DIR, f"{model_choice}_confusion_matrix.png")
        
    with col_a2:
        st.markdown("#### Normalized Confusion Matrix")
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.info("Confusion matrix plot not found.")

    col_a3, col_a4 = st.columns(2)
    roc_path = os.path.join(config.PLOTS_DIR, f"{model_choice}_roc_curves.png")
    pr_path = os.path.join(config.PLOTS_DIR, f"{model_choice}_per_class_metrics.png")
    
    with col_a3:
        st.markdown("#### Multi-Class ROC Curves")
        if os.path.exists(roc_path):
            st.image(roc_path, use_column_width=True)
        else:
            st.info("ROC curves plot not found.")
            
    with col_a4:
        st.markdown("#### Per-Class Performance Metrics")
        if os.path.exists(pr_path):
            st.image(pr_path, use_column_width=True)
        else:
            st.info("Per-class metrics plot not found.")
