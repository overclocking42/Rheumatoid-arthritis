"""
Configuration file for the Clinical RA Assessment System with authentication.
Customize settings here without modifying app code.
"""

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# Path to SQLite database (relative to app.py location)
DATABASE_PATH = "data/app_users.db"

# Enable automatic database backups
AUTO_BACKUP = True
BACKUP_DIRECTORY = "data/backups"

# ============================================================================
# SECURITY SETTINGS
# ============================================================================

# Password requirements
MIN_PASSWORD_LENGTH = 6
REQUIRE_UPPERCASE = False
REQUIRE_NUMBERS = False
REQUIRE_SPECIAL_CHARS = False

# PBKDF2 hashing
PBKDF2_ITERATIONS = 100000
PBKDF2_ALGORITHM = 'sha256'
SALT_LENGTH = 32

# Session timeout (seconds, 0 = never timeout)
SESSION_TIMEOUT = 0

# ============================================================================
# UI CUSTOMIZATION
# ============================================================================

# Application title and description
APP_TITLE = "🏥 Clinical RA Assessment System"
APP_DESCRIPTION = "Evidence-Based Diagnosis Support | Numerical + Imaging Analysis"

# Tab names
TABS = {
    "lab": "📋 Lab Assessment",
    "xray": "🖼️ X-Ray Analysis",
    "combined": "📊 Combined Results",
    "performance": "📈 Model Performance",
    "history": "📜 Prediction History"
}

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Model paths (relative to project root)
NUMERIC_MODEL_PATH = "models/ann_model.pth"
NUMERIC_SCALER_PATH = "models/ann_scaler.pkl"
IMAGING_MODEL_PATH = "models/swin_fold4_best.pth"

# Model names for display
NUMERIC_MODEL_NAME = "ANN (Artificial Neural Network)"
IMAGING_MODEL_NAME = "Swin Transformer"

# ============================================================================
# LAB ASSESSMENT SETTINGS
# ============================================================================

# Default values for lab inputs
LAB_DEFAULTS = {
    "age": 55,
    "gender": "Female",
    "rf": 25.0,
    "anti_ccp": 15.0,
    "crp": 8.0,
    "esr": 30.0
}

# Input ranges and limits
LAB_RANGES = {
    "age": {"min": 0, "max": 120},
    "rf": {"min": 0.0, "max": 500.0},
    "anti_ccp": {"min": 0.0, "max": 500.0},
    "crp": {"min": 0.0, "max": 100.0},
    "esr": {"min": 0.0, "max": 150.0}
}

# Class names for predictions
LAB_CLASS_NAMES = ["Healthy", "Seropositive RA", "Seronegative RA"]

# ============================================================================
# IMAGING ASSESSMENT SETTINGS
# ============================================================================

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ['bmp', 'png', 'jpg', 'jpeg']

# Image preprocessing
IMAGE_SIZE = 224
IMAGE_NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],  # ImageNet-21k
    "std": [0.229, 0.224, 0.225]
}

# Erosion classification threshold
EROSION_THRESHOLD = 0.5

# Class names for imaging
IMAGING_CLASS_NAMES = ["Non-Erosive", "Erosive"]

# ============================================================================
# PREDICTION HISTORY SETTINGS
# ============================================================================

# How many predictions to show by default
HISTORY_DEFAULT_LIMIT = 100

# Maximum predictions to retrieve
HISTORY_MAX_LIMIT = 1000

# Export formats
EXPORT_FORMATS = {
    "csv": {"extension": "csv", "mime": "text/csv"},
    "json": {"extension": "json", "mime": "application/json"}
}

# ============================================================================
# MODEL PERFORMANCE DISPLAY
# ============================================================================

# Numeric model performance metrics
NUMERIC_MODEL_METRICS = {
    "cv_accuracy": 0.8892,
    "cv_std": 0.0022,
    "test_accuracy": 0.9126,
    "f1_scores": {
        "healthy": 0.6818,
        "seropositive": 0.9659,
        "seronegative": 0.7803
    }
}

# Imaging model performance metrics
IMAGING_MODEL_METRICS = {
    "test_accuracy": 0.8583,
    "test_auc": 0.95,
    "f1_erosive": 0.9171,
    "f1_non_erosive": 0.8879,
    "recall": 0.9495,
    "precision": 0.8879
}

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable features
FEATURES = {
    "authentication": True,
    "prediction_history": True,
    "export_csv": True,
    "export_json": True,
    "export_pdf": False,  # Future
    "user_statistics": True,
    "performance_charts": True,
    "email_notifications": False,  # Future
    "api_endpoints": False  # Future
}

# ============================================================================
# LOGGING AND MONITORING
# ============================================================================

# Enable logging
ENABLE_LOGGING = True
LOG_FILE = "data/app.log"

# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# Log predictions to file
LOG_PREDICTIONS = True

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Device for model inference
DEVICE = "cpu"  # or "cuda" for GPU

# Number of threads for PyTorch
PYTORCH_NUM_THREADS = 4

# Cache models in memory
CACHE_MODELS = True

# Maximum cache size (MB)
MAX_CACHE_SIZE = 500

# Enable debug mode
DEBUG_MODE = False

# ============================================================================
# CLINICAL GUIDELINES
# ============================================================================

# Treatment recommendations based on results
CLINICAL_GUIDELINES = {
    "healthy": {
        "title": "No RA Indicators",
        "recommendation": "Annual follow-up recommended"
    },
    "seronegative_non_erosive": {
        "title": "RA Without Bone Damage",
        "recommendation": "Early DMARD initiation recommended"
    },
    "seronegative_erosive": {
        "title": "Advanced RA Without Serology",
        "recommendation": "Aggressive DMARD therapy + imaging"
    },
    "seropositive_non_erosive": {
        "title": "RA Without Bone Damage",
        "recommendation": "Early DMARD initiation to prevent erosions"
    },
    "seropositive_erosive": {
        "title": "Advanced RA With Bone Damage",
        "recommendation": "Biologic DMARD therapy urgent"
    }
}

# ============================================================================
# DEPLOYMENT SETTINGS
# ============================================================================

# Port for Streamlit
STREAMLIT_PORT = 8501

# Public URL (for sharing)
PUBLIC_URL = "http://localhost:8501"

# Enable sharing
ENABLE_SHARING = False

# ============================================================================
# GEMINI API CONFIGURATION (For PDF Lab Report Extraction)
# ============================================================================

# IMPORTANT: Get your free API key from: https://makersuite.google.com/app/apikeys
# 
# Method 1: Set environment variable (RECOMMENDED)
#   export GOOGLE_API_KEY="your-api-key-here"
#
# Method 2: Paste here (LESS SECURE - don't commit to git)
#   GEMINI_API_KEY = "your-api-key-here"
#
# The system will check both locations automatically
GEMINI_API_KEY = ""

# Alternate API key (fallback if primary doesn't work)
GEMINI_API_KEY_ALTERNATE = ""

# ============================================================================
# GROQ API CONFIGURATION (Preferred for Ask AI chat support)
# ============================================================================

# Method 1: Set environment variable (RECOMMENDED)
#   export GROQ_API_KEY="your-api-key-here"
#
# Method 2: Paste here (LESS SECURE - don't commit to git)
#   GROQ_API_KEY = "your-api-key-here"
#
# The app prefers Groq for Ask AI when this key is available.
GROQ_API_KEY = ""

# PDF extraction settings
PDF_EXTRACTION_ENABLED = True  # Set to False to disable PDF feature
PDF_MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB max file size
SUPPORTED_REPORT_FORMATS = ['pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp']

# ============================================================================
# DATA RETENTION
# ============================================================================

# Keep predictions older than X days (0 = keep all)
DATA_RETENTION_DAYS = 0

# Automatically delete old predictions
AUTO_DELETE_OLD = False

# Archive predictions instead of deleting
ARCHIVE_OLD = True
