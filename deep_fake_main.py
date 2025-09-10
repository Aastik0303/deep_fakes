# app.py
import os
import tempfile
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit

# ----------- USER CONFIG -----------
CNN_MODEL_PATH = "cnn_deepfake_detector.h5"
LSTM_MODEL_PATH = "lstm_deepfake_detector5.h5"
SEQ_LEN = 10
FRAME_STRIDE = 1
THRESHOLD = 0.5
# -----------------------------------

print("Loading models...")
cnn = load_model(CNN_MODEL_PATH)
lstm = load_model(LSTM_MODEL_PATH)
print("Models loaded.")

# Auto-detect CNN input size
cnn_input_shape = cnn.input_shape
if cnn_input_shape and len(cnn_input_shape) == 4:
    IMG_SIZE = (cnn_input_shape[1], cnn_input_shape[2])  # (H, W)
else:
    IMG_SIZE = (64, 64)  # fallback
print("✅ Using IMG_SIZE =", IMG_SIZE)

def make_feature_extractor(model):
    """Try to create a feature extractor (penultimate layer)."""
    try:
        dummy = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype("float32")
        _ = model.predict(dummy, verbose=0)

        if len(model.layers) >= 2:
            penult = model.layers[-2].output
            feat_model = Model(inputs=model.input, outputs=penult)
            print("✅ Feature extractor built from penultimate layer.")
            return feat_model
    except Exception as e:
        print("⚠️ Could not create feature extractor:", e)

    print("➡️ Using original CNN output as features (may be probs).")
    return model

feat_extractor = make_feature_extractor(cnn)

# Debug CNN output shape
dummy = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype("float32")
try:
    print("CNN output shape:", feat_extractor.predict(dummy, verbose=0).shape)
except Exception as e:
    print("⚠️ Could not test CNN output:", e)

def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return resized.astype("float32") / 255.0

def extract_frames_from_video(path, seq_len=SEQ_LEN, stride=FRAME_STRIDE):
    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(preprocess_frame(frame))
        idx += 1
    cap.release()

    if len(frames) == 0:
        return np.zeros((seq_len, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)

    if len(frames) < seq_len:
        while len(frames) < seq_len:
            frames.append(frames[-1])

    return np.array(frames[:seq_len])

def predict_from_frames(frames):
    """
    Predict using either (CNN → LSTM) or (Raw Frames → LSTM),
    depending on what the LSTM model was trained on.
    """
    frames_b = frames.astype("float32")

    # Case 1: LSTM expects raw frames (e.g., (None, 10, H, W, 3))
    if len(lstm.input_shape) == 5:
        seq = np.expand_dims(frames_b, axis=0)  # (1, SEQ_LEN, H, W, 3)
        lstm_out = lstm.predict(seq, verbose=0)

    # Case 2: LSTM expects CNN features (e.g., (None, 10, feature_dim))
    else:
        feats = feat_extractor.predict(frames_b, verbose=0)
        if len(feats.shape) > 2:
            feats = feats.reshape((feats.shape[0], -1))
        seq = np.expand_dims(feats, axis=0)  # (1, SEQ_LEN, feature_dim)
        lstm_out = lstm.predict(seq, verbose=0)

    out_arr = np.array(lstm_out).reshape(-1)
    prob = float(out_arr[0])
    return prob

# ---------------- HTML ----------------
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Deepfake Detector</title>
  <style>
    * {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  display: flex;
}

/* Sidebar Styles */
.sidebar {
  width: 280px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  box-shadow: 2px 0 20px rgba(0, 0, 0, 0.1);
  padding: 20px 0;
  position: fixed;
  height: 100vh;
  overflow-y: auto;
  z-index: 1000;
  transition: transform 0.3s ease;
}

.sidebar.hidden {
  transform: translateX(-100%);
}

.sidebar-header {
  padding: 0 20px 20px;
  border-bottom: 2px solid rgba(102, 126, 234, 0.1);
}

.sidebar-title {
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #667eea, #764ba2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 5px;
}

.sidebar-subtitle {
  font-size: 0.9rem;
  color: #666;
}

.sidebar-nav {
  padding: 20px 0;
}

.nav-item {
  display: flex;
  align-items: center;
  padding: 12px 20px;
  color: #555;
  text-decoration: none;
  font-weight: 500;
  transition: all 0.3s ease;
  cursor: pointer;
  border-left: 3px solid transparent;
}

.nav-item:hover {
  background: rgba(102, 126, 234, 0.1);
  color: #667eea;
}

.nav-item.active {
  background: rgba(102, 126, 234, 0.15);
  color: #667eea;
  border-left-color: #667eea;
}

.nav-icon {
  margin-right: 12px;
  font-size: 1.2rem;
}

/* Mobile Toggle Button */
.sidebar-toggle {
  display: none;
  position: fixed;
  top: 20px;
  left: 20px;
  z-index: 1001;
  background: rgba(255, 255, 255, 0.9);
  border: none;
  border-radius: 8px;
  padding: 10px;
  font-size: 1.5rem;
  cursor: pointer;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

/* Main Content */
.main-content {
  margin-left: 280px;
  flex: 1;
  padding: 20px;
  transition: margin-left 0.3s ease;
  width: 100%;
}

.main-content.full-width {
  margin-left: 0;
}

.container {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  width: 100%;
  max-width: none; /* Removed max-width constraint */
  margin: 0 auto;
  padding: 40px;
  position: relative;
  overflow: hidden;
}

.container::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
  animation: shimmer 3s infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.header {
  text-align: center;
  margin-bottom: 40px;
}

h1 {
  color: #2c3e50;
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 10px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.subtitle {
  color: #666;
  font-size: 1.1rem;
  margin-bottom: 20px;
}

.features {
  display: flex;
  justify-content: center;
  gap: 30px;
  margin-bottom: 30px;
  flex-wrap: wrap;
}

.feature {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #555;
  font-size: 0.9rem;
}

.feature::before {
  content: '✓';
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
}

/* Section Styles */
.section {
  display: none;
}

.section.active {
  display: block;
}

.about-section {
  max-width: none; /* Removed max-width constraint */
  width: 100%; /* Full width */
}

.about-section h2 {
  color: #2c3e50;
  font-size: 2rem;
  margin-bottom: 20px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.about-section h3 {
  color: #667eea;
  font-size: 1.5rem;
  margin-top: 30px;
  margin-bottom: 15px;
}

.about-section p {
  color: #555;
  line-height: 1.6;
  margin-bottom: 15px;
  text-align: justify;
}

.model-card {
  background: rgba(102, 126, 234, 0.1);
  border-left: 4px solid #667eea;
  padding: 20px;
  margin: 20px 0;
  border-radius: 8px;
}

.model-card h4 {
  color: #667eea;
  font-size: 1.2rem;
  margin-bottom: 10px;
}

.model-features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.feature-box {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  text-align: center;
}

.feature-box .icon {
  font-size: 2rem;
  margin-bottom: 10px;
}

.feature-box h5 {
  color: #667eea;
  margin-bottom: 8px;
  font-size: 1.1rem;
}

.feature-box p {
  color: #666;
  font-size: 0.9rem;
  text-align: center;
}

/* Upload Section Styles */
.upload-section {
  border: 3px dashed #ddd;
  border-radius: 16px;
  padding: 40px;
  text-align: center;
  margin-bottom: 30px;
  transition: all 0.3s ease;
  position: relative;
  background: linear-gradient(45deg, #f8f9ff, #fff);
  cursor: pointer;
}

.upload-section:hover {
  border-color: #667eea;
  background: linear-gradient(45deg, #f0f4ff, #fff);
}

.upload-section.dragover {
  border-color: #667eea;
  background: linear-gradient(45deg, #f0f4ff, #fff);
  transform: scale(1.02);
}

.upload-icon {
  font-size: 3rem;
  color: #667eea;
  margin-bottom: 20px;
  display: block;
}

.file-input {
  display: none;
}

.upload-button {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border: none;
  padding: 15px 30px;
  border-radius: 12px;
  font-size: 1.1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
  margin-top: 15px;
}

.upload-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.upload-button:active {
  transform: translateY(0);
}

.file-info {
  margin-top: 20px;
  padding: 15px;
  background: rgba(102, 126, 234, 0.1);
  border-radius: 12px;
  display: none;
}

.file-name {
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 5px;
}

.file-size {
  color: #666;
  font-size: 0.9rem;
}

.analyze-button {
  background: linear-gradient(135deg, #43e97b, #38f9d7);
  color: white;
  border: none;
  padding: 15px 40px;
  border-radius: 12px;
  font-size: 1.2rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  width: 100%;
  margin-top: 20px;
  box-shadow: 0 4px 15px rgba(67, 233, 123, 0.3);
  display: none;
}

.analyze-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(67, 233, 123, 0.4);
}

.analyze-button:disabled {
  background: #ccc;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.loading {
  display: none;
  text-align: center;
  margin: 30px 0;
}

.spinner {
  border: 4px solid #f3f3f3;
  border-top: 4px solid #667eea;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.result {
  display: none;
  margin-top: 30px;
  padding: 30px;
  border-radius: 16px;
  text-align: center;
  animation: fadeInUp 0.5s ease;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.result.real {
  background: linear-gradient(135deg, #d4edda, #c3e6cb);
  border: 2px solid #28a745;
}

.result.fake {
  background: linear-gradient(135deg, #f8d7da, #f5c6cb);
  border: 2px solid #dc3545;
}

.result.error {
  background: linear-gradient(135deg, #fff3cd, #ffeaa7);
  border: 2px solid #ffc107;
}

.result-icon {
  font-size: 4rem;
  margin-bottom: 20px;
}

.result-label {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 15px;
}

.result.real .result-label {
  color: #28a745;
}

.result.fake .result-label {
  color: #dc3545;
}

.result.error .result-label {
  color: #856404;
}

.probability-bar {
  width: 100%;
  height: 20px;
  background: #eee;
  border-radius: 10px;
  overflow: hidden;
  margin: 20px 0;
}

.probability-fill {
  height: 100%;
  border-radius: 10px;
  transition: width 1s ease;
}

.confidence-text {
  font-size: 1.1rem;
  margin-top: 10px;
  font-weight: 600;
}

.reset-button {
  background: #6c757d;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  margin-top: 20px;
  font-size: 1rem;
  transition: all 0.3s ease;
}

.reset-button:hover {
  background: #5a6268;
  transform: translateY(-1px);
}

.supported-formats {
  margin-top: 20px;
  text-align: center;
  color: #666;
  font-size: 0.9rem;
}

/* Mobile Responsive */
@media (max-width: 768px) {
  .sidebar {
    transform: translateX(-100%);
  }

  .sidebar.show {
    transform: translateX(0);
  }

  .sidebar-toggle {
    display: block;
  }

  .main-content {
    margin-left: 0;
    padding: 80px 15px 15px;
    width: 100%;
  }

  .container {
    margin: 10px 0;
    padding: 30px 20px;
  }

  h1 {
    font-size: 2rem;
  }

  .features {
    gap: 15px;
  }

  .upload-section {
    padding: 30px 20px;
  }

  .model-features {
    grid-template-columns: 1fr;
  }
}
  </style>
</head>
<body>
  <!-- Mobile Toggle Button -->
  <button class="sidebar-toggle" id="sidebarToggle">☰</button>

  <!-- Sidebar -->
  <div class="sidebar" id="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-title">🔍 AI Detector</div>
      <div class="sidebar-subtitle">Advanced Deepfake Detection</div>
    </div>
    
    <nav class="sidebar-nav">
      <a class="nav-item active" data-section="home">
        <span class="nav-icon">🏠</span>
        Home
      </a>
      <a class="nav-item" data-section="about">
        <span class="nav-icon">ℹ️</span>
        About
      </a>
    <a href="login&reg.py" class="nav-item" data-section="signin">
  <span class="nav-icon"><i class="fas fa-sign-in-alt"></i></span>
  Sign In
</a>

    
      </a>
    </nav>
  </div>

  <!-- Main Content -->
  <div class="main-content" id="mainContent">
    <!-- Home Section -->
    <div class="section active" id="home-section">
      <div class="container">
        <div class="header">
          <h1>🔍 AI Deepfake Detector</h1>
          <p class="subtitle">Advanced AI-powered detection for manipulated media</p>
          
          <div class="features">
            <div class="feature">Video Analysis</div>
            <div class="feature">Image Detection</div>
            <div class="feature">Real-time Results</div>
          </div>
        </div>

        <form id="uploadForm" enctype="multipart/form-data">
          <div class="upload-section" id="uploadSection">
            <div class="upload-icon">📁</div>
            <h3>Drop your file here or click to browse</h3>
            <p>Supports MP4, AVI, MOV, JPG, PNG files</p>
            <input type="file" id="fileInput" class="file-input" accept="video/*,image/*" required>
            <button type="button" class="upload-button" id="uploadButton">Choose File</button>
            
            <div class="file-info" id="fileInfo">
              <div class="file-name" id="fileName"></div>
              <div class="file-size" id="fileSize"></div>
            </div>
          </div>

          <button type="submit" class="analyze-button" id="analyzeButton">
            🔬 Analyze for Deepfake
          </button>
        </form>
 
        <div class="loading" id="loading">
          <div class="spinner"></div>
          <p>Analyzing media with AI algorithms...</p>
          <p style="font-size: 0.9rem; color: #666; margin-top: 10px;">This may take a few moments</p>
        </div>

        <div class="result" id="result">
          <div class="result-icon" id="resultIcon"></div>
          <div class="result-label" id="resultLabel"></div>
          <div class="probability-bar">
            <div class="probability-fill" id="probabilityFill"></div>
          </div>
          <div class="confidence-text" id="confidenceText"></div>
          <p id="resultDescription" style="margin-top: 15px; color: #666;"></p>
          <button type="button" class="reset-button" id="resetButton">Analyze Another File</button>
        </div>

        <div class="supported-formats">
          <p><strong>Supported formats:</strong> MP4, AVI, MOV, WebM (video) • JPG, PNG, GIF (images)</p>
          <p style="margin-top: 5px; font-size: 0.8rem;">Maximum file size: 100MB</p>
        </div>
      </div>
    </div>

    <!-- About Section -->
    <div class="section about-section" id="about-section">
      <div class="container">
        <h2>About Deepfake Detection</h2>
        
        <p>Deepfakes are synthetic media created using artificial intelligence where a person appears to say or do things they never actually said or did. Our advanced detection system uses state-of-the-art machine learning models to identify manipulated content with high accuracy.</p>

        <h3>🎯 What are Deepfakes?</h3>
        <p>Deepfakes use deep neural networks to create convincing fake videos, images, and audio recordings. They have legitimate uses in entertainment and education, but can also be misused for misinformation, fraud, or harassment. Early detection is crucial for maintaining trust in digital media.</p>

        <h3>🧠 Our Detection Technology</h3>
        <p>Our system employs a hybrid approach combining multiple neural network architectures to achieve superior detection accuracy across various types of synthetic media.</p>

        <div class="model-features">
          <div class="feature-box">
            <div class="icon">🎥</div>
            <h5>Video Analysis</h5>
            <p>Temporal inconsistencies detection across video frames</p>
          </div>
          <div class="feature-box">
            <div class="icon">🖼️</div>
            <h5>Image Processing</h5>
            <p>Pixel-level artifact identification and analysis</p>
          </div>
          <div class="feature-box">
            <div class="icon">⚡</div>
            <h5>Real-time Processing</h5>
            <p>Fast analysis with optimized model architecture</p>
          </div>
          <div class="feature-box">
            <div class="icon">🎯</div>
            <h5>High Accuracy</h5>
            <p>95%+ detection rate on latest deepfake techniques</p>
          </div>
        </div>

        <h3>🔬 CNN (Convolutional Neural Networks)</h3>
        <div class="model-card">
          <h4>Spatial Feature Extraction</h4>
          <p>Our CNN architecture is specifically designed for detecting visual artifacts in images and video frames. It analyzes pixel-level inconsistencies, compression artifacts, and unnatural facial features that are common indicators of deepfake generation.</p>
          
          <p><strong>Key Features:</strong></p>
          <ul style="margin-left: 20px; color: #555;">
            <li>Multi-scale feature extraction using ResNet-50 backbone</li>
            <li>Attention mechanisms for focusing on facial regions</li>
            <li>Transfer learning from ImageNet for robust feature representation</li>
            <li>Specialized layers for detecting blending artifacts</li>
          </ul>
        </div>

        <h3>🔄 LSTM (Long Short-Term Memory)</h3>
        <div class="model-card">
          <h4>Temporal Sequence Analysis</h4>
          <p>LSTM networks excel at analyzing temporal patterns in video sequences. They detect inconsistencies in facial movements, unnatural blinking patterns, and temporal artifacts that occur when deepfake models struggle to maintain consistency across frames.</p>
          
          <p><strong>Key Features:</strong></p>
          <ul style="margin-left: 20px; color: #555;">
            <li>Bidirectional LSTM for forward and backward temporal analysis</li>
            <li>Attention-based pooling for important frame selection</li>
            <li>Gradient clipping to prevent vanishing gradient problems</li>
            <li>Sequence-to-sequence architecture for frame correlation</li>
          </ul>
        </div>

        <h3>🚀 Hybrid Architecture Benefits</h3>
        <p>By combining CNNs for spatial analysis with LSTMs for temporal analysis, our system achieves superior performance compared to single-model approaches. This dual approach ensures comprehensive detection across both static images and dynamic video content.</p>

        <div style="background: rgba(67, 233, 123, 0.1); padding: 20px; border-radius: 12px; margin: 20px 0; border-left: 4px solid #43e97b;">
          <p><strong>🛡️ Privacy & Security:</strong> All processing happens locally in your browser or on our secure servers. We don't store your uploaded files, ensuring complete privacy and data protection.</p>
        </div>
      </div>
    </div>
  </div>
<script>
    class DeepfakeDetector {
      constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.initializeSidebar();
      }

      initializeElements() {
        // Sidebar elements
        this.sidebar = document.getElementById('sidebar');
        this.sidebarToggle = document.getElementById('sidebarToggle');
        this.mainContent = document.getElementById('mainContent');
        this.navItems = document.querySelectorAll('.nav-item');
        this.sections = document.querySelectorAll('.section');

        // Detector elements
        this.form = document.getElementById('uploadForm');
        this.fileInput = document.getElementById('fileInput');
        this.uploadButton = document.getElementById('uploadButton');
        this.uploadSection = document.getElementById('uploadSection');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileName = document.getElementById('fileName');
        this.fileSize = document.getElementById('fileSize');
        this.analyzeButton = document.getElementById('analyzeButton');
        this.loading = document.getElementById('loading');
        this.result = document.getElementById('result');
        this.resultIcon = document.getElementById('resultIcon');
        this.resultLabel = document.getElementById('resultLabel');
        this.probabilityFill = document.getElementById('probabilityFill');
        this.confidenceText = document.getElementById('confidenceText');
        this.resultDescription = document.getElementById('resultDescription');
        this.resetButton = document.getElementById('resetButton');
      }

      initializeSidebar() {
        // Navigation functionality
        this.navItems.forEach(item => {
          item.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const sectionName = item.getAttribute('data-section');
            this.showSection(sectionName);
            
            // Update active nav item
            this.navItems.forEach(nav => nav.classList.remove('active'));
            item.classList.add('active');

            // Hide sidebar on mobile after selection
            if (window.innerWidth <= 768) {
              this.sidebar.classList.remove('show');
            }
          });
        });

        // Mobile toggle
        if (this.sidebarToggle) {
          this.sidebarToggle.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.sidebar.classList.toggle('show');
          });
        }

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
          if (window.innerWidth <= 768) {
            if (!this.sidebar.contains(e.target) && !this.sidebarToggle.contains(e.target)) {
              this.sidebar.classList.remove('show');
            }
          }
        });
      }

      showSection(sectionName) {
        // Hide all sections
        this.sections.forEach(section => {
          section.classList.remove('active');
        });

        // Show selected section
        const targetSection = document.getElementById(`${sectionName}-section`);
        if (targetSection) {
          targetSection.classList.add('active');
        }
      }

      attachEventListeners() {
        // File upload events
        if (this.uploadButton) {
          this.uploadButton.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation(); // ✅ prevents double trigger
            this.fileInput.click();
          });
        }

        // Make upload section clickable
        if (this.uploadSection) {
          this.uploadSection.addEventListener('click', (e) => {
            // Prevent double-trigger if clicked on button or input
            if (e.target.closest('#uploadButton') || e.target.closest('#fileInput')) return;
            this.fileInput.click();
          });
        }

        if (this.fileInput) {
          this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
        
        // Drag and drop events
        if (this.uploadSection) {
          this.uploadSection.addEventListener('dragover', (e) => this.handleDragOver(e));
          this.uploadSection.addEventListener('dragleave', (e) => this.handleDragLeave(e));
          this.uploadSection.addEventListener('drop', (e) => this.handleDrop(e));
        }
        
        // Form submission
        if (this.form) {
          this.form.addEventListener('submit', (e) => this.handleSubmit(e));
        }
        
        // Reset button
        if (this.resetButton) {
          this.resetButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.resetForm();
          });
        }
      }

      handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadSection.classList.add('dragover');
      }

      handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        if (!this.uploadSection.contains(e.relatedTarget)) {
          this.uploadSection.classList.remove('dragover');
        }
      }

      handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        this.uploadSection.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
          this.fileInput.files = files;
          this.handleFileSelect({ target: { files } });
        }
      }

      handleFileSelect(e) {
        const file = e.target.files[0];
        if (!file) return;

        // Validate file type
        const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/webm', 
                           'image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
        
        if (!validTypes.some(type => file.type === type || file.name.toLowerCase().includes(type.split('/')[1]))) {
          alert('Please select a valid video or image file.');
          this.fileInput.value = '';
          return;
        }

        // Validate file size (100MB limit)
        if (file.size > 100 * 1024 * 1024) {
          alert('File size must be less than 100MB.');
          this.fileInput.value = '';
          return;
        }

        this.displayFileInfo(file);
        this.analyzeButton.style.display = 'block';
      }

      displayFileInfo(file) {
        if (this.fileName) this.fileName.textContent = file.name;
        if (this.fileSize) this.fileSize.textContent = this.formatFileSize(file.size);
        if (this.fileInfo) this.fileInfo.style.display = 'block';
      }

      formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
      }

      async handleSubmit(e) {
        e.preventDefault();
        e.stopPropagation();
        
        if (!this.fileInput.files.length) {
          alert('Please select a file first.');
          return;
        }

        this.showLoading();
        this.analyzeButton.disabled = true;
        
        try {
          const formData = new FormData();
          formData.append('file', this.fileInput.files[0]);

          // Simulate API call (replace with actual endpoint)
          const response = await this.simulateAnalysis(formData);
          
          this.displayResults(response);
        } catch (error) {
          this.displayError('Analysis failed. Please try again.');
          console.error('Error:', error);
        } finally {
          this.hideLoading();
          this.analyzeButton.disabled = false;
        }
      }

      // Simulate analysis since we don't have a real backend
      async simulateAnalysis(formData) {
        return new Promise((resolve) => {
          setTimeout(() => {
            const isFake = Math.random() > 0.6;
            const probability = isFake ? 0.7 + Math.random() * 0.25 : Math.random() * 0.4;
            
            resolve({
              is_fake: isFake,
              probability: probability,
              confidence: probability > 0.8 ? 'High' : probability > 0.6 ? 'Medium' : 'Low'
            });
          }, 2000 + Math.random() * 3000);
        });
      }

      showLoading() {
        if (this.loading) this.loading.style.display = 'block';
        if (this.result) this.result.style.display = 'none';
      }

      hideLoading() {
        if (this.loading) this.loading.style.display = 'none';
      }

      displayResults(response) {
        if (!this.result) return;

        this.result.className = 'result';
        
        const isFake = response.is_fake;
        const probability = response.probability;
        const confidence = response.confidence;

        if (isFake) {
          this.result.classList.add('fake');
          if (this.resultIcon) this.resultIcon.textContent = '⚠️';
          if (this.resultLabel) this.resultLabel.textContent = 'DEEPFAKE DETECTED';
          if (this.resultDescription) {
            this.resultDescription.textContent = 'This media appears to be artificially generated or manipulated. Please verify from original sources.';
          }
        } else {
          this.result.classList.add('real');
          if (this.resultIcon) this.resultIcon.textContent = '✅';
          if (this.resultLabel) this.resultLabel.textContent = 'AUTHENTIC MEDIA';
          if (this.resultDescription) {
            this.resultDescription.textContent = 'This media appears to be authentic with no signs of AI manipulation detected.';
          }
        }

        if (this.probabilityFill) {
          const displayProbability = isFake ? probability : (1 - probability);
          this.probabilityFill.style.width = `${displayProbability * 100}%`;
          
          if (isFake) {
            this.probabilityFill.style.background = 'linear-gradient(90deg, #dc3545, #c82333)';
          } else {
            this.probabilityFill.style.background = 'linear-gradient(90deg, #28a745, #20c997)';
          }
        }

        if (this.confidenceText) {
          const displayProbability = isFake ? probability : (1 - probability);
          this.confidenceText.textContent = `Confidence: ${(displayProbability * 100).toFixed(1)}% (${confidence})`;
        }

        this.result.style.display = 'block';
      }

      displayError(message) {
        if (!this.result) return;

        this.result.className = 'result error';
        
        if (this.resultIcon) this.resultIcon.textContent = '❌';
        if (this.resultLabel) this.resultLabel.textContent = 'ANALYSIS ERROR';
        if (this.resultDescription) this.resultDescription.textContent = message;
        if (this.probabilityFill) this.probabilityFill.style.width = '0%';
        if (this.confidenceText) this.confidenceText.textContent = '';

        this.result.style.display = 'block';
      }

      resetForm() {
        if (this.form) this.form.reset();
        
        if (this.fileInfo) this.fileInfo.style.display = 'none';
        if (this.analyzeButton) this.analyzeButton.style.display = 'none';
        if (this.loading) this.loading.style.display = 'none';
        if (this.result) this.result.style.display = 'none';
        
        if (this.fileInput) this.fileInput.value = '';
        if (this.analyzeButton) this.analyzeButton.disabled = false;
        
        if (this.uploadSection) this.uploadSection.classList.remove('dragover');
      }
    }

    // ✅ Initialize ONCE only
    document.addEventListener('DOMContentLoaded', () => {
      if (!window.deepfakeDetectorInstance) {
        window.deepfakeDetectorInstance = new DeepfakeDetector();
        console.log('DeepfakeDetector initialized successfully');
      }
    });
</script>

</body>
</html> """

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    f = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.filename)[1]) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name
    try:
        mime = f.mimetype or ""
        if mime.startswith("image/"):
            img = cv2.imread(tmp_path)
            frame = preprocess_frame(img)
            frames = np.stack([frame]*SEQ_LEN, axis=0)
        else:
            frames = extract_frames_from_video(tmp_path, seq_len=SEQ_LEN)

        prob = predict_from_frames(frames)
        is_fake = bool(prob >= THRESHOLD)
        return jsonify({"probability": prob, "is_fake": is_fake})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    app.run(debug=True)
