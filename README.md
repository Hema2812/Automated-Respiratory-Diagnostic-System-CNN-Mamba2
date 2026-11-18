# Automated-Respiratory-Diagnostic-System-CNN-Mamba2# Automated Respiratory Diagnostic System (Hybrid CNN + Mamba-2)

An AI-based diagnostic framework that classifies pulmonary conditions (**COPD, Pneumonia, Bronchiolitis, and Healthy**) from lung sound recordings using a hybrid deep learning and multi-modal fusion architecture.

## Key Architectural Modules (`main.py`)
1. **Signal Enhancement (TKEO):** Applies the **Teager-Kaiser Energy Operator** to highlight transient acoustic events and normalize respiratory cycles.
2. **Dual-Channel Spectrogram Generation:** Generates a 2-channel feature map (`44.1 kHz`, `128 Mel bands`) stacking standard Mel-Spectrograms with TKEO-enhanced Mel-Spectrograms.
3. **CNN Spatial Feature Extraction:** Uses a TensorFlow/Keras Conv2D + BatchNormalization + GlobalAveragePooling2D pipeline to extract spatial representations.
4. **Mamba-2 Sequence Modeling:** Implements a Selective State Space Model (SSM) recurrence scan to capture long-range temporal dependencies across breathing cycles.
5. **Multi-Modal Metadata Fusion:** Fuses acoustic embeddings with patient clinical metadata before final softmax classification.
6. **Evaluation:** Trained with Adam (`lr=0.001`), EarlyStopping, and evaluated using Accuracy, Precision, Recall, and Confusion Matrix heatmaps.

## Tech Stack
* **Deep Learning:** TensorFlow, Keras, Selective State Space Models (Mamba-2)
* **Audio Processing:** Librosa, NumPy, Teager-Kaiser Energy Operator (TKEO)
* **Evaluation & Visualization:** Scikit-learn, Matplotlib, Seaborn

## Setup & Usage
```bash
pip install -r requirements.txt
```
