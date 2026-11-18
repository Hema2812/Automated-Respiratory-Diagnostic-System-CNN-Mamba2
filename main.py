import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Signal Enhancement (Teager-Kaiser Energy Operator - TKEO)
def tkeo_operator(signal):
    # Calculate Teager-Kaiser Energy: x[n]^2 - x[n-1]*x[n+1]
    tkeo = np.copy(signal)
    tkeo[1:-1] = signal[1:-1]**2 - signal[:-2] * signal[2:]
    
    # Handle boundaries and normalize
    tkeo[0], tkeo[-1] = tkeo[1], tkeo[-2]
    return (tkeo - np.min(tkeo)) / (np.max(tkeo) - np.min(tkeo))

# 2. Dual-Channel Spectrogram Generation
def generate_dual_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    
    # Channel 1: Standard Mel-Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    
    # Channel 2: TKEO-Enhanced Mel-Spectrogram
    y_tkeo = tkeo_operator(y)
    mel_tkeo = librosa.feature.melspectrogram(y=y_tkeo, sr=sr, n_mels=128)
    
    # Stack into a (128, 128, 2) feature map
    return np.stack([librosa.power_to_db(mel), librosa.power_to_db(mel_tkeo)], axis=-1)

# 3. CNN Feature Extraction Layer
def cnn_spatial_block(input_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D()
    ])
    return model

# 4. Mamba-2 Sequence Modeling (Selective State Space Layer)
def mamba_selective_scan(x, delta, A, B, C):
    # x: input sequence, delta: step size, A, B, C: state parameters
    # Computes linear-time state recurrence (Selective SSM)
    state = 0
    outputs = []
    for xt in x:
        state = A * state + B * xt
        yt = C * state
        outputs.append(yt)
    return np.array(outputs)

# 5. Multi-Modal Metadata Fusion & Classification
def build_hybrid_classifier(mamba_output, metadata_dim, num_classes):
    # Fusing CNN-Mamba features with Metadata
    acoustic_features = layers.Dense(128, activation='relu')(mamba_output)
    metadata_input = layers.Input(shape=(metadata_dim,))
    metadata_features = layers.Dense(16, activation='relu')(metadata_input)
    
    # Concatenate and classify
    fused = layers.Concatenate()([acoustic_features, metadata_features])
    output = layers.Dense(num_classes, activation='softmax')(fused)
    return output

# 6. Training and Optimization Setup
def compile_and_train(model, train_generator, val_generator):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
    )
    return history

# 7. Performance Visualization (Confusion Matrix)
def plot_confusion_matrix(model, test_data, test_labels, class_names):
    y_pred = model.predict(test_data)
    cm = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(y_pred, axis=1))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Pathology')
    plt.ylabel('Actual Label')
    plt.title('Respiratory Diagnosis Confusion Matrix')
    plt.show()
