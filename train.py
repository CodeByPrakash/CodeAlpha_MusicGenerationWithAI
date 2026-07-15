"""
Step 4: Train the LSTM Model
=============================
Loads preprocessed data, builds the model, trains it, and saves:
  - Best model weights to models/
  - Training loss plot to output/
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from model import build_model
from preprocess import SEQUENCE_LENGTH

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def load_training_data():
    """Load preprocessed training arrays from disk."""
    with open(os.path.join(DATA_DIR, 'X_train.pkl'), 'rb') as f:
        X = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'y_train.pkl'), 'rb') as f:
        y = pickle.load(f)
    return X, y


def train(epochs=20, batch_size=64):
    """
    Train the LSTM model on preprocessed music data.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
    """
    # Load data
    print("[train] Loading preprocessed data...")
    X, y = load_training_data()
    n_vocab = y.shape[1]

    print(f"[train] Training samples: {X.shape[0]}")
    print(f"[train] Vocabulary size:  {n_vocab}")
    print(f"[train] Sequence length:  {SEQUENCE_LENGTH}")

    # Build model
    model = build_model(sequence_length=SEQUENCE_LENGTH, n_vocab=n_vocab)
    model.summary()

    # Prepare output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Callbacks
    weights_path = os.path.join(MODEL_DIR, 'best_weights.keras')
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Train
    print(f"\n[train] Starting training for {epochs} epochs...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    # Save final weights as well
    final_path = os.path.join(MODEL_DIR, 'final_weights.keras')
    model.save(final_path)
    print(f"[train] Final model saved to {final_path}")

    # Plot training loss
    plot_training_loss(history)

    return model, history


def plot_training_loss(history):
    """Save a training loss curve to output/training_loss.png."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], linewidth=2, color='#6366f1', label='Training Loss')
    if 'accuracy' in history.history:
        ax2 = plt.twinx()
        ax2.plot(history.history['accuracy'], linewidth=2, color='#10b981', linestyle='--', label='Accuracy')
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(loc='upper right')

    plt.title('LSTM Music Model — Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, 'training_loss.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[train] Training loss plot saved to {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the LSTM music model.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)
