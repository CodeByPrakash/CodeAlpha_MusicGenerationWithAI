"""
Step 3: LSTM Model Architecture
================================
Defines a 2-layer LSTM model with dropout for music sequence prediction.
The model learns to predict the next note/chord given a sequence of previous ones.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_model(sequence_length, n_vocab, lstm_units=256):
    """
    Build and compile the LSTM model for music generation.

    Architecture:
        Input → LSTM(256) → Dropout(0.3) → LSTM(256) → Dropout(0.3)
              → Dense(256, ReLU) → Dense(n_vocab, Softmax)

    Args:
        sequence_length: Number of time steps in each input sequence.
        n_vocab: Number of unique notes/chords (output classes).
        lstm_units: Number of units in each LSTM layer.

    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        LSTM(
            lstm_units,
            input_shape=(sequence_length, 1),
            return_sequences=True,
            name='lstm_1'
        ),
        Dropout(0.3, name='dropout_1'),

        LSTM(
            lstm_units,
            return_sequences=False,
            name='lstm_2'
        ),
        Dropout(0.3, name='dropout_2'),

        Dense(lstm_units, activation='relu', name='dense_hidden'),
        Dense(n_vocab, activation='softmax', name='output'),
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    # Quick test: build a model with dummy dimensions and print summary
    test_model = build_model(sequence_length=100, n_vocab=300)
    test_model.summary()
