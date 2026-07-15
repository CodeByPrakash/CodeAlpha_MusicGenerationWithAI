"""
Step 2: Preprocess MIDI Data
============================
Parses all MIDI files into note/chord sequences, builds a vocabulary mapping,
and creates fixed-length input/output training arrays for the LSTM model.
Saves everything as pickle files for fast reloading.
"""

import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

# Paths
BASE_DIR = os.path.dirname(__file__)
MIDI_DIR = os.path.join(BASE_DIR, 'data', 'midi')
DATA_DIR = os.path.join(BASE_DIR, 'data')

SEQUENCE_LENGTH = 100  # Number of previous notes the model sees


def extract_notes(midi_dir=MIDI_DIR):
    """
    Parse every MIDI file and extract a flat list of note/chord string tokens.

    Returns:
        List[str]: e.g. ['C4', 'E4.G4.B4', 'D5', ...]
    """
    all_notes = []
    midi_files = glob.glob(os.path.join(midi_dir, '*.mid'))

    print(f"[preprocess] Parsing {len(midi_files)} MIDI files...")

    for i, filepath in enumerate(midi_files):
        try:
            midi = converter.parse(filepath)
            parts = instrument.partitionByInstrument(midi)

            # Use the first instrument part, or fall back to the flat stream
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    all_notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    # Represent chords as dot-separated pitch strings
                    all_notes.append('.'.join(str(p) for p in element.pitches))

        except Exception as e:
            print(f"  Skipping {filepath}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  Parsed {i + 1}/{len(midi_files)} files...")

    print(f"[preprocess] Extracted {len(all_notes)} total note tokens.")
    return all_notes


def create_sequences(notes, sequence_length=SEQUENCE_LENGTH):
    """
    Build training input/output arrays from the note list.

    Args:
        notes: List of note/chord string tokens.
        sequence_length: How many past notes form one input sample.

    Returns:
        X: np.ndarray of shape (n_samples, sequence_length, 1), normalized.
        y: np.ndarray of shape (n_samples, n_vocab), one-hot encoded.
        note_to_int: dict mapping note strings → integers.
        int_to_note: dict mapping integers → note strings.
    """
    # Build vocabulary
    unique_notes = sorted(set(notes))
    n_vocab = len(unique_notes)

    note_to_int = {n: i for i, n in enumerate(unique_notes)}
    int_to_note = {i: n for i, n in enumerate(unique_notes)}

    print(f"[preprocess] Vocabulary size: {n_vocab} unique notes/chords.")

    # Create input-output pairs
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_samples = len(network_input)
    print(f"[preprocess] Created {n_samples} training sequences.")

    # Reshape and normalize input for LSTM: (samples, sequence_length, 1)
    X = np.reshape(network_input, (n_samples, sequence_length, 1))
    X = X / float(n_vocab)  # Normalize to [0, 1]

    # One-hot encode the output
    y = np.zeros((n_samples, n_vocab), dtype=np.float32)
    for i, idx in enumerate(network_output):
        y[i, idx] = 1.0

    return X, y, note_to_int, int_to_note


def preprocess_and_save():
    """
    Full preprocessing pipeline: extract notes → build sequences → save to disk.
    """
    notes = extract_notes()

    if len(notes) < SEQUENCE_LENGTH + 1:
        raise ValueError(
            f"Not enough notes ({len(notes)}) to create sequences. "
            f"Need at least {SEQUENCE_LENGTH + 1}. Add more MIDI files."
        )

    X, y, note_to_int, int_to_note = create_sequences(notes)

    # Save everything
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, 'notes.pkl'), 'wb') as f:
        pickle.dump(notes, f)

    with open(os.path.join(DATA_DIR, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X, f)

    with open(os.path.join(DATA_DIR, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y, f)

    with open(os.path.join(DATA_DIR, 'note_to_int.pkl'), 'wb') as f:
        pickle.dump(note_to_int, f)

    with open(os.path.join(DATA_DIR, 'int_to_note.pkl'), 'wb') as f:
        pickle.dump(int_to_note, f)

    print(f"\n[preprocess] Saved all data to {DATA_DIR}/")
    print(f"  X_train shape: {X.shape}")
    print(f"  y_train shape: {y.shape}")

    return X, y, note_to_int, int_to_note


if __name__ == '__main__':
    preprocess_and_save()
