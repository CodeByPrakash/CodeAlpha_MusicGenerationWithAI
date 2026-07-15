"""
Step 5: Generate New Music
===========================
Loads trained model weights and vocabulary mappings, then generates
a new musical sequence using temperature-based sampling.
Outputs a MIDI file to output/generated_music.mid.
"""

import os
import pickle
import argparse
import numpy as np
from music21 import instrument, note, chord, stream

from model import build_model
from preprocess import SEQUENCE_LENGTH

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def sample_with_temperature(predictions, temperature=1.0):
    """
    Sample an index from the prediction probability array using temperature scaling.

    Lower temperature → more conservative (picks the most likely note).
    Higher temperature → more creative/random.

    Args:
        predictions: Probability distribution over vocabulary.
        temperature: Sampling temperature (default 1.0).

    Returns:
        Sampled index (int).
    """
    predictions = np.asarray(predictions).astype('float64')
    # Apply temperature
    log_preds = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(log_preds)
    probabilities = exp_preds / np.sum(exp_preds)
    # Sample from the distribution
    return np.random.choice(len(probabilities), p=probabilities)


def generate_music(num_notes=200, temperature=0.8, output_file=None):
    """
    Generate a new musical composition using the trained LSTM model.

    Args:
        num_notes: Number of notes/chords to generate.
        temperature: Sampling temperature for variety.
        output_file: Path for the output MIDI file.

    Returns:
        Path to the generated MIDI file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, 'generated_music.mid')

    # Load vocabulary mappings
    print("[generate] Loading vocabulary mappings...")
    with open(os.path.join(DATA_DIR, 'notes.pkl'), 'rb') as f:
        notes = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'int_to_note.pkl'), 'rb') as f:
        int_to_note = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'note_to_int.pkl'), 'rb') as f:
        note_to_int = pickle.load(f)

    n_vocab = len(set(notes))

    # Build model and load weights
    print("[generate] Building model and loading weights...")
    model = build_model(sequence_length=SEQUENCE_LENGTH, n_vocab=n_vocab)

    # Try best weights first, fall back to final weights
    weights_path = os.path.join(MODEL_DIR, 'best_weights.keras')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(MODEL_DIR, 'final_weights.keras')

    model.load_weights(weights_path)
    print(f"[generate] Loaded weights from {weights_path}")

    # Pick a random starting seed sequence from the original notes
    start_idx = np.random.randint(0, len(notes) - SEQUENCE_LENGTH)
    seed_sequence = notes[start_idx:start_idx + SEQUENCE_LENGTH]
    pattern = [note_to_int[n] for n in seed_sequence]

    print(f"[generate] Generating {num_notes} notes with temperature={temperature}...")

    generated_notes = []

    for i in range(num_notes):
        # Prepare input
        input_seq = np.reshape(pattern, (1, len(pattern), 1))
        input_seq = input_seq / float(n_vocab)

        # Predict
        prediction = model.predict(input_seq, verbose=0)[0]

        # Sample using temperature
        idx = sample_with_temperature(prediction, temperature)

        result_note = int_to_note[idx]
        generated_notes.append(result_note)

        # Slide the window forward
        pattern.append(idx)
        pattern = pattern[1:]

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_notes} notes...")

    # Convert generated note tokens into a music21 stream
    print("[generate] Converting to MIDI...")
    output_stream = create_midi_stream(generated_notes)

    # Write to MIDI file
    output_stream.write('midi', fp=output_file)
    print(f"\n[generate] Done! MIDI file saved to: {output_file}")

    return output_file


def create_midi_stream(generated_notes):
    """
    Convert a list of note/chord string tokens into a music21 Stream.

    Args:
        generated_notes: List of note strings like 'C4' or chord strings like 'C4.E4.G4'.

    Returns:
        music21.stream.Stream object.
    """
    output_notes = []
    offset = 0

    for token in generated_notes:
        try:
            if '.' in token:
                # It's a chord
                pitches = token.split('.')
                chord_notes = [note.Note(p) for p in pitches]
                new_chord = chord.Chord(chord_notes)
                new_chord.offset = offset
                new_chord.storedInstrument = instrument.Piano()
                output_notes.append(new_chord)
            else:
                # It's a single note
                new_note = note.Note(token)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
        except Exception:
            # Skip any tokens that can't be parsed
            pass

        offset += 0.5  # Each note/chord is half a beat apart

    midi_stream = stream.Stream(output_notes)
    return midi_stream


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate music from a trained LSTM model.')
    parser.add_argument('--notes', type=int, default=200, help='Number of notes to generate (default: 200)')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (default: 0.8)')
    parser.add_argument('--output', type=str, default=None, help='Output MIDI file path')
    args = parser.parse_args()

    generate_music(num_notes=args.notes, temperature=args.temperature, output_file=args.output)
