"""
Step 1: Prepare MIDI Data
=========================
Extracts classical MIDI files from music21's built-in corpus (Bach chorales)
and saves them into the data/midi/ directory.
No external downloads needed — everything comes bundled with music21.
"""

import os
from music21 import corpus

# Directory to save extracted MIDI files
MIDI_DIR = os.path.join(os.path.dirname(__file__), 'data', 'midi')


def prepare_data(max_files=50):
    """
    Extract Bach chorale MIDI files from the music21 corpus.

    Args:
        max_files: Maximum number of MIDI files to extract.

    Returns:
        List of saved file paths.
    """
    os.makedirs(MIDI_DIR, exist_ok=True)

    # Get paths to Bach chorale files from the built-in corpus
    bach_paths = corpus.getComposer('bach')
    
    saved_files = []
    count = 0

    print(f"[prepare_data] Found {len(bach_paths)} Bach pieces in music21 corpus.")
    print(f"[prepare_data] Extracting up to {max_files} files...\n")

    for path in bach_paths:
        if count >= max_files:
            break

        try:
            # Parse the score from the corpus
            score = corpus.parse(path)

            # Create a safe filename
            filename = f"bach_{count:03d}.mid"
            filepath = os.path.join(MIDI_DIR, filename)

            # Write to MIDI
            score.write('midi', fp=filepath)
            saved_files.append(filepath)
            count += 1

            if count % 10 == 0:
                print(f"  Extracted {count} files...")

        except Exception as e:
            print(f"  Skipping {path}: {e}")
            continue

    print(f"\n[prepare_data] Done! Saved {len(saved_files)} MIDI files to {MIDI_DIR}")
    return saved_files


if __name__ == '__main__':
    prepare_data()
