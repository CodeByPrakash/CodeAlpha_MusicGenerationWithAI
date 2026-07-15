"""
Music Generation with AI — Unified Pipeline
=============================================
One-click script that runs the full pipeline:
  1. Prepare data (extract MIDI from music21 corpus)
  2. Preprocess (parse notes, build sequences)
  3. Train the LSTM model
  4. Generate new music

Usage:
    python main.py                     # Full pipeline (default 20 epochs)
    python main.py --epochs 50         # Full pipeline with 50 epochs
    python main.py --generate-only     # Skip training, generate from saved weights
"""

import argparse
import time


def run_pipeline(epochs=20, batch_size=64, generate_only=False, num_notes=200, temperature=0.8):
    """
    Execute the full music generation pipeline.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        generate_only: If True, skip data prep + training and go straight to generation.
        num_notes: Number of notes to generate.
        temperature: Sampling temperature for generation.
    """
    total_start = time.time()

    if not generate_only:
        # ── Step 1: Prepare MIDI Data ──────────────────────────────────
        print("=" * 60)
        print("  STEP 1/5: Preparing MIDI Data")
        print("=" * 60)
        from prepare_data import prepare_data
        prepare_data(max_files=50)

        # ── Step 2: Preprocess Data ────────────────────────────────────
        print("\n" + "=" * 60)
        print("  STEP 2/5: Preprocessing Data")
        print("=" * 60)
        from preprocess import preprocess_and_save
        preprocess_and_save()

        # ── Step 3 & 4: Build & Train Model ────────────────────────────
        print("\n" + "=" * 60)
        print("  STEP 3-4/5: Building & Training LSTM Model")
        print("=" * 60)
        from train import train
        train(epochs=epochs, batch_size=batch_size)

    # ── Step 5: Generate Music ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5/5: Generating New Music")
    print("=" * 60)
    from generate import generate_music
    output_path = generate_music(num_notes=num_notes, temperature=temperature)

    # ── Done ───────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE in {minutes}m {seconds}s")
    print(f"  Generated MIDI: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AI Music Generation Pipeline — CodeAlpha Internship Task 3'
    )
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--generate-only', action='store_true',
                        help='Skip data prep & training, generate from saved weights')
    parser.add_argument('--notes', type=int, default=200,
                        help='Number of notes to generate (default: 200)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    args = parser.parse_args()

    run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        generate_only=args.generate_only,
        num_notes=args.notes,
        temperature=args.temperature
    )
