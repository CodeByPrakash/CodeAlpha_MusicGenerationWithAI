# 🎵 Music Generation with AI (LSTM)

An AI-powered music composition tool that uses a deep learning LSTM model to learn patterns from classical music and generate new original compositions as MIDI files.

## Internship Details

| Detail        | Info                         |
|---------------|------------------------------|
| **Organization** | CodeAlpha                 |
| **Role**         | AI Engineer Intern        |
| **Duration**     | 1 Month                   |
| **Project**      | Task 3 — Music Generation with AI |

---

## Features

- **Self-Contained Dataset** — Automatically extracts ~50 Bach chorale MIDI files from `music21`'s built-in corpus. No manual downloads required.
- **Intelligent Preprocessing** — Parses notes and chords, builds a vocabulary, and creates windowed training sequences with one-hot encoding.
- **2-Layer LSTM Architecture** — A stacked LSTM network (256 units per layer) with dropout regularization learns long-range musical patterns.
- **Temperature-Based Sampling** — Controls the creativity of generated music (lower = predictable, higher = experimental).
- **MIDI Output** — Generated compositions are saved as `.mid` files, playable in any MIDI player or DAW.
- **Training Visualization** — Loss and accuracy plots are automatically saved after training.
- **One-Click Pipeline** — A single `main.py` script runs everything from data prep to music generation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LSTM Music Model                      │
├─────────────────────────────────────────────────────────┤
│  Input (100 notes)                                       │
│       ↓                                                  │
│  LSTM Layer 1 (256 units, return_sequences=True)         │
│       ↓                                                  │
│  Dropout (0.3)                                           │
│       ↓                                                  │
│  LSTM Layer 2 (256 units)                                │
│       ↓                                                  │
│  Dropout (0.3)                                           │
│       ↓                                                  │
│  Dense (256, ReLU)                                       │
│       ↓                                                  │
│  Dense (n_vocab, Softmax) → Predicted Next Note          │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

- **Language:** Python 3.10+
- **Deep Learning:** TensorFlow / Keras
- **Music Processing:** music21
- **Data:** NumPy, Pickle
- **Visualization:** Matplotlib

---

## Project Structure

```
CodeAlpha_MusicGenerationWithAI/
├── .venv/               # Virtual environment
├── data/
│   ├── midi/            # Extracted MIDI files from music21 corpus
│   ├── notes.pkl        # Extracted note tokens
│   ├── X_train.pkl      # Training input sequences
│   ├── y_train.pkl      # Training target labels
│   ├── note_to_int.pkl  # Note → integer mapping
│   └── int_to_note.pkl  # Integer → note mapping
├── models/
│   ├── best_weights.keras   # Best checkpoint during training
│   └── final_weights.keras  # Final model after training
├── output/
│   ├── generated_music.mid  # Generated MIDI composition
│   └── training_loss.png    # Training loss/accuracy plot
├── prepare_data.py      # Step 1: Extract MIDI from corpus
├── preprocess.py        # Step 2: Parse & tokenize notes
├── model.py             # Step 3: LSTM model definition
├── train.py             # Step 4: Model training pipeline
├── generate.py          # Step 5: Music generation
├── main.py              # One-click full pipeline
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher

### 2. Create Virtual Environment
```powershell
cd CodeAlpha_MusicGenerationWithAI
python -m venv .venv
```

### 3. Activate Environment
```powershell
# PowerShell
.venv\Scripts\Activate.ps1

# CMD
.venv\Scripts\activate.bat
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Usage

### Full Pipeline (Recommended)
Run everything from data preparation to music generation in one command:
```powershell
python main.py --epochs 20
```

### Generate Only (after training)
Skip training and generate from previously saved weights:
```powershell
python main.py --generate-only --temperature 0.8 --notes 300
```

### Run Individual Steps
```powershell
python prepare_data.py        # Step 1: Extract MIDI data
python preprocess.py          # Step 2: Preprocess into sequences
python train.py --epochs 20   # Step 3-4: Train the model
python generate.py            # Step 5: Generate music
```

### CLI Arguments
| Argument          | Default | Description                                |
|-------------------|---------|--------------------------------------------|
| `--epochs`        | 20      | Number of training epochs                  |
| `--batch-size`    | 64      | Training batch size                        |
| `--generate-only` | False   | Skip training, use saved weights           |
| `--notes`         | 200     | Number of notes to generate                |
| `--temperature`   | 0.8     | Creativity control (0.2 = safe, 1.5 = wild)|

---

## Output

After running the pipeline:
- **`output/generated_music.mid`** — Open in any MIDI player (e.g., Windows Media Player, VLC, or a DAW like FL Studio / GarageBand).
- **`output/training_loss.png`** — Visual chart of model convergence during training.
