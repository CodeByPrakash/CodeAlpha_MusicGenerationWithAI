# Music Generation with AI — Implementation Plan

Build an LSTM-based deep learning model that learns patterns from classical MIDI music and generates new original compositions, using Python, TensorFlow/Keras, and music21.

## User Review Required

> [!IMPORTANT]
> **MIDI Dataset**: We will programmatically generate a small curated classical piano MIDI dataset using `music21`'s built-in corpus (Bach chorales, etc.). This avoids manual downloads and keeps the project self-contained and reproducible. No external downloads are required.

> [!IMPORTANT]
> **GPU/Training Time**: Training an LSTM on CPU will be slow. The model will be configured for a lightweight architecture (256-unit LSTM, ~20 epochs) suitable for CPU training in ~5–15 minutes. Results will be musically interesting but not production-grade. The user can increase epochs/layers for better results if they have a GPU.

> [!WARNING]
> **Dependencies**: This project requires `tensorflow`, `music21`, `numpy`, and `matplotlib`. TensorFlow can be a large download (~400MB+). We will install these in a local `.venv`.

## Proposed Changes

### Project Structure

```
CodeAlpha_MusicGenerationWithAI/
├── .venv/                  # Virtual environment
├── data/
│   └── midi/               # MIDI files extracted from music21 corpus
├── output/                 # Generated MIDI output files
├── models/                 # Saved trained model weights
├── prepare_data.py         # Step 1: Extract MIDI from music21 corpus
├── preprocess.py           # Step 2: Parse MIDI → note sequences → training arrays
├── model.py                # Step 3: Define LSTM model architecture
├── train.py                # Step 4: Train the model
├── generate.py             # Step 5: Generate new music & save as MIDI
├── main.py                 # One-click pipeline: prep → preprocess → train → generate
├── README.md               # Documentation
└── requirements.txt        # Dependencies
```

---

### [Data Collection Component]

#### [NEW] prepare_data.py
- Uses `music21.corpus` to load ~30–50 Bach chorale MIDI files (bundled with music21, no download needed).
- Saves them as `.mid` files into `data/midi/`.

---

### [Preprocessing Component]

#### [NEW] preprocess.py
- Parses all MIDI files in `data/midi/` using `music21`.
- Extracts **notes** (e.g., `C4`) and **chords** (e.g., `C4.E4.G4`) from piano parts.
- Maps each unique note/chord to an integer index (vocabulary).
- Creates fixed-length input sequences (sliding window of 100 notes) and their corresponding next-note targets.
- Normalizes inputs to `[0, 1]` range and one-hot encodes targets.
- Saves processed arrays (`X_train`, `y_train`) and the vocabulary mappings as `.pkl` files.

---

### [Model Architecture Component]

#### [NEW] model.py
- Builds a Sequential Keras model:
  1. **LSTM layer** (256 units, `return_sequences=True`) — captures long-range temporal patterns.
  2. **Dropout** (0.3) — prevents overfitting.
  3. **LSTM layer** (256 units) — deeper pattern extraction.
  4. **Dropout** (0.3).
  5. **Dense** (256, ReLU) — non-linear feature mixing.
  6. **Dense** (n_vocab, Softmax) — probability distribution over all possible next notes.
- Loss: `categorical_crossentropy`, Optimizer: `Adam`.

---

### [Training Component]

#### [NEW] train.py
- Loads preprocessed data from `.pkl` files.
- Builds the model via `model.py`.
- Trains for **20 epochs** (configurable), batch size 64.
- Uses `ModelCheckpoint` callback to save the best weights to `models/`.
- Prints loss per epoch and saves a training loss plot to `output/training_loss.png`.

---

### [Generation Component]

#### [NEW] generate.py
- Loads the trained model weights and vocabulary mappings.
- Picks a random seed sequence from the training data.
- Iteratively predicts the next note 200 times using temperature-based sampling for variety.
- Converts the predicted integer sequence back to `music21` Note/Chord objects.
- Writes the result to `output/generated_music.mid`.
- Prints a confirmation message with the file path.

---

### [Pipeline Component]

#### [NEW] main.py
- A single entry-point script that runs all 5 steps sequentially:
  1. `prepare_data` → 2. `preprocess` → 3. `build model` → 4. `train` → 5. `generate`
- Includes argument parsing: `--epochs`, `--generate-only` (skip training, just generate from saved weights).

---

### [Documentation & Config]

#### [NEW] requirements.txt
```
tensorflow
music21
numpy
matplotlib
```

#### [NEW] README.md
- Project overview, internship context (CodeAlpha, AI Engineer, 1-month).
- Architecture diagram, feature list, installation, and usage instructions.

---

## Verification Plan

### Automated Tests
1. Run `python prepare_data.py` → verify MIDI files appear in `data/midi/`.
2. Run `python preprocess.py` → verify `.pkl` files are created and contain valid arrays.
3. Run `python train.py --epochs 2` → verify model trains without errors and weights are saved.
4. Run `python generate.py` → verify `output/generated_music.mid` is created and is a valid MIDI file.

### Manual Verification
- Open the generated `.mid` file in a MIDI player or import into a DAW to listen.
- Inspect the training loss plot for convergence.
