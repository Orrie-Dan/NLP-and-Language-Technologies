# Research-Informed Sequential Models for Swahili Spoken-Word Classification

Formative Assignment 2 (NLP and Language Technologies). We compare five substantially different approaches for recognising 12 isolated Swahili spoken words from the Zindi Swahili Audio Classification dataset. Two are classical baselines and three are neural sequence models.

**Research question:** *How effectively can sequential modelling approaches recognise isolated Swahili spoken words, and what evidence supports the strengths and limitations of each approach?*

## Results at a glance

All models use the same shared split and are scored once on the same internal test split (630 clips) after their configuration was locked on validation. Every number below is read from a saved results file in this repository.

| Approach | Type | Test accuracy | Test macro-F1 | Macro ROC-AUC | Parameters | Results |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | Classical baseline | 0.381 | 0.380 | – | 78-D input | `results/classical/` |
| Linear SVM | Classical baseline | 0.395 | 0.386 | – | 78-D input | `results/classical/` |
| Temporal CNN | Neural sequence model | 0.821 | 0.820 | 0.975 | 25,548 | `results/person3_cnn_60epochs/` |
| Transformer encoder | Neural sequence model | 0.929 | 0.929 | 0.997 | 414,092 | `results/models/person4_model/` |
| **Bidirectional GRU** | Neural sequence model | **0.959** | **0.959** | **0.998** | 1,764,876 | `results/models/recurrent/` |

The BiGRU was also retrained with seeds 42, 43 and 44, which gave a test macro-F1 of 0.955 ± 0.006 (`results/models/recurrent/seed_robustness.csv`). The other rows are single-seed results.

The classical baselines summarise each clip as the mean and standard deviation of its MFCC frames, which discards the order of the sounds. All three neural models read the frames in order. That difference, roughly 0.39 against 0.82–0.96 macro-F1, is the main finding the report discusses.

## Dataset

Zindi Weekendz Swahili audio: 4,200 labelled 16 kHz mono WAV clips, 350 per class. The 12 classes are *hapana* (no), *kumi* (ten), *mbili* (two), *moja* (one), *nane* (eight), *ndio* (yes), *nne* (four), *saba* (seven), *sita* (six), *tano* (five), *tatu* (three) and *tisa* (nine). Clip durations range from 2.48 s to 62.24 s (median 4.32 s). Full findings are in `results/eda/eda_findings.md` and `reports/figures/draft/dataset_eda_preprocessing.md`.

The competition files are not in the repository (they are gitignored). Download them from Zindi and place them, unchanged and unextracted, in `data/raw/`:

| File | Contents |
| --- | --- |
| `Train.csv` | 4,200 rows: `Word_id`, `Swahili_word` (the label), `English_translation` |
| `Test.csv` | 1,800 unlabelled ids. **Not** our research test set. |
| `SampleSubmission.csv` | Zindi submission template |
| `Swahili_words.zip` | 6,000 WAV files, read directly from the zip |

## Experimental protocol

- **One shared split.** `data/splits/shared_split.csv` holds 2,940 train, 630 validation and 630 test clips (70/15/15, stratified, seed 42), cut from `Train.csv` only. No model creates its own split.
- **Labels** come from `Train.csv` → `Swahili_word`, joined on `Word_id`.
- **Validation only for decisions.** Preprocessing selection, hyperparameter tuning, model selection and early stopping all use the validation split.
- **Test once.** The internal test split (`split == "test"`) is evaluated once per approach, after its configuration is locked.
- **No leakage.** Feature standardisation statistics are fitted on training clips only.
- **Official test set unused.** The unlabelled Zindi `Test.csv` is never used for evaluation.
- **Primary metric** is macro-F1. Accuracy, weighted F1, macro precision and recall, and one-vs-rest macro ROC-AUC are also reported, all from the shared `src/evaluation/metrics.py`.
- **Balanced classes.** Every class has 350 clips, so no class weighting or oversampling is used.

## The five approaches

All models share one acoustic front end: 13 MFCCs from a 25 ms Hann window with a 10 ms hop, 512-point FFT and 40 mel filters (`src/preprocessing/`). Clips longer than the window are truncated, keeping the start.

| Approach | Owner | Input | Selection | Code | Notebook |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression, Linear SVM | Person 1 | MFCC + Δ + ΔΔ, 8 s window, mean + std over frames (78-D) | 4 feature configurations × 4 values of C on validation | `src/preprocessing/pipeline.py`, `scripts/run_phase5_classical_baselines.py` | `person1_eda_baselines.ipynb` |
| Bidirectional GRU | Person 2 | MFCC + Δ + ΔΔ frame sequence, 8 s window, 3 frames stacked (30 ms steps) | One-factor-at-a-time search over 13 runs: cell and direction, Δ features, frame stacking, pooling, hidden size, dropout. Plus 3 seeds. | `src/models/recurrent.py`, `src/preprocessing/sequences.py`, `scripts/run_person2_recurrent.py` | `person2_recurrent_model.ipynb` |
| Temporal CNN | Person 3 | Static 13-MFCC frame sequence, 6 s window (shared defaults) | Two 1-D convolution layers (64 channels, kernel 5). Training budget of 30 vs 60 epochs compared. | `src/models/temporal_cnn.py` | `person3_model.ipynb` |
| Transformer encoder | Person 4 | MFCC + Δ + ΔΔ frame sequence, 8 s window, 3 frames stacked | 4 runs: baseline, smaller model, deeper model, more dropout | `src/models/transformer.py` | `person4_model.ipynb` |

Every neural model pads variable-length batches and masks the padded steps, so each clip contributes only its real frames.

## How to run

### Option A: Google Colab (recommended for the neural models)

The BiGRU needs a GPU. A full search took 39 minutes on a Colab T4; on a laptop CPU it would take days.

1. Upload the four data files to a Google Drive folder, for example `My Drive/swahili_audio/`.
2. In Colab, open a notebook with **File → Open notebook → GitHub**, choosing this repository and the `main` branch.
3. Select **Runtime → Change runtime type → T4 GPU**.
4. `person2_recurrent_model.ipynb` sets itself up: it mounts Drive, clones the repository, copies the data, installs the requirements, runs the experiment and saves the results back to Drive. Set `BRANCH = "main"` and `DRIVE_DATA_DIR` in its first code cell.
5. For the other notebooks, run this cell first, then run the notebook as normal:

```python
from google.colab import drive
import os, shutil, subprocess
drive.mount("/content/drive")
DATA = "/content/drive/MyDrive/swahili_audio"   # folder holding the four data files
REPO = "/content/NLP-and-Language-Technologies"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "https://github.com/Orrie-Dan/NLP-and-Language-Technologies.git", REPO], check=True)
os.chdir(REPO)
for f in ["Train.csv", "Test.csv", "SampleSubmission.csv", "Swahili_words.zip"]:
    if not os.path.exists(f"data/raw/{f}"):
        shutil.copy(f"{DATA}/{f}", f"data/raw/{f}")
%pip install -q -r requirements.txt
```

The Colab disk is wiped when the session ends, so copy anything you want to keep to Drive.

### Option B: Locally

```bash
git clone https://github.com/Orrie-Dan/NLP-and-Language-Technologies.git
cd NLP-and-Language-Technologies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the four data files in data/raw/
```

Run the notebooks from the repository root or from `notebooks/`. Command-line entry points, run from the repository root:

The EDA and classical scripts take no arguments and overwrite their committed outputs in `results/eda/` and `results/classical/` when run. `run_phase2_eda` also expects Zindi's `Swahili_Audio_StarterNotebook.ipynb` in `data/raw/`.

```bash
python -m scripts.run_phase2_eda                     # EDA tables and figures
python -m scripts.run_phase3_investigation           # duration, energy and MFCC investigation
python -m scripts.run_phase5_classical_baselines     # Logistic Regression and Linear SVM
python scripts/run_person2_recurrent.py              # BiGRU search + locked test (GPU recommended)
python scripts/run_person2_recurrent.py --quick      # 1-minute smoke test on a few clips
python -m unittest discover tests                    # feature-pipeline tests
```

## Repository structure

| Path | Purpose |
| --- | --- |
| `data/raw/` | Original competition files (gitignored) |
| `data/splits/shared_split.csv` | The single fixed train/validation/test split |
| `src/data/` | Loading, the split, audio metadata and EDA helpers |
| `src/preprocessing/` | Shared MFCC front end. `pipeline.py` gives aggregated vectors for the classical models; `sequences.py` gives frame sequences, deltas, frame stacking and train-only standardisation for the neural models. |
| `src/models/` | `recurrent.py` (BiGRU/BiLSTM), `temporal_cnn.py` (CNN), `transformer.py` (Transformer encoder) |
| `src/evaluation/metrics.py` | Shared metrics, classification reports and confusion matrices |
| `src/utils/reproducibility.py` | Shared seed (42) and environment records |
| `scripts/` | Repeatable command-line experiments |
| `notebooks/` | One notebook per person, calling the shared code |
| `results/eda/` | EDA tables and findings |
| `results/classical/` | Classical baseline validation search and test results |
| `results/models/recurrent/` | BiGRU search table, training histories, test metrics, per-clip predictions, seed robustness |
| `results/person3_cnn_60epochs/` | Temporal CNN histories, test metrics, confusion matrix |
| `results/models/person4_model/` | Transformer search table, histories, test metrics, per-clip predictions, error pairs |
| `results/figures/` | EDA figures, plus `classical/`, `recurrent/` and `person4_model/` model figures |
| `reports/` | Report drafts |
| `tests/` | Unit tests for the feature pipeline, using synthetic audio |

Model checkpoints (`*.pt`), feature caches and the raw data are gitignored. They are regenerated by re-running the notebooks or scripts.

## Reproducibility

- **Seeds.** Python, NumPy and PyTorch are seeded with 42. Each results folder contains an `environment.json` with package versions, device and run time.
- **Hardware.** The BiGRU was trained on a Colab Tesla T4 (CUDA). The Transformer was trained on Apple Silicon (MPS) in about 7 minutes. The CNN was trained on CPU.
- **Determinism.** Results can differ slightly across hardware because GPU recurrent and attention kernels are not fully deterministic. The BiGRU seed study shows a spread of about ±0.006 macro-F1.

## Known limitations

- **Speakers may overlap.** The dataset has no speaker labels, so the same speakers may appear in training and test. Scores may overestimate performance on new speakers.
- **Long clips are truncated.** Clips longer than the window are cut, and accuracy is lower on them (BiGRU: 0.83 on clips of 8 s or more, vs. about 0.96 overall).
- **Unequal search budgets.** Inputs and tuning effort differ between models (13 BiGRU runs, 4 Transformer runs, 2 CNN runs; the CNN uses a 6 s window without deltas). The report discusses this when comparing approaches.

## Team

| Role | Responsibilities |
| --- | --- |
| Dan Nkusi(Person 1) | Data loading, shared split, EDA, preprocessing, classical baselines; Dataset and EDA report sections; repository organisation |
| Eddy Irasetsa(Person 2) | Bidirectional GRU model; Introduction; recurrent-model methodology and results; references formatting; |
| Prince Mbonyumugisha(Person 3) | Temporal CNN model; evaluation-metric justification; results comparison tables; related-work synthesis |
| Anthony Ariik(Person 4) | Transformer model; error analysis; limitations, conclusion and future work |
| Shared | Discussion, references, demo video, contribution tracking, Report |
