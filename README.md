# Research-Informed Sequential Models for NLP and Language Technologies

Formative Assignment 2. The challenge files are the Zindi Weekendz Swahili audio set supplied by the group. This README describes the shared project structure, what those files contain, and the experimental rules. It does not report model results. Counts below were read from the copied CSV and zip files. They are not modelling results.

## Project objective

This project investigates how different sequential modelling approaches perform on an approved NLP or language-technology challenge. The work emphasises:

- sequential characteristics of the data
- justification of model selection
- an empirical comparison of substantially different approaches
- strengths and limitations of each approach
- error analysis

The comparison will cover five approaches, including at least three neural sequential architectures. Those five approaches are not chosen yet. Classical baselines will use an audio representation that has not been chosen. No model has been trained.

## Dataset

Source files, copied unchanged into `data/raw/`:

| File | What it is |
| --- | --- |
| `Train.csv` | 4,200 rows. Columns: `Word_id`, `Swahili_word`, `English_translation`. No duplicate ids and no missing values. |
| `Test.csv` | 1,800 rows. Column: `Word_id` only. No labels. |
| `SampleSubmission.csv` | 1,800 rows, same ids as `Test.csv`. Columns are `Word_id` plus 12 class names, filled with zeros. |
| `Swahili_words.zip` | 6,000 `.wav` files, one per train and test id. No extra files. Not extracted yet. |
| `Swahili_Audio_StarterNotebook.ipynb` | Official starter notebook. Reference only. It is not project code and it contains no completed training results. |

The starter notebook states the objective as an automatic speech recognition solution that classifies simple Swahili audio into text, for uses such as voice or text prompts, translation, public health, or emergency services.

Each `Word_id` is a `.wav` filename. Train and test ids do not overlap. Every id in the two CSV files is present in the zip, and every zip entry is listed in one of the two CSV files.

`Train.csv` has 12 Swahili labels. Each label occurs 350 times, and each label has one English translation:

| Swahili | English | Rows |
| --- | --- | --- |
| hapana | no | 350 |
| kumi | ten | 350 |
| mbili | two | 350 |
| moja | one | 350 |
| nane | eight | 350 |
| ndio | yes | 350 |
| nne | four | 350 |
| saba | seven | 350 |
| sita | six | 350 |
| tano | five | 350 |
| tatu | three | 350 |
| tisa | nine | 350 |

The shared research split is `data/splits/shared_split.csv`. It was cut only from the 4,200 labeled rows in `Train.csv`: 70% train, 15% validation, and 15% internal test, stratified on `Swahili_word`, with random seed 42 (scikit-learn 1.8.0). Exact sizes and per-class counts are in `results/eda/split_summary.json`. Because 15% of 350 is not an integer, validation and internal test have 52 or 53 clips per class. `Test.csv` is not part of this split. It remains an unlabeled external file.

The internal test split is reserved for final comparison. It is not for tuning, model selection, preprocessing fits, threshold selection, or early stopping.

The input is audio, not documents. Text TF-IDF is not a baseline for these files. Classical baselines still need an audio representation, which has not been chosen. The starter notebook's spectrogram-and-image-classifier example is not a chosen project method. Measured audio properties are in `results/eda/eda_findings.md`.

## Repository structure

| Path | Purpose |
| --- | --- |
| `data/raw/` | Original dataset files, unchanged after download. Contents are gitignored. |
| `data/interim/` | Partially cleaned data, before the shared split is applied. Contents are gitignored. |
| `data/processed/` | Data after agreed preprocessing, if a processed copy is stored. Contents are gitignored. |
| `data/splits/` | The single fixed train/validation/test split in `shared_split.csv`. Every model must load this file. |
| `notebooks/` | Person-specific notebooks that call shared code. They are not a second implementation of the pipeline. |
| `src/data/` | Dataset loading. Dan. |
| `src/preprocessing/` | Shared preprocessing decisions. Dan. |
| `src/models/classical.py` | Classical audio baselines. Dan. Representation not chosen yet. |
| `src/models/neural.py` | Recurrent and other neural sequential models. Persons 2, 3, and 4. |
| `src/evaluation/` | Metrics used by every model. |
| `src/utils/` | Reproducibility helpers, including the shared random seed. |
| `scripts/` | Repeatable command-line entry points, once experiments exist. |
| `results/eda/` | Tables and summaries from exploratory analysis. |
| `results/baselines/` | Outputs from the classical baselines. |
| `results/models/` | Outputs from neural models. Checkpoints are gitignored. |
| `results/figures/` | Figures generated from experiments. |
| `reports/figures/` | Figures selected for the written report. |

Intended workflow:

```text
Raw dataset
    ↓
Dataset loading
    ↓
Basic validation / cleaning
    ↓
ONE shared fixed train/validation/test split
    ↓
Shared preprocessing decisions
    ↓
    ├── audio features → classical baseline (representation not chosen yet)
    ├── audio features → second classical baseline (not chosen yet)
    ├── sequence representation → recurrent model
    ├── neural sequential model
    └── neural sequential model
```

Dan's classical models must not create a separate train/validation/test split.

## Team structure

Names are not recorded here. Roles are by person number.

**Dan**

- data loading and the shared split
- exploratory data analysis
- preprocessing
- classical baselines
- class-imbalance investigation
- Dataset and Exploratory Analysis section of the report
- preprocessing portion of Methodology

**Person 2**

- BiLSTM/BiGRU recurrent model
- recurrent-model methodology and results
- introduction
- final report assembly

**Person 3**

- assigned neural sequential model (architecture not chosen yet)
- related-work synthesis
- discussion and analysis coordination

**Person 4**

- assigned neural sequential model (architecture not chosen yet)
- discussion and analysis with Person 3

**Shared**

- final model comparison
- discussion
- references
- demo video
- contribution tracking

## Experimental principles

- One fixed shared split is created once and reused by every model.
- Experiments must be reproducible: seeds, split files, and preprocessing choices are recorded.
- The test set is not used to choose preprocessing, features, architectures, or hyperparameters.
- The validation set is used for model and tuning decisions.
- The test set is reserved for final evaluation.
- Preprocessing choices must be documented and justified.
- Models are compared on the same held-out test set.
- Reported numbers must come from actual experiments run on this dataset.
- Statistics, tables, and results must not be fabricated or filled in before the code has been run.

## Current status

Phase 3 duration, energy, MFCC, and fixed-window investigation is complete. The shared split was not rewritten. No model has been trained, and no preprocessing pipeline has been adopted. Evidence is in `results/eda/eda_findings.md` and `results/eda/preprocessing_investigation.json`. Neural-model libraries are still absent from `requirements.txt`.
