# Emotion classification on Reddit comments (GoEmotions)

ESILV – Machine Learning project 2026-2027, Group XX.

Multi-label classification of short English Reddit comments into 27 emotions + neutral, using the GoEmotions dataset (Demszky et al., 2020, simplified version: 43,410 train / 5,426 validation / 5,427 test).

## Setup

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The GoEmotions simplified splits are downloaded automatically by the notebooks with `datasets.load_dataset("google-research-datasets/go_emotions", "simplified")`. No manual download is needed.

## How to run

Run the notebooks from the `notebooks/` folder, in order. All paths are relative to the project root.

| Notebook | Content | Outputs |
|---|---|---|
| `00_data_check.ipynb` | Loading sanity check | – |
| `01_eda_simple.ipynb` | First look: label counts, labels per comment, text length, frequent words, duplicates | `results/eda_simple/` |

## Structure

```
├── requirements.txt          pinned dependencies
├── notebooks/                analysis and modelling notebooks
├── results/                  figures, tables and findings produced by the notebooks
├── report/                   Report_Group_XX.pdf (+ LaTeX sources)
└── ai_governance/
    ├── CONTRIBUTIONS.md      ownership matrix
    ├── PROMPT_LOG.md         4–6 documented AI interactions
    └── AI_AUDIT.md           ≥ 4 flaws caught in AI outputs
```
