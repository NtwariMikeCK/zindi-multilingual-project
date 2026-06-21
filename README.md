# Multilingual Health Question Answering in Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/multilingual-health-qa/blob/main/notebook/mhqa_experiments_final.ipynb)

**Competition:** [Multilingual Health QA Challenge — Zindi / ITU](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge)  
**Best Public Score:** 0.5864 | **Rank:** 177 / 595 active participants  
**Leaderboard Username:** Wagner_Mushayija

---

## Project Overview

This project builds a multilingual retrieval-augmented question answering system for health-related questions across eight African language subsets: Akan (Ghana), Amharic (Ethiopia), English (Ethiopia/Ghana/Kenya/Uganda), Luganda (Uganda), and Swahili (Kenya).

Rather than relying on generative fine-tuning alone, the system uses a hybrid retrieval pipeline combining semantic sentence embeddings (LaBSE) and TF-IDF keyword matching, with per-language weight tuning. Nine experiments were conducted, progressing from a character n-gram TF-IDF baseline (ROUGE-1: 0.454) to an exact match + LaBSE hybrid system (Zindi: 0.5864).

---

## Repository Structure

```
multilingual-health-qa/
├── notebook/
│   └── mhqa_experiments_final.ipynb   # Full experiment pipeline (Exp 1–9)
├── submissions/                        # CSV files submitted to Zindi
├── figures/                            # All plots and visualizations
├── screenshots/                        # Zindi leaderboard and submission screenshots
├── requirements.txt
└── README.md
```

---

## Setup and Reproduction

### Requirements

All experiments run on **Google Colab** (free tier, GPU only needed for Section C).

```bash
pip install -r requirements.txt
```

### Data

Download the competition data from [Zindi](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge/data) and place the files as follows:

```
/content/drive/MyDrive/MHQA/
├── Train.csv
├── Val.csv
└── Test.csv
```

### Running the Notebook

1. Open `notebook/mhqa_experiments_final.ipynb` in Google Colab (use the badge above)
2. Mount Google Drive when prompted
3. Run **Section 0** (setup) once per session — restart runtime after the pip install cell
4. Run sections A, B, and C in order. Each section has a **RESUME POINT** cell — if your session crashes, start from the resume point instead of re-running everything

**Estimated runtimes:**
- Section A (Exp 1–4): ~40 minutes, no GPU needed
- Section B (Exp 5–7): ~40 minutes, no GPU needed  
- Section C (Exp 8–9): ~15 minutes, no GPU needed

---

## Experiments Summary

| Exp | Name | Val ROUGE-1 | Val ROUGE-L | Zindi Score |
|-----|------|:-----------:|:-----------:|:-----------:|
| 1 | TF-IDF char ngram baseline | 0.454 | 0.380 | — |
| 2 | Semantic-only (mpnet) | 0.470 | 0.399 | — |
| 3 | Hybrid mpnet + TF-IDF (default) | 0.482 | 0.406 | 0.490 |
| 4 | Hybrid mpnet + per-language tuning | 0.509 | 0.439 | 0.490 |
| 5 | LaBSE semantic-only | 0.491 | 0.424 | 0.526 |
| 6 | LaBSE hybrid + tuned weights | 0.514 | 0.446 | 0.533 |
| 7 | LaBSE + char TF-IDF (non-Latin) | 0.521 | 0.454 | 0.557 |
| 8 | Full corpus (train+val) at test time | 0.521 | 0.454 | — |
| **9** | **Exact match + LaBSE fallback** | **0.521** | **0.454** | **0.586** |

---

## Key Results

- **Best Zindi score:** 0.5864 (rank 177/595)
- **Largest single gain:** Per-language weight tuning in Exp 4 (+2.8 pp ROUGE-1)
- **Hardest language:** Amharic (Amh_Eth) — ROUGE-1 consistently ~0.03 due to Ge'ez script and small training pool
- **Best embedding model:** LaBSE outperformed mpnet on every configuration, especially for Swahili (+12 pp) and Luganda

---

## Evaluation Metrics

- **ROUGE-1 F1** — unigram overlap between prediction and reference
- **ROUGE-L F1** — longest common subsequence F1
- **LLM-as-a-Judge** — fluency and factual quality scored by a language model

All three are averaged into the Zindi public score.

---

## Report and Demo

- Full academic report: `Wagner_Mushayija_FinalProject.pdf` (submitted via Canvas)
- Demo video: [link]

---

## Citation

If you use this code or approach, please cite the competition:

> ITU. (2026). Multilingual Health Question Answering in Low-Resource African Languages Challenge. Zindi Africa. https://zindi.africa
