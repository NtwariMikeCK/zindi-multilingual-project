# Multilingual Health Question Answering in Low-Resource African Languages

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/multilingual-health-qa/blob/main/notebook/mhqa_experiments_final.ipynb)

**Competition:** [Multilingual Health QA Challenge — Zindi / ITU](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge)  
**Best Public Score:** 0.586735 | **Rank:** 182 / 595 active participants  
**Leaderboard Username:**Ntwari Mike Chris Kevin

---

## Project Overview

This project builds a multilingual retrieval-augmented question answering system for health-related questions across eight African language subsets: Akan (Ghana), Amharic (Ethiopia), English (Ethiopia/Ghana/Kenya/Uganda), Luganda (Uganda), and Swahili (Kenya).

Rather than relying on generative fine-tuning alone, the system uses a hybrid retrieval pipeline combining semantic sentence embeddings (LaBSE) and TF-IDF keyword matching, with per-language weight tuning. Nine experiments were conducted, progressing from a character n-gram TF-IDF baseline (ROUGE-1: 0.454) to an exact match + LaBSE hybrid system (Zindi: 0.5864).

---

## Repository Structure

```
multilingual-health-qa/
├── notebook/
│   └── Experiment 1 to 11
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

---

## Experiments Summary

Exp
Method
ROUGE-1 F1
ROUGE-L F1
LLM Judge
1
TF-IDF Character N-Gram Retrieval
0.4773
0.3969
0.6504
2
Zero-Shot mT5
0.0401
0.0348
—
3
Fine-Tuned mT5
0.0682
0.0552
0.4799
4
E5 Semantic Retrieval
0.5525
0.4790
0.7321
5
Language-Aware FAISS + Generator
0.0100
0.0095
—
6
Aya Expanse Fine-Tuning
Not Completed
Not Completed
Not Completed
7
Aya Expanse + RAG
Not Completed
Not Completed
Not Completed
8
LaBSE Semantic Retrieval
0.5176
0.4475
0.6485
9
Character TF-IDF for Amharic/Luganda
0.5603
0.4901
0.6970
10
Train + Validation Retrieval Corpus
0.5768
0.5071
0.7130
11
Exact Match + LaBSE Retrieval
0.5768
0.5071
0.7142


---

## Key Results

- **Largest single gain:** Per-language weight tuning in Exp 4 (+2.8 pp ROUGE-1)
- **Hardest language:** Amharic (Amh_Eth) — ROUGE-1 consistently ~0.03 due to Ge'ez script and small training pool
- **Best embedding model:** LaBSE outperformed E5 on every configuration, especially for Swahili (+12 pp) and Luganda

---

## Evaluation Metrics

- **ROUGE-1 F1** — unigram overlap between prediction and reference
- **ROUGE-L F1** — longest common subsequence F1
- **LLM-as-a-Judge** — fluency and factual quality scored by a language model

All three are averaged into the Zindi public score.

---


## Citation

If you use this code or approach, please cite the competition:

> ITU. (2026). Multilingual Health Question Answering in Low-Resource African Languages Challenge. Zindi Africa. https://zindi.africa
