# zindi-multilingual-project


# Multilingual Health Question Answering in Low-Resource African Languages

Final project for Machine Learning Techniques I — Zindi competition: *Multilingual Health Question
Answering in Low-Resource African Languages*.

This repository contains three Colab notebooks implementing and comparing three approaches to the task:

| Notebook | Experiment | Approach | Depends on |
|---|---|---|---|
| `Experiment1_Pure_RAG.ipynb` | Experiment 1 | Language-aware FAISS retrieval over training Q/A pairs, with two variants (nearest-neighbour copy vs. retrieval + zero-shot generation) | Nothing (run first) |
| `Experiment2_Finetune_Only.ipynb` | Experiment 2 | QLoRA fine-tuning of `CohereForAI/aya-expanse-8b`, used alone with no retrieval context | Nothing (can run independently of Notebook 1) |
| `Experiment3_Finetune_Plus_RAG.ipynb` | Experiment 3 | A second, RAG-aware QLoRA adapter trained on prompts that include retrieved reference examples, combined with the same retrieval index at inference time | Notebook 1 (FAISS indexes) |
| `Experiment4_Strong_ZeroShot_RAG.ipynb` | Experiment 4 | The "strongest RAG" condition: zero-shot (no fine-tuning) `CohereForAI/aya-expanse-8b`, used through its chat template, with retrieval depth raised to 15 reference Q&A pairs | Notebook 1 (FAISS indexes) |

## How to run (Google Colab)

1. Open each notebook in Colab (free T4 GPU runtime is sufficient for all three: `Runtime > Change runtime
   type > T4 GPU`).
2. Place `train.csv`, `val.csv`, `test.csv`, and `sample_submission.csv` in
   `MyDrive/health_qa_project/data/` in your Google Drive before running any notebook — every notebook
   mounts Drive in its first cell and reads from this fixed path.
3. Run notebooks in this order: **1 → 2 → 3 → 4**. Notebooks 3 and 4 will both fail their setup
   assertion if Notebook 1 has not been run at least once, since they load the FAISS indexes Notebook 1
   builds. Notebook 4 does not depend on Notebooks 2 or 3.
4. Each notebook writes its outputs (submission CSV, validation metrics JSON, plots, and model
   artifacts) to `MyDrive/health_qa_project/`, under `submissions/` and `artifacts/<experiment_name>/`
   respectively. These persist across sessions and across notebooks.

## Repository / Drive structure

```
health_qa_project/
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── sample_submission.csv
├── artifacts/
│   ├── exp1_rag/
│   │   ├── faiss_<language>.index        (one FAISS index per language)
│   │   ├── meta_<language>.pkl           (questions/answers/IDs per index)
│   │   ├── manifest.json                 (embedding model name, languages, top_k used)
│   │   ├── exp1_validation_metrics.json
│   │   └── *.png                         (language distribution, ROUGE-by-language, variant comparison)
│   ├── exp2_finetune/
│   │   ├── lora_adapter/                 (trained LoRA weights, NOT a merged full model)
│   │   ├── training_logs.json
│   │   ├── adapter_manifest.json
│   │   ├── exp2_validation_metrics.json
│   │   └── *.png                         (loss curve, token length histogram, ROUGE-by-language)
│   ├── exp3_finetune_rag/
│   │   ├── lora_adapter_rag/             (second LoRA adapter, trained on RAG-formatted prompts)
│   │   ├── training_logs.json
│   │   ├── adapter_manifest.json
│   │   ├── exp3_validation_metrics.json
│   │   ├── cross_experiment_comparison.csv
│   │   └── *.png                         (loss curve, ROUGE-by-language, cross-experiment comparison)
│   └── exp4_strong_zero_shot_rag/
│       ├── exp4_validation_metrics.json
│       ├── cross_experiment_comparison_1to4.csv
│       └── *.png                         (token length histogram, ROUGE-by-language, 1-4 comparison)
└── submissions/
    ├── submission_exp1_pure_rag.csv
    ├── submission_exp2_finetuned_only.csv
    ├── submission_exp3_finetuned_plus_rag.csv
    └── submission_exp4_strong_zero_shot_rag.csv
```

## Design notes

**Why Aya Expanse 8B.** Among 8B-class instruction-tuned multilingual models, Aya Expanse has
comparatively strong coverage of African languages, which matters directly for Amharic, Akan, and Luganda,
the lowest-resource languages in this dataset.

**Why QLoRA.** Free-tier Colab provides a single T4 GPU (~15GB VRAM). An 8B model in 4-bit NF4
quantisation occupies roughly 5–6GB, leaving enough headroom for LoRA adapter training with gradient
checkpointing, a small per-device batch size, and gradient accumulation. Full fine-tuning of an 8B model is
not feasible on this hardware tier.

**Why two separate LoRA adapters (Experiment 2 vs 3) instead of one.** The project brief's
training/inference alignment principle (retrieved examples should appear in the prompt during *both*
training and inference) means the RAG-aware adapter is trained on a different prompt distribution
(longer, with reference Q/A pairs prepended) than the non-RAG adapter. Reusing the Experiment 2 adapter as
a RAG-time-only addition, without ever training on RAG-formatted prompts, would violate that principle and
likely produce a model that ignores its retrieved context. Training a second adapter from the same base
model isolates the actual effect of adding retrieval.

**Language-aware retrieval.** All FAISS indexes are built per-language (one index per value in the
mapped `subset` column), so a Luganda question can never retrieve a Swahili reference example, per the
project brief's Section 10 guidance.

**Self-retrieval leakage guard.** When building RAG training prompts in Notebook 3, a training row's own
question is excluded from its own retrieved context (`retrieve_top_k_excluding_self`), preventing the model
from learning to trivially copy a retrieved answer that is identical to its own training target.

**Why a fourth, zero-shot experiment (Experiment 4).** Experiments 1-3 alone can't cleanly separate two
different effects: generator model strength, and fine-tuning itself. Experiment 4 uses the same base model
as Experiments 2/3 (`aya-expanse-8b`) but with no fine-tuning and a much deeper retrieval context (15
reference Q&A pairs vs. Notebook 1's 5 or Notebook 3's 3). This gives two useful comparisons: Exp 1 vs Exp
4 isolates the effect of generator strength (holding "no fine-tuning + RAG" constant), and Exp 4 vs Exp 3
isolates the effect of fine-tuning (holding the base model and presence of RAG constant, though at
different retrieval depths - noted as a caveat in Notebook 4 itself). Aya Expanse is instruction-tuned, so
Notebook 4 routes prompts through its chat template (`apply_chat_template`) rather than raw completion,
since zero-shot instruction-following quality depends on using the model's intended prompt format.

## Evaluation

ROUGE-1 F1 and ROUGE-L F1 are computed with whitespace tokenisation (`rouge-score` library with a custom
tokenizer), which avoids assuming any language-specific tokenisation rules — important across five
languages with different scripts and orthographies. The competition's third metric, LLM-as-a-Judge, is
computed server-side on Zindi from the submitted answers and is not reproduced locally.

## Submission format

Per the competition format, all three target columns (`TargetRLF1`, `TargetR1F1`, `TargetLLM`) contain the
**same** generated answer for each row; the platform computes all three metrics from that single column.
Every notebook's `make_submission()` function asserts this before writing the CSV.

## Reproducibility

- Random seed fixed to `42` for all numpy/torch operations.
- All hyperparameters are declared in a single "Configuration" cell near the top of each notebook.
- Every experiment's full configuration (model name, LoRA settings, training settings, retrieval settings)
  is saved alongside its validation metrics in a `*_validation_metrics.json` file, so results can be
  traced back to the exact settings that produced them.

## Suggested additional experiments (for the "10 meaningful experiments" report requirement)

Starting from the notebook skeletons here, these are documented variations worth running and logging:

1. `TOP_K` retrieval depth: 1 vs 3 vs 5 vs 10.
2. `LORA_R`: 8 vs 16 vs 32 (trainable parameter count vs. performance trade-off).
3. `MAX_SEQ_LENGTH`: effect of truncating longer RAG prompts.
4. Embedding model swap: `intfloat/multilingual-e5-large` vs `sentence-transformers/LaBSE`.
5. Retrieval-only (Variant A) vs retrieval+generation (Variant B) in Notebook 1 — already built in as a
   documented comparison in that notebook.
6. Number of training epochs: 2 vs 3 vs 5, watching for validation loss divergence (overfitting on a
   ~30k-example training set).
7. Effect of the self-exclusion leakage guard: train one RAG adapter *without* excluding self-matches and
   compare validation ROUGE (expect inflated training performance, degraded generalisation).
8. Greedy decoding vs. beam search (`num_beams`) at inference time.
9. Per-language breakdown across all three experiments — already produced via
   `compute_rouge_by_language()` in each notebook — to identify which languages benefit most from RAG vs.
   fine-tuning.
10. Prompt wording ablation: with vs without the explicit `"Answer only in {language}."` instruction, to
    quantify how much it affects language-consistency failures.
    
