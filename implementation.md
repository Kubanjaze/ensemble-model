# Phase 46 — Ensemble Model (RF + LGB + SVM)

**Version:** 1.1 | **Tier:** Standard | **Date:** 2026-03-26

## Goal
Build a probability-average ensemble of RF + SVM classifiers (excluding LightGBM which was below random).
Compare ensemble ROC-AUC, PR-AUC, and EF@K vs individual models.

CLI: `python main.py --input data/compounds.csv --threshold 7.0`

## Logic
- Average predicted probabilities (not ranks) from RF + SVM LOO-CV
- Exclude LightGBM (ROC-AUC=0.44 from Phase 38/43 — below random, contaminates)
- Ensemble = mean(p_rf, p_svm) per compound

## Actual Results (v1.1)

| Model | ROC-AUC | PR-AUC | EF@10% | EF@20% |
|---|---|---|---|---|
| RF | 0.8267 | 0.9101 | 1.50× | 1.50× |
| SVM | 0.7889 | 0.8897 | 1.50× | 1.50× |
| Ensemble (RF+SVM) | **0.8444** | **0.9227** | 1.50× | 1.50× |

**Key insight:** Probability-average ensemble modestly improves ROC-AUC (+0.018 over RF) and PR-AUC (+0.013). EF@K unchanged — both models already achieve maximum possible EF at these cutoffs. Ensemble works when combining good models (unlike Phase 43 where LightGBM contaminated).
