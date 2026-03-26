# ensemble-model — Phase 46

Probability-average ensemble of RF + SVM classifiers (LightGBM excluded — below random in Phase 43).
Evaluates whether probability averaging outperforms individual models.

## Usage

```bash
PYTHONUTF8=1 python main.py --input data/compounds.csv --threshold 7.0
```
