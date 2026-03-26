import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
RDLogger.DisableLog("rdApp.*")

FAMILY_COLORS = {"benz": "#4C72B0", "naph": "#DD8452", "ind": "#55A868",
                 "quin": "#C44E52", "pyr": "#8172B2", "bzim": "#937860", "other": "#808080"}

def load_compounds(path, threshold):
    df = pd.read_csv(path)
    records, n_bad = [], 0
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None: n_bad += 1; continue
        try:
            pic50 = float(row["pic50"])
        except (KeyError, ValueError):
            continue
        if np.isnan(pic50): continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        fam = str(row["compound_name"]).split("_")[0]
        records.append({"compound_name": str(row["compound_name"]),
                        "family": fam if fam in FAMILY_COLORS else "other",
                        "active": int(pic50 >= threshold), "fp": list(fp)})
    print(f"  {len(records)} valid ({n_bad} skipped)")
    return pd.DataFrame(records)

def loo_proba(model_fn, X, y):
    loo = LeaveOneOut()
    scores = np.zeros(len(y))
    for tr, te in loo.split(X):
        m = model_fn()
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])
        if 1 in m.classes_:
            scores[te] = proba[0, list(m.classes_).index(1)]
        else:
            scores[te] = 0.0
    return scores

def compute_ef(y_true, y_score, k_frac):
    n = len(y_true); k = max(1, int(np.round(n * k_frac)))
    total_hits = y_true.sum()
    if total_hits == 0: return 0.0
    order = np.argsort(y_score)[::-1]
    return round(float((y_true[order[:k]].sum() / k) / (total_hits / n)), 3)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading: {args.input}")
    df = load_compounds(args.input, args.threshold)
    X = np.array(df["fp"].tolist(), dtype=float)
    y = df["active"].values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    print("LOO-CV RF...")
    rf_scores = loo_proba(lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), X, y)
    print("LOO-CV SVM...")
    svm_scores = loo_proba(lambda: SVC(kernel="rbf", probability=True, random_state=42, C=1.0), X_sc, y)
    ensemble_scores = (rf_scores + svm_scores) / 2

    models = {"RF": rf_scores, "SVM": svm_scores, "Ensemble (RF+SVM)": ensemble_scores}

    # Evaluate
    hit_rate = y.mean()
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    summary = []
    for (name, scores), color in zip(models.items(), colors):
        roc_auc = roc_auc_score(y, scores)
        pr_auc = average_precision_score(y, scores)
        ef10 = compute_ef(y, scores, 0.10)
        ef20 = compute_ef(y, scores, 0.20)
        summary.append({"model": name, "roc_auc": round(roc_auc, 4),
                        "pr_auc": round(pr_auc, 4), "ef_10": ef10, "ef_20": ef20})
        fpr, tpr, _ = roc_curve(y, scores)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        prec, rec, _ = precision_recall_curve(y, scores)
        ax_pr.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={pr_auc:.3f})")

    ax_roc.plot([0,1],[0,1],"k--",lw=1); ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curve",fontweight="bold"); ax_roc.legend(fontsize=9)
    ax_roc.spines["top"].set_visible(False); ax_roc.spines["right"].set_visible(False)
    ax_pr.axhline(hit_rate,color="k",linestyle="--",lw=1); ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR Curve",fontweight="bold"); ax_pr.legend(fontsize=9)
    ax_pr.spines["top"].set_visible(False); ax_pr.spines["right"].set_visible(False)
    plt.suptitle("Ensemble (RF+SVM) vs Individual Models", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "ensemble_curves.png"), dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {args.output_dir}/ensemble_curves.png")

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(os.path.join(args.output_dir, "ensemble_summary.csv"), index=False)
    print(f"Saved: {args.output_dir}/ensemble_summary.csv")

    score_df = pd.DataFrame({"compound_name": df["compound_name"], "family": df["family"],
                              "y_true": y, "rf_score": rf_scores, "svm_score": svm_scores,
                              "ensemble_score": ensemble_scores})
    score_df.to_csv(os.path.join(args.output_dir, "ensemble_scores.csv"), index=False)

    print(f"\n--- Ensemble Results ---")
    print(f"  {'Model':<20}  {'ROC-AUC':>8}  {'PR-AUC':>8}  {'EF@10%':>7}  {'EF@20%':>7}")
    for r in summary:
        print(f"  {r['model']:<20}  {r['roc_auc']:>8.4f}  {r['pr_auc']:>8.4f}  {r['ef_10']:>7.3f}x  {r['ef_20']:>7.3f}x")
    print("\nDone.")

if __name__ == "__main__":
    main()
