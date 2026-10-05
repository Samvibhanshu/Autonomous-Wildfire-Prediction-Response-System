"""Figures 8-15 of the disclosure, generated from the data and the trained models."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .config import GRID_X, GRID_Y, MONTHS

COL = {"random_forest": "#2563eb", "gradient_boosting": "#16a34a", "logistic_regression": "#9333ea", "ensemble": "#111827"}
LABEL = {"random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting", "logistic_regression": "Logistic Regression", "ensemble": "Ensemble (mean)"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_all(raw, X, y, Xte, yte, test_p, rf, cv, baseline, out):
    out.mkdir(parents=True, exist_ok=True)
    # Fig 8: ROC on the test split
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for n, p in test_p.items():
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, color=COL[n], lw=2 if n == "ensemble" else 1.4, label=f"{LABEL[n]} (AUC {roc_auc_score(yte, p):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Chance (AUC 0.50)")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"Fig 8. ROC, test split (n={len(yte)})")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, out / "fig08_roc.png")

    # Fig 9: accuracy, test split vs 5x10 CV, against the majority baseline
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    names = list(test_p)
    x = np.arange(len(names)); w = .38
    tacc = [np.mean((test_p[n] >= .5) == yte.values) for n in names]
    cacc = [cv[n]["accuracy"]["mean"] for n in names]; cerr = [cv[n]["accuracy"]["std"] for n in names]
    ax.bar(x - w / 2, tacc, w, label=f"Test split (n={len(yte)})", color="#93c5fd")
    ax.bar(x + w / 2, cacc, w, yerr=cerr, capsize=3, label="Cross-validation (mean ± sd)", color="#2563eb")
    ax.axhline(baseline, color="#dc2626", ls="--", lw=1.2, label=f"Always predict majority class ({baseline:.2f})")
    ax.set_xticks(x); ax.set_xticklabels([LABEL[n].replace(" ", "\n") for n in names], fontsize=8)
    ax.set(ylim=(0.4, 0.8), ylabel="Accuracy", title="Fig 9. Model accuracy")
    ax.legend(fontsize=7.5, loc="upper right")
    _save(fig, out / "fig09_accuracy.png")

    # Fig 10: RF feature importance
    imp = dict(sorted(zip(X.columns, rf.feature_importances_), key=lambda kv: kv[1]))
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.barh(list(imp), list(imp.values()), color="#2563eb")
    ax.set(xlabel="Importance (Random Forest, impurity)", title="Fig 10. Feature importance")
    _save(fig, out / "fig10_feature_importance.png")

    # Fig 11/12: scatter plots against burned area
    for fname, col, label, title in (("fig11_temp_vs_area.png", "temp", "Temperature (°C)", "Fig 11. Temperature vs burned area"),
                                     ("fig12_rh_vs_area.png", "RH", "Relative humidity (%)", "Fig 12. Relative humidity vs burned area")):
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        burned = raw[raw.area > 0]
        ax.scatter(burned[col], burned.area, s=14, alpha=.55, color="#dc2626")
        ax.set_yscale("log"); ax.set(xlabel=label, ylabel="Burned area (ha, log scale; area > 0 only)", title=title)
        r = np.corrcoef(burned[col], np.log(burned.area))[0, 1]
        ax.text(.03, .95, f"corr with log(area): {r:+.2f}", transform=ax.transAxes, va="top", fontsize=8)
        _save(fig, out / fname)

    # Fig 13: heatmap of recorded fire events by park-grid cell
    grid = np.full((len(GRID_Y), len(GRID_X)), np.nan)
    for (gx, gy), g in raw.groupby(["X", "Y"]):
        grid[gy - min(GRID_Y), gx - min(GRID_X)] = (g.area > 0).sum()
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(grid, origin="upper", cmap="YlOrRd")
    ax.set_xticks(range(len(GRID_X)), list(GRID_X)); ax.set_yticks(range(len(GRID_Y)), list(GRID_Y))
    ax.set(xlabel="X (park grid)", ylabel="Y (park grid)", title="Fig 13. Recorded fire events per grid cell")
    fig.colorbar(im, ax=ax, label="fire events")
    _save(fig, out / "fig13_risk_heatmap.png")

    # Fig 14: temperature distribution
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.hist(raw.temp, bins=20, color="#f59e0b", edgecolor="white")
    ax.set(xlabel="Temperature (°C)", ylabel="Records", title="Fig 14. Temperature distribution")
    _save(fig, out / "fig14_temp_distribution.png")

    # Fig 15: monthly fire occurrence
    fires = raw[raw.area > 0].groupby("month").size().reindex(MONTHS, fill_value=0)
    total = raw.groupby("month").size().reindex(MONTHS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(MONTHS, total, color="#e5e7eb", label="All records")
    ax.bar(MONTHS, fires, color="#dc2626", label="Fire recorded (area > 0)")
    ax.set(ylabel="Records", title="Fig 15. Monthly fire occurrence"); ax.legend(fontsize=8)
    _save(fig, out / "fig15_monthly_fires.png")
