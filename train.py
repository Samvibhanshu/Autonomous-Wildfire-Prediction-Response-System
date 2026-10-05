"""python train.py [--quick]   trains the models, writes artifacts/model_bundle.joblib, metrics.json and figures."""
import argparse
import json

from wfire import training

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="smaller search and fewer CV repeats (for smoke tests)")
ap.add_argument("--no-figures", action="store_true")
a = ap.parse_args()
m = training.run(quick=a.quick, make_figs=not a.no_figures)

print(f"\nModel version {m['version']}   ({m['seconds']} s)")
print(f"Records {m['dataset']['records']}, fire rate {m['dataset']['positive_rate']:.1%}, "
      f"split {m['split']['train']}/{m['split']['validation']}/{m['split']['test']}")
print(f"\nSingle 70/15/15 split, test n={m['split']['test']} (majority-class baseline accuracy {m['test_majority_baseline_accuracy']:.3f}):")
for n, r in m["test"].items():
    print(f"  {n:20s} acc {r['accuracy']:.3f}  AUC {r['auc']:.3f} (95% CI {r['auc_ci95'][0]:.2f}-{r['auc_ci95'][1]:.2f})  "
          f"prec {r['precision']:.2f}  recall {r['recall']:.2f}")
print(f"\n{m['cross_validation']['scheme']}  (majority baseline {m['cross_validation']['majority_baseline_accuracy']:.3f}):")
for n, r in m["cross_validation"]["results"].items():
    print(f"  {n:20s} acc {r['accuracy']['mean']:.3f} +/- {r['accuracy']['std']:.3f}   AUC {r['auc']['mean']:.3f} +/- {r['auc']['std']:.3f}")
g = m["confidence_gating"]
print("\nObserved fire rate by risk level (out-of-fold):")
for lv, r in g["by_level"].items():
    print(f"  {lv:9s} n={r['n']:3d}  fire rate {('%.2f' % r['observed_fire_rate']) if r['observed_fire_rate'] is not None else '-'}")
print("Alert precision:")
for k, r in g["alerts"].items():
    print(f"  {k:38s} n={r['n']:3d}  precision {('%.2f' % r['precision']) if r['precision'] is not None else '-'}")
print("\nNote:", m["disclosure_projection_note"])
