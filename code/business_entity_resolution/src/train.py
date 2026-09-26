"""Training: blocking -> features -> grouped CV (+ S1 self-negatives) -> threshold
tuning (flat vs expected-F0.5 decoder) -> final model -> experiment log.

    python -m src.train --data-dir dataset --work-dir work --model-dir models
"""
import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .blocking import blocking_report, generate_candidates, s1_self_negative_pairs
from .config import LGB_PARAMS, BlockingConfig
from .decode import calibrate, decode, decode_v2, macro_f05, tune
from .features import attach_ce, build_features
from .io_utils import candidate_table, load_split
from .representations import build_reps


def analyse_ground_truth(truth, s1, cands):
    country = dict(zip(s1.entity_id, s1.country.str.strip().str.lower()))
    country.update(zip(cands.entity_id, cands.country.str.strip().str.lower()))
    owners = Counter(c for cs in truth.values() for c in cs)
    n_match = [len(v) for v in truth.values()]
    pairs = [(s, c) for s, cs in truth.items() for c in cs]
    stats = {
        "n_s1": len(truth),
        "singleton_fraction": float(np.mean([n == 0 for n in n_match])),
        "avg_matches_per_s1": float(np.mean(n_match)),
        "max_matches_per_s1": int(max(n_match)),
        "frac_records_with_multiple_s1_owners":
            float(np.mean([v > 1 for v in owners.values()])) if owners else 0.0,
        "frac_matches_s3": float(np.mean([c.startswith("S3") for _, c in pairs])) if pairs else 0.0,
        "frac_matches_cross_country":
            float(np.mean([country.get(s) != country.get(c) for s, c in pairs])) if pairs else 0.0,
        "s1_by_country": dict(Counter(country[s] for s in truth)),
    }
    return stats


def new_model():
    return lgb.LGBMClassifier(**LGB_PARAMS)


LOG_COLS = ["exp_id", "timestamp", "blocking_cfg", "n_self_neg", "decode_method",
            "cv_f05_flat", "cv_f05_expected", "cv_f05_used", "loco_min_gap",
            "blocking_recall", "n_train_pairs", "n_features", "notes"]


def log_experiment(work_dir, row):
    """Phase 5: append one row per training run to work/experiments.csv so later
    you can see exactly which change moved the score, instead of "I changed five
    things and it went up.\""""
    path = Path(work_dir) / "experiments.csv"
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLS})
    print(f"logged experiment {row.get('exp_id')} -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--no-dense", action="store_true", help="TF-IDF/key blocking only (CPU-friendly)")
    ap.add_argument("--embed-model", default=BlockingConfig.embed_model)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1, help="processes for string features")
    ap.add_argument("--ce-scores", default=None, help="optional cross-encoder OOF scores (.pkl)")
    ap.add_argument("--skip-loco", action="store_true")
    ap.add_argument("--no-self-neg", action="store_true",
                    help="disable S1-vs-S1 hard-negative mining (Phase 2)")
    ap.add_argument("--self-neg-k", type=int, default=8)
    ap.add_argument("--self-neg-min-sim", type=float, default=0.15)
    ap.add_argument("--notes", default="", help="free-text note stored in experiments.csv")
    a = ap.parse_args()
    exp_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    work, mdir = Path(a.work_dir), Path(a.model_dir)
    work.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    cfg = BlockingConfig(use_dense=not a.no_dense, embed_model=a.embed_model)

    print("== loading train data")
    s1, s2, s3, truth = load_split(a.data_dir, "train")
    cands = candidate_table(s2, s3)
    gt_stats = analyse_ground_truth(truth, s1, cands)
    print(json.dumps(gt_stats, indent=2))
    one_owner = gt_stats["frac_records_with_multiple_s1_owners"] <= 0.01
    print(f"one_owner decoding = {one_owner}")

    print("== representations + blocking")
    r = build_reps(s1, cands, cfg, work, "train")
    pairs = generate_candidates(r, cfg)
    stats, missed = blocking_report(pairs, truth, list(s1.entity_id), len(cands))
    print(json.dumps(stats, indent=2))
    txt = dict(zip(cands.entity_id, cands.business_name + " | " + cands.business_address))
    s1txt = dict(zip(s1.entity_id, s1.business_name + " | " + s1.business_address))
    pd.DataFrame([(s, c, s1txt[s], txt[c]) for s, c in missed],
                 columns=["s1", "cand", "s1_text", "cand_text"]).to_csv(
        work / "blocking_missed_pairs.tsv", sep="\t", index=False)
    if stats["pair_recall"] < 0.97:
        print("WARNING: blocking recall < 97% - inspect work/blocking_missed_pairs.tsv "
              "and add a generator / raise K before tuning the model.")

    if a.ce_scores:
        pairs = attach_ce(pairs, a.ce_scores)

    print("== features")
    X = build_features(r, pairs, n_jobs=a.n_jobs)
    key = pairs.s1_id + "|" + pairs.cand_id
    true_keys = {f"{s}|{c}" for s, cs in truth.items() for c in cs}
    y = key.isin(true_keys).astype(int).values
    print(f"pairs={len(X):,} features={X.shape[1]} positives={y.sum():,} ({y.mean():.3%})")
    pairs.assign(y=y).to_pickle(work / "train_pairs.pkl")

    # ---- Phase 2: S1-vs-S1 hard negatives (distinct S1 entities are guaranteed
    # non-matches; near-duplicate ones teach the model the real decision boundary).
    # Added ONLY to each fold's training rows / the final fit - never to a
    # validation fold, so the OOF score and everything derived from it (threshold
    # tuning, calibration, LOCO) stays an honest estimate of real performance. ----
    X_self, y_self, n_self = None, None, 0
    if not a.no_self_neg:
        print("== Phase 2: mining S1-vs-S1 hard negatives")
        pseudo_r, self_pairs = s1_self_negative_pairs(r, k=a.self_neg_k, min_sim=a.self_neg_min_sim)
        if len(self_pairs):
            X_self = build_features(pseudo_r, self_pairs, n_jobs=a.n_jobs)
            y_self = np.zeros(len(X_self), dtype=int)
            n_self = len(X_self)
        print(f"   {n_self:,} self-negative pairs added to every training fold")

    def fit_with_self_neg(Xtr, ytr):
        if X_self is not None:
            Xtr = pd.concat([Xtr, X_self], ignore_index=True)
            ytr = np.concatenate([ytr, y_self])
        return new_model().fit(Xtr, ytr)

    print(f"== {a.folds}-fold grouped CV (grouped by S1 entity)")
    oof = np.zeros(len(X))
    for k, (tr, va) in enumerate(GroupKFold(n_splits=a.folds).split(X, y, pairs.s1_id.values)):
        m = fit_with_self_neg(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        print(f"  fold {k} done")
    oof_df = pairs[["s1_id", "cand_id", "src"]].copy()
    oof_df["p"], oof_df["y"] = oof, y
    oof_df.to_csv(work / "oof_predictions.tsv", sep="\t", index=False)

    s1_ids = list(s1.entity_id)
    best = tune(oof_df, truth, s1_ids, one_owner)
    print(f"== tuned flat-threshold decode on OOF: {best}")
    base = macro_f05(decode(oof_df, 0.5, one_owner), truth, s1_ids)
    print(f"   (F0.5 at fixed thr 0.5 = {base:.4f})")

    # ---- Phase 4: entity-level expected-F0.5 decoder, compared against the flat
    # threshold on the SAME out-of-fold predictions. Keep whichever wins - the
    # extra complexity only earns its place if it actually beats the baseline. ----
    print("== Phase 4: calibrating + comparing expected-F0.5 decoder")
    iso = calibrate(oof_df)
    score_v2 = macro_f05(decode_v2(oof_df, iso, one_owner), truth, s1_ids)
    print(f"   expected-F0.5 entity-level decode OOF score: {score_v2:.4f} "
          f"vs flat-threshold {best['score']:.4f}")
    use_v2 = score_v2 > best["score"]
    decode_method = "expected_f05" if use_v2 else "flat_threshold"
    cv_f05_used = score_v2 if use_v2 else best["score"]
    print(f"   -> using '{decode_method}' decoder (OOF F0.5 = {cv_f05_used:.4f})")

    report = {"ground_truth": gt_stats, "blocking": stats, "oof": best,
              "oof_expected_f05": score_v2, "decode_method": decode_method, "loco": {}}
    countries = s1.country.str.strip().str.lower()
    loco_scores = []
    if not a.skip_loco and countries.nunique() > 1:
        print("== leave-one-country-out (proxy for the unseen France test country)")
        s1c = pairs.s1_id.map(dict(zip(s1.entity_id, countries))).values
        for ctry in sorted(countries.unique()):
            te = s1c == ctry
            if te.sum() == 0 or (~te).sum() == 0:
                continue
            m = fit_with_self_neg(X[~te], y[~te])
            d = pairs.loc[te, ["s1_id", "cand_id", "src"]].copy()
            d["p"] = m.predict_proba(X[te])[:, 1]
            ids = list(s1.entity_id[(countries == ctry).values])
            sc_fixed = macro_f05(decode(d, best["thr"], one_owner, best["margin"]), truth, ids)
            sc_own = tune(d, truth, ids, one_owner)
            report["loco"][ctry] = {"F0.5_at_global_thr": sc_fixed, "best_possible": sc_own}
            loco_scores.append(sc_fixed)
            print(f"  held-out {ctry}: F0.5 at tuned thr = {sc_fixed:.4f} | oracle = {sc_own}")
        # Phase 6 groundwork: flag (don't yet hard-block) a large CV-vs-LOCO gap,
        # the concrete symptom of features that don't survive an unseen country.
        gap = cv_f05_used - min(loco_scores) if loco_scores else 0.0
        report["loco_min_gap"] = gap
        if gap > 0.10:
            print(f"WARNING: CV F0.5 ({cv_f05_used:.4f}) exceeds the worst held-out "
                  f"country by {gap:.3f}. This is the exact symptom of features that "
                  f"won't transfer to France - inspect feature importances and the "
                  f"per-country feature distributions before trusting the CV score.")

    print("== final fit on all training pairs")
    model = fit_with_self_neg(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(imp.head(15).to_string())
    joblib.dump({
        "model": model, "features": list(X.columns), "thr": best["thr"],
        "margin": best["margin"], "one_owner": one_owner, "blocking": asdict(cfg),
        "uses_ce": bool(a.ce_scores), "decode_method": decode_method,
        "calibrator": iso if use_v2 else None,
    }, mdir / "bundle.joblib")
    (work / "train_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"saved {mdir/'bundle.joblib'}")

    log_experiment(work, {
        "exp_id": exp_id, "timestamp": exp_id, "blocking_cfg": json.dumps(asdict(cfg)),
        "n_self_neg": n_self, "decode_method": decode_method,
        "cv_f05_flat": round(best["score"], 4), "cv_f05_expected": round(score_v2, 4),
        "cv_f05_used": round(cv_f05_used, 4),
        "loco_min_gap": round(report.get("loco_min_gap", 0.0), 4),
        "blocking_recall": round(stats["pair_recall"], 4),
        "n_train_pairs": len(X), "n_features": X.shape[1], "notes": a.notes,
    })


if __name__ == "__main__":
    main()
