"""Batched, memory-bounded training/inference pipeline for datasets too large
for the original in-memory train.py/infer.py (millions of S1 entities /
multi-million-row candidate corpus).

Architecture:
    S1 batch -> blocking against a resident CandidateIndex (candidate_index.py)
    -> feature computation for just that batch's (small) candidate set ->
    accumulate/write results -> discard batch RAM -> next batch.

The candidate corpus (S2+S3) is represented ONCE (cheap string columns + TF-IDF
matrices, held resident throughout) - this is the one genuinely corpus-sized
cost and is unavoidable regardless of how S1 is batched, since every S1 batch
needs to query against the full corpus. What batching removes is (a) ever
materializing normalize.prepare()'s expensive per-row Python objects for the
full multi-million-row corpus at once (candidate_index.py computes those only
for the small number of rows actually selected as candidates), and (b) ever
holding the full S1 x candidate pair/feature table in memory at once.

Ground-truth positives are force-included in each S1's candidate set (via
generate_candidates_batch's force_ids) so blocking can never silently cost you
training recall.

TRAINING: builds a bounded training sample (all positives + capped sampled
negatives per S1) across all batches, then runs the same GroupKFold CV /
threshold-tuning / calibration / decoder-comparison / LOCO / final-fit code as
train.py, just on this sample instead of the full pair set - a large enough
sample is sufficient to tune a threshold or fit a boosted-tree classifier; you
do not need every negative pair for that.

INFERENCE: streams every S1 batch's scored candidates through select_candidates
(decode.py) into a temporary tentative-matches file, then does ONE global
apply_one_owner pass at the end - necessary because a S2/S3 record can be
competed for by S1 entities in different batches.

NOT YET SUPPORTED in this path (use train.py/infer.py, or ask for these to be
ported): dense embedding blocking (g_dense), S1-vs-S1 hard negatives, the
cross-encoder feature. None of these require changing the file formats used
here, so they can be added later without disruption.

    python -m src.batch_pipeline --mode train --data-dir s3://bucket/dataset --work-dir work --model-dir models
    python -m src.batch_pipeline --mode infer --data-dir s3://bucket/dataset --work-dir work --model-dir models --out-dir output
"""
import argparse
import csv
import gc
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .blocking import generate_candidates_batch, prep_features_for_batch
from .candidate_index import CandidateIndex
from .config import LGB_PARAMS, BlockingConfig
from .decode import apply_one_owner, calibrate, decode, decode_v2, macro_f05, select_candidates, tune
from .features import build_features
from .io_utils import candidate_table, load_split, write_id_lists
from .normalize import prepare_heavy, prepare_light


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def new_model():
    return lgb.LGBMClassifier(**LGB_PARAMS)


def batch_ranges(n, size):
    for s in range(0, n, size):
        yield s, min(s + size, n)


def prepare_s1_batch(s1_raw_batch, cidx):
    light = prepare_light(s1_raw_batch)
    heavy = prepare_heavy(light)
    X1 = {c: cidx.vecs[c].transform(light[c]) for c in ("name_core", "joint_txt", "addr_norm")}
    return light, heavy, X1


def append_id_lists_streaming(path, s1_ids_batch, cand_id_lists_batch, col_name, is_first):
    """Streaming per-batch append for a potentially huge id-list TSV (e.g.
    candidate_pairs.tsv can be up to n_s1 * max_cands rows worth of ids). Never
    holds more than one batch in memory."""
    with open(path, "w" if is_first else "a", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        if is_first:
            w.writerow(["source1_entity_id", col_name])
        for sid, ids in zip(s1_ids_batch, cand_id_lists_batch):
            w.writerow([sid, ",".join(sorted(set(ids)))])


def cap_negatives_per_entity(X, y, s1_ids, cand_ids, neg_per_s1, rng):
    """Keep all positives + at most `neg_per_s1` randomly-sampled negatives per
    S1 entity, bounding the training-sample size regardless of corpus size."""
    df = pd.DataFrame({"s1_id": s1_ids, "y": y})
    df["row"] = np.arange(len(df))
    keep_rows = [df.loc[df.y == 1, "row"].to_numpy()]
    neg = df[df.y == 0]
    if len(neg):
        sampled = (neg.groupby("s1_id", sort=False)
                  .apply(lambda g: g.sample(min(len(g), neg_per_s1), random_state=rng.integers(1 << 30)))
                  ["row"].to_numpy())
        keep_rows.append(sampled)
    keep = np.concatenate(keep_rows)
    return (X.iloc[keep].reset_index(drop=True), y[keep],
           np.asarray(s1_ids)[keep], np.asarray(cand_ids)[keep])


def log_experiment(work_dir, row):
    cols = ["exp_id", "timestamp", "mode", "s1_batch_size", "n_s1", "n_train_sample",
            "decode_method", "cv_f05_flat", "cv_f05_expected", "cv_f05_used",
            "loco_min_gap", "pair_recall", "notes"]
    path = Path(work_dir) / "experiments.csv"
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if is_new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})
    log(f"logged experiment {row.get('exp_id')} -> {path}")


def run_train(a):
    exp_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work, mdir = Path(a.work_dir), Path(a.model_dir)
    work.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    cfg = BlockingConfig(use_dense=False, max_cands=a.max_cands)

    log("loading train data")
    s1, s2, s3, truth = load_split(a.data_dir, "train")
    cands = candidate_table(s2, s3)
    n_s1 = len(s1)
    log(f"S1={n_s1:,}  S2+S3 candidates={len(cands):,}")

    vocab_sample = s1.sample(min(len(s1), 200_000), random_state=0)
    cidx = CandidateIndex(cands, cfg, s1_sample_for_vocab=vocab_sample, log=log)
    del cands
    gc.collect()

    owner_count = {}
    for cs in truth.values():
        for c in cs:
            owner_count[c] = owner_count.get(c, 0) + 1
    one_owner = (np.mean([v > 1 for v in owner_count.values()]) if owner_count else 0.0) <= 0.01
    log(f"one_owner decoding = {one_owner}")

    rng = np.random.default_rng(0)
    sample_X, sample_y, sample_s1, sample_cand = [], [], [], []
    tp_total, gt_total = 0, 0
    n_batches = (n_s1 + a.s1_batch_size - 1) // a.s1_batch_size
    for bi, (s, e) in enumerate(batch_ranges(n_s1, a.s1_batch_size)):
        batch_raw = s1.iloc[s:e].reset_index(drop=True)
        light, heavy, X1 = prepare_s1_batch(batch_raw, cidx)
        force_ids = {sid: truth[sid] for sid in batch_raw.entity_id if truth.get(sid)}

        pairs = generate_candidates_batch(light, heavy, X1, cidx, cfg, force_ids=force_ids)
        got = set(zip(pairs.s1_id, pairs.cand_id))
        for sid, cs in force_ids.items():
            gt_total += len(cs)
            tp_total += sum(1 for c in cs if (sid, c) in got)

        r, remapped = prep_features_for_batch(pairs, heavy, X1, cidx)
        X_batch = build_features(r, remapped, n_jobs=a.n_jobs)
        key = pairs.s1_id.astype(str) + "|" + pairs.cand_id.astype(str)
        true_keys = {f"{sid}|{c}" for sid, cs in force_ids.items() for c in cs}
        y_batch = key.isin(true_keys).astype(int).to_numpy()

        if rng.random() < a.train_s1_frac:
            Xb, yb, sb, cb = cap_negatives_per_entity(
                X_batch, y_batch, pairs.s1_id.values, pairs.cand_id.values, a.neg_per_s1, rng)
            sample_X.append(Xb)
            sample_y.append(yb)
            sample_s1.append(sb)
            sample_cand.append(cb)

        if (bi + 1) % max(1, a.log_every) == 0 or bi + 1 == n_batches:
            n_sample_rows = sum(len(x) for x in sample_X)
            log(f"batch {bi+1}/{n_batches}  pairs={len(pairs):,}  "
               f"recall_so_far={tp_total}/{gt_total}  sample_rows={n_sample_rows:,}")
        del batch_raw, light, heavy, X1, pairs, r, remapped, X_batch, key, true_keys, y_batch
        if (bi + 1) % 20 == 0:
            gc.collect()

    pair_recall = tp_total / max(gt_total, 1)
    log(f"blocking pair recall across all batches: {pair_recall:.4f} ({tp_total}/{gt_total})")
    if pair_recall < 0.97:
        log("WARNING: blocking recall < 97% - candidates are missing true matches "
           "before the model even sees them. Consider raising k_name/k_joint/max_cands "
           "in BlockingConfig, or lowering min_tfidf_sim.")

    X = pd.concat(sample_X, ignore_index=True)
    y = np.concatenate(sample_y)
    s1_ids = np.concatenate(sample_s1)
    cand_ids = np.concatenate(sample_cand)
    del sample_X, sample_y, sample_s1, sample_cand
    gc.collect()
    log(f"bounded training sample: {len(X):,} rows, {y.sum():,} positives ({y.mean():.3%})")

    log(f"{a.folds}-fold grouped CV")
    oof = np.zeros(len(X))
    for k, (tr, va) in enumerate(GroupKFold(n_splits=a.folds).split(X, y, s1_ids)):
        m = new_model().fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        log(f"  fold {k} done")
    oof_df = pd.DataFrame({"s1_id": s1_ids, "cand_id": cand_ids, "p": oof, "y": y})
    sample_ids = list(pd.unique(s1_ids))

    best = tune(oof_df, truth, sample_ids, one_owner=one_owner)
    log(f"tuned flat-threshold decode on sample OOF: {best}")

    iso = calibrate(oof_df)
    score_v2 = macro_f05(decode_v2(oof_df, iso, one_owner=one_owner), truth, sample_ids)
    log(f"expected-F0.5 decode on sample OOF: {score_v2:.4f} vs flat-threshold {best['score']:.4f}")
    use_v2 = score_v2 > best["score"]
    decode_method = "expected_f05" if use_v2 else "flat_threshold"
    cv_f05_used = score_v2 if use_v2 else best["score"]
    log(f"using '{decode_method}' decoder (sample OOF F0.5 = {cv_f05_used:.4f})")

    report = {"n_s1": n_s1, "n_train_sample": len(X), "pair_recall": pair_recall,
             "oof_flat": best, "oof_expected_f05": score_v2, "decode_method": decode_method,
             "loco": {}}
    loco_scores = []
    countries = pd.Series(s1.set_index("entity_id").country.str.lower().reindex(s1_ids).values)
    if not a.skip_loco and countries.nunique() > 1:
        log("leave-one-country-out on the bounded sample")
        for ctry in sorted(countries.dropna().unique()):
            te = (countries == ctry).to_numpy()
            if te.sum() == 0 or (~te).sum() == 0:
                continue
            m = new_model().fit(X[~te], y[~te])
            d = pd.DataFrame({"s1_id": s1_ids[te], "cand_id": cand_ids[te],
                              "p": m.predict_proba(X[te])[:, 1]})
            ids = list(pd.unique(s1_ids[te]))
            sc = macro_f05(decode(d, best["thr"], one_owner, best["margin"]), truth, ids)
            report["loco"][ctry] = sc
            loco_scores.append(sc)
            log(f"  held-out {ctry}: F0.5 = {sc:.4f}")
        gap = cv_f05_used - min(loco_scores) if loco_scores else 0.0
        report["loco_min_gap"] = gap
        if gap > 0.10:
            log(f"WARNING: sample CV F0.5 ({cv_f05_used:.4f}) exceeds the worst held-out "
               f"country by {gap:.3f} - features may not transfer to an unseen country (France).")

    log("final fit on the full bounded sample")
    model = new_model().fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    log("top features:\n" + imp.head(15).to_string())

    joblib.dump({
        "model": model, "features": list(X.columns), "thr": best["thr"],
        "margin": best["margin"], "one_owner": one_owner, "blocking": asdict(cfg),
        "decode_method": decode_method, "calibrator": iso if use_v2 else None,
        "batched": True,
    }, mdir / "bundle.joblib")
    (work / "train_report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"saved {mdir/'bundle.joblib'}")

    log_experiment(work, {
        "exp_id": exp_id, "timestamp": exp_id, "mode": "train",
        "s1_batch_size": a.s1_batch_size, "n_s1": n_s1, "n_train_sample": len(X),
        "decode_method": decode_method, "cv_f05_flat": round(best["score"], 4),
        "cv_f05_expected": round(score_v2, 4), "cv_f05_used": round(cv_f05_used, 4),
        "loco_min_gap": round(report.get("loco_min_gap", 0.0), 4),
        "pair_recall": round(pair_recall, 4), "notes": a.notes,
    })


def run_infer(a):
    work, mdir, out = Path(a.work_dir), Path(a.model_dir), Path(a.out_dir)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(mdir / "bundle.joblib")
    cfg = BlockingConfig(**bundle["blocking"])

    log("loading test data")
    s1, s2, s3, _ = load_split(a.data_dir, "test")
    cands = candidate_table(s2, s3)
    n_s1 = len(s1)
    log(f"test S1={n_s1:,}  S2+S3 candidates={len(cands):,}  "
       f"countries={sorted(s1.country.str.lower().unique())}")

    vocab_sample = s1.sample(min(len(s1), 200_000), random_state=0)
    cidx = CandidateIndex(cands, cfg, s1_sample_for_vocab=vocab_sample, log=log)
    del cands
    gc.collect()

    cand_path = out / "candidate_pairs.tsv"
    tentative_path = work / "tentative_matches.tsv"
    tentative_is_first = True
    n_batches = (n_s1 + a.s1_batch_size - 1) // a.s1_batch_size
    for bi, (s, e) in enumerate(batch_ranges(n_s1, a.s1_batch_size)):
        batch_raw = s1.iloc[s:e].reset_index(drop=True)
        light, heavy, X1 = prepare_s1_batch(batch_raw, cidx)
        pairs = generate_candidates_batch(light, heavy, X1, cidx, cfg)

        grouped = pairs.groupby("s1_id").cand_id.apply(list)
        grouped = grouped.reindex(batch_raw.entity_id, fill_value=[])
        append_id_lists_streaming(cand_path, grouped.index, grouped.values,
                                  "candidate_entity_ids", is_first=(bi == 0))

        if len(pairs):
            r, remapped = prep_features_for_batch(pairs, heavy, X1, cidx)
            X_batch = build_features(r, remapped, n_jobs=a.n_jobs)
            scored = pairs[["s1_id", "cand_id"]].copy()
            scored["p"] = bundle["model"].predict_proba(X_batch)[:, 1]
            if bundle["decode_method"] == "expected_f05":
                tentative = select_candidates(scored, calibrator=bundle["calibrator"])
            else:
                tentative = select_candidates(scored, thr=bundle["thr"])
            mode = "w" if tentative_is_first else "a"
            tentative.to_csv(tentative_path, sep="\t", index=False,
                             mode=mode, header=tentative_is_first)
            tentative_is_first = False

        if (bi + 1) % max(1, a.log_every) == 0 or bi + 1 == n_batches:
            log(f"batch {bi+1}/{n_batches} scored and appended")
        del batch_raw, light, heavy, X1, pairs, grouped
        if (bi + 1) % 20 == 0:
            gc.collect()

    log("global one-owner resolution across all batches")
    if tentative_path.exists():
        tentative_all = pd.read_csv(tentative_path, sep="\t", dtype={"s1_id": str, "cand_id": str})
        margin = bundle["margin"] if bundle["decode_method"] == "flat_threshold" else 0.0
        match_map = apply_one_owner(tentative_all, margin=margin)
    else:
        match_map = {}
    write_id_lists(out / "matching_results.tsv", s1.entity_id, match_map, "matched_entity_ids")

    country = dict(zip(s1.entity_id, s1.country.str.lower()))
    cand_counts = pd.read_csv(cand_path, sep="\t", dtype=str)
    n_cands_map = dict(zip(cand_counts.source1_entity_id,
                          cand_counts.candidate_entity_ids.fillna("").apply(
                              lambda s: 0 if not s else len(s.split(",")))))
    summ = pd.DataFrame({
        "country": [country[s] for s in s1.entity_id],
        "n_cands": [n_cands_map.get(s, 0) for s in s1.entity_id],
        "n_matches": [len(match_map.get(s, [])) for s in s1.entity_id],
    })
    summ["has_match"] = summ.n_matches > 0
    log(f"decode_method={bundle['decode_method']} margin={bundle['margin']}")
    log("\n" + summ.groupby("country").agg(
        n=("has_match", "size"), match_rate=("has_match", "mean"),
        avg_matches=("n_matches", "mean"), avg_cands=("n_cands", "mean")).round(3).to_string())
    log("-> compare France's match_rate/avg_matches with US/India; a large gap "
       "means the threshold or features do not transfer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "infer"], required=True)
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--s1-batch-size", type=int, default=5000)
    ap.add_argument("--max-cands", type=int, default=80)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--neg-per-s1", type=int, default=5,
                    help="max sampled negatives kept per S1 entity in the training sample")
    ap.add_argument("--train-s1-frac", type=float, default=1.0,
                    help="fraction of S1 batches whose pairs enter the training sample "
                         "(coarse subsampling to bound sample size further; every S1 "
                         "batch is still fully blocked/scored regardless)")
    ap.add_argument("--skip-loco", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-s1", type=int, default=None,
                    help="process only the first N S1 entities - for a quick throughput/"
                         "sanity check on real data before committing to a full run")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    if a.mode == "train":
        run_train(a)
    else:
        run_infer(a)


if __name__ == "__main__":
    main()