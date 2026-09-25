"""Inference on the test set -> output/matching_results.tsv + output/candidate_pairs.tsv

    python -m src.infer --data-dir dataset --work-dir work --model-dir models --out-dir output
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from .blocking import generate_candidates
from .config import BlockingConfig
from .decode import decode, decode_v2
from .features import attach_ce, build_features
from .io_utils import candidate_table, load_split, write_id_lists
from .representations import build_reps


def validate(match, cand, s1_ids, all_cand_ids):
    """Mirror of the challenge's rules (the official validator is still the referee)."""
    errs = []
    for name, m in (("matching", match), ("candidate", cand)):
        if set(m) - set(s1_ids):
            errs.append(f"{name}: unknown S1 ids")
        for s, ids in m.items():
            if len(ids) != len(set(ids)):
                errs.append(f"{name}: duplicate ids for {s}")
            bad = [x for x in ids if x not in all_cand_ids]
            if bad:
                errs.append(f"{name}: {s} has ids not in test S2/S3: {bad[:3]}")
    for s, ids in match.items():
        if set(ids) - set(cand.get(s, [])):
            errs.append(f"matching not subset of candidates for {s}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--ce-scores", default=None)
    ap.add_argument("--thr", type=float, default=None, help="override tuned threshold")
    a = ap.parse_args()

    out, work = Path(a.out_dir), Path(a.work_dir)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(Path(a.model_dir) / "bundle.joblib")
    cfg = BlockingConfig(**bundle["blocking"])       # identical blocking to training
    if bundle["uses_ce"] and not a.ce_scores:
        raise SystemExit("model was trained with cross-encoder scores: pass --ce-scores")

    s1, s2, s3, _ = load_split(a.data_dir, "test")
    cands = candidate_table(s2, s3)
    print(f"test: S1={len(s1)} S2={len(s2)} S3={len(s3)}; "
          f"countries={sorted(s1.country.str.lower().unique())}")

    r = build_reps(s1, cands, cfg, work, "test")
    pairs = generate_candidates(r, cfg)
    # ---- this exact DataFrame is what the model scores: save it as candidate_pairs ----
    cand_map = pairs.groupby("s1_id").cand_id.apply(list).to_dict()
    write_id_lists(out / "candidate_pairs.tsv", s1.entity_id, cand_map, "candidate_entity_ids")
    pairs.to_pickle(work / "test_pairs.pkl")
    if a.ce_scores:
        pairs = attach_ce(pairs, a.ce_scores)

    X = build_features(r, pairs, n_jobs=a.n_jobs).reindex(columns=bundle["features"])
    scored = pairs[["s1_id", "cand_id", "src"]].copy()
    scored["p"] = bundle["model"].predict_proba(X)[:, 1]

    decode_method = bundle.get("decode_method", "flat_threshold")
    if a.thr is not None:                          # explicit override always wins
        match_map = decode(scored, a.thr, bundle["one_owner"], bundle["margin"])
        print(f"decode: flat_threshold (overridden thr={a.thr})")
    elif decode_method == "expected_f05":
        match_map = decode_v2(scored, bundle["calibrator"], bundle["one_owner"])
        print("decode: expected_f05 (Phase 4 entity-level decoder, chosen on OOF at train time)")
    else:
        match_map = decode(scored, bundle["thr"], bundle["one_owner"], bundle["margin"])
        print(f"decode: flat_threshold (thr={bundle['thr']})")
    write_id_lists(out / "matching_results.tsv", s1.entity_id, match_map, "matched_entity_ids")

    errs = validate(match_map, cand_map, list(s1.entity_id), set(cands.entity_id))
    print("self-check:", "PASS" if not errs else errs[:10])

    country = dict(zip(s1.entity_id, s1.country.str.lower()))
    summ = pd.DataFrame({"country": [country[s] for s in s1.entity_id],
                         "n_cands": [len(cand_map.get(s, [])) for s in s1.entity_id],
                         "n_matches": [len(match_map.get(s, [])) for s in s1.entity_id]})
    summ["has_match"] = summ.n_matches > 0
    print("decode_method =", decode_method, "| margin =", bundle["margin"],
          "| one_owner =", bundle["one_owner"])
    print(summ.groupby("country").agg(n=("has_match", "size"), match_rate=("has_match", "mean"),
                                      avg_matches=("n_matches", "mean"),
                                      avg_cands=("n_cands", "mean")).round(3).to_string())
    print("-> compare France's match_rate/avg_matches with US/India; a large gap "
          "means the threshold or features do not transfer.")


if __name__ == "__main__":
    main()
