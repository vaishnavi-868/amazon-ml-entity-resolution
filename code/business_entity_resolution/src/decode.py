"""Turn pair probabilities into final match lists, and score them (macro F0.5)."""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .config import MARGIN_GRID, THRESHOLD_GRID


def select_candidates(df, thr=None, calibrator=None, n_mc=400):
    """Per-entity candidate selection WITHOUT cross-entity (one-owner)
    resolution. Split out from decode()/decode_v2() so the batched pipeline can
    call this once per S1 batch (bounded memory) and defer the one-owner
    resolution - which needs to compare across ALL S1 entities that want a given
    S2/S3 record, including ones in other batches - to a single global pass
    (apply_one_owner) over the much smaller set of tentatively-selected pairs.
    Returns DataFrame[s1_id, cand_id, score]. Pass `thr` for the flat-threshold
    decoder, or `calibrator` (from calibrate()) for the expected-F0.5 decoder."""
    if calibrator is not None:
        d = df.copy()
        d["score"] = calibrator.predict(d.p.values)
        selected = []
        for s1_id, g in d.sort_values("score", ascending=False).groupby("s1_id", sort=False):
            k = expected_f05_best_k(g.score.values, n_mc=n_mc)
            for c, sc in zip(g.cand_id.values[:k], g.score.values[:k]):
                selected.append((s1_id, c, sc))
        return pd.DataFrame(selected, columns=["s1_id", "cand_id", "score"])
    d = df[df.p >= thr]
    return d.rename(columns={"p": "score"})[["s1_id", "cand_id", "score"]]


def apply_one_owner(tentative, margin=0.0):
    """Global cross-entity conflict resolution over a (possibly batch-assembled)
    tentative-selections table: each S2/S3 record ends up owned by its single
    highest-scoring S1 entity, full stop - regardless of which batch either side
    came from. margin > 0: if the runner-up S1 for a record is within `margin`
    of the winner, drop the record entirely (too ambiguous - precision over
    recall). Returns {s1_id: [cand_id, ...]}."""
    if not len(tentative):
        return defaultdict(list)
    d = tentative.sort_values("score", ascending=False)
    rank = d.groupby("cand_id").cumcount()
    top = d[rank == 0]
    if margin > 0:
        second = d[rank == 1].set_index("cand_id").score
        gap = top.cand_id.map(second)
        top = top[gap.isna() | ((top.score - gap) > margin).values]
    out = defaultdict(list)
    for s, c in zip(top.s1_id.values, top.cand_id.values):
        out[s].append(c)
    return out


def _to_dict(tentative):
    out = defaultdict(list)
    for s, c in zip(tentative.s1_id.values, tentative.cand_id.values):
        out[s].append(c)
    return out


def decode(df, thr, one_owner=True, margin=0.0):
    """df columns: s1_id, cand_id, p.  Returns {s1_id: [cand_id, ...]}.
    Thin wrapper over select_candidates()/apply_one_owner() for the in-memory
    (non-batched) case; see those two for the streaming/batched equivalent."""
    tentative = select_candidates(df, thr=thr)
    return apply_one_owner(tentative, margin) if one_owner else _to_dict(tentative)


def f05_single(pred, true, beta2=0.25):
    if not pred and not true:
        return 1.0                      # correct singleton
    if not pred or not true:
        return 0.0                      # false merge on singleton / missed everything
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + beta2) * p * r / (beta2 * p + r)


def macro_f05(pred, truth, s1_ids):
    return float(np.mean([f05_single(set(pred.get(s, ())), truth.get(s, set()))
                          for s in s1_ids]))


def calibrate(oof_df):
    """Phase 4: fit p(true match) from raw model scores on OUT-OF-FOLD predictions
    only (never on training-fold scores, which are overconfident). Isotonic
    regression is monotone, so it preserves ranking (and therefore recall) while
    making probabilities usable directly as "expected number of true matches"."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_df.p.values, oof_df.y.values)
    return iso


def expected_f05_best_k(probs_desc, n_mc=400, beta2=0.25, seed=0):
    """Given one S1 entity's candidate probabilities (already sorted descending),
    pick the prediction-set size k (0 = predict nothing / singleton) that maximises
    E[F0.5] under a Monte-Carlo simulation of which candidates are true matches.
    This is the entity-level decision the metric actually rewards: F0.5 is scored
    per S1 entity and a correct empty prediction on a true singleton scores 1.0, so
    a flat global threshold is provably suboptimal versus this per-entity choice."""
    n = len(probs_desc)
    if n == 0:
        return 0
    rng = np.random.default_rng(seed)
    draws = rng.random((n_mc, n)) < np.asarray(probs_desc)[None, :]
    n_true = draws.sum(1)
    best_k, best_val = 0, float(np.mean(n_true == 0))     # k=0: correct iff no true match
    for k in range(1, n + 1):
        tp = draws[:, :k].sum(1)
        prec = tp / k
        rec = np.divide(tp, n_true, out=np.zeros_like(tp, dtype=float), where=n_true > 0)
        f = np.where(tp > 0, (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-12), 0.0)
        val = float(f.mean())
        if val > best_val:
            best_k, best_val = k, val
    return best_k


def decode_v2(df, calibrator=None, one_owner=True, n_mc=400):
    """Entity-level expected-F0.5 decoder (replaces the flat-threshold `decode`).
    Thin wrapper over select_candidates()/apply_one_owner(), same relationship
    as decode() above."""
    tentative = select_candidates(df, calibrator=calibrator, n_mc=n_mc)
    return apply_one_owner(tentative) if one_owner else _to_dict(tentative)


def tune(oof, truth, s1_ids, one_owner, thr_grid=THRESHOLD_GRID, margins=MARGIN_GRID):
    """Grid-search threshold (and margin) on out-of-fold predictions."""
    best = (-1.0, None, None)
    for m in (margins if one_owner else [0.0]):
        for t in thr_grid:
            sc = macro_f05(decode(oof, t, one_owner, m), truth, s1_ids)
            if sc > best[0]:
                best = (sc, t, m)
    return {"score": best[0], "thr": best[1], "margin": best[2]}
