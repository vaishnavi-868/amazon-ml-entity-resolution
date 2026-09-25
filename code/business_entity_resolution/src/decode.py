"""Turn pair probabilities into final match lists, and score them (macro F0.5)."""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .config import MARGIN_GRID, THRESHOLD_GRID


def decode(df, thr, one_owner=True, margin=0.0):
    """df columns: s1_id, cand_id, p.  Returns {s1_id: [cand_id, ...]}.

    1. keep pairs with p >= thr
    2. one_owner: a S2/S3 record is assigned to its single best S1 entity
       (only valid if the training data shows records are never shared between
       S1 entities - train.py checks this and stores the decision)
    3. margin > 0: if the runner-up S1 for the same record is within `margin`
       of the winner, the record is ambiguous -> drop it (precision over recall)."""
    d = df[df.p >= thr]
    if one_owner and len(d):
        d = d.sort_values("p", ascending=False)
        rank = d.groupby("cand_id").cumcount()
        top = d[rank == 0]
        if margin > 0:
            second = d[rank == 1].set_index("cand_id").p
            gap = top.cand_id.map(second)
            top = top[gap.isna() | ((top.p - gap) > margin).values]
        d = top
    out = defaultdict(list)
    for s, c in zip(d.s1_id.values, d.cand_id.values):
        out[s].append(c)
    return out


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
    df columns: s1_id, cand_id, p. `calibrator` is the output of `calibrate()`;
    pass None to use raw probabilities directly (only sensible if they are already
    calibrated, e.g. from a model trained with a proper scoring loss)."""
    d = df.copy()
    d["pc"] = calibrator.predict(d.p.values) if calibrator is not None else d.p.values
    selected = []
    for s1_id, g in d.sort_values("pc", ascending=False).groupby("s1_id", sort=False):
        k = expected_f05_best_k(g.pc.values, n_mc=n_mc)
        for c, pc in zip(g.cand_id.values[:k], g.pc.values[:k]):
            selected.append((s1_id, c, pc))
    sel = pd.DataFrame(selected, columns=["s1_id", "cand_id", "pc"])
    if one_owner and len(sel):
        # a S2/S3 record can only end up owned by its single best-scoring S1 entity
        sel = sel.sort_values("pc", ascending=False).drop_duplicates("cand_id", keep="first")
    out = defaultdict(list)
    for s, c in zip(sel.s1_id, sel.cand_id):
        out[s].append(c)
    return out


def tune(oof, truth, s1_ids, one_owner, thr_grid=THRESHOLD_GRID, margins=MARGIN_GRID):
    """Grid-search threshold (and margin) on out-of-fold predictions."""
    best = (-1.0, None, None)
    for m in (margins if one_owner else [0.0]):
        for t in thr_grid:
            sc = macro_f05(decode(oof, t, one_owner, m), truth, s1_ids)
            if sc > best[0]:
                best = (sc, t, m)
    return {"score": best[0], "thr": best[1], "margin": best[2]}
