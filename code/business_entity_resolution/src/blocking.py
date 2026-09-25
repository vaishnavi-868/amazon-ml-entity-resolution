"""Candidate generation = union of four generators, capped per S1 entity.

  g_name  : char 3-4gram TF-IDF kNN on the core name   (typos, transliteration)
  g_joint : char 3-4gram TF-IDF kNN on name + address
  g_dense : multilingual embedding kNN (optional)      (semantic / word-order)
  g_key   : inverted-index keys: rare name tokens, consonant skeletons,
            postal-code + name-token                   (cheap exact recall)

kNN is run separately for S2 and S3 so one source cannot crowd out the other.
The returned DataFrame is *exactly* what the matching model scores, and is what
gets written to candidate_pairs.tsv."""
from collections import defaultdict

import numpy as np
import pandas as pd

from .representations import rowdot

FLAGS = ["g_name", "g_joint", "g_dense", "g_key"]


def _topk(Q, D, k, chunk=256):
    n, k = Q.shape[0], min(k, D.shape[0])
    idx = np.empty((n, k), dtype=np.int64)
    sc = np.empty((n, k), dtype=np.float32)
    sparse = hasattr(Q, "multiply")
    DT = D.T.tocsc() if sparse else D.T
    for s in range(0, n, chunk):
        sim = Q[s:s + chunk] @ DT
        sim = sim.toarray() if sparse else sim
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(sim, part, 1)
        order = np.argsort(-ps, axis=1)
        idx[s:s + chunk] = np.take_along_axis(part, order, 1)
        sc[s:s + chunk] = np.take_along_axis(ps, order, 1)
    return idx, sc


def _knn_pairs(Q, D, src, k, min_sim, flag):
    """kNN for each source separately; returns DataFrame(i, j, flag)."""
    parts = []
    for s in ("S2", "S3"):
        cols = np.where(src == s)[0]
        if len(cols) == 0:
            continue
        idx, sc = _topk(Q, D[cols], k)
        ii = np.repeat(np.arange(Q.shape[0]), idx.shape[1])
        jj = cols[idx.ravel()]
        keep = sc.ravel() > min_sim
        parts.append(pd.DataFrame({"i": ii[keep], "j": jj[keep], flag: 1}))
    return pd.concat(parts, ignore_index=True)


def _keys(core, postal, skel_set, skel_first):
    ks = set()
    for t in core:
        if len(t) >= 3:
            ks.add("t:" + t)
            if postal:
                ks.add(f"pt:{postal}:{t}")
    for t in skel_set:
        if len(t) >= 3:
            ks.add("s:" + t)
    if postal and skel_first:
        ks.add(f"ps:{postal}:{skel_first}")
    return ks


def _key_pairs(r, cfg):
    index = defaultdict(list)
    for j, args in enumerate(zip(r.c.core, r.c.postal, r.c.skel_set, r.c.skel_first)):
        for k in _keys(*args):
            index[k].append(j)
    ii, jj = [], []
    for i, args in enumerate(zip(r.s1.core, r.s1.postal, r.s1.skel_set, r.s1.skel_first)):
        for k in _keys(*args):
            lst = index.get(k)
            if not lst:
                continue
            limit = cfg.key_rare_df if k[:2] in ("t:", "s:") else cfg.key_max_block
            if len(lst) <= limit:
                ii.extend([i] * len(lst))
                jj.extend(lst)
    return pd.DataFrame({"i": np.array(ii, dtype=np.int64),
                         "j": np.array(jj, dtype=np.int64), "g_key": 1})


def generate_candidates(r, cfg) -> pd.DataFrame:
    src = r.c.src.values
    parts = [
        _knn_pairs(r.X1_name, r.Xc_name, src, cfg.k_name, cfg.min_tfidf_sim, "g_name"),
        _knn_pairs(r.X1_joint, r.Xc_joint, src, cfg.k_joint, cfg.min_tfidf_sim, "g_joint"),
        _key_pairs(r, cfg),
    ]
    if r.dense:
        parts.append(_knn_pairs(r.E1_joint, r.Ec_joint, src, cfg.k_dense, -1.0, "g_dense"))

    pairs = pd.concat(parts, ignore_index=True)
    for f in FLAGS:
        if f not in pairs:
            pairs[f] = 0
    pairs[FLAGS] = pairs[FLAGS].fillna(0).astype(np.int8)
    pairs = pairs.groupby(["i", "j"], as_index=False)[FLAGS].max()

    i, j = pairs.i.values, pairs.j.values
    pairs["cos_name"] = rowdot(r.X1_name, r.Xc_name, i, j)
    pairs["cos_joint"] = rowdot(r.X1_joint, r.Xc_joint, i, j)
    score = [pairs.cos_name, pairs.cos_joint]
    if r.dense:
        pairs["emb_joint"] = rowdot(r.E1_joint, r.Ec_joint, i, j)
        score.append(pairs.emb_joint)
    pairs["n_gens"] = pairs[FLAGS].sum(axis=1)
    pairs["base"] = sum(score) / len(score) + 0.03 * pairs.n_gens

    # final cap (part of blocking, so what we save == what the model scores)
    pairs = (pairs.sort_values(["i", "base"], ascending=[True, False])
             .groupby("i", sort=False).head(cfg.max_cands)
             .sort_values(["i", "base"], ascending=[True, False]).reset_index(drop=True))
    pairs["s1_id"] = r.s1.entity_id.values[pairs.i.values]
    pairs["cand_id"] = r.c.entity_id.values[pairs.j.values]
    pairs["src"] = src[pairs.j.values]
    return pairs


def s1_self_negative_pairs(r, k=8, min_sim=0.15):
    """Phase 2 (hard negatives): every pair of DISTINCT Source-1 entities is a
    guaranteed non-match (S1 is the deduplicated reference source). Lexically
    similar S1 pairs (near-duplicate names/addresses across different real
    businesses) are exactly the hard negatives the classifier needs to learn a
    tight decision boundary, and they cost nothing to label.

    Returns (pseudo_r, pairs) where pseudo_r has the same attribute shape as a
    normal Reps object (so build_features() works unmodified on it) but with
    both "sides" pointing at r.s1 / r.s1's own vectors."""
    import types
    idx, sc = _topk(r.X1_name, r.X1_name, k + 1)   # +1: index 0 is always self
    ii, jj = [], []
    for row in range(idx.shape[0]):
        for col, sim in zip(idx[row], sc[row]):
            if col != row and sim >= min_sim:
                ii.append(row)
                jj.append(int(col))
    i = np.array(ii, dtype=np.int64)
    j = np.array(jj, dtype=np.int64)
    keep = i < j                     # dedup unordered pairs, keep one direction
    i, j = i[keep], j[keep]

    pairs = pd.DataFrame({"i": i, "j": j})
    pairs["s1_id"] = r.s1.entity_id.values[i]
    pairs["cand_id"] = r.s1.entity_id.values[j]
    pairs["src"] = "S1"                                    # not S2/S3; harmless flag value
    pairs["g_name"] = 1
    for f in ("g_joint", "g_dense", "g_key"):
        pairs[f] = 0
    pairs["n_gens"] = 1
    pairs["cos_name"] = rowdot(r.X1_name, r.X1_name, i, j)
    pairs["cos_joint"] = rowdot(r.X1_joint, r.X1_joint, i, j)
    if r.dense:
        pairs["emb_joint"] = rowdot(r.E1_joint, r.E1_joint, i, j)

    pseudo_r = types.SimpleNamespace(
        s1=r.s1, c=r.s1,
        X1_name=r.X1_name, Xc_name=r.X1_name,
        X1_joint=r.X1_joint, Xc_joint=r.X1_joint,
        X1_addr=r.X1_addr, Xc_addr=r.X1_addr,
        name_idf=r.name_idf, name_idf_default=r.name_idf_default,
        addr_idf=r.addr_idf, addr_idf_default=r.addr_idf_default,
        dense=r.dense,
    )
    if r.dense:
        pseudo_r.E1_name = pseudo_r.Ec_name = r.E1_name
        pseudo_r.E1_addr = pseudo_r.Ec_addr = r.E1_addr
        pseudo_r.E1_joint = pseudo_r.Ec_joint = r.E1_joint
    return pseudo_r, pairs


def blocking_report(pairs, truth, s1_ids, n_cands_total):
    """Recall ceiling / reduction ratio; also returns the list of missed pairs."""
    got = set(zip(pairs.s1_id, pairs.cand_id))
    tp = sum(1 for s, cs in truth.items() for c in cs if (s, c) in got)
    total = sum(len(cs) for cs in truth.values())
    per_s1 = []
    for s in s1_ids:
        t = truth.get(s, set())
        if t:
            per_s1.append(len({c for c in t if (s, c) in got}) / len(t))
    stats = {
        "pair_recall": tp / max(total, 1),
        "macro_recall_nonsingleton": float(np.mean(per_s1)) if per_s1 else 1.0,
        "avg_cands_per_s1": len(pairs) / max(len(s1_ids), 1),
        "reduction_ratio": 1 - len(pairs) / max(len(s1_ids) * n_cands_total, 1),
        "n_pairs": int(len(pairs)),
    }
    missed = [(s, c) for s, cs in truth.items() for c in cs if (s, c) not in got]
    return stats, missed
