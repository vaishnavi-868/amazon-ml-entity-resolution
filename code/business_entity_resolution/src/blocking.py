"""Candidate generation = union of four generators, capped per S1 entity.

  g_name  : char 3-4gram TF-IDF kNN on the core name   (typos, transliteration)
  g_joint : char 3-4gram TF-IDF kNN on name + address
  g_dense : multilingual embedding kNN (optional)      (semantic / word-order)
  g_key   : inverted-index keys: rare name tokens, consonant skeletons,
            postal-code + name-token                   (cheap exact recall)

kNN is run separately for S2 and S3 so one source cannot crowd out the other.
The returned DataFrame is *exactly* what the matching model scores, and is what
gets written to candidate_pairs.tsv.

MEMORY: the sparse (TF-IDF) top-k NEVER densifies a query x corpus matrix. At
millions of candidate rows that matrix would be tens of gigabytes even for a
small query chunk - this is the exact failure mode of a naive
`(Q @ D.T).toarray()` blocking implementation. `_sparse_topn` uses
sparse_dot_topn (github.com/ing-bank/sparse_dot_topn, MIT), which computes the
sparse product while keeping only the top-n entries per row, with a pure-numpy
fallback that blocks BOTH dimensions if the library isn't installed."""
from collections import defaultdict

import numpy as np
import pandas as pd

from .representations import rowdot

FLAGS = ["g_name", "g_joint", "g_dense", "g_key"]


def _sparse_topn(Q, D, k, min_sim, q_chunk=200_000, DT=None, n_threads=-1):
    """Q, D: sparse CSR matrices (Q = queries, D = corpus). Returns (row, col,
    score) arrays for each query's up-to-k highest-cosine corpus rows with
    score > min_sim. Batches over Q rows only for progress/latency, not for
    memory - sparse_dot_topn's own memory use is bounded by top_n regardless of
    corpus size, which is the whole point of using it here.

    n_threads=-1 (use all but one core) matters a lot: sp_matmul_topn defaults
    to SEQUENTIAL (single-core) processing if n_threads is left as None. That
    default is exactly what caused single-core-pinned, 20-30-minutes-per-batch
    behavior on a 64-vCPU instance where 63 cores sat idle - not a memory or
    algorithmic problem, just an unset threading parameter.

    Pass a precomputed `DT` (= D.T.tocsr()) when this is called repeatedly
    against the SAME corpus (e.g. once per S1 batch in batch_pipeline.py), to
    avoid re-transposing a multi-million-row matrix on every call."""
    n_d = DT.shape[1] if DT is not None else D.shape[0]
    k = min(k, n_d)
    if k == 0 or Q.shape[0] == 0:
        z = np.empty(0, dtype=np.int64)
        return z, z.copy(), np.empty(0, dtype=np.float32)
    try:
        from sparse_dot_topn import sp_matmul_topn
    except ImportError:
        return _sparse_topn_fallback(Q, D, k, min_sim)
    if DT is None:
        DT = D.T.tocsr()
    thr = float(min_sim) if min_sim > 0 else None
    rows, cols, vals = [], [], []
    for s in range(0, Q.shape[0], q_chunk):
        C = sp_matmul_topn(Q[s:s + q_chunk].tocsr(), DT, top_n=k,
                           threshold=thr, sort=False, n_threads=n_threads).tocoo()
        rows.append(C.row.astype(np.int64) + s)
        cols.append(C.col.astype(np.int64))
        vals.append(C.data.astype(np.float32))
    return (np.concatenate(rows) if rows else np.empty(0, dtype=np.int64),
            np.concatenate(cols) if cols else np.empty(0, dtype=np.int64),
            np.concatenate(vals) if vals else np.empty(0, dtype=np.float32))


def _sparse_topn_fallback(Q, D, k, min_sim, q_chunk=20_000, d_chunk=200_000):
    """Pure numpy/scipy fallback if sparse_dot_topn isn't installed: blocks BOTH
    Q and D so no single dense chunk exceeds q_chunk * d_chunk cells, merging a
    running top-k across D blocks. Slower than the library path but still
    memory-bounded regardless of corpus size."""
    DT = D.T.tocsc()
    n = Q.shape[0]
    rows, cols, vals = [], [], []
    for qs in range(0, n, q_chunk):
        Qc = Q[qs:qs + q_chunk]
        best_idx = np.full((Qc.shape[0], k), -1, dtype=np.int64)
        best_val = np.full((Qc.shape[0], k), -np.inf, dtype=np.float32)
        for ds in range(0, D.shape[0], d_chunk):
            sim = (Qc @ DT[:, ds:ds + d_chunk]).toarray()
            kk = min(k, sim.shape[1])
            part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
            ps = np.take_along_axis(sim, part, 1)
            merged_idx = np.concatenate([best_idx, part + ds], axis=1)
            merged_val = np.concatenate([best_val, ps], axis=1)
            order = np.argsort(-merged_val, axis=1)[:, :k]
            best_idx = np.take_along_axis(merged_idx, order, 1)
            best_val = np.take_along_axis(merged_val, order, 1)
        for r in range(Qc.shape[0]):
            keep = (best_idx[r] >= 0) & (best_val[r] > min_sim)
            rows.append(np.full(int(keep.sum()), qs + r, dtype=np.int64))
            cols.append(best_idx[r][keep])
            vals.append(best_val[r][keep])
    return (np.concatenate(rows) if rows else np.empty(0, dtype=np.int64),
            np.concatenate(cols) if cols else np.empty(0, dtype=np.int64),
            np.concatenate(vals) if vals else np.empty(0, dtype=np.float32))


def _dense_topk(Q, D, k, q_chunk=500, d_chunk=100_000):
    """Dense (L2-normalised) embeddings: dot product = cosine. Blocks BOTH Q and
    D so peak memory per chunk is q_chunk * d_chunk floats (~200MB at the
    defaults), never the full query-chunk x whole-corpus matrix."""
    n, k = Q.shape[0], min(k, D.shape[0])
    idx = np.full((n, k), -1, dtype=np.int64)
    sc = np.full((n, k), -np.inf, dtype=np.float32)
    for qs in range(0, n, q_chunk):
        Qc = Q[qs:qs + q_chunk]
        best_idx = np.full((Qc.shape[0], k), -1, dtype=np.int64)
        best_val = np.full((Qc.shape[0], k), -np.inf, dtype=np.float32)
        for ds in range(0, D.shape[0], d_chunk):
            block = D[ds:ds + d_chunk]
            sim = Qc @ block.T
            kk = min(k, sim.shape[1])
            part = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
            ps = np.take_along_axis(sim, part, 1)
            merged_idx = np.concatenate([best_idx, part + ds], axis=1)
            merged_val = np.concatenate([best_val, ps], axis=1)
            order = np.argsort(-merged_val, axis=1)[:, :k]
            best_idx = np.take_along_axis(merged_idx, order, 1)
            best_val = np.take_along_axis(merged_val, order, 1)
        idx[qs:qs + Qc.shape[0]] = best_idx
        sc[qs:qs + Qc.shape[0]] = best_val
    return idx, sc


def _knn_pairs(Q, D, src, k, min_sim, flag):
    """kNN for each source separately; returns DataFrame(i, j, flag)."""
    sparse = hasattr(Q, "multiply")
    parts = []
    for s in ("S2", "S3"):
        cols = np.where(src == s)[0]
        if len(cols) == 0:
            continue
        if sparse:
            ii, jj_local, sc = _sparse_topn(Q, D[cols], k, min_sim)
            jj = cols[jj_local]
        else:
            idx, sc2 = _dense_topk(Q, D[cols], k)
            valid = idx >= 0
            ii = np.repeat(np.arange(Q.shape[0]), idx.shape[1])[valid.ravel()]
            jj = cols[idx[valid]]
            sc = sc2[valid]
            keep = sc > min_sim
            ii, jj, sc = ii[keep], jj[keep], sc[keep]
        parts.append(pd.DataFrame({"i": ii, "j": jj, flag: 1}))
    return (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=["i", "j", flag]))


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


def prep_features_for_batch(pairs, s1_heavy, X1, cidx):
    """Bridge between the batched pipeline and the existing build_features():
    pairs.j holds GLOBAL candidate row indices (positions in the full corpus).
    This fetches heavy (Python-object) features for just the small set of
    candidates actually touched by this batch - never the full corpus - and
    remaps pairs.j to local positions in that small slice, so build_features()
    works completely unmodified. Returns (pseudo_r, remapped_pairs)."""
    import types
    uniq_j, local_j = np.unique(pairs.j.values, return_inverse=True)
    c_heavy = cidx.get_heavy(uniq_j)
    c_heavy["entity_id"] = cidx.entity_id[uniq_j]
    c_heavy["src"] = cidx.src[uniq_j]
    Xc_addr_local = cidx.Xc["addr_norm"][uniq_j]

    remapped = pairs.copy()
    remapped["j"] = local_j

    r = types.SimpleNamespace(
        s1=s1_heavy, c=c_heavy,
        X1_addr=X1["addr_norm"], Xc_addr=Xc_addr_local,
        name_idf=cidx.name_idf, name_idf_default=cidx.name_idf_default,
        addr_idf=cidx.addr_idf, addr_idf_default=cidx.addr_idf_default,
        dense=False,
    )
    return r, remapped


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
    ii_raw, jj_raw, sc_raw = _sparse_topn(r.X1_name, r.X1_name, k + 1, min_sim)
    keep = (ii_raw != jj_raw) & (ii_raw < jj_raw)   # drop self, dedup unordered pairs
    i, j = ii_raw[keep], jj_raw[keep]

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


def _knn_pairs_cached(Q, cidx, field, k, min_sim, flag):
    """Batched-pipeline counterpart to _knn_pairs: uses CandidateIndex's
    precomputed per-source transposed matrices and column-index arrays (built
    ONCE at index-construction time, see CandidateIndex.__init__) instead of
    re-slicing and re-transposing the full corpus matrix on every call. This
    function runs once per S1 BATCH - potentially hundreds of times - so that
    recomputation was the actual root cause of an "O(corpus) cost per batch"
    slowdown (each batch redoing multi-million-row sparse transposes)."""
    parts = []
    for s in ("S2", "S3"):
        cols = cidx.src_cols.get(s)
        if cols is None or len(cols) == 0:
            continue
        ii, jj_local, sc = _sparse_topn(Q, None, k, min_sim, DT=cidx.DT[field][s])
        jj = cols[jj_local]
        parts.append(pd.DataFrame({"i": ii, "j": jj, flag: 1}))
    return (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=["i", "j", flag]))


def _key_pairs_batch(s1_heavy, cidx, cfg):
    """Like _key_pairs, but looks up an already-built CandidateIndex's inverted
    index for just this S1 batch, instead of rebuilding an index every call."""
    ii, jj = [], []
    for i, args in enumerate(zip(s1_heavy.core, s1_heavy.postal,
                                 s1_heavy.skel_set, s1_heavy.skel_first)):
        for k in _keys(*args):
            lst = cidx.key_index.get(k)
            if not lst:
                continue
            limit = cfg.key_rare_df if k[:2] in ("t:", "s:") else cfg.key_max_block
            if len(lst) <= limit:
                ii.extend([i] * len(lst))
                jj.extend(lst)
    return pd.DataFrame({"i": np.array(ii, dtype=np.int64),
                         "j": np.array(jj, dtype=np.int64), "g_key": 1})


def generate_candidates_batch(s1_light, s1_heavy, X1, cidx, cfg,
                              force_ids=None):
    """Candidate generation for one S1 batch against a resident CandidateIndex
    (candidate_index.py) - the batched-pipeline counterpart to
    generate_candidates(). Uses _knn_pairs_cached (precomputed per-source
    transposes, see CandidateIndex) rather than _knn_pairs directly, since this
    runs once per S1 batch rather than once per whole dataset. Dense embedding
    blocking (g_dense) is not yet supported in the batched path.

    force_ids: optional {s1_id: set(cand_id)} of pairs (typically ground-truth
    positives) that must survive the max_cands cap regardless of their blocking
    score, so training positives are never silently dropped by blocking."""
    src = cidx.src
    parts = [
        _knn_pairs_cached(X1["name_core"], cidx, "name_core", cfg.k_name, cfg.min_tfidf_sim, "g_name"),
        _knn_pairs_cached(X1["joint_txt"], cidx, "joint_txt", cfg.k_joint, cfg.min_tfidf_sim, "g_joint"),
        _key_pairs_batch(s1_heavy, cidx, cfg),
    ]
    pairs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["i", "j"])
    for f in ("g_name", "g_joint", "g_dense", "g_key"):
        if f not in pairs:
            pairs[f] = 0
    pairs[list(FLAGS)] = pairs[list(FLAGS)].fillna(0).astype(np.int8)
    pairs = pairs.groupby(["i", "j"], as_index=False)[list(FLAGS)].max()

    i, j = pairs.i.values, pairs.j.values
    pairs["cos_name"] = rowdot(X1["name_core"], cidx.Xc["name_core"], i, j)
    pairs["cos_joint"] = rowdot(X1["joint_txt"], cidx.Xc["joint_txt"], i, j)
    pairs["n_gens"] = pairs[list(FLAGS)].sum(axis=1)
    pairs["base"] = (pairs.cos_name + pairs.cos_joint) / 2 + 0.03 * pairs.n_gens
    pairs["forced"] = False

    if force_ids:
        s1_ids = s1_light.entity_id.values
        extra_i, extra_j = [], []
        have = set(zip(pairs.i.values, pairs.j.values))
        for local_i, sid in enumerate(s1_ids):
            for cid in force_ids.get(sid, ()):
                gj = cidx.id_to_idx.get(cid)
                if gj is not None and (local_i, gj) not in have:
                    extra_i.append(local_i)
                    extra_j.append(gj)
        if extra_i:
            ei, ej = np.array(extra_i, dtype=np.int64), np.array(extra_j, dtype=np.int64)
            extra = pd.DataFrame({"i": ei, "j": ej})
            for f in FLAGS:
                extra[f] = 0
            extra["n_gens"] = 0
            extra["cos_name"] = rowdot(X1["name_core"], cidx.Xc["name_core"], ei, ej)
            extra["cos_joint"] = rowdot(X1["joint_txt"], cidx.Xc["joint_txt"], ei, ej)
            extra["base"] = -np.inf     # sorts last, but `forced` exempts it from the cap
            extra["forced"] = True
            pairs = pd.concat([pairs, extra], ignore_index=True)

    # cap per S1, but always keep forced (ground-truth) pairs regardless of cap
    capped = (pairs[~pairs.forced].sort_values(["i", "base"], ascending=[True, False])
             .groupby("i", sort=False).head(cfg.max_cands))
    pairs = pd.concat([capped, pairs[pairs.forced]], ignore_index=True) \
             .drop_duplicates(["i", "j"]).reset_index(drop=True)

    pairs["s1_id"] = s1_light.entity_id.values[pairs.i.values]
    pairs["cand_id"] = cidx.entity_id[pairs.j.values]
    pairs["src"] = src[pairs.j.values]
    return pairs.drop(columns="forced")


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
