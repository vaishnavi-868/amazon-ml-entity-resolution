"""Pair features. Design rules for generalising to an unseen country:
  * no country one-hot; only 'country strings equal?'
  * TF-IDF / IDF fitted on the split's own corpus
  * scale-free relative features (rank / margin inside the candidate set)
  * missing address parts -> NaN (LightGBM handles NaN natively)"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .representations import rowdot

NAN = float("nan")

STR_FEATS = [
    # raw, untouched fields (Phase 1: don't let normalization destroy information -
    # a verbatim copy-paste record is free precision that survives even if a
    # normalization rule is later found to be too aggressive)
    "raw_name_exact", "raw_addr_exact",
    # raw normalised name
    "nm_jw", "nm_lev", "nm_tset", "nm_tsort", "nm_part",
    # core name (legal suffixes removed)
    "nc_jw", "nc_lev", "nc_tset", "nc_tsort",
    "nc_idf_jac", "nc_tok_jac", "nc_skel_jac", "nc_exact", "nc_first_eq", "nc_first_jw",
    "nc_acr_match", "nc_acr_eq", "nc_contains", "nc_len_ratio", "nc_ntok_diff",
    # address
    "ad_jw", "ad_lev", "ad_tset", "ad_tsort", "ad_part", "ad_idf_jac", "ad_tok_jac",
    "ad_num_jac", "ad_num_any", "ad_postal_eq", "ad_postal3_eq", "ad_miss1", "ad_miss2",
    "ad_landmark_any",
]


def _sims(a, b):
    if not a or not b:
        return (NAN,) * 5
    return (JaroWinkler.similarity(a, b), Levenshtein.normalized_similarity(a, b),
            fuzz.token_set_ratio(a, b) / 100, fuzz.token_sort_ratio(a, b) / 100,
            fuzz.partial_ratio(a, b) / 100)


def _jac(s1, s2):
    if not s1 or not s2:
        return NAN
    return len(s1 & s2) / len(s1 | s2)


def _idf_jac(s1, s2, idf, d):
    if not s1 or not s2:
        return NAN
    den = sum(idf.get(t, d) for t in (s1 | s2))
    return sum(idf.get(t, d) for t in (s1 & s2)) / den if den else NAN


def _rec(df):
    cols = ["name_norm", "name_core", "core_set", "addr_norm", "addr_set", "postal",
            "nums", "acronym", "skel_set", "core_first", "has_landmark",
            "business_name", "business_address"]
    return list(zip(*[df[c].values for c in cols]))


def _pair_feats(chunk, R1, Rc, name_idf, nd, addr_idf, addr_d):
    out = np.empty((len(chunk), len(STR_FEATS)), dtype=np.float32)
    for row, (i, j) in enumerate(chunk):
        n1, c1, cs1, a1, as1, p1, nu1, ac1, sk1, f1, lm1, rn1, ra1 = R1[i]
        n2, c2, cs2, a2, as2, p2, nu2, ac2, sk2, f2, lm2, rn2, ra2 = Rc[j]
        raw_name_exact = float(rn1 == rn2)
        raw_addr_exact = float(ra1 == ra2 and bool(ra1))
        nm = _sims(n1, n2)
        nc = _sims(c1, c2)
        l1, l2 = len(c1), len(c2)
        acr_match = float(bool((ac1 and ac1 == c2.replace(" ", "")) or
                               (ac2 and ac2 == c1.replace(" ", ""))))
        acr_eq = float(len(ac1) >= 3 and ac1 == ac2)
        contains = float(min(l1, l2) >= 3 and (c1 in c2 or c2 in c1))
        ad = _sims(a1, a2)
        if p1 and p2:
            p_eq, p3 = float(p1 == p2), float(p1[:3] == p2[:3])
        else:
            p_eq = p3 = NAN
        num_any = float(bool(nu1 & nu2)) if (nu1 and nu2) else NAN
        vals = [
            raw_name_exact, raw_addr_exact,
            *nm, nc[0], nc[1], nc[2], nc[3],
            _idf_jac(cs1, cs2, name_idf, nd), _jac(cs1, cs2), _jac(sk1, sk2),
            float(c1 == c2 and l1 > 0), float(f1 == f2 and f1 != ""),
            JaroWinkler.similarity(f1, f2) if f1 and f2 else NAN,
            acr_match, acr_eq, contains,
            min(l1, l2) / max(l1, l2, 1), abs(len(cs1) - len(cs2)),
            *ad, _idf_jac(as1, as2, addr_idf, addr_d),
            _jac(as1, as2), _jac(nu1, nu2), num_any, p_eq, p3,
            float(not a1), float(not a2), float(bool(lm1 or lm2)),
        ]
        out[row] = vals
    return out


def _relative(X, pairs, cols):
    """Rank / margin of each pair among the candidates of its S1 entity
    (forward) and among the S1 entities competing for the same candidate (reverse)."""
    for grp_name, g in (("s1", pairs.s1_id.values), ("c", pairs.cand_id.values)):
        for c in cols:
            s = pd.Series(X[c].values).fillna(-1.0)
            grp = s.groupby(g)
            top1 = grp.transform("max")
            # second largest per group
            rank = grp.rank(ascending=False, method="first")
            second = s.where(rank == 2).groupby(g).transform("max").fillna(-1.0)
            other_max = np.where(s.values == top1.values, second.values, top1.values)
            X[f"{c}_rank_{grp_name}"] = grp.rank(ascending=False, method="min").values
            X[f"{c}_margin_{grp_name}"] = s.values - other_max
    X["n_cands_s1"] = pairs.groupby("s1_id")["i"].transform("size").values
    X["n_s1_for_cand"] = pairs.groupby("cand_id")["i"].transform("size").values
    return X


def attach_ce(pairs, path):
    ce = pd.read_pickle(path)[["s1_id", "cand_id", "ce_score"]]
    return pairs.merge(ce, on=["s1_id", "cand_id"], how="left")


def build_features(r, pairs, n_jobs=1, chunk=100_000):
    i, j = pairs.i.values, pairs.j.values
    R1, Rc = _rec(r.s1), _rec(r.c)
    ij = list(zip(i.tolist(), j.tolist()))
    chunks = [ij[s:s + chunk] for s in range(0, len(ij), chunk)]
    if n_jobs == 1:
        res = [_pair_feats(c, R1, Rc, r.name_idf, r.name_idf_default, r.addr_idf,
                           r.addr_idf_default) for c in chunks]
    else:
        res = Parallel(n_jobs=n_jobs)(delayed(_pair_feats)(
            c, R1, Rc, r.name_idf, r.name_idf_default, r.addr_idf, r.addr_idf_default)
            for c in chunks)
    X = pd.DataFrame(np.vstack(res) if res else np.empty((0, len(STR_FEATS))),
                     columns=STR_FEATS)

    miss1 = (r.s1.addr_norm.values[i] == "")
    missc = (r.c.addr_norm.values[j] == "")
    X["cos_name"] = pairs.cos_name.values
    X["cos_joint"] = pairs.cos_joint.values
    cos_addr = rowdot(r.X1_addr, r.Xc_addr, i, j)
    X["cos_addr"] = np.where(miss1 | missc, np.nan, cos_addr)
    rel_cols = ["cos_name", "cos_joint", "nc_jw", "ad_jw"]

    if r.dense:
        X["emb_joint"] = pairs.emb_joint.values
        X["emb_name"] = rowdot(r.E1_name, r.Ec_name, i, j)
        ea = rowdot(r.E1_addr, r.Ec_addr, i, j)
        X["emb_addr"] = np.where(miss1 | missc, np.nan, ea)
        rel_cols += ["emb_joint", "emb_name"]

    X["is_s3"] = (pairs.src.values == "S3").astype(np.float32)
    X["country_match"] = (r.s1.country.values[i] == r.c.country.values[j]).astype(np.float32)
    for f in ("g_name", "g_joint", "g_dense", "g_key", "n_gens"):
        X[f] = pairs[f].values
    if "ce_score" in pairs:
        X["ce_score"] = pairs.ce_score.values
        rel_cols.append("ce_score")

    X = _relative(X, pairs.reset_index(drop=True), rel_cols)
    return X.astype(np.float32)
