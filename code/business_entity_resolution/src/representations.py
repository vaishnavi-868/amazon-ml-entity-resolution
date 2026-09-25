"""Builds every vector representation for one split (train or test).

TF-IDF vocabularies / IDF weights are fitted on the split's own corpus (labels are
never used), so they adapt to new countries at test time without retraining."""
import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import prepare


def token_idf(token_lists):
    n = len(token_lists)
    df = Counter()
    for toks in token_lists:
        df.update(set(toks))
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    return idf, math.log(n + 1) + 1.0


def rowdot(A, B, i, j, chunk=200_000):
    """Row-wise dot product of A[i] and B[j] for sparse or dense matrices."""
    out = np.empty(len(i), dtype=np.float32)
    sparse = hasattr(A, "multiply")
    for s in range(0, len(i), chunk):
        ii, jj = i[s:s + chunk], j[s:s + chunk]
        if sparse:
            out[s:s + chunk] = np.asarray(A[ii].multiply(B[jj]).sum(axis=1)).ravel()
        else:
            out[s:s + chunk] = np.einsum("ij,ij->i", A[ii], B[jj])
    return out


class Embedder:
    """Lazy sentence-transformers wrapper with an on-disk cache."""

    def __init__(self, model_path, batch_size=128, cache_dir=None):
        self.model_path, self.batch_size = model_path, batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None

    def encode(self, texts, tag):
        key = hashlib.md5(("\x1f".join(texts)).encode("utf-8")).hexdigest()[:12]
        path = self.cache_dir / f"emb_{tag}_{key}.npy" if self.cache_dir else None
        if path is not None and path.exists():
            return np.load(path)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_path)
        arr = self._model.encode(texts, batch_size=self.batch_size,
                                 normalize_embeddings=True,
                                 show_progress_bar=True).astype(np.float32)
        if path is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(path, arr)
        return arr


class Reps:
    pass


def _tfidf():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True,
                           dtype=np.float32)


def build_reps(s1, cands, cfg, work_dir, tag):
    r = Reps()
    r.s1, r.c = prepare(s1), prepare(cands)
    r.c["src"] = cands["src"].values

    def fit_transform(col):
        v = _tfidf()
        v.fit(list(r.s1[col]) + list(r.c[col]))
        return v.transform(r.s1[col]), v.transform(r.c[col])

    r.X1_name, r.Xc_name = fit_transform("name_core")
    r.X1_joint, r.Xc_joint = fit_transform("joint_txt")
    r.X1_addr, r.Xc_addr = fit_transform("addr_norm")

    r.name_idf, r.name_idf_default = token_idf(list(r.s1.core) + list(r.c.core))
    tok = lambda s: s.split()
    r.addr_idf, r.addr_idf_default = token_idf(
        [tok(a) for a in r.s1.addr_norm] + [tok(a) for a in r.c.addr_norm])

    r.dense = bool(cfg.use_dense)
    if r.dense:
        emb = Embedder(cfg.embed_model, cfg.embed_batch, Path(work_dir) / "emb_cache")
        raw = lambda df: (df.business_name + " | " + df.business_address).tolist()
        r.E1_joint, r.Ec_joint = emb.encode(raw(r.s1), f"{tag}_s1j"), emb.encode(raw(r.c), f"{tag}_cj")
        r.E1_name = emb.encode(r.s1.business_name.tolist(), f"{tag}_s1n")
        r.Ec_name = emb.encode(r.c.business_name.tolist(), f"{tag}_cn")
        r.E1_addr = emb.encode(r.s1.business_address.tolist(), f"{tag}_s1a")
        r.Ec_addr = emb.encode(r.c.business_address.tolist(), f"{tag}_ca")
    return r
