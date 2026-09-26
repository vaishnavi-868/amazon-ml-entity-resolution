"""Memory-bounded representation of the full S2+S3 candidate corpus, for the
batched pipeline (batch_pipeline.py). This is the fix for the ~65GB RSS blowup
seen when running the non-batched pipeline at ~2.2M S1 / ~10M candidate scale:
that blowup came from normalize.prepare() building millions of Python lists and
frozensets for the ENTIRE candidate corpus at once. Batching S1 alone does not
fix this, because the corpus side still has to be represented once regardless
of how S1 is chunked - so this module keeps only cheap (string / sparse-matrix)
representations of the full corpus resident, and computes the expensive
per-row Python-object columns (normalize.prepare_heavy) only on-demand, for the
small number of rows actually selected as candidates for one S1 batch."""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .blocking import _keys
from .normalize import prepare_heavy, prepare_light
from .representations import token_idf

TEXT_FIELDS = ("name_core", "joint_txt", "addr_norm")


def _tfidf(max_features):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True,
                           max_features=max_features, dtype=np.float32)


class CandidateIndex:
    """Built once from the full S2+S3 corpus; read-only and reused across every
    S1 batch. Holds: fitted TF-IDF vectorizers + the full corpus transformed
    through them (sparse, held resident - the one genuinely corpus-sized cost,
    but far cheaper than materializing prepare_heavy() for every row), token IDF
    weights, a key-blocking inverted index, and the light-prepared corpus
    dataframe (cheap string columns only) for on-demand heavy-feature lookup."""

    def __init__(self, cands_raw, cfg, s1_sample_for_vocab=None,
                fit_sample=500_000, index_chunk=200_000, log=print, seed=0):
        self.raw = cands_raw.reset_index(drop=True)
        n = len(self.raw)
        self.entity_id = self.raw.entity_id.to_numpy()
        self.src = (self.raw.src.to_numpy() if "src" in self.raw.columns
                   else np.array([str(e)[:2] for e in self.entity_id]))
        self.id_to_idx = {eid: i for i, eid in enumerate(self.entity_id)}

        log(f"[CandidateIndex] preparing light columns for {n:,} candidate rows")
        self.light = prepare_light(self.raw)
        self.postal = self.light.postal.to_numpy()
        self.country = self.light.country.to_numpy()

        rng = np.random.default_rng(seed)
        samp_idx = rng.choice(n, size=min(fit_sample, n), replace=False)
        fit_txt = {c: self.light[c].iloc[samp_idx] for c in TEXT_FIELDS}
        if s1_sample_for_vocab is not None:
            s1l = prepare_light(s1_sample_for_vocab)
            for c in TEXT_FIELDS:
                fit_txt[c] = pd.concat([fit_txt[c], s1l[c]], ignore_index=True)

        self.vecs = {}
        for c in TEXT_FIELDS:
            log(f"[CandidateIndex] fitting TF-IDF vectorizer: {c} "
               f"(on {len(fit_txt[c]):,}-doc sample)")
            v = _tfidf(cfg.tfidf_max_features)
            v.fit(fit_txt[c])
            self.vecs[c] = v

        log(f"[CandidateIndex] transforming full corpus ({n:,} rows) in chunks of {index_chunk:,}")
        chunks = {c: [] for c in TEXT_FIELDS}
        for s in range(0, n, index_chunk):
            block = self.light.iloc[s:s + index_chunk]
            for c in TEXT_FIELDS:
                chunks[c].append(self.vecs[c].transform(block[c]))
            log(f"[CandidateIndex]   transformed {min(s + index_chunk, n):,}/{n:,}")
        self.Xc = {c: vstack(m).tocsr() for c, m in chunks.items()}

        self.name_idf, self.name_idf_default = token_idf(
            [t.split() for t in self.light.name_core.iloc[samp_idx]])
        self.addr_idf, self.addr_idf_default = token_idf(
            [t.split() for t in self.light.addr_norm.iloc[samp_idx]])

        log("[CandidateIndex] building key-blocking inverted index (streamed, "
           "heavy columns discarded per chunk)")
        self.key_index = defaultdict(list)
        self.skel_first = np.empty(n, dtype=object)
        for s in range(0, n, index_chunk):
            block_heavy = prepare_heavy(self.light.iloc[s:s + index_chunk])
            for local_j, args in enumerate(zip(block_heavy.core, block_heavy.postal,
                                               block_heavy.skel_set, block_heavy.skel_first)):
                gj = s + local_j
                self.skel_first[gj] = args[3]
                for k in _keys(*args):
                    self.key_index[k].append(gj)
            log(f"[CandidateIndex]   indexed {min(s + index_chunk, n):,}/{n:,}")
        del block_heavy

        total_nnz = sum(m.nnz for m in self.Xc.values())
        log(f"[CandidateIndex] done: {n:,} candidate rows, {total_nnz/1e6:.1f}M total "
           f"TF-IDF nnz, {len(self.key_index):,} distinct blocking keys")

    def get_heavy(self, idx):
        """Recompute normalize.prepare_heavy() for just these candidate row
        positions. Only ever called on an already-selected slice (bounded by
        batch_size * max_cands), never the full corpus. Returned frame is
        indexed 0..len(idx)-1 in the same order as `idx`."""
        idx = np.asarray(idx)
        return prepare_heavy(self.light.iloc[idx].reset_index(drop=True))
