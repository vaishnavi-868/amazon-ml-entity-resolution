"""TSV reading / writing. Always tab-separated, never NaN-coerced.

--data-dir accepts either a local folder or an s3://bucket/prefix URI transparently
(reads go straight to S3 via s3fs; no local sync/download needed). Credentials come
from the normal AWS chain - on a SageMaker notebook/Studio that's the execution
role, so no keys need to be configured for this to work."""
from pathlib import Path

import pandas as pd


def _is_s3(path) -> bool:
    return str(path).startswith("s3://")


def _join(base, *parts) -> str:
    """Path join that works for both local paths and s3:// URIs (pathlib mangles
    the double slash in 's3://', so plain string joining is used for S3)."""
    base = str(base)
    if _is_s3(base):
        return "/".join([base.rstrip("/"), *parts])
    return str(Path(base, *parts))


def _exists(path) -> bool:
    if _is_s3(path):
        import s3fs
        return s3fs.S3FileSystem().exists(path)
    return Path(path).exists()


def read_tsv(path) -> pd.DataFrame:
    kwargs = {"storage_options": {"anon": False}} if _is_s3(path) else {}
    try:
        # pyarrow-backed strings: modestly lower memory than python-object strings,
        # and identical behaviour for everything downstream (tested against
        # normalize.prepare()). Falls back cleanly if pyarrow isn't installed.
        return pd.read_csv(str(path), sep="\t", dtype_backend="pyarrow",
                           keep_default_na=False, **kwargs)
    except (ImportError, ValueError):
        return pd.read_csv(str(path), sep="\t", dtype=str, keep_default_na=False, **kwargs)


def parse_ids(s: str):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def read_tsv_sample(path, n=200_000, chunksize=50_000):
    """Memory-bounded read for exploration on small instances: streams the file in
    chunks and keeps only the first `n` rows, so peak memory is ~chunksize rows
    rather than the whole file. Good enough for profiling (Phase 0); NOT used by
    train.py/infer.py, which need the full data and therefore need enough RAM."""
    kwargs = {"storage_options": {"anon": False}} if _is_s3(path) else {}
    parts, seen = [], 0
    for chunk in pd.read_csv(str(path), sep="\t", dtype=str, keep_default_na=False,
                             chunksize=chunksize, **kwargs):
        parts.append(chunk)
        seen += len(chunk)
        if seen >= n:
            break
    return pd.concat(parts, ignore_index=True).head(n)


def load_split_sample(data_dir, split, n=200_000):
    """Sampled version of load_split() for exploring huge files on a small
    instance. Returns the same 4-tuple shape, but truth is filtered to only the
    S1 ids that made it into the sample."""
    d = _join(data_dir, split)
    s1 = read_tsv_sample(_join(d, f"{split}_source1.tsv"), n)
    s2 = read_tsv_sample(_join(d, f"{split}_source2.tsv"), n)
    s3 = read_tsv_sample(_join(d, f"{split}_source3.tsv"), n)
    truth = None
    gt_path = _join(d, f"{split}_ground_truth.tsv")
    if _exists(gt_path):
        keep = set(s1.entity_id)
        gt = read_tsv_sample(gt_path, n * 2)
        truth = {r.source1_entity_id: set(parse_ids(r.matched_entity_ids))
                 for r in gt.itertuples(index=False) if r.source1_entity_id in keep}
        for sid in s1.entity_id:
            truth.setdefault(sid, set())
    return s1, s2, s3, truth


def load_split(data_dir, split):
    """Returns (s1, s2, s3, truth) where truth is {s1_id: set(ids)} or None.
    data_dir may be a local folder or an s3://bucket/prefix URI; either way it
    must contain a `<split>/` folder with the challenge's standard file names."""
    d = _join(data_dir, split)
    s1 = read_tsv(_join(d, f"{split}_source1.tsv"))
    s2 = read_tsv(_join(d, f"{split}_source2.tsv"))
    s3 = read_tsv(_join(d, f"{split}_source3.tsv"))
    truth = None
    gt_path = _join(d, f"{split}_ground_truth.tsv")
    if _exists(gt_path):
        gt = read_tsv(gt_path)
        truth = {r.source1_entity_id: set(parse_ids(r.matched_entity_ids))
                 for r in gt.itertuples(index=False)}
        for sid in s1.entity_id:          # S1 entities missing from GT = singletons
            truth.setdefault(sid, set())
    return s1, s2, s3, truth


def candidate_table(s2: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    c = pd.concat([s2, s3], ignore_index=True)
    c["src"] = c.entity_id.str.slice(0, 2)  # 'S2' / 'S3' from the id prefix
    return c


def write_id_lists(path, s1_ids, mapping, col_name):
    """One row per S1 entity (in file order); empty string when no ids.
    `path` is always local (final outputs are written to disk, then presented /
    uploaded / validated from there)."""
    rows = [(sid, ",".join(sorted(mapping.get(sid, [])))) for sid in s1_ids]
    pd.DataFrame(rows, columns=["source1_entity_id", col_name]).to_csv(
        path, sep="\t", index=False)

