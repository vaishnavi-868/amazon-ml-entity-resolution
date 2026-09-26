# Business Entity Resolution: reproducible pipeline

Blocking (TF-IDF + dense + key generators) → pair features → LightGBM → one-owner decoding with an
F0.5-tuned threshold → `matching_results.tsv` + `candidate_pairs.tsv`.

Everything is computed from the provided training/test files only. **No external lookups**
(no geocoding, no entity-resolution APIs, no AWS Entity Resolution / Location / Comprehend / Bedrock).

| Component | Licence | Size |
|---|---|---|
| `BAAI/bge-m3` (dense blocking + embedding features) | MIT | ~568M params |
| `xlm-roberta-base` (optional cross-encoder) | MIT | ~280M params |
| LightGBM, rapidfuzz, scikit-learn | MIT / MIT / BSD | n/a |

All are MIT/Apache-2.0 and well under 8B parameters.

## Layout

```
src/
  config.py            blocking + LightGBM settings (blocking config is saved inside the model bundle)
  io_utils.py          TSV read/write
  normalize.py         ascii-folding, abbreviation expansion, legal-suffix stripping, postal code, skeleton keys
  representations.py   char TF-IDF, IDF weights, (cached) dense embeddings
  blocking.py          4 candidate generators, union, per-S1 cap, recall report,
                        + S1-vs-S1 hard-negative mining (Phase 2)
  features.py          ~65 pair features: raw-exact (Phase 1), string, TF-IDF, embedding,
                        relative rank/margin, metadata
  decode.py            flat threshold + margin decoder, isotonic calibration and the
                        entity-level expected-F0.5 decoder (Phase 4), macro F0.5
  train.py             grouped CV w/ hard negatives, both decoders compared on OOF
                        (best one wins automatically), leave-one-country-out, final fit,
                        experiment log (Phase 5)
  infer.py             test-time pipeline; uses whichever decoder won at train time;
                        writes both output files + self-check
  cross_encoder.py     OPTIONAL fine-tuned cross-encoder feature (GPU)
scripts/
  run_all.sh                 train -> infer
  make_synthetic_data.py     synthetic data for smoke tests only
```

## 1. Environment

Python 3.11+ (3.12 was used for the smoke test).

```bash
cd code/business_entity_resolution
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the dense model **once** and keep a local copy (reproducible, no network needed later):

```bash
python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-m3").save("models/bge-m3")
EOF
```

CPU-only? Everything works with `--no-dense` (TF-IDF + key blocking). Expect somewhat lower recall.

## 2. Run end-to-end

`DATA` is the folder that contains `train/` and `test/` (i.e. `student_resource/dataset`). It can be a **local path or an
`s3://bucket/prefix` URI** — `--data-dir` accepts either transparently, no manual download/sync needed.

```bash
DATA=/path/to/student_resource/dataset          # local
# DATA=s3://your-bucket/dataset                 # or read directly from S3

# with dense blocking (GPU recommended)
python -m src.train --data-dir $DATA --work-dir work --model-dir models \
       --embed-model models/bge-m3 --n-jobs 4
python -m src.infer --data-dir $DATA --work-dir work --model-dir models --out-dir output --n-jobs 4

# CPU-only variant
python -m src.train --data-dir $DATA --no-dense
python -m src.infer --data-dir $DATA --out-dir output
```

**Using S3 directly:** the bucket must contain `train/` and `test/` subfolders with the challenge's standard file names
(e.g. `s3://your-bucket/dataset/train/train_source1.tsv`). Reads use `s3fs` under the hood and pick up credentials from
the normal AWS chain — on a SageMaker notebook/Studio instance that's the attached **execution role**, so no access keys
need to be configured; the role just needs `s3:GetObject` / `s3:ListBucket` on that bucket (see the IAM policy earlier in
this project's setup notes). `work/`, `models/` and `output/` always stay local regardless of where `--data-dir` points.

(`scripts/run_all.sh $DATA output [--no-dense]` runs both.) Then validate with the official script:

```bash
cd /path/to/student_resource
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`.

## 3. What to read in the training log

`train.py` prints and stores (`work/train_report.json`):

1. **Ground-truth stats.** Singleton fraction, matches per S1, and whether any S2/S3 record belongs to more than one
   S1 entity. The one-owner decoding rule is enabled automatically only if this is ≤ 1%.
2. **Blocking recall.** Recall ceiling, average candidates per S1 and reduction ratio. Missed pairs are written to
   `work/blocking_missed_pairs.tsv`: **read them**, then add a generator or raise K in `config.py`.
   The target is ≥ 98% recall.
3. **OOF F0.5** (5-fold, grouped by S1 entity) with the tuned threshold/margin, including singletons.
4. **Leave-one-country-out F0.5.** This is the closest proxy for the unseen test country (France).
   If it is far below the OOF score, the model relies on country-specific features; inspect the feature importances.

At inference, `infer.py` prints the match rate per country. France's rate should be in the same ballpark as
US/India; a large gap means the threshold or features don't transfer.

## 4. What's automatic vs what you should check

- **Hard negatives (Phase 2).** On by default. Distinct S1 entities are guaranteed
  non-matches, so lexically-similar S1 pairs are mined and added to every training
  fold (never to a validation fold - the OOF score stays honest). Disable with
  `--no-self-neg`; tune aggressiveness with `--self-neg-k` / `--self-neg-min-sim`.
- **Decoder choice (Phase 4).** `train.py` always computes both the flat-threshold
  decoder and the entity-level expected-F0.5 decoder on the *same* out-of-fold
  predictions, and stores whichever one scored higher in `models/bundle.joblib`.
  `infer.py` reads that choice automatically - you don't pick it by hand. Check the
  printed comparison (`expected-F0.5 entity-level decode OOF score: ... vs
  flat-threshold ...`) to see which one actually won on your data.
- **Experiment log.** Every `train.py` run appends one row to `work/experiments.csv`
  (blocking config, decoder used, both OOF scores, LOCO gap, blocking recall,
  `--notes`). Use it to compare runs instead of trusting memory.
- **Raw-exact features (Phase 1).** Always on, no flag needed.

## 5. Optional: cross-encoder feature

Requires a GPU. See the docstring in `src/cross_encoder.py` for the 5-command workflow. Use it only after the baseline
is solid. It trains on labelled pairs from the training data only, and scores are out-of-fold to avoid leakage.

## 6. At multi-million-row scale: the batched pipeline

If your dataset is large enough that `train.py`/`infer.py` need more RAM than you
have (millions of S1 entities, multi-million-row S2+S3 corpus), use
`src/batch_pipeline.py` instead. It processes S1 in batches against a
memory-bounded index of the full candidate corpus, so peak memory no longer
scales with total dataset size the way the original pipeline's did.

**Why the original pipeline runs out of memory at this scale, specifically:**
`normalize.prepare()` builds Python lists and frozensets for every row (needed
for exact-token/skeleton features and key-blocking). At a few thousand rows
this is nothing; at multi-million-row scale, millions of these objects' per-object
overhead alone can reach tens of GB, independent of blocking's own memory use.
Batching S1 alone does not fix this, because the candidate corpus side still has
to be represented once regardless of how S1 is chunked.

**What `batch_pipeline.py` does differently:**
- `src/candidate_index.py`'s `CandidateIndex` holds only cheap string columns and
  TF-IDF matrices for the full candidate corpus, resident throughout. The
  expensive per-row Python-object columns (`normalize.prepare_heavy`) are computed
  only for the small number of candidate rows actually selected for one S1 batch
  (`CandidateIndex.get_heavy`), and for the key-blocking inverted index they're
  computed and discarded one chunk at a time during index construction - never
  materialized for the full corpus at once.
- Sparse TF-IDF top-k blocking uses `sparse_dot_topn` (never builds a dense
  query-chunk × full-corpus matrix - see `blocking.py`'s module docstring for why
  that was the original OOM's exact cause).
- Ground-truth positives are force-included in each S1's candidate set
  (`generate_candidates_batch(..., force_ids=...)`), so blocking can never
  silently cost you training recall regardless of batch size.
- Training uses a bounded sample (all positives + capped sampled negatives per
  S1, see `--neg-per-s1`) for CV/threshold-tuning/calibration/LOCO - a large
  representative sample is sufficient for this, you don't need every negative
  pair from a 100M+-row candidate table.
- Inference streams every batch's scored candidates into a temporary file, then
  does one global one-owner conflict-resolution pass at the end (necessary
  because a S2/S3 record can be competed for by S1 entities in different
  batches).

**Usage:**
```bash
python -m src.batch_pipeline --mode train \
    --data-dir s3://your-bucket/dataset --work-dir work --model-dir models \
    --s1-batch-size 5000 --n-jobs 4 --neg-per-s1 5

python -m src.batch_pipeline --mode infer \
    --data-dir s3://your-bucket/dataset --work-dir work --model-dir models --out-dir output \
    --s1-batch-size 5000 --n-jobs 4
```
Tune `--s1-batch-size` down if you still see memory pressure (smaller batches
= smaller per-batch working set, at some cost to blocking throughput);
`--neg-per-s1` controls the bounded training-sample size; `--train-s1-frac`
coarsely subsamples which S1 batches contribute to the training sample at all,
if the sample itself is still too large.

**Not yet supported here** (present in `train.py`/`infer.py`, not yet ported):
dense embedding blocking (`g_dense`), S1-vs-S1 hard-negative mining, the
cross-encoder feature. None of these require changing the file formats this
pipeline writes, so they can be added later without disruption - ask if you
need one of them at this scale.

**Before a multi-hour run on your real data:** validate on a small subsample
first (e.g. write a truncated copy of your real train/test folders, or use
`scripts/make_synthetic_data.py --n-train 20000 --n-test 8000` for a synthetic
sanity check) and confirm blocking recall, sample size, and wall-clock time per
batch before committing a large instance to a full run.

## 7. Design notes / known limits

* **Precision-first decoding.** The threshold is tuned on out-of-fold predictions using the exact macro-F0.5 metric
  (singletons = 1.0 if you predict nothing). Optional margin rule: ambiguous records are dropped.
* **Generalising to France.** No country one-hot; accent folding; IDF/TF-IDF fitted on the corpus being processed;
  rank/margin features are scale-free; `country_match` only compares strings.
* **Transductive step.** None by default. If you add test pseudo-labelling, disclose it in the methodology doc.
* **Testing status.** The core pipeline (train → infer → output files, with and without a stubbed dense encoder,
  and with `--n-jobs 2`) was smoke-tested on synthetic data (`scripts/make_synthetic_data.py`). It has **not** been run on the
  real challenge data, on real `bge-m3`, or `cross_encoder.py`. Expect to tune `BlockingConfig` on real data.
* Runtime is dominated by string features (~60–100 µs/pair per core). With 50 candidates per S1 and 50k S1 entities,
  use `--n-jobs` ≥ 4.
