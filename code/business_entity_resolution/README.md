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

## 6. Design notes / known limits

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
