# Business Entity Resolution: Methodology

**Team:** `<team_name>`  **Members:** `<names>`
Items marked **`<FILL>`** must be replaced with numbers from your own run (`work/train_report.json`, the training log).

## 1. Summary

We resolve entities with a three-stage pipeline: (1) **hybrid blocking**, a union of char-TF-IDF kNN, dense multilingual
embedding kNN and inverted-index key blocking, capped per Source-1 entity; (2) a **LightGBM pair classifier** over string,
TF-IDF, embedding and relative (rank/margin) features; (3) **precision-first decoding**, a tuned probability threshold plus a
one-owner constraint, since F0.5 punishes false merges. Singletons are handled implicitly: an S1 entity receives an empty list
when no candidate clears the threshold.

Headline validation results (5-fold CV grouped by S1 entity, macro F0.5 incl. singletons): **`<FILL>`**
Leave-one-country-out (train US → test India and reverse): **`<FILL>`**

## 2. Compliance

* **No external data lookup.** Only the provided training/test files are used. No geocoding, entity-resolution or registry
  APIs; no AWS Entity Resolution / Location / Comprehend / Bedrock. AWS was used only as compute/storage (S3, SageMaker notebook).
* **Models and licences.**

| Model | Use | Licence | Params |
|---|---|---|---|
| BAAI/bge-m3 | dense blocking + embedding features | MIT | ~568M |
| LightGBM | pair classifier | MIT | n/a |
| xlm-roberta-base *(only if `cross_encoder.py` was used: **`<FILL yes/no>`**)* | cross-encoder feature | MIT | ~280M |

* All pretrained weights are used as released, or fine-tuned only on the provided training data.
* Transductive/pseudo-labelling on test data: **`<FILL: none / describe>`**

## 3. Data exploration findings

`<FILL from train_report.json>`: singleton fraction, average/max matches per S1, S2 vs S3 share of matches, fraction of
S2/S3 records claimed by more than one S1 entity (decides the one-owner rule), fraction of matches that cross country labels.

## 4. Preprocessing (`src/normalize.py`)

* NFKD accent folding to ASCII (fallback to original characters for non-Latin scripts), lower-casing, `&` → `and`,
  punctuation removal.
* Token-level abbreviation expansion (corp/co/ltd/pvt/inc, rd/st/ave/blvd, nr/opp, …). It is applied identically to both sides
  of every pair, so ambiguous expansions are harmless.
* **Core name:** legal suffixes and stop-words removed (limited, private, sarl, sas, "the", "de", …).
* **Postal code:** last 5/6-digit token in the address (US ZIP, India PIN, French code postal).
* **Consonant skeleton** of each name token (vowels and `h` removed, repeated letters collapsed) to absorb transliteration variants.
* Acronym of multi-word names, address number tokens, landmark flag (near/opposite/behind/…).
* No country-specific rules, so unseen countries pass through the same code.

## 5. Candidate generation / blocking (`src/blocking.py`)

Candidates for each S1 entity = union of:

| Generator | Signal | Setting |
|---|---|---|
| `g_name` | char 3–4-gram TF-IDF cosine on core name | top-30 per source (S2, S3 separately) |
| `g_joint` | char 3–4-gram TF-IDF on core name + address | top-30 per source |
| `g_dense` | bge-m3 embedding of "name \| address" | top-30 per source |
| `g_key` | inverted index on rare name tokens, name skeletons, postal+name-token | block size ≤ 15 / 40 |

The union is capped at **80 candidates per S1** (ranked by mean similarity + a bonus per generator that proposed the pair).
This capped set is exactly what the model scores and is what is written to `candidate_pairs.tsv`.

Blocking quality on training data: pair recall **`<FILL>`**, non-singleton macro recall **`<FILL>`**, avg candidates/S1 **`<FILL>`**,
reduction ratio **`<FILL>`**. Missed-pair analysis and fixes: **`<FILL>`**

## 6. Model and features (`src/features.py`, `src/train.py`)

Classifier: LightGBM (500 trees, 63 leaves, lr 0.05, subsample/colsample 0.8), trained on candidate pairs. Positives = ground-truth pairs,
negatives = all other blocked candidates, plus S1-vs-S1 hard negatives (below), so the training distribution matches inference.

**Hard-negative mining.** Distinct Source-1 entities are guaranteed non-matches by construction (S1 is the deduplicated reference source).
We mine lexically-similar S1 pairs (TF-IDF kNN within S1's own core names, k=`<FILL>`, min similarity=`<FILL>`) and add them as extra
negatives to every training fold's *training* rows only (never to a validation fold, so the reported OOF score stays an honest estimate).
`<FILL>` self-negative pairs were added per run.

Feature groups (~65):
* **Raw exact match:** verbatim (case-sensitive) equality of the untouched `business_name` / `business_address` strings, kept alongside every
  normalized feature so a normalization rule that is later found too aggressive can never destroy a free, high-precision signal.
* **Name (raw + core):** Jaro-Winkler, Levenshtein similarity, token-set/token-sort/partial ratios, IDF-weighted and plain token Jaccard,
  skeleton Jaccard, exact/first-token match, acronym match, containment, length ratio, token-count difference.
* **Address:** the same string similarities, IDF-Jaccard, house-number overlap, postal equality / 3-digit prefix equality, missing-address
  flags (NaN when a side is missing), landmark flag.
* **Vector similarities:** char-TF-IDF cosine (name / joint / address), bge-m3 cosine (name / address / joint).
* **Relative features:** rank and margin-to-runner-up of key similarities within the S1's candidates (forward) and among the S1 entities
  competing for the same S2/S3 record (reverse); candidate-set sizes. These are scale-free and transfer to a new country.
* **Metadata:** source (S2/S3), `country_match` (string equality only, no country one-hot), generator flags.
* *(Optional)* fine-tuned cross-encoder score, out-of-fold for training pairs.

TF-IDF vocabularies and IDF weights are fitted on the corpus being processed (labels are never used), so they adapt to French text at test time.

## 7. Decoding and threshold selection (`src/decode.py`)

We evaluate two decoders on the same out-of-fold predictions and keep whichever wins — this comparison is automatic in `train.py` and the
choice is stored with the model, so `infer.py` never has to guess which one to use.

**Decoder A — flat threshold.** Keep pairs with p ≥ threshold, then the **one-owner rule** (each S2/S3 record goes to its single best-scoring
S1 entity; enabled only when the training ground truth shows records are not shared — `<FILL: enabled/disabled>`), then an optional margin
rule that drops a record whose runner-up S1 is within `margin` of the winner. Threshold and margin are grid-searched on OOF predictions
against the exact metric. Chosen: threshold **`<FILL>`**, margin **`<FILL>`**.

**Decoder B — entity-level expected-F0.5.** Per S1 entity, calibrate raw model scores with isotonic regression fit on OOF predictions, then
for that entity's candidates (sorted by calibrated probability) choose the prediction-set size k — including k = 0, i.e. predict nothing —
that maximises E[F0.5] under a Monte-Carlo simulation of which candidates are true matches. This targets the metric directly: F0.5 is scored
per S1 entity, and a correct empty prediction on a true singleton scores 1.0, which a single global cutoff cannot express as precisely per
entity. The one-owner rule is still applied afterward, resolving any record claimed by more than one S1's selection in favour of the
higher-calibrated-probability owner.

**Result:** OOF F0.5 was **`<FILL>`** for the flat threshold and **`<FILL>`** for the expected-F0.5 decoder; we used **`<FILL: A / B>`**.

## 8. Validation protocol

* 5-fold GroupKFold by S1 entity (no leakage of an entity's pairs across folds).
* Leave-one-country-out as a proxy for the unseen France test set: **`<FILL>`**.
* Test-time sanity check: per-country match rate / matches per S1 (France vs US/India): **`<FILL>`**.

## 9. Results

| Stage | Score |
|---|---|
| OOF macro F0.5 (tuned) | `<FILL>` |
| Leave-one-country-out (US→India / India→US) | `<FILL>` |
| Public leaderboard | `<FILL>` |

Ablations you may want to report (optional): no dense generator, no relative features, no one-owner rule, with/without cross-encoder.

## 10. Limitations and what we would do next

`<FILL>`. Examples: heavy landmark-only addresses, businesses with several near-identical branches (ambiguity → dropped by design),
French-specific abbreviations not seen in training, threshold sensitivity for the unseen country.

## 11. Reproduction

See `code/business_entity_resolution/README.md`. Hardware used: `<FILL: instance type>`; wall-clock: `<FILL>`.
