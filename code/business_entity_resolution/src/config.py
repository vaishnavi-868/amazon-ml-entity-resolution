"""Central configuration. Everything that affects candidate generation lives in
BlockingConfig and is stored inside the trained model bundle, so inference is
guaranteed to use exactly the same blocking as training."""
from dataclasses import dataclass


@dataclass
class BlockingConfig:
    k_name: int = 30            # top-K by char TF-IDF on the core business name
    k_joint: int = 30           # top-K by char TF-IDF on name + address
    k_dense: int = 30           # top-K by dense embedding (if use_dense)
    min_tfidf_sim: float = 0.05  # ignore TF-IDF neighbours with (almost) no overlap
    key_rare_df: int = 15       # a name token/skeleton is a blocking key only if it
                                # occurs in <= this many S2/S3 records
    key_max_block: int = 40     # max block size for postal-code based keys
    max_cands: int = 80         # hard cap on candidates kept per S1 entity
    use_dense: bool = True
    # Local path (recommended, see README) or HF id. MIT licence, 568M params.
    embed_model: str = "BAAI/bge-m3"
    embed_batch: int = 128


LGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

THRESHOLD_GRID = [round(0.30 + 0.01 * k, 2) for k in range(0, 69)]  # 0.30 .. 0.98
MARGIN_GRID = [0.0, 0.05, 0.10, 0.20]  # 0.0 = margin rule disabled
