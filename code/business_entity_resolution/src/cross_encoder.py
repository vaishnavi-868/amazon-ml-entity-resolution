"""OPTIONAL stage: fine-tune a cross-encoder (default xlm-roberta-base, MIT licence)
and export its scores as an extra LightGBM feature ("ce_score").

Leakage-safe stacking: the training-pair scores are out-of-fold (2-fold by S1 entity);
test scores come from a model trained on all training pairs.

Workflow (needs a GPU; see README):
    python -m src.train  ...                       # baseline; writes work/train_pairs.pkl
    python -m src.infer  ...                       # baseline; writes work/test_pairs.pkl
    python -m src.cross_encoder --data-dir dataset --work-dir work
    python -m src.train  ... --ce-scores work/ce_train.pkl
    python -m src.infer  ... --ce-scores work/ce_test.pkl

Only the top-N candidates per S1 (by blocking score) are scored; the others get NaN,
which LightGBM handles natively. Not exercised in the CPU smoke test."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import candidate_table, load_split


def _text(name, addr):
    return f"{name} | {addr}"


def _fit(model_name, a, b, y, epochs, bs, lr, max_len, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(a)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    y = torch.tensor(y, dtype=torch.float32)
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            enc = tok([a[i] for i in idx], [b[i] for i in idx], truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                logits = model(**enc).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), y[idx].to(device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        print(f"  epoch {ep + 1}/{epochs} last loss {loss.item():.4f}")
    return tok, model


def _score(tok, model, a, b, bs, max_len, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(a), bs):
            enc = tok(a[s:s + bs], b[s:s + bs], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            out.append(torch.sigmoid(model(**enc).logits.squeeze(-1)).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--model-name", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--neg-per-s1", type=int, default=5, help="hardest negatives kept for training")
    ap.add_argument("--score-topn", type=int, default=15)
    a = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    work = Path(a.work_dir)

    def texts(split):
        s1, s2, s3, _ = load_split(a.data_dir, split)
        c = candidate_table(s2, s3)
        return (dict(zip(s1.entity_id, s1.business_name + " | " + s1.business_address)),
                dict(zip(c.entity_id, c.business_name + " | " + c.business_address)))

    tr = pd.read_pickle(work / "train_pairs.pkl")
    t1, tc = texts("train")
    tr["rk"] = tr.groupby("s1_id").base.rank(ascending=False, method="first")
    pos = tr[tr.y == 1]
    neg = tr[(tr.y == 0) & (tr.rk <= a.neg_per_s1)]
    fit_df = pd.concat([pos, neg], ignore_index=True)
    score_df = tr[tr.rk <= a.score_topn].copy()
    ids = np.array(sorted(tr.s1_id.unique()))
    fold = dict(zip(ids, np.random.RandomState(0).randint(0, 2, len(ids))))
    fit_df["fold"] = fit_df.s1_id.map(fold)
    score_df["fold"] = score_df.s1_id.map(fold)

    ce_train = []
    for k in (0, 1):                      # train on fold != k, score fold k
        sub = fit_df[fit_df.fold != k]
        tok, model = _fit(a.model_name, [t1[s] for s in sub.s1_id], [tc[c] for c in sub.cand_id],
                          sub.y.values, a.epochs, a.batch_size, a.lr, a.max_len, device)
        held = score_df[score_df.fold == k]
        held = held.assign(ce_score=_score(tok, model, [t1[s] for s in held.s1_id],
                                           [tc[c] for c in held.cand_id], 128, a.max_len, device))
        ce_train.append(held[["s1_id", "cand_id", "ce_score"]])
    pd.concat(ce_train).to_pickle(work / "ce_train.pkl")

    print("final cross-encoder on all training pairs -> test scores")
    tok, model = _fit(a.model_name, [t1[s] for s in fit_df.s1_id], [tc[c] for c in fit_df.cand_id],
                      fit_df.y.values, a.epochs, a.batch_size, a.lr, a.max_len, device)
    te = pd.read_pickle(work / "test_pairs.pkl")
    e1, ec = texts("test")
    te["rk"] = te.groupby("s1_id").base.rank(ascending=False, method="first")
    te = te[te.rk <= a.score_topn].copy()
    te["ce_score"] = _score(tok, model, [e1[s] for s in te.s1_id], [ec[c] for c in te.cand_id],
                            128, a.max_len, device)
    te[["s1_id", "cand_id", "ce_score"]].to_pickle(work / "ce_test.pkl")
    print("wrote", work / "ce_train.pkl", work / "ce_test.pkl")


if __name__ == "__main__":
    main()
