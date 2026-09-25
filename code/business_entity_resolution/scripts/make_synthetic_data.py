"""Generates a small synthetic dataset in the challenge's file format, ONLY to smoke-test
the pipeline end-to-end (train has US+India, test adds France). Not used for results.

    python scripts/make_synthetic_data.py --out dataset_synth
"""
import argparse
import random
from pathlib import Path

import pandas as pd

W = ["alpha", "bharat", "cosmos", "delta", "eagle", "fortune", "global", "horizon", "indus",
     "jupiter", "kalyan", "lotus", "metro", "nova", "orient", "pioneer", "quantum", "royal",
     "sunrise", "titan", "united", "vertex", "western", "zenith", "shree", "sai", "ganga",
     "atlas", "bleu", "soleil", "lumiere", "petit", "grand", "maison", "central", "prime"]
K = ["traders", "textiles", "foods", "motors", "pharma", "logistics", "solutions", "systems",
     "boulangerie", "cafe", "hardware", "electronics", "consulting", "builders", "medicals"]
LEG = {"US": ["Inc", "LLC", "Corp", "Co"], "India": ["Pvt Ltd", "Private Limited", "Ltd"],
       "France": ["SARL", "SAS", "SA"]}
ST = {"US": [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd")],
      "India": [("Road", "Rd"), ("Nagar", "Nagar"), ("Marg", "Marg"), ("Street", "St")],
      "France": [("Rue", "R."), ("Avenue", "Av"), ("Boulevard", "Bd")]}
CITY = {"US": ["Seattle", "Austin", "Denver", "Boston"], "India": ["Pune", "Mumbai", "Delhi", "Chennai"],
        "France": ["Paris", "Lyon", "Lille", "Nantes"]}


def typo(s, rng):
    if len(s) < 5 or rng.random() > 0.35:
        return s
    i = rng.randrange(1, len(s) - 1)
    return s[:i] + s[i + 1:] if rng.random() < 0.5 else s[:i] + s[i] + s[i:]


def make_entity(c, rng):
    name = " ".join(rng.sample(W, rng.choice([1, 2, 2, 3])) + [rng.choice(K)]).title()
    leg = rng.choice(LEG[c])
    st = rng.choice(ST[c])
    city = rng.choice(CITY[c])
    zipc = str(rng.randrange(10000, 99999)) if c != "India" else str(rng.randrange(400000, 599999))
    return dict(name=name, leg=leg, num=rng.randrange(1, 999), st=st, street=rng.choice(W).title(),
                city=city, zipc=zipc, country=c)


def render(e, rng, noisy):
    name = e["name"]
    if noisy:
        name = " ".join(typo(w, rng) for w in name.split())
        if rng.random() < 0.3:
            name = " ".join(reversed(name.split()))
        name = name.replace(" And ", " & ")
    leg = e["leg"] if (not noisy or rng.random() > 0.3) else ""
    if noisy and rng.random() < 0.4:
        leg = leg.replace("Private Limited", "Pvt Ltd").replace("Limited", "Ltd").replace("Corp", "Corporation")
    full = f"{name} {leg}".strip()
    stw = e["st"][1] if (noisy and rng.random() < 0.5) else e["st"][0]
    parts = [f"{e['num']} {e['street']} {stw}", e["city"], e["zipc"]]
    if noisy:
        if rng.random() < 0.3:
            parts = [f"Near SBI ATM, {parts[0]}"] + parts[1:]
        if rng.random() < 0.25:
            parts = parts[:-1]
        if rng.random() < 0.15:
            parts = parts[:1] + parts[2:]
    return full, ", ".join(parts)


def build(n, countries, rng, prefix_offset=0):
    r1, r2, r3, gt = [], [], [], []
    a2 = a3 = 0
    for k in range(n):
        c = countries[k % len(countries)]
        e = make_entity(c, rng)
        sid = f"S1-{k:05d}"
        nm, ad = render(e, rng, False)
        r1.append((sid, nm, ad, c))
        ms = []
        if rng.random() > 0.25:          # 25% singletons
            for src in (2, 3):
                for _ in range(rng.choice([0, 1, 1, 2]) if src == 2 else rng.choice([0, 1])):
                    nm, ad = render(e, rng, True)
                    if src == 2:
                        i = f"S2-{a2:05d}"; a2 += 1; r2.append((i, nm, ad, c))
                    else:
                        i = f"S3-{a3:05d}"; a3 += 1; r3.append((i, nm, ad, c))
                    ms.append(i)
        gt.append((sid, ",".join(ms)))
    for _ in range(n // 3):                 # distractor records (no S1 owner)
        c = rng.choice(countries)
        nm, ad = render(make_entity(c, rng), rng, True)
        i = f"S2-{a2:05d}"; a2 += 1; r2.append((i, nm, ad, c))
    cols = ["entity_id", "business_name", "business_address", "country"]
    return (pd.DataFrame(r1, columns=cols), pd.DataFrame(r2, columns=cols),
            pd.DataFrame(r3, columns=cols), pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_synth")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=900)
    a = ap.parse_args()
    rng = random.Random(7)
    for split, n, cs in (("train", a.n_train, ["US", "India"]), ("test", a.n_test, ["US", "India", "France"])):
        d = Path(a.out) / split
        d.mkdir(parents=True, exist_ok=True)
        s1, s2, s3, gt = build(n, cs, rng)
        s1.to_csv(d / f"{split}_source1.tsv", sep="\t", index=False)
        s2.to_csv(d / f"{split}_source2.tsv", sep="\t", index=False)
        s3.to_csv(d / f"{split}_source3.tsv", sep="\t", index=False)
        if split == "train":
            gt.to_csv(d / "train_ground_truth.tsv", sep="\t", index=False)
        else:   # keep hidden truth outside the dataset dir for optional scoring in the smoke test
            gt.to_csv(Path(a.out) / "test_truth_hidden.tsv", sep="\t", index=False)
    print("written to", a.out)
