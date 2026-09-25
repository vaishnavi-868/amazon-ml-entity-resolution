"""Text normalisation. No country-specific branching: the same rules run for every
record, so an unseen country (France) is handled by the same code path. Abbreviation
maps are applied identically to both sides of a pair, so ambiguous expansions
(st = street/saint) are harmless."""
import re
import unicodedata

import pandas as pd

ABBR = {
    # legal / name
    "corp": "corporation", "co": "company", "ltd": "limited", "pvt": "private",
    "inc": "incorporated", "svcs": "services", "svc": "service", "intl": "international",
    "assoc": "associates", "mfg": "manufacturing", "ent": "enterprises",
    "bros": "brothers", "dept": "department",
    # address
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bvd": "boulevard", "bldg": "building", "nr": "near",
    "opp": "opposite", "ste": "suite", "apt": "apartment", "fl": "floor", "dr": "drive",
    "ln": "lane", "hwy": "highway", "ct": "court", "pl": "place", "sq": "square",
    "mkt": "market", "sec": "sector", "ph": "phase", "cross": "cross",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ctr": "center", "centre": "center", "nagar": "nagar", "gali": "lane",
}
LEGAL = {
    "limited", "private", "incorporated", "corporation", "company", "llc", "llp", "lp",
    "pllc", "plc", "sarl", "sas", "sa", "eurl", "sci", "gmbh", "inc", "corp", "ltd",
    "pvt", "co", "and", "the", "of", "de", "la", "le", "les", "du", "des", "et", "opc",
}
LANDMARK = {"near", "opposite", "behind", "beside", "next", "landmark", "adjacent",
            "nearby", "beyond", "off"}
_POSTAL_RE = re.compile(r"(?<!\d)(\d{6}|\d{5})(?:-\d{4})?(?!\d)")


def ascii_lower(s: str) -> str:
    s = str(s or "").replace("&", " and ")
    a = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    if not re.search(r"[a-z0-9]", a):        # non-Latin script: keep original chars
        a = s.lower()
    return a


def tokens(s: str):
    s = re.sub(r"[^\w\s]|_", " ", ascii_lower(s))
    return [ABBR.get(t, t) for t in s.split()]


def skel(t: str) -> str:
    """Consonant skeleton: absorbs vowel/transliteration variants (shree ~ sri)."""
    if len(t) <= 2 or not t.isalpha():
        return t
    s = t[0] + re.sub(r"[aeiouyh]", "", t[1:])
    return re.sub(r"(.)\1+", r"\1", s)


def postal_code(addr: str) -> str:
    m = _POSTAL_RE.findall(str(addr or ""))
    return m[-1] if m else ""     # postal codes sit at the end of an address


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived columns used by blocking and features."""
    df = df.copy()
    name_tok = [tokens(x) for x in df.business_name]
    core_tok = [[t for t in ts if t not in LEGAL] or ts for ts in name_tok]
    addr_tok = [tokens(x) for x in df.business_address]
    postal = [postal_code(a) for a in df.business_address]

    df["name_norm"] = [" ".join(t) for t in name_tok]
    df["core"] = core_tok
    df["name_core"] = [" ".join(t) for t in core_tok]
    df["core_set"] = [frozenset(t) for t in core_tok]
    df["core_first"] = [t[0] if t else "" for t in core_tok]
    df["skel_set"] = [frozenset(skel(x) for x in t) for t in core_tok]
    df["skel_first"] = [skel(t[0]) if t else "" for t in core_tok]
    df["acronym"] = [("".join(x[0] for x in t) if len(t) >= 2 else "") for t in core_tok]

    df["postal"] = postal
    df["addr_norm"] = [" ".join(t) for t in addr_tok]
    df["addr_set"] = [frozenset(t) for t in addr_tok]
    df["nums"] = [frozenset(x for x in t if any(c.isdigit() for c in x) and x != p)
                  for t, p in zip(addr_tok, postal)]
    df["has_landmark"] = [int(any(x in LANDMARK for x in t)) for t in addr_tok]
    df["joint_txt"] = df.name_core + " | " + df.addr_norm
    df["country"] = df["country"].str.strip().str.lower()
    return df
