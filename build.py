"""Step 2: one clean table of complaints, the 12 categories, and the fixed evaluation sets.

    ./venv/bin/python build.py

data/complaints.csv.gz   every unique complaint (82k), with its category when it has exactly one in-scope component
data/sets.json           ids for: dev (60, from 2025: for writing the prompt) and test (480, from 2026: scored once)

Categories are the 12 most common components owners file under. Two near-duplicate NHTSA codes are merged
(ENGINE AND ENGINE COOLING -> ENGINE, VISIBILITY/WIPER -> VISIBILITY); UNKNOWN OR OTHER is left out.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
MERGE = {"ENGINE AND ENGINE COOLING": "ENGINE", "VISIBILITY/WIPER": "VISIBILITY"}
CATEGORIES = ["POWER TRAIN", "ENGINE", "ELECTRICAL SYSTEM", "VISIBILITY", "STEERING", "FORWARD COLLISION AVOIDANCE",
              "STRUCTURE", "FUEL/PROPULSION SYSTEM", "SERVICE BRAKES", "AIR BAGS", "EXTERIOR LIGHTING", "SUSPENSION"]
PER_CLASS_TEST, PER_CLASS_DEV = 40, 5


def load_raw():
    rows = {}
    for f in sorted((HERE / "data" / "raw").glob("*.json")):
        for r in json.loads(f.read_text()):
            rows[r["odiNumber"]] = r          # a complaint can list several vehicles; keep it once
    d = pd.DataFrame(rows.values())
    return pd.DataFrame({
        "odi": d["odiNumber"].astype(int),
        "filed": pd.to_datetime(d["dateComplaintFiled"], format="%m/%d/%Y").dt.date.astype(str),
        "make": d["manufacturer"], "components": d["components"],
        "crash": d["crash"].astype(bool), "fire": d["fire"].astype(bool),
        "injuries": d["numberOfInjuries"].astype(int), "summary": d["summary"].fillna("").str.strip(),
    }).sort_values("odi")


def main():
    d = load_raw()
    single = ~d["components"].str.contains(",")
    d["category"] = d["components"].where(single).map(lambda c: MERGE.get(c, c) if isinstance(c, str) else None)
    d.loc[~d["category"].isin(CATEGORIES), "category"] = None
    d = d[d["summary"].str.len() >= 40]                                 # a sentence at least
    d.to_csv(HERE / "data" / "complaints.csv.gz", index=False)

    scoped = d[d["category"].notna()]
    test = scoped[scoped["filed"] >= "2026-01-01"].groupby("category").sample(PER_CLASS_TEST, random_state=0)
    dev = scoped[scoped["filed"].between("2025-01-01", "2025-12-31")].groupby("category").sample(PER_CLASS_DEV, random_state=0)
    sets = {"categories": CATEGORIES, "dev": sorted(dev["odi"].tolist()), "test": sorted(test["odi"].tolist())}
    (HERE / "data" / "sets.json").write_text(json.dumps(sets, indent=1) + "\n")
    print(f"{len(d):,} complaints; {len(scoped):,} in the 12 categories "
          f"({(scoped['filed'] < '2026-01-01').sum():,} before 2026 for training the classic model); "
          f"dev {len(dev)}, test {len(test)}")
    print(scoped["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
