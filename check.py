"""The project's check: fair sets, nothing leaks from test into training, every score recomputes, the demo is clean.

    ./venv/bin/python check.py
"""
import json
import re
from pathlib import Path

import pandas as pd

from demo import EMAIL, LOCATION, PHONE

HERE = Path(__file__).parent


def main():
    d = pd.read_csv(HERE / "data" / "complaints.csv.gz").set_index("odi")
    sets = json.loads((HERE / "data" / "sets.json").read_text())
    cats, test, dev = sets["categories"], d.loc[sets["test"]], d.loc[sets["dev"]]
    assert not set(sets["test"]) & set(sets["dev"]), "dev and test overlap"
    assert (test["category"].value_counts() == 40).all() and len(test) == 480 and set(test["category"]) == set(cats)
    assert (dev["category"].value_counts() == 5).all() and dev["filed"].str.startswith("2025").all()
    assert (test["filed"] >= "2026-01-01").all(), "test complaints must come after everything the baseline learnt from"

    llm = pd.DataFrame([json.loads(l) for l in (HERE / "results" / "llm_test.jsonl").open()]).set_index("odi")
    assert sorted(llm.index) == sorted(sets["test"]) and llm.index.is_unique and llm["category"].isin(cats).all()
    base = pd.read_csv(HERE / "results" / "baseline_test.csv").set_index("odi")
    y = test["category"]
    l, b = llm.loc[y.index, "category"], base.loc[y.index, "predicted"]

    ev = json.loads((HERE / "results" / "evaluation.json").read_text())
    assert abs((l == y).mean() - ev["llm"]["accuracy"]) < 1e-9 and abs((b == y).mean() - ev["baseline"]["accuracy"]) < 1e-9
    assert ev["paired"]["llm_right_baseline_wrong"] == int(((l == y) & (b != y)).sum())
    assert ev["paired"]["baseline_right_llm_wrong"] == int(((b == y) & (l != y)).sum())
    flag = test["crash"] | test["fire"]
    assert abs((flag == llm.loc[y.index, "crash_or_fire"]).mean() - ev["crash_or_fire"]["agreement"]) < 1e-9

    review = pd.read_csv(HERE / "results" / "review.csv")
    assert set(review["odi"]) == set(y.index[l != y]), "every disagreement reviewed, nothing else"
    assert review["verdict"].isin(["llm", "label", "both", "no_info"]).all() and review["reason"].str.len().gt(10).all()

    demo = json.loads((HERE / "demo" / "demo.json").read_text())
    assert len(demo["complaints"]) == 480 and demo["evaluation"] == ev
    for r in demo["complaints"]:
        for pat in (PHONE, EMAIL, LOCATION):
            assert not pat.search(r["text"]), f"personal detail left in demo text for {r['id']}"
    assert not re.search(r"sk-[A-Za-z0-9]{20}", json.dumps(demo)), "no API key in the demo"

    cost = json.loads((HERE / "results" / "cost.json").read_text())
    assert cost["test"]["calls"] == 480 and cost["test"]["usd"] < 1.0
    print(f"OK: 480 test complaints (40 per category) after all training data, LLM {ev['llm']['accuracy']:.1%} vs "
          f"baseline {ev['baseline']['accuracy']:.1%} recompute, {len(review)} disagreements reviewed, demo text scrubbed")


if __name__ == "__main__":
    main()
