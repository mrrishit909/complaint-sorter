"""Step 5: score the LLM and the baseline on the same 480 test complaints from 2026.

    ./venv/bin/python evaluate.py

results/evaluation.json, results/per_category.csv, results/confusion_llm.csv, results/disagreements.csv
"""
import json
from math import comb
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

HERE = Path(__file__).parent


def load():
    d = pd.read_csv(HERE / "data" / "complaints.csv.gz").set_index("odi")
    sets = json.loads((HERE / "data" / "sets.json").read_text())
    llm = pd.DataFrame([json.loads(l) for l in (HERE / "results" / "llm_test.jsonl").open()]).set_index("odi")
    base = pd.read_csv(HERE / "results" / "baseline_test.csv").set_index("odi")
    t = d.loc[sets["test"]].join(llm.add_prefix("llm_")).join(base[["predicted"]].rename(columns={"predicted": "base"}))
    return t, sets["categories"]


def main():
    t, cats = load()
    y, l, b = t["category"], t["llm_category"], t["base"]
    owner_flag = t["crash"] | t["fire"]
    agree = l == b
    cost = json.loads((HERE / "results" / "cost.json").read_text())["test"]
    # Paired test on the same complaints (exact McNemar): only the ones where exactly one method is right count.
    llm_only, base_only = int(((l == y) & (b != y)).sum()), int(((b == y) & (l != y)).sum())
    k, n = min(llm_only, base_only), llm_only + base_only
    mcnemar_p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    out = {
        "test_complaints": len(t),
        "llm": {"accuracy": accuracy_score(y, l), "macro_f1": f1_score(y, l, average="macro")},
        "baseline": {"accuracy": accuracy_score(y, b), "macro_f1": f1_score(y, b, average="macro")},
        "paired": {"llm_right_baseline_wrong": llm_only, "baseline_right_llm_wrong": base_only, "mcnemar_exact_p": mcnemar_p},
        "both_agree": {"share": agree.mean(), "accuracy_when_agree": (l[agree] == y[agree]).mean(),
                       "accuracy_when_disagree_llm": (l[~agree] == y[~agree]).mean(),
                       "accuracy_when_disagree_base": (b[~agree] == y[~agree]).mean()},
        "crash_or_fire": {"owner_ticked": int(owner_flag.sum()), "llm_said": int(t["llm_crash_or_fire"].sum()),
                          "agreement": (owner_flag == t["llm_crash_or_fire"]).mean(),
                          "llm_yes_owner_no": int((t["llm_crash_or_fire"] & ~owner_flag).sum()),
                          "llm_no_owner_yes": int((~t["llm_crash_or_fire"] & owner_flag).sum())},
        "llm_cost_usd": cost["usd"], "llm_cost_per_1000": 1000 * cost["usd"] / cost["calls"],
        "llm_median_seconds": float(t["llm_seconds"].median()),
    }
    p_l, r_l, f_l, _ = precision_recall_fscore_support(y, l, labels=cats, zero_division=0)
    p_b, r_b, f_b, _ = precision_recall_fscore_support(y, b, labels=cats, zero_division=0)
    per = pd.DataFrame({"category": cats, "llm_f1": f_l, "llm_precision": p_l, "llm_recall": r_l,
                        "baseline_f1": f_b}).sort_values("llm_f1", ascending=False)
    per.round(4).to_csv(HERE / "results" / "per_category.csv", index=False)
    pd.DataFrame(confusion_matrix(y, l, labels=cats), index=cats, columns=cats).to_csv(HERE / "results" / "confusion_llm.csv")
    dis = t[l != y][["category", "llm_category", "base"]].reset_index()      # ids only: the text stays in data/
    dis.to_csv(HERE / "results" / "disagreements.csv", index=False)
    (HERE / "results" / "evaluation.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(json.dumps(out, indent=2, default=float))
    print(per.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
