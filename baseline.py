"""Step 3: the classic baseline the LLM has to beat: word counts (TF-IDF) + logistic regression.

    ./venv/bin/python baseline.py

Trained on every in-scope complaint filed before 2026 (minus the 60 dev complaints), so it has seen ~35,000
labelled examples; the LLM sees none. Scored on the same 480 test complaints from 2026.

results/baseline_test.csv
"""
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).parent


def load():
    d = pd.read_csv(HERE / "data" / "complaints.csv.gz")
    sets = json.loads((HERE / "data" / "sets.json").read_text())
    return d, sets


def main():
    d, sets = load()
    scoped = d[d["category"].notna()]
    train = scoped[(scoped["filed"] < "2026-01-01") & ~scoped["odi"].isin(sets["dev"])]
    test = d[d["odi"].isin(sets["test"])]
    t0 = time.time()
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
                          LogisticRegression(max_iter=3000, C=4))
    model.fit(train["summary"], train["category"])
    pred = model.predict(test["summary"])
    out = test[["odi", "category"]].assign(predicted=pred)
    (HERE / "results").mkdir(exist_ok=True)
    out.to_csv(HERE / "results" / "baseline_test.csv", index=False)
    print(f"trained on {len(train):,} complaints in {time.time() - t0:.0f} s; test accuracy "
          f"{accuracy_score(out['category'], pred):.1%}, macro-F1 {f1_score(out['category'], pred, average='macro'):.3f}")


if __name__ == "__main__":
    main()
