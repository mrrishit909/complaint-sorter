"""Step 7: save the 480 recorded test results for the static demo page (demo/index.html reads demo/demo.json).

    ./venv/bin/python demo.py

The page never calls the API: it shows what the LLM and the baseline already answered.
"""
import json
import re
import zlib
from pathlib import Path

import pandas as pd

from evaluate import load

HERE = Path(__file__).parent
# A few complaints start with the filer's town + ZIP ("KohlerWI53044United States"); public, but not repeated here.
LOCATION = re.compile(r'^[^"\[]{0,60}?[A-Z]{2}\s?\d{3,5}\s?United States')
PHONE = re.compile(r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}\b")
EMAIL = re.compile(r"\S+@\S+\.\w+")


def scrub(text):
    return EMAIL.sub("[email removed]", PHONE.sub("[phone removed]", LOCATION.sub("[location removed]", text)))


def main():
    t, cats = load()
    review = pd.read_csv(HERE / "results" / "review.csv").set_index("odi")
    ev = json.loads((HERE / "results" / "evaluation.json").read_text())
    rv = json.loads((HERE / "results" / "review.json").read_text())
    rows = []
    for odi, r in t.iterrows():
        rows.append({
            "id": int(odi), "filed": r["filed"], "make": r["make"], "owner": r["category"], "llm": r["llm_category"],
            "base": r["base"], "llm_summary": r["llm_summary"], "owner_crash_fire": bool(r["crash"] or r["fire"]),
            "llm_crash_fire": bool(r["llm_crash_or_fire"]), "text": scrub(r["summary"])[:1500],
            "verdict": review["verdict"].get(odi), "reason": review["reason"].get(odi),
        })
    rows.sort(key=lambda r: zlib.crc32(str(r["id"]).encode()))         # fixed shuffle: mixed categories up front
    out = {"categories": cats, "evaluation": ev, "review": rv, "complaints": rows}
    (HERE / "demo").mkdir(exist_ok=True)
    (HERE / "demo" / "demo.json").write_text(json.dumps(out, default=float))
    print(f"demo/demo.json: {len(rows)} complaints, {len(json.dumps(out, default=float)) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
