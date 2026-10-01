# Complaint Sorter: an LLM vs a classic classifier on real vehicle-safety complaints

Car owners file safety complaints with NHTSA in their own words and pick the part that failed. Can an LLM, with
**no labelled examples**, read a complaint and sort it into the right one of 12 components, flag crashes and
fires, and write a one-line summary? And is it any better than a classic word-count model trained on 35,000
older complaints?

**Demo (recorded results, no live API):** https://mrrishit909.github.io/projects/complaint-sorter/demo/

*Planned with the CFPB consumer-complaint database, but the CFPB no longer publishes complaint text: its API
rejects the narrative field and the bulk file has no text column (checked 2026-10-01). NHTSA's complaints are
the closest public substitute: real consumer complaints, in the filer's words, with the filer's own category.*

## Steps

| Step | File | What it does |
|---|---|---|
| 1 | `download.py` | 114,912 complaint records for popular models from 10 makes (model years 2019–2024) from api.nhtsa.gov; ~4 min |
| 2 | `build.py` | 81,324 unique complaints; the 12 most common components; fixed sets: 60 dev complaints (2025) and 480 test complaints (2026, 40 per component) |
| 3 | `baseline.py` | TF-IDF + logistic regression trained on 35,187 complaints filed before 2026 |
| 4 | `sort.py`, `llm.py` | gpt-5.4-mini with structured output (category, crash/fire, summary); the prompt was written on the dev set, then run once on test |
| 5 | `evaluate.py` | accuracy, macro-F1, per-category scores, paired test, crash/fire agreement, cost |
| 6 | `review.py` | every LLM-vs-owner disagreement read and given a verdict with a one-line reason |
| 7 | `demo.py`, `demo/` | static page with all 480 recorded results |
| 8 | `check.py` | fair sets, test after all training data, scores recompute, every disagreement reviewed, demo text scrubbed |

## Run it

```
python3 -m venv venv && ./venv/bin/pip install pandas scikit-learn
./venv/bin/python download.py && ./venv/bin/python build.py
./venv/bin/python baseline.py
echo "OPENAI_API_KEY=..." > .env         # type it in your own terminal; .env is git-ignored
./venv/bin/python sort.py dev && ./venv/bin/python sort.py test     # about $0.53 in total
./venv/bin/python evaluate.py && ./venv/bin/python review.py && ./venv/bin/python demo.py
./venv/bin/python check.py
```

## Results (480 test complaints filed Jan–Sep 2026, 40 per component)

| | Matches the owner's category | Macro-F1 | Labelled examples used |
|---|---:|---:|---:|
| LLM (gpt-5.4-mini, zero-shot) | **88.1%** | 0.884 | 0 |
| TF-IDF + logistic regression | 85.4% | 0.858 | 35,187 |

- **The LLM's lead isn't proven.** On the same complaints it was right on 32 that the classic model missed and wrong
  on 19 it got, so the gap could be chance (exact McNemar p = 0.09).
- **Together they're a triage rule.** They agree on 87.9% of complaints and are then right 92.7% of the time. The
  other 12% could go to a person.
- **"Wrong" is often not wrong.** Of the LLM's 57 disagreements with the owner: 11 are clear LLM mistakes. In 11 the text
  fits the LLM's category better, 28 could go either way (a part that spans two categories, two problems in one complaint,
  a filing convention), and 7 don't say what failed (a recall number or a redaction). So clear errors were 11 of 480 (2.3%).
- **Weakest category: electrical system** (F1 0.72). The LLM over-uses it for anything with a warning light or software.
  Strongest: exterior lighting (0.97) and air bags (0.96).
- **Crash or fire:** the LLM agreed with the owner's tick boxes on 475 of 480 (99.0%).
- **Cost and speed:** $0.43 for the 480 test complaints ($0.89 per 1,000); median 1.3 s each.

## Honest notes

- The prompt changed once, after reading the dev misses: sunroof glass moved to visibility, and adaptive cruise / ADAS
  were added to forward collision avoidance. Both match how 2019–2025 complaints are filed (66% and 90% of keyword
  matches). Dev went from 53/60 to 54/60. The test set was run once, after that.
- The verdicts in `review.py` are judgement calls, each with its reason, so they can be disputed line by line.
- The "owner's category" is what the filer picked, not an expert's diagnosis.
- Complaint text is NHTSA's public record. This repo doesn't store it (the scripts re-download it). The demo shows the
  480 test complaints with phone numbers, e-mail addresses and town/ZIP prefixes removed.

## Data

NHTSA Office of Defects Investigation, vehicle complaints API, https://api.nhtsa.gov/complaints (US government,
public domain), downloaded 2026-10-01.

## Not done

- Complaints listing several components (34% of all) and the less common components are left out.
- No fine-tuned or few-shot LLM comparison; no run of the full 81,000 complaints.
- The summaries aren't graded; the demo shows them next to the owner's text so you can judge.
