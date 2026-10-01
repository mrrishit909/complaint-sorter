"""Step 4: the LLM reads each complaint and returns a category, a crash/fire flag and a one-line summary.

    ./venv/bin/python sort.py dev      # 60 complaints from 2025: where the prompt was written and adjusted
    ./venv/bin/python sort.py test     # 480 complaints from 2026: run once, after the prompt was final

Zero-shot: the model gets the category list with a one-line description each, and no labelled examples.
Answers are cached in results/llm_<set>.jsonl (re-running only calls the API for missing ones); cost is logged.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import llm

HERE = Path(__file__).parent
# v2: after reading the dev misses, sunroof glass moved to VISIBILITY and adaptive cruise / ADAS added to
# FORWARD COLLISION AVOIDANCE; both match how 2019-2025 complaints are filed (66% and 90% of keyword matches).
DESCRIBE = {
    "POWER TRAIN": "transmission, gear shifting, clutch, axles, drive shafts, transfer case, rolling away in park",
    "ENGINE": "the engine itself and its cooling: oil, overheating, coolant, engine damage or failure, timing chain",
    "ELECTRICAL SYSTEM": "battery, wiring, alternator, screens and infotainment, instrument cluster, keys, software or electrical faults",
    "VISIBILITY": "windshield, sunroof and other glass, wipers and washers, mirrors, defroster, anything blocking the driver's view",
    "STEERING": "steering wheel, power-steering assist, steering column, steering that is hard, loose or locks",
    "FORWARD COLLISION AVOIDANCE": "automatic emergency braking, collision warnings, adaptive cruise and other driver-assistance (ADAS) features, including braking for no reason",
    "STRUCTURE": "body and frame, hood, doors and panels, rust, parts coming off the body",
    "FUEL/PROPULSION SYSTEM": "fuel pump, fuel leaks and smell, fuel gauge, and losing power because of the fuel or drive system",
    "SERVICE BRAKES": "the brakes the driver uses: pedal, pads, rotors, ABS, brake fluid, brakes that fail or grind",
    "AIR BAGS": "air bags and their sensors or warning light; deploying when they shouldn't or not deploying",
    "EXTERIOR LIGHTING": "headlights, tail lights, brake lights, turn signals, daytime running lights",
    "SUSPENSION": "springs, shocks and struts, control arms, ball joints, sway bars, wheel alignment from suspension damage",
}
SYSTEM = (
    "You sort vehicle-safety complaints that car owners filed with NHTSA, the US road-safety agency.\n"
    "Pick the ONE category the complaint is mainly about: the part that failed, not its consequences.\n"
    "Categories:\n" + "\n".join(f"- {k}: {v}" for k, v in DESCRIBE.items()) + "\n"
    "crash_or_fire: true only if the owner says the vehicle actually crashed, hit something, or caught fire "
    "(not a near miss, smoke alone, or a fear that it might).\n"
    "summary: one plain sentence of at most 25 words saying what went wrong; no names, VINs or locations."
)
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["category", "crash_or_fire", "summary"],
    "properties": {"category": {"type": "string", "enum": list(DESCRIBE)},
                   "crash_or_fire": {"type": "boolean"}, "summary": {"type": "string"}},
}


def run(which):
    d = pd.read_csv(HERE / "data" / "complaints.csv.gz")
    ids = json.loads((HERE / "data" / "sets.json").read_text())[which]
    todo = d[d["odi"].isin(ids)]
    out = HERE / "results" / f"llm_{which}.jsonl"
    done = {json.loads(l)["odi"] for l in out.open()} if out.exists() else set()
    todo = todo[~todo["odi"].isin(done)]

    def one(row):
        t0 = time.time()
        a = llm.chat_json(SYSTEM, row.summary[:4000], SCHEMA, name="complaint")
        return {"odi": int(row.odi), **a, "seconds": round(time.time() - t0, 2)}

    with ThreadPoolExecutor(6) as pool, out.open("a") as f:
        for r in pool.map(one, todo.itertuples()):
            f.write(json.dumps(r) + "\n")
    cost = HERE / "results" / "cost.json"
    log = json.loads(cost.read_text()) if cost.exists() else {}
    log[which] = {k: log.get(which, {}).get(k, 0) + v for k, v in llm.spent.items()}
    cost.write_text(json.dumps(log, indent=2) + "\n")
    print(f"{which}: {len(todo)} new answers; this run ${llm.spent['usd']:.3f} "
          f"({llm.spent['in_tokens']:,} in / {llm.spent['out_tokens']:,} out tokens)")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "dev")
