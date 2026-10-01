"""Step 6: a person reads every complaint where the LLM disagreed with the owner's label.

    ./venv/bin/python review.py

Verdicts (written by reading each complaint in results/disagreements.csv; reasons kept to one line):
  llm      the text points to the LLM's category more than to the owner's
  label    the owner's category fits better: a real LLM mistake
  both     either is reasonable (overlapping parts, two problems, or a filing convention)
  no_info  the text doesn't say what failed (a recall number only, or redacted)

results/review.csv and results/review.json
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
VERDICTS = {  # odi: (verdict, reason)
    11708141: ("no_info", "text is a FOIA redaction notice"),
    11708662: ("both", "exploding sunroof glass; owners file it under either"),
    11710290: ("label", "random stops tied to a software TSB: the emergency-braking system, not wiring"),
    11712270: ("both", "lag on the gas then a jump into gear: fuel/propulsion or transmission"),
    11712295: ("label", "the failure is water leaking in through the body; air bags are the worry, not the fault"),
    11714018: ("llm", "oil pressure switch and oil burning are the engine"),
    11715403: ("both", "lurched forward with foot on the brake: brakes or unintended acceleration"),
    11716165: ("both", "sudden acceleration; the owner's label names the damage, the LLM the cause"),
    11717090: ("both", "loss of power with check-engine light: engine or fuel/propulsion"),
    11717632: ("both", "backup camera blurred by water: visibility, though cameras are usually filed as electrical"),
    11719238: ("label", "auto-up power window is an electrical fault, not the glass"),
    11721957: ("label", "camera shows no image: electrical, matching how camera faults are filed"),
    11722669: ("llm", "lower control arm ball joint is suspension"),
    11723476: ("both", "loose liftgate hinge cover (structure) shattered the back glass (visibility)"),
    11723486: ("label", "hybrid system shut down and lost propulsion"),
    11726459: ("llm", "low oil pressure seized the engine"),
    11728957: ("llm", "oil disappearing and blue smoke: engine"),
    11729276: ("llm", "catalytic converters are part of the engine's exhaust, not the transmission"),
    11730920: ("both", "parking brake stuck because of a wiring fault"),
    11731512: ("both", "'emergency brake activates' could be the parking brake or automatic emergency braking"),
    11731597: ("both", "stop-start shut the engine off and it would not restart"),
    11732964: ("both", "sputtering, flashing check-engine light, limp mode"),
    11733615: ("both", "'death wobble': steering linkage or suspension"),
    11735791: ("no_info", "recall part unavailable; component redacted"),
    11737292: ("both", "leaking, jammed sunroof"),
    11737659: ("no_info", "recall with no remedy; part not named"),
    11738465: ("llm", "mechanic found the power transfer unit and transfer case failed"),
    11739375: ("label", "roof glass the owner calls a structural part, not a view problem"),
    11740883: ("label", "pinched sunroof drains leaking water: body/structure"),
    11741710: ("no_info", "open recall with no parts; part not named"),
    11742845: ("label", "ADAS controller fault after windshield work: driver assistance"),
    11743103: ("no_info", "legal complaint about a recall; part not named"),
    11743762: ("both", "rear differential mount bushing: drivetrain or suspension"),
    11744006: ("both", "liftgate glass cracked on its own"),
    11744694: ("both", "sunroof would not close"),
    11745994: ("llm", "gear shifter came loose and would not go into park"),
    11746394: ("llm", "car shuts off in gear, blank speedometer: electrical, not the service brakes"),
    11746841: ("both", "two problems: a suspension thump and grinding rear brakes"),
    11747269: ("both", "plug-in hybrid electrical fault with high RPM and a burning smell from the engine"),
    11747445: ("llm", "brakes activating by themselves is automatic emergency braking"),
    11747761: ("both", "surging while braking: engine/throttle or brakes"),
    11749879: ("both", "stop-start would not restart even with a new battery"),
    11752219: ("both", "rearview camera recall for moisture; filed as visibility, cameras usually electrical"),
    11752343: ("both", "engine-top fire; mechanic suspects a high-pressure fuel valve"),
    11753045: ("both", "PCV valve recall and fumes: engine part, fuel-system recall"),
    11755168: ("llm", "Full Self-Driving drove toward a closed rail crossing: driver assistance"),
    11755543: ("label", "road-sign-assist warning disabling the safety features: driver assistance"),
    11756210: ("no_info", "recall paperwork for a converted van; part not named"),
    11761087: ("both", "two problems: worn rear brakes and worn shocks"),
    11761202: ("label", "turn-signal lever: exterior lighting"),
    11762567: ("llm", "a flashing brake warning light points to the brakes"),
    11763843: ("both", "power sliding doors not latching"),
    11765120: ("label", "sunroof glass shattering: visibility, as the prompt itself says"),
    11765644: ("both", "high-voltage battery recall reduced power: electrical or propulsion"),
    11765835: ("no_info", "'not performing' since a recall; nothing more"),
    11766472: ("both", "software pop-up covering the display"),
    11766595: ("both", "Full Self-Driving pop-up distraction"),
}


def main():
    dis = pd.read_csv(HERE / "results" / "disagreements.csv")
    assert set(dis["odi"]) == set(VERDICTS), "review covers exactly the disagreements"
    dis["verdict"] = dis["odi"].map(lambda o: VERDICTS[o][0])
    dis["reason"] = dis["odi"].map(lambda o: VERDICTS[o][1])
    dis.drop(columns="summary").to_csv(HERE / "results" / "review.csv", index=False)
    counts = dis["verdict"].value_counts().to_dict()
    n = 480
    right = n - len(dis)
    out = {"disagreements": len(dis), "verdicts": counts,
           "accuracy_vs_owner_labels": right / n,
           "accuracy_if_llm_and_both_count": (right + counts.get("llm", 0) + counts.get("both", 0)) / n,
           "clear_llm_mistakes": counts.get("label", 0)}
    (HERE / "results" / "review.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
