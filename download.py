"""Step 1: download real vehicle-safety complaints from NHTSA's public API (no key needed).

    ./venv/bin/python download.py

Popular models from 10 makes, model years 2019-2024. Each complaint has the owner's own words ("summary"), the
component(s) the owner filed it under, and crash / fire boxes. Saved as one JSON file per make/model/year in
data/raw/ (not committed); build.py turns them into data/complaints.csv.gz.
"""
import json
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
API = "https://api.nhtsa.gov"
CTX = ssl.create_default_context(cafile="/etc/ssl/cert.pem")     # python.org builds ship without system certs
YEARS = range(2019, 2025)
FAMILIES = {
    "toyota": ["CAMRY", "COROLLA", "RAV4", "HIGHLANDER", "TACOMA", "TUNDRA", "SIENNA"],
    "honda": ["CIVIC", "ACCORD", "CR-V", "PILOT", "ODYSSEY", "HR-V"],
    "ford": ["F-150", "EXPLORER", "ESCAPE", "BRONCO", "MUSTANG MACH-E", "EDGE"],
    "chevrolet": ["SILVERADO 1500", "EQUINOX", "MALIBU", "TRAVERSE", "BOLT"],
    "nissan": ["ROGUE", "ALTIMA", "SENTRA", "PATHFINDER"],
    "hyundai": ["ELANTRA", "TUCSON", "SANTA FE", "SONATA", "PALISADE"],
    "kia": ["SORENTO", "TELLURIDE", "FORTE", "SPORTAGE", "SOUL"],
    "jeep": ["GRAND CHEROKEE", "WRANGLER", "CHEROKEE", "COMPASS"],
    "tesla": ["MODEL 3", "MODEL Y"],
    "subaru": ["OUTBACK", "FORESTER", "CROSSTREK", "ASCENT"],
}


def get(path, **params):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "complaint-sorter (student project)"}), timeout=60, context=CTX) as r:
                return json.loads(r.read())
        except Exception as e:
            print(f"retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed: {url}")


def models(make, year):
    names = {r["model"] for r in get("products/vehicle/models", make=make, modelYear=year, issueType="c")["results"]}
    return sorted(n for n in names if any(n.startswith(f) for f in FAMILIES[make]))


def fetch(job):
    make, model, year = job
    out = RAW / f"{make}_{model.replace('/', '-').replace(' ', '_')}_{year}.json"
    if not out.exists():
        out.write_text(json.dumps(get("complaints/complaintsByVehicle", make=make, model=model, modelYear=year)["results"]))
        time.sleep(0.2)                                   # be polite to a free public API
    return len(json.loads(out.read_text()))


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(mk, m, y) for mk in FAMILIES for y in YEARS for m in models(mk, y)]
    with ThreadPoolExecutor(4) as pool:
        counts = list(pool.map(fetch, jobs))
    print(f"{len(jobs)} make/model/year files, {sum(counts):,} complaints -> {RAW}")
