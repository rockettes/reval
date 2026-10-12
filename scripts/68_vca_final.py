"""VCA Final — unified VCA strict + VCA±1yr for all 5 generators.

Uses ONLY existing cached data (zero API calls):
- doi_4way_v2.json (CrossRef existence + cited_year/cr_year from script 46b)
- vca_70b_cache.json (Llama 70B semantic verdicts from script 64)
- Response text files (for year extraction when 46b cited_year is None)

Definitions:
- VCA (strict): DOI exists on CrossRef AND 70B=VALID AND cited_year == cr_year
- VCA±1yr:      DOI exists on CrossRef AND 70B=VALID AND |cited_year - cr_year| <= 1
- If cited_year cannot be determined: counts for VCA±1yr only (70B confirmed ±1yr)
- Denominator: total DOIs cited (including fabricated)
"""
import json
import re
from pathlib import Path
from collections import defaultdict

DRY = Path(__file__).resolve().parents[1]

# Load cached data
cr_data = json.load(open(DRY / "doi_4way_v2.json"))
vca_cache = json.load(open(DRY / "vca_70b_cache.json"))

# Load response texts for year extraction fallback
RESPONSE_FILES = {
    "Ours": "responses_350_RAG.json",
    "GLM": "responses_350_glm_v2.json",
    "Gemini": "responses_350_gemini_v2.json",
    "Grok": "responses_350_grok_v2.json",
    "GemSearch": "responses_350_gemsearch.json",
    "GPT54": "responses_350_gpt54_v2.json",
}
responses = {}
for gen, fname in RESPONSE_FILES.items():
    data = json.load(open(DRY / fname))
    responses[gen] = {r["id"]: r.get("answer", "") for r in data}


def extract_year_near_doi(text, doi):
    """Extract year from text in 200-char window before DOI."""
    pos = text.lower().find(doi.lower())
    if pos < 0:
        return None
    window = text[max(0, pos - 200):pos]
    years = re.findall(r'\b(19[5-9]\d|20[0-3]\d)\b', window)
    if years:
        return years[-1]  # closest year before DOI
    return None


def year_diff(cited_y, cr_y):
    """Return absolute year difference, or None if can't compute."""
    if not cited_y or not cr_y:
        return None
    try:
        return abs(int(cited_y) - int(cr_y))
    except (ValueError, TypeError):
        return None


def process_generator(gen):
    """Process one generator, return stats dict."""
    stats = {
        "total_dois": 0,
        "exist": 0,
        "fabricated": 0,
        "vca_strict": 0,
        "vca_lenient": 0,
        "seventy_b_valid": 0,
        "seventy_b_invalid": 0,
        "year_exact": 0,
        "year_drift": 0,
        "year_unknown": 0,
    }
    by_type = defaultdict(lambda: {
        "total_dois": 0, "exist": 0, "fabricated": 0,
        "vca_strict": 0, "vca_lenient": 0,
    })

    # Get question types
    resp_data = json.load(open(DRY / RESPONSE_FILES[gen]))
    q_types = {r["id"]: r.get("question_type", "UNK") for r in resp_data}

    for q in cr_data[gen]["results"]:
        qid = q["qid"]
        qtype = q_types.get(qid, "UNK")
        answer = responses[gen].get(qid, "")

        for d in q["dois"]:
            doi = d["doi"]
            stats["total_dois"] += 1
            by_type[qtype]["total_dois"] += 1

            # Step 1: CrossRef existence
            if not d.get("cr_valid"):
                stats["fabricated"] += 1
                by_type[qtype]["fabricated"] += 1
                continue

            stats["exist"] += 1
            by_type[qtype]["exist"] += 1

            # Step 2: 70B verdict (author validation)
            cache_key = f"{gen}_{qid}_{doi}"
            verdict = vca_cache.get(cache_key)
            if verdict is None:
                # Try alternative key formats
                for alt in [f"{gen}_{qid}_{doi.lower()}"]:
                    verdict = vca_cache.get(alt)
                    if verdict:
                        break

            if verdict != "VALID":
                stats["seventy_b_invalid"] += 1
                continue

            stats["seventy_b_valid"] += 1

            # Step 3: Year check (separate strict from ±1yr)
            cr_year = d.get("cr_year")
            cited_year = d.get("cited_year")

            # Fallback: extract year from text if 46b didn't get it
            if not cited_year:
                cited_year = extract_year_near_doi(answer, doi)

            yd = year_diff(cited_year, cr_year)

            if yd is not None and yd == 0:
                # Exact year match → both VCA and VCA±1yr
                stats["vca_strict"] += 1
                stats["vca_lenient"] += 1
                stats["year_exact"] += 1
                by_type[qtype]["vca_strict"] += 1
                by_type[qtype]["vca_lenient"] += 1
            elif yd is not None and yd == 1:
                # Year off by 1 → only VCA±1yr
                stats["vca_lenient"] += 1
                stats["year_drift"] += 1
                by_type[qtype]["vca_lenient"] += 1
            elif yd is not None and yd > 1:
                # Year off by >1 — 70B said VALID but year is too far
                # This shouldn't happen often (70B has ±1yr in prompt)
                # Don't count for either metric
                stats["year_drift"] += 1
            else:
                # Can't determine year → count as VCA±1yr only
                # (70B already confirmed author + year ±1yr)
                stats["vca_lenient"] += 1
                stats["year_unknown"] += 1
                by_type[qtype]["vca_lenient"] += 1

    return stats, dict(by_type)


def pct(n, d):
    return round(n / d * 100, 1) if d > 0 else 0.0


def main():
    print("=" * 80)
    print("VCA FINAL — 6 generators, zero API calls")
    print("=" * 80)

    all_stats = {}
    all_by_type = {}
    gens = ["Ours", "GPT54", "Gemini", "GemSearch", "Grok", "GLM"]

    for gen in gens:
        if gen not in cr_data:
            print(f"SKIP {gen}: not in doi_4way_v2.json")
            continue
        stats, by_type = process_generator(gen)
        all_stats[gen] = stats
        all_by_type[gen] = by_type

    # Main table
    print(f"\n{'Gen':<12} {'DOIs':>6} {'Exist%':>8} {'Fabr%':>8} {'VCA':>8} {'VCA±1yr':>9}")
    print("-" * 53)
    for gen in gens:
        if gen not in all_stats:
            continue
        s = all_stats[gen]
        t = s["total_dois"]
        print(f"{gen:<12} {t:>6} {pct(s['exist'],t):>7.1f}% {pct(s['fabricated'],t):>7.1f}% {pct(s['vca_strict'],t):>7.1f}% {pct(s['vca_lenient'],t):>8.1f}%")

    # Detail breakdown
    print(f"\n{'Gen':<12} {'70B_VALID':>10} {'yr_exact':>9} {'yr_drift':>9} {'yr_unk':>7}")
    print("-" * 49)
    for gen in gens:
        if gen not in all_stats:
            continue
        s = all_stats[gen]
        print(f"{gen:<12} {s['seventy_b_valid']:>10} {s['year_exact']:>9} {s['year_drift']:>9} {s['year_unknown']:>7}")

    # By question type
    types_order = ["DIVERGENT", "POPULATION", "MULTI_HOP", "COMP", "TEMPORAL", "MECHANISM", "NUM", "ADV_OOS"]

    for metric_name, key in [("VCA (strict)", "vca_strict"), ("VCA±1yr", "vca_lenient")]:
        print(f"\n=== {metric_name} by question type ===")
        header = f"{'Tipo':<12} {'N':>3}"
        for g in gens:
            if g in all_stats:
                header += f" {g:>12}"
        print(header)
        print("-" * (18 + 13 * len([g for g in gens if g in all_stats])))

        for t in types_order:
            n_q = sum(1 for r in json.load(open(DRY / RESPONSE_FILES["Ours"])) if r.get("question_type") == t)
            row = f"{t:<12} {n_q:>3}"
            for g in gens:
                if g not in all_by_type:
                    continue
                bt = all_by_type[g].get(t, {"total_dois": 0, key: 0})
                td = bt["total_dois"]
                val = bt[key]
                row += f" {pct(val, td):>11.1f}%"
            print(row)

        # Total row
        row = f"{'TOTAL':<12} {350:>3}"
        for g in gens:
            if g not in all_stats:
                continue
            s = all_stats[g]
            row += f" {pct(s[key], s['total_dois']):>11.1f}%"
        print("-" * (18 + 13 * len([g for g in gens if g in all_stats])))
        print(row)

    # Save
    out = {
        "summary": {},
        "by_type": {},
        "methodology": {
            "vca_strict": "DOI exists AND 70B=VALID AND cited_year == cr_year",
            "vca_lenient": "DOI exists AND 70B=VALID AND |cited_year - cr_year| <= 1 (or year unknown)",
            "denominator": "total DOIs cited (including fabricated)",
            "author_validation": "Llama 3.3 70B (temperature 0.0)",
            "api_calls": 0,
        },
    }
    for gen in gens:
        if gen not in all_stats:
            continue
        s = all_stats[gen]
        t = s["total_dois"]
        out["summary"][gen] = {
            "total_dois": t,
            "exist": s["exist"],
            "fabricated": s["fabricated"],
            "exist_pct": pct(s["exist"], t),
            "fabr_pct": pct(s["fabricated"], t),
            "vca_strict": s["vca_strict"],
            "vca_strict_pct": pct(s["vca_strict"], t),
            "vca_lenient": s["vca_lenient"],
            "vca_lenient_pct": pct(s["vca_lenient"], t),
            "seventy_b_valid": s["seventy_b_valid"],
            "year_exact": s["year_exact"],
            "year_drift": s["year_drift"],
            "year_unknown": s["year_unknown"],
        }
        out["by_type"][gen] = {}
        for qtype, bt in all_by_type[gen].items():
            td = bt["total_dois"]
            out["by_type"][gen][qtype] = {
                "total_dois": td,
                "exist": bt["exist"],
                "fabricated": bt["fabricated"],
                "vca_strict": bt["vca_strict"],
                "vca_strict_pct": pct(bt["vca_strict"], td),
                "vca_lenient": bt["vca_lenient"],
                "vca_lenient_pct": pct(bt["vca_lenient"], td),
            }

    out_path = DRY / "vca_final_5way.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    main()
