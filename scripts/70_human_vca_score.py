"""Human validation of the VCA author/year matcher (300 stratified DOIs).

Usage: python 70_human_vca_score.py

Reads the two annotators' labels (results/human_vca_annotator{1,2}_300.csv) and
the automated matcher's verdict per item (results/human_vca_300_validator_verdicts.csv).

Outputs: inter-annotator Cohen's κ; Cohen's κ between the human consensus
(items where both annotators agree) and the matcher, with false positives
(matcher VALID, humans INVALID) and false negatives (matcher INVALID, humans
VALID). Writes results/human_vca_300_results.json.
"""
import json
import sys
from pathlib import Path
from collections import Counter

DRY = Path(__file__).resolve().parents[1]

def cohens_kappa(y1, y2):
    """Compute Cohen's κ for two lists of binary labels."""
    assert len(y1) == len(y2)
    n = len(y1)
    # Confusion matrix
    tp = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 0)

    po = (tp + tn) / n  # observed agreement
    # Expected agreement
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((tn + fn) / n) * ((tn + fp) / n)
    pe = p_yes + p_no

    if pe == 1.0:
        return 1.0
    kappa = (po - pe) / (1 - pe)
    return kappa


def _load_verdicts(path):
    import csv
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get("human_verdict", "").strip().upper()
            if v in ("VALID", "INVALID"):
                out[int(row["item_id"])] = 1 if v == "VALID" else 0
    return out


def main():
    # Inter-annotator agreement on the 300-item blinded validation.
    # Both annotators' labels are released; no system-identity mapping is needed.
    a1 = _load_verdicts(DRY / "results" / "human_vca_annotator1_300.csv")
    a2 = _load_verdicts(DRY / "results" / "human_vca_annotator2_300.csv")

    items = sorted(set(a1) & set(a2))
    y1 = [a1[i] for i in items]
    y2 = [a2[i] for i in items]
    n = len(items)

    kappa = cohens_kappa(y1, y2)
    agree = sum(1 for a, b in zip(y1, y2) if a == b)

    print(f"{'='*50}")
    print("HUMAN VCA VALIDATION — INTER-ANNOTATOR AGREEMENT")
    print(f"{'='*50}")
    print(f"N = {n} (blind, stratified DOIs; 2 independent annotators)")
    print(f"Raw agreement: {agree}/{n} ({agree/n*100:.1f}%)")
    print(f"Cohen's κ = {kappa:.3f}")

    if kappa >= 0.81:
        print("Interpretation: ALMOST PERFECT agreement")
    elif kappa >= 0.61:
        print("Interpretation: SUBSTANTIAL agreement")
    elif kappa >= 0.41:
        print("Interpretation: MODERATE agreement")
    else:
        print("Interpretation: FAIR or lower")

    # Human consensus vs automated matcher
    import csv
    matcher = {}
    with open(DRY / "results" / "human_vca_300_validator_verdicts.csv") as f:
        for row in csv.DictReader(f):
            matcher[int(row["item_id"])] = 1 if row["validator_verdict"].strip().upper() == "VALID" else 0
    consensus = [i for i in items if a1[i] == a2[i] and i in matcher]
    hc = [a1[i] for i in consensus]
    mv = [matcher[i] for i in consensus]
    kappa_hv = cohens_kappa(hc, mv)
    fp = sum(1 for h, v in zip(hc, mv) if v == 1 and h == 0)
    fn = sum(1 for h, v in zip(hc, mv) if v == 0 and h == 1)
    agree_hv = sum(1 for h, v in zip(hc, mv) if h == v)
    print(f"\nHuman consensus vs matcher: N = {len(consensus)}, agreement {agree_hv}/{len(consensus)}, "
          f"Cohen's κ = {kappa_hv:.3f}, false positives = {fp}, false negatives = {fn}")

    out = {
        "n": n, "annotators": 2,
        "inter_annotator_kappa": round(kappa, 3),
        "inter_annotator_agreement_pct": round(agree / n * 100, 1),
        "human_vs_validator_kappa": round(kappa_hv, 3),
        "human_vs_validator_agreement_pct": round(agree_hv / len(consensus) * 100, 1),
        "disagreements_human_human": n - agree,
        "disagreements_consensus_vs_validator": len(consensus) - agree_hv,
        "false_negatives": fn, "false_positives": fp,
    }
    (DRY / "results" / "human_vca_300_results.json").write_text(json.dumps(out, indent=2))
    print("Saved results/human_vca_300_results.json")


if __name__ == "__main__":
    main()
