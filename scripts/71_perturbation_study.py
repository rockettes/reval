"""Controlled citation-perturbation study (rebuttal experiment, Reviewer rxLv).

Takes the 350 GemSearch responses (fully reproducible public system), replaces every
CrossRef-valid DOI with a fabricated (non-resolving) DOI of identical format, keeping
the response text otherwise byte-identical. Re-judges the perturbed variants with the
same 4 judges (Atla, GLM, Grok, GPT54 — Gemini excluded, same base model) and compares
paired REVAL against the original scores.

Prediction (from the paper's TOST result): judges cannot distinguish valid from
fabricated DOIs -> Delta REVAL ~ 0, including on the Citation Verifiability dimension.

Phases:
  A  build   -> responses_350_gemsearch_perturbFAB.json (+ fabrication log)
  B  judge   -> scores_perturbFAB.json (1400 calls) + scores_retest50.json (200 calls)
  C  analyze -> perturbation_analysis.json (paired deltas, bootstrap CI, TOST)

Usage: python 71_perturbation_study.py A|B|C [fab|min]
  fab -> hash-based digit substitution (arm 1; breaks publisher surface patterns:
         zero-padding, embedded years). Judges detected these (CiteVerif -1.58) —
         evidence of surface-form checking, not verification.
  min -> minimal edit (arm 2; flip trailing digits only, preserving length, padding
         and year fields; 404-verified). Surface-indistinguishable from the original.
"""
# NOTE (release): this script is published as run. It reads and writes the working
# directory layout used during the experiments ("responses_350_<system>.json",
# "scores_*.json"); see the file-name mapping in README.md to match these against
# the released responses/, scores/ and results/ folders. API keys are read from a
# local .env; no credentials are included here.
import hashlib
import importlib.util
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DRY = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent

GEMSEARCH = DRY / "responses_350_gemsearch.json"
DOI4WAY = DRY / "doi_4way_v2.json"
MERGED = DRY / "scores_REVAL_merged.json"

ARM = (sys.argv[2] if len(sys.argv) > 2 else "fab").lower()
assert ARM in ("fab", "min")
SUF = "FAB" if ARM == "fab" else "MIN"

PERTURBED = DRY / f"responses_350_gemsearch_perturb{SUF}.json"
FABLOG = DRY / ("perturbation_fabrication_log.json" if ARM == "fab"
                else "perturbation_fabrication_log_min.json")
NONEXIST_CACHE = DRY / "crossref_nonexistence_cache.json"
SCORES_PERTURB = DRY / f"scores_perturb{SUF}.json"
SCORES_RETEST = DRY / "scores_retest50.json"  # shared across arms (noise floor)
ANALYSIS = DRY / ("perturbation_analysis.json" if ARM == "fab"
                  else "perturbation_analysis_min.json")

JUDGES = ["Atla", "GLM", "Grok", "GPT54"]  # Gemini excluded (same base model as GemSearch)
RETEST_N = 50
UA = {"User-Agent": "REVAL-rebuttal/1.0 (mailto:anonymous@review.org)"}


def load_judge_module():
    """Import 44_judge_REVAL.py (filename starts with a digit -> importlib)."""
    spec = importlib.util.spec_from_file_location("judge44", SCRIPTS / "44_judge_REVAL.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- Phase A: build perturbed responses ----------

def mutate_doi(doi, salt=0):
    """Deterministically mutate a DOI suffix into a same-format, different string.

    Salts 0-4: in-place digit substitution (keeps length). Dense namespaces
    (e.g. 10.7717/peerj.NNNN, where nearly every NNNN resolves) defeat this,
    so salts 5+: append hash digits, lengthening the trailing number beyond
    the namespace's populated range.
    """
    prefix, _, suffix = doi.partition("/")
    h = hashlib.sha256(f"{doi}|{salt}".encode()).hexdigest()
    digits = [c for c in h if c.isdigit()] or ["7", "3"]
    if salt >= 5:
        extra = "".join(digits[:2 + (salt - 5)])
        return prefix + "/" + suffix + extra
    di = 0
    out = []
    for ch in suffix:
        if ch.isdigit():
            # replace with a different digit, keeping length/format
            repl = digits[di % len(digits)]
            if repl == ch:
                repl = str((int(ch) + 3) % 10)
            out.append(repl)
            di += 1
        else:
            out.append(ch)
    new = prefix + "/" + "".join(out)
    if new == doi:  # no digits in suffix -> append deterministic digits
        new = doi + digits[0] + digits[1]
    return new


def crossref_exists(doi, cache):
    if doi in cache:
        return cache[doi]
    for attempt in range(3):
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=30)
            if r.status_code == 200:
                cache[doi] = True
                return True
            if r.status_code == 404:
                cache[doi] = False
                return False
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None  # unknown -> caller must pick another mutation


def mutate_doi_minimal(doi, salt=0):
    """Arm 2: flip a single trailing digit, preserving length, padding and year fields.

    salt walks (position_from_end, replacement_digit) pairs deterministically so
    surface form stays indistinguishable: jsc.0000000000003936 -> ...3938.
    """
    prefix, _, suffix = doi.partition("/")
    idxs = [i for i, ch in enumerate(suffix) if ch.isdigit()]
    if not idxs:
        return doi + str((salt % 9) + 1)
    # walk digits from last to first; 9 replacement digits per position
    pos = idxs[-1 - (salt // 9) % len(idxs)]
    old_d = int(suffix[pos])
    new_d = (old_d + 1 + (salt % 9)) % 10
    if new_d == old_d:
        new_d = (old_d + 1) % 10
    s = suffix[:pos] + str(new_d) + suffix[pos + 1:]
    return prefix + "/" + s


def fabricate(doi, cache):
    """Return a DOI (mutated per ARM strategy) verified to NOT exist on CrossRef."""
    mut = mutate_doi_minimal if ARM == "min" else mutate_doi
    # min arm: walk digit positions from the end; dense namespaces (sequential
    # article numbers) need deeper positions — a flipped mid-string zero in a
    # zero-padded ID keeps the surface format but lands in unpopulated ranges.
    n_digits = sum(ch.isdigit() for ch in doi.partition("/")[2])
    max_salt = max(9 * n_digits, 30) if ARM == "min" else 10
    tried = set()
    for salt in range(max_salt):
        cand = mut(doi, salt)
        if cand == doi or cand in tried:
            continue
        tried.add(cand)
        exists = crossref_exists(cand, cache)
        if exists is False:
            return cand
    raise RuntimeError(f"could not fabricate for {doi}")


def replace_doi_in_text(text, old, new):
    """Replace DOI occurrences case-insensitively, preserving count."""
    rx = re.compile(re.escape(old), re.IGNORECASE)
    return rx.subn(new, text)


def phase_a():
    responses = json.loads(GEMSEARCH.read_text())
    d4 = json.loads(DOI4WAY.read_text())["GemSearch"]["results"]
    valid_by_qid = {}
    for r in d4:
        valid_by_qid[r["qid"]] = [d["doi"] for d in r.get("dois", []) if d.get("cr_valid")]

    cache = json.loads(NONEXIST_CACHE.read_text()) if NONEXIST_CACHE.exists() else {}
    fablog = json.loads(FABLOG.read_text()) if FABLOG.exists() else {}

    out = []
    n_replaced = 0
    n_qs_touched = 0
    for resp in responses:
        qid = resp["id"]
        text = resp["answer"]
        valids = sorted(set(valid_by_qid.get(qid, [])), key=len, reverse=True)
        replacements = {}
        for doi in valids:
            if doi in fablog:
                fake = fablog[doi]
            else:
                fake = fabricate(doi, cache)
                fablog[doi] = fake
                NONEXIST_CACHE.write_text(json.dumps(cache, indent=0))
                FABLOG.write_text(json.dumps(fablog, indent=1))
            text, n = replace_doi_in_text(text, doi, fake)
            if n:
                replacements[doi] = {"fake": fake, "n_occurrences": n}
                n_replaced += n
        if replacements:
            n_qs_touched += 1
        new_resp = dict(resp)
        new_resp["answer"] = text
        new_resp["perturbed"] = bool(replacements)
        new_resp["replacements"] = replacements
        out.append(new_resp)
        print(f"Q{qid}: {len(replacements)} valid DOIs -> fabricated ({sum(v['n_occurrences'] for v in replacements.values())} occurrences)", flush=True)

    PERTURBED.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nDone. {n_qs_touched}/350 responses perturbed, {n_replaced} DOI occurrences replaced.")
    print(f"Sanity: text identical except DOI strings; verify with phase C.")


# ---------- Phase B: judge ----------

save_lock = threading.Lock()


def live_judges():
    """Drop judges whose API is unusable (dead key OR exhausted quota) — resume fills later.

    GET /models returns 200 even with zero credits, so probe with a real
    minimal completion and inspect the error code.
    """
    import os
    alive = list(JUDGES)
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY','')}",
                               "Content-Type": "application/json"},
                      json={"model": "gpt-5.4-2026-03-05",
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_completion_tokens": 16},
                      timeout=60)
    if r.status_code != 200:
        reason = ""
        try:
            reason = r.json().get("error", {}).get("code", "")
        except Exception:
            pass
        print(f"WARN: OpenAI unusable (HTTP {r.status_code} {reason}) -> skipping GPT54 judge lane this run")
        alive.remove("GPT54")
    return alive


def phase_b():
    mod = load_judge_module()
    perturbed = {r["id"]: r for r in json.loads(PERTURBED.read_text())}
    originals = {r["id"]: r for r in json.loads(GEMSEARCH.read_text())}
    judges = live_judges()

    # retest subsample: 50 qids, stratified by id order (deterministic)
    retest_qids = sorted(originals)[::len(originals) // RETEST_N][:RETEST_N]

    jobs = []  # (tag, judge, qid, question, answer)
    scores_p = json.loads(SCORES_PERTURB.read_text()) if SCORES_PERTURB.exists() else []
    scores_r = json.loads(SCORES_RETEST.read_text()) if SCORES_RETEST.exists() else []
    done_p = {(s["judge"], s["qid"]) for s in scores_p if s.get("scores")}
    done_r = {(s["judge"], s["qid"]) for s in scores_r if s.get("scores")}

    for j in judges:
        for qid, r in perturbed.items():
            if r.get("perturbed") and (j, qid) not in done_p:
                jobs.append(("perturb", j, qid, r["question"], r["answer"]))
        for qid in retest_qids:
            if (j, qid) not in done_r:
                r = originals[qid]
                jobs.append(("retest", j, qid, r["question"], r["answer"]))

    print(f"Pending judge calls: {len(jobs)} (perturb + retest)")
    t0 = time.time()
    counter = [0]

    def run(tag, judge, qid, question, answer):
        try:
            prompt = mod.JUDGE_PROMPT.format(question=question, answer=answer)
            text = mod.JUDGE_FNS[judge](prompt)
            scores = mod.parse_scores(text)
            if scores is None:
                return tag, {"judge": judge, "generator": "GemSearchPerturb" if tag == "perturb" else "GemSearchRetest",
                             "qid": qid, "error": "parse_fail", "raw": text[:300]}
            return tag, {"judge": judge, "generator": "GemSearchPerturb" if tag == "perturb" else "GemSearchRetest",
                         "qid": qid, "scores": scores, "reval": round(mod.reval_score(scores), 4)}
        except Exception as e:
            return tag, {"judge": judge, "generator": tag, "qid": qid,
                         "error": f"{type(e).__name__}: {str(e)[:120]}"}

    def save(tag, entry):
        with save_lock:
            (scores_p if tag == "perturb" else scores_r).append(entry)
            counter[0] += 1
            i = counter[0]
            eta = (time.time() - t0) / i * (len(jobs) - i)
            err = " ERR" if "error" in entry else f" reval={entry.get('reval','?')}"
            print(f"  [{i:5d}/{len(jobs)}] {tag:7s} {entry['judge']:6s} Q{entry['qid']:3d}{err} | ETA {eta/60:.0f}min", flush=True)
            if i % 25 == 0:
                SCORES_PERTURB.write_text(json.dumps(scores_p, ensure_ascii=False, indent=1))
                SCORES_RETEST.write_text(json.dumps(scores_r, ensure_ascii=False, indent=1))

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run, *j) for j in jobs]
        for f in as_completed(futs):
            tag, entry = f.result()
            save(tag, entry)

    SCORES_PERTURB.write_text(json.dumps(scores_p, ensure_ascii=False, indent=1))
    SCORES_RETEST.write_text(json.dumps(scores_r, ensure_ascii=False, indent=1))
    errs = sum(1 for s in scores_p + scores_r if "error" in s)
    print(f"\nDONE. perturb={len(scores_p)}, retest={len(scores_r)}, errors={errs}")


# ---------- Phase C: analysis ----------

def paired_stats(deltas, margin=0.15, B=10000, seed=42):
    import statistics
    rnd = random.Random(seed)
    n = len(deltas)
    mean = sum(deltas) / n
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    # bootstrap CI
    means = []
    for _ in range(B):
        s = [deltas[rnd.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo, hi = means[int(0.025 * B)], means[int(0.975 * B)]
    # TOST via bootstrap: equivalence if 90% CI within +-margin
    lo90, hi90 = means[int(0.05 * B)], means[int(0.95 * B)]
    tost_pass = (lo90 > -margin) and (hi90 < margin)
    return {"n": n, "mean_delta": round(mean, 4), "sd": round(sd, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "ci90": [round(lo90, 4), round(hi90, 4)],
            "tost_margin": margin, "tost_equivalent": tost_pass}


def phase_c():
    mod = load_judge_module()
    merged = json.loads(MERGED.read_text())
    orig = {(s["judge"], s["qid"]): s for s in merged
            if s["generator"] == "GemSearch" and s.get("scores")}
    pert = {(s["judge"], s["qid"]): s for s in json.loads(SCORES_PERTURB.read_text())
            if s.get("scores")}
    retest = {(s["judge"], s["qid"]): s for s in json.loads(SCORES_RETEST.read_text())
              if s.get("scores")}

    report = {}

    # main contrast: perturbed - original (paired per judge x qid)
    keys = sorted(set(orig) & set(pert))
    deltas = [pert[k]["reval"] - orig[k]["reval"] for k in keys]
    report["perturb_vs_original_REVAL"] = paired_stats(deltas)

    # per-dimension
    report["per_dimension"] = {}
    for dim, _, _ in mod.DIMENSIONS:
        dd = [pert[k]["scores"][dim] - orig[k]["scores"][dim] for k in keys]
        report["per_dimension"][dim] = paired_stats(dd)

    # per-judge
    report["per_judge"] = {}
    for j in JUDGES:
        dj = [pert[k]["reval"] - orig[k]["reval"] for k in keys if k[0] == j]
        if dj:
            report["per_judge"][j] = paired_stats(dj)

    # noise floor: retest - original on unperturbed responses
    rkeys = sorted(set(orig) & set(retest))
    rdeltas = [retest[k]["reval"] - orig[k]["reval"] for k in rkeys]
    if rdeltas:
        report["retest_noise_floor_REVAL"] = paired_stats(rdeltas)

    # how many fabricated DOIs are in play
    fablog = json.loads(FABLOG.read_text())
    report["n_dois_fabricated"] = len(fablog)
    perturbed = json.loads(PERTURBED.read_text())
    report["n_responses_perturbed"] = sum(1 for r in perturbed if r.get("perturbed"))

    ANALYSIS.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    phase = (sys.argv[1] if len(sys.argv) > 1 else "A").upper()
    {"A": phase_a, "B": phase_b, "C": phase_c}[phase]()
