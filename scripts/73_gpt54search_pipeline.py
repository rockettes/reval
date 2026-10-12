"""Downstream pipeline for the GPT-5.4 + web search rebuttal baselines (Reviewer ySda).

Reuses the canonical functions from 46b (CrossRef), 64 (VCA 70B semantic) and 44
(REVAL judges) via importlib so numbers are protocol-identical to the paper.

Stages:
  crossref          -> extend doi_4way_v2.json with GPT54S_auto / GPT54S_forced
  vca               -> extend vca_70b_cache.json + compute VCA strict / +-1yr
  judge [auto|forced|both] -> scores_gpt54search.json (judges: Atla, GLM, Gemini, Grok;
                       GPT54 excluded = same base model)
  report            -> consolidated table (DOIs, fabr, VCA, VCA+-1, REVAL, search rate)

Usage: python 73_gpt54search_pipeline.py crossref|vca|judge|report [judge-mode]
"""
# NOTE (release): this script is published as run. It reads and writes the working
# directory layout used during the experiments ("responses_350_<system>.json",
# "scores_*.json"); see the file-name mapping in README.md to match these against
# the released responses/, scores/ and results/ folders. API keys are read from a
# local .env; no credentials are included here.
import importlib.util
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DRY = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent

NEW_GENS = {
    "GPT54S_auto":   DRY / "responses_350_gpt54search_auto.json",
    "GPT54S_forced": DRY / "responses_350_gpt54search_forced.json",
}
DOI4WAY = DRY / "doi_4way_v2.json"
VCA_CACHE = DRY / "vca_70b_cache.json"
SCORES_OUT = DRY / "scores_gpt54search.json"
JUDGES = ["Atla", "GLM", "Gemini", "Grok"]  # GPT54 excluded (same base model)


def load_mod(fname, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- stage: crossref ----------

def stage_crossref():
    m46 = load_mod("46b_crossref_v2.py", "cr46b")
    out = json.loads(DOI4WAY.read_text())
    for label, path in NEW_GENS.items():
        if label in out:
            print(f"SKIP {label}: already in doi_4way_v2.json")
            continue
        if not path.exists():
            print(f"SKIP {label}: {path.name} not found (collection still running?)")
            continue
        out[label] = m46.run_model(label, path)
        DOI4WAY.write_text(json.dumps(out, ensure_ascii=False, indent=2))
        s = out[label]["summary"]
        print(f"{label}: {s['total_dois']} DOIs, exist {s['vca_existence']}%, fabr {s['hr_existence']}%")


# ---------- stage: vca (70B semantic) ----------

def stage_vca():
    m64 = load_mod("64_vca_llm_70b.py", "vca64")
    v2 = json.loads(DOI4WAY.read_text())
    cache = json.loads(VCA_CACHE.read_text()) if VCA_CACHE.exists() else {}
    cr_cache_path = DRY / "crossref_title_cache.json"
    cr_cache = json.loads(cr_cache_path.read_text()) if cr_cache_path.exists() else {}

    for gen, path in NEW_GENS.items():
        if gen not in v2:
            print(f"SKIP {gen}: run crossref stage first")
            continue
        responses = {r["id"]: r.get("answer", "") for r in json.loads(path.read_text())}
        n_new = 0
        for q in v2[gen]["results"]:
            qid = q["qid"]
            answer = responses.get(qid, "")
            for d in q["dois"]:
                if not d.get("cr_valid"):
                    continue
                doi = d["doi"]
                key = f"{gen}_{qid}_{doi}"
                if key in cache:
                    continue
                if doi in cr_cache:
                    cr_title = cr_cache[doi].get("title", "Unknown")
                else:
                    info = m64.get_crossref_info(doi)
                    cr_title = (info or {}).get("title", "Unknown")
                    cr_cache[doi] = info or {}
                    cr_cache_path.write_text(json.dumps(cr_cache, ensure_ascii=False, indent=0))
                pos = answer.lower().find(doi.lower())
                excerpt = answer[max(0, pos - 300):pos + len(doi) + 50] if pos >= 0 else answer[:350]
                user = m64.USER_TEMPLATE.format(
                    doi=doi, cr_title=cr_title,
                    cr_author=d.get("cr_first_author") or "unknown",
                    cr_year=d.get("cr_year") or "unknown",
                    response_excerpt=excerpt)
                verdict = m64.call_groq(m64.SYSTEM_PROMPT, user)
                cache[key] = verdict
                n_new += 1
                if n_new % 25 == 0:
                    VCA_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
                    print(f"  [{gen}] {n_new} verdicts...", flush=True)
        VCA_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
        print(f"{gen}: {n_new} new 70B verdicts")


# ---------- VCA final math (copied from 68_vca_final.py) ----------

def extract_year_near_doi(text, doi):
    pos = text.lower().find(doi.lower())
    if pos < 0:
        return None
    window = text[max(0, pos - 200):pos]
    years = re.findall(r'\b(19[5-9]\d|20[0-3]\d)\b', window)
    return years[-1] if years else None


def year_diff(cited_y, cr_y):
    if not cited_y or not cr_y:
        return None
    try:
        return abs(int(cited_y) - int(cr_y))
    except (ValueError, TypeError):
        return None


def vca_stats(gen, path):
    cr_data = json.loads(DOI4WAY.read_text())
    vca_cache = json.loads(VCA_CACHE.read_text())
    responses = {r["id"]: r.get("answer", "") for r in json.loads(path.read_text())}
    st = defaultdict(int)
    for q in cr_data[gen]["results"]:
        qid = q["qid"]
        answer = responses.get(qid, "")
        for d in q["dois"]:
            st["total_dois"] += 1
            if not d.get("cr_valid"):
                st["fabricated"] += 1
                continue
            st["exist"] += 1
            verdict = vca_cache.get(f"{gen}_{qid}_{d['doi']}")
            if verdict != "VALID":
                st["seventy_b_invalid"] += 1
                continue
            st["seventy_b_valid"] += 1
            cited_year = d.get("cited_year") or extract_year_near_doi(answer, d["doi"])
            yd = year_diff(cited_year, d.get("cr_year"))
            if yd == 0:
                st["vca_strict"] += 1
                st["vca_lenient"] += 1
            elif yd == 1:
                st["vca_lenient"] += 1
            elif yd is None:
                st["vca_lenient"] += 1
    return dict(st)


# ---------- stage: judge ----------

def stage_judge(mode="forced"):
    m44 = load_mod("44_judge_REVAL.py", "judge44")
    targets = [f"GPT54S_{m}" for m in (("auto", "forced") if mode == "both" else (mode,))]

    scores = json.loads(SCORES_OUT.read_text()) if SCORES_OUT.exists() else []
    done = {(s["judge"], s["generator"], s["qid"]) for s in scores if s.get("scores")}

    jobs = []
    for gen in targets:
        path = NEW_GENS[gen]
        if not path.exists():
            print(f"SKIP {gen}: responses not found")
            continue
        for r in json.loads(path.read_text()):
            if not r.get("answer"):
                continue
            for j in JUDGES:
                if (j, gen, r["id"]) not in done:
                    jobs.append((j, gen, r["id"], r["question"], r["answer"]))

    print(f"Pending judge calls: {len(jobs)}")
    lock = threading.Lock()
    t0 = time.time()
    n = [0]

    def run(j, gen, qid, question, answer):
        try:
            text = m44.JUDGE_FNS[j](m44.JUDGE_PROMPT.format(question=question, answer=answer))
            sc = m44.parse_scores(text)
            if sc is None:
                return {"judge": j, "generator": gen, "qid": qid, "error": "parse_fail", "raw": text[:300]}
            return {"judge": j, "generator": gen, "qid": qid, "scores": sc,
                    "reval": round(m44.reval_score(sc), 4)}
        except Exception as e:
            return {"judge": j, "generator": gen, "qid": qid, "error": f"{type(e).__name__}: {str(e)[:120]}"}

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run, *j) for j in jobs]
        for f in as_completed(futs):
            entry = f.result()
            with lock:
                scores.append(entry)
                n[0] += 1
                eta = (time.time() - t0) / n[0] * (len(jobs) - n[0])
                err = " ERR" if "error" in entry else f" reval={entry.get('reval','?')}"
                print(f"  [{n[0]:5d}/{len(jobs)}] {entry['judge']:6s} {entry['generator']:14s} Q{entry['qid']:3d}{err} | ETA {eta/60:.0f}min", flush=True)
                if n[0] % 25 == 0:
                    SCORES_OUT.write_text(json.dumps(scores, ensure_ascii=False, indent=1))

    SCORES_OUT.write_text(json.dumps(scores, ensure_ascii=False, indent=1))
    errs = sum(1 for s in scores if "error" in s)
    print(f"DONE. total={len(scores)}, errors={errs}")


# ---------- stage: report ----------

def stage_report():
    print(f"{'System':<16} {'DOIs':>5} {'Exist%':>7} {'Fabr%':>6} {'VCA':>6} {'VCA+-1':>7} {'REVAL':>6} {'Search%':>8}")
    # reference rows from the paper
    for name, dois, ex, fab, vca, vca1, reval in [
        ("SYSTEM-R", 695, 98.1, 1.9, 82.9, 86.3, 3.346),
        ("GemSearch", 1076, 92.2, 7.8, 63.7, 70.2, 4.604),
        ("GPT54 (base)", 1231, 71.4, 28.6, 28.4, 34.1, 4.587),
    ]:
        print(f"{name:<16} {dois:>5} {ex:>7} {fab:>6} {vca:>6} {vca1:>7} {reval:>6}")
    scores = json.loads(SCORES_OUT.read_text()) if SCORES_OUT.exists() else []
    for gen, path in NEW_GENS.items():
        if not path.exists():
            continue
        resp = json.loads(path.read_text())
        search_rate = 100 * sum(1 for r in resp if r.get("search_calls", 0) > 0) / len(resp)
        st = vca_stats(gen, path)
        t = st.get("total_dois", 0)
        revals = [s["reval"] for s in scores if s["generator"] == gen and s.get("scores")]
        reval = sum(revals) / len(revals) if revals else float("nan")
        print(f"{gen:<16} {t:>5} {100*st.get('exist',0)/t:>7.1f} {100*st.get('fabricated',0)/t:>6.1f} "
              f"{100*st.get('vca_strict',0)/t:>6.1f} {100*st.get('vca_lenient',0)/t:>7.1f} "
              f"{reval:>6.3f} {search_rate:>7.1f}%  (n_scores={len(revals)})")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage == "crossref":
        stage_crossref()
    elif stage == "vca":
        stage_vca()
    elif stage == "judge":
        stage_judge(sys.argv[2] if len(sys.argv) > 2 else "forced")
    elif stage == "report":
        stage_report()
    else:
        raise SystemExit(f"unknown stage {stage}")
