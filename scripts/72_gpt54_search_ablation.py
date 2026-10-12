"""Ablation: GPT-5.4 with OpenAI web search vs without (rebuttal experiment, Reviewer ySda).

Mirrors 66_gemini_search_ablation.py but for GPT-5.4 via the OpenAI Responses API
with the built-in web_search tool. Same Appendix-A prompt, temperature 0.0.

Phase 1: 10 questions -> check DOI-format compliance (Appendix H concern: models
         may emit URLs instead of DOIs under search affordances).
Phase 2: all 350 questions.

Usage: python 72_gpt54_search_ablation.py [1|2] [auto|forced]
  auto   -> tool available, model decides (mirrors GemSearch protocol)
  forced -> tool_choice forces >=1 web search per question (strong-baseline condition)
Output: responses_350_gpt54search_<mode>.json (same schema as responses_350_*.json)
"""
# NOTE (release): this script is published as run. It reads and writes the working
# directory layout used during the experiments ("responses_350_<system>.json",
# "scores_*.json"); see the file-name mapping in README.md to match these against
# the released responses/, scores/ and results/ folders. API keys are read from a
# local .env; no credentials are included here.
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

MODE = (sys.argv[2] if len(sys.argv) > 2 else "auto").lower()
assert MODE in ("auto", "forced")

DRY = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[1]

ENV = ROOT / ".env"
if ENV.exists():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

OPENAI_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-5.4-2026-03-05"
URL = "https://api.openai.com/v1/responses"

PROMPT_TEMPLATE = """Você é um especialista em ciência do exercício, nutrição esportiva e fisiculturismo baseado em evidências.

PERGUNTA: {question}

INSTRUÇÕES:
1. Responda à pergunta com base em evidências científicas
2. Cite todas as fontes utilizadas no formato: (Autor, Ano, tipo de estudo, N=X, população, DOI: 10.xxxx/xxxxx)
3. Se a evidência é conflitante, apresente os diferentes achados
4. Inclua dados quantitativos quando disponíveis (doses, ranges, percentuais)
5. Máximo de 250 palavras"""

CACHE_FILE = DRY / f"gpt54_search_cache_{MODE}.json"
OUT_FILE = DRY / f"responses_350_gpt54search_{MODE}.json"

# Tool type fallbacks: Responses API renamed the tool across versions
TOOL_TYPES = ["web_search", "web_search_preview"]


def extract_text(data):
    """Pull assistant text out of a Responses API payload."""
    # Convenience field first
    if data.get("output_text"):
        return data["output_text"]
    parts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    parts.append(c.get("text", ""))
    return "\n".join(p for p in parts if p)


def count_search_calls(data):
    return sum(1 for item in data.get("output", [])
               if item.get("type", "").startswith("web_search"))


def call_gpt54_with_search(question, max_retries=4):
    prompt = PROMPT_TEMPLATE.format(question=question)
    base_payload = {
        "model": MODEL,
        "input": prompt,
        "max_output_tokens": 8192,
        "temperature": 0.0,
    }
    tool_idx = 0
    drop_temperature = False
    for attempt in range(max_retries):
        payload = dict(base_payload)
        payload["tools"] = [{"type": TOOL_TYPES[tool_idx]}]
        if MODE == "forced":
            payload["tool_choice"] = {"type": TOOL_TYPES[tool_idx]}
        if drop_temperature:
            payload.pop("temperature", None)
        try:
            r = requests.post(URL,
                              headers={"Authorization": f"Bearer {OPENAI_KEY}",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=600)
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage", {})
                return {
                    "answer": extract_text(data),
                    "finish_reason": data.get("status"),
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "search_calls": count_search_calls(data),
                    "tool_type": TOOL_TYPES[tool_idx],
                }
            body = r.text[:300]
            if r.status_code == 400 and "temperature" in body and not drop_temperature:
                print("  400 on temperature -> retrying without it")
                drop_temperature = True
                continue
            if r.status_code == 400 and TOOL_TYPES[tool_idx] in body and tool_idx + 1 < len(TOOL_TYPES):
                print(f"  400 on tool type {TOOL_TYPES[tool_idx]} -> trying {TOOL_TYPES[tool_idx+1]}")
                tool_idx += 1
                continue
            if r.status_code == 429:
                wait = 2 ** attempt * 10
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  Error {r.status_code}: {body}")
            time.sleep(3)
        except Exception as e:
            print(f"  Exception: {e}")
            time.sleep(3)
    return None


def extract_dois(text):
    if not text:
        return []
    rx = re.compile(r'(10\.\d{4,9}/\S+)', re.IGNORECASE)
    return [d.rstrip('.,;:)]}>').lower() for d in rx.findall(text)]


def extract_urls(text):
    if not text:
        return []
    return re.findall(r'https?://\S+', text)


def main():
    questions = json.loads((DRY / "responses_350_gemini_v2.json").read_text())

    phase = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    subset = questions[:10] if phase == 1 else questions
    print(f"Phase {phase}: {len(subset)} questions, model {MODEL}")

    cache = {}
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())

    # parallel collection into cache (4 workers), then assemble in order
    import threading
    lock = threading.Lock()
    todo = [q for q in subset if str(q["id"]) not in cache]
    done_n = [0]

    def collect(q):
        t0 = time.time()
        res = call_gpt54_with_search(q["question"])
        if res is not None:
            res["client_latency_s"] = round(time.time() - t0, 2)
        with lock:
            done_n[0] += 1
            if res is not None:
                cache[str(q["id"])] = res
                if done_n[0] % 5 == 0:
                    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
            print(f"[{done_n[0]}/{len(todo)}] Q{q['id']}: "
                  f"{'FAILED' if res is None else str(res.get('search_calls',0)) + ' searches'}", flush=True)

    if todo:
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(collect, todo))
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))

    results = []
    n_dois_total = 0
    n_urls_total = 0
    for q in subset:
        qid = str(q["id"])
        if qid not in cache:
            continue
        res = cache[qid]
        ans = res.get("answer", "")
        dois = extract_dois(ans)
        urls = extract_urls(ans)
        n_dois_total += len(dois)
        n_urls_total += len(urls)
        results.append({
            "id": q["id"],
            "question_type": q.get("question_type"),
            "category": q.get("category"),
            "difficulty": q.get("difficulty"),
            "question": q["question"],
            "answer": ans,
            "client_latency_s": res.get("client_latency_s", 0),
            "char_count": len(ans),
            "word_count": len(ans.split()),
            "input_tokens": res.get("input_tokens", 0),
            "output_tokens": res.get("output_tokens", 0),
            "total_tokens": res.get("total_tokens", 0),
            "finish_reason": res.get("finish_reason"),
            "search_calls": res.get("search_calls", 0),
        })
    n_searched = sum(1 for r in results if r["search_calls"] > 0)
    print(f"\nMode: {MODE}")
    print(f"Totals: {n_dois_total} DOIs, {n_urls_total} raw URLs over {len(results)} responses")
    print(f"Search invocation: {n_searched}/{len(results)} responses used >=1 web search")
    print(f"DOI-format compliance: {sum(1 for r in results if extract_dois(r['answer']))}/{len(results)} responses have >=1 DOI")

    if phase == 2:
        OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"Wrote {OUT_FILE}")
    else:
        (DRY / "gpt54_search_pilot10.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"Wrote pilot file gpt54_search_pilot10.json — inspect DOI compliance before Phase 2")


if __name__ == "__main__":
    main()
