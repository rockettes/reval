"""F2b-v2: Run 350 baselines with EXPLICIT DOI/ISBN request in prompt.

Prompt difference vs v1:
- v1: "cite os estudos científicos relevantes"
- v2: "cite os estudos científicos relevantes. Inclua o DOI (ou ISBN para livros)"

Goal: show that when forced to cite DOIs, generalists fabricate them.
Combined with v1 (no DOI request → no DOI cited), demonstrates that
generalists are NEVER usable for researchers needing verifiable sources.

Outputs:
- responses_350_gemini_v2.json
- responses_350_grok_v2.json
- responses_350_glm_v2.json
"""
# NOTE (release): this script is published as run. It reads and writes the working
# directory layout used during the experiments ("responses_350_<system>.json",
# "scores_*.json"); see the file-name mapping in README.md to match these against
# the released responses/, scores/ and results/ folders. API keys are read from a
# local .env; no credentials are included here.
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

DRY = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "benchmark/bodybuilding_qa_350.jsonl"

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
XAI_API_KEY = os.environ["XAI_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# v2 PROMPT — explicit DOI/ISBN request
PROMPT_V2 = """Você é um especialista em ciência do exercício e nutrição esportiva. Responda à pergunta de forma cientificamente fundamentada, citando os estudos científicos relevantes. Inclua o DOI (ou ISBN para livros) de cada fonte citada. Máximo 200 palavras.

PERGUNTA: {question}"""


def call_gemini(question):
    prompt = PROMPT_V2.format(question=question)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    cand = data["candidates"][0]
    text = cand.get("content", {}).get("parts", [{}])[0].get("text", "")
    usage = data.get("usageMetadata", {})
    return {
        "answer": text,
        "input_tokens": usage.get("promptTokenCount", 0),
        "output_tokens": usage.get("candidatesTokenCount", 0),
        "thoughts_tokens": usage.get("thoughtsTokenCount", 0),
        "total_tokens": usage.get("totalTokenCount", 0),
        "finish_reason": cand.get("finishReason"),
    }


def call_grok(question):
    prompt = PROMPT_V2.format(question=question)
    payload = {
        "model": "grok-4.20-0309-reasoning",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    r = requests.post("https://api.x.ai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                      json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})
    return {
        "answer": msg.get("content") or "",
        "reasoning_content": (msg.get("reasoning_content") or "")[:500],
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "finish_reason": choice.get("finish_reason"),
    }


def call_glm(question):
    prompt = PROMPT_V2.format(question=question)
    payload = {
        "model": "zai-org/GLM-5.1:together",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    r = requests.post("https://router.huggingface.co/v1/chat/completions",
                      headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                      json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})
    return {
        "answer": msg.get("content") or msg.get("reasoning") or "",
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "finish_reason": choice.get("finish_reason"),
    }


def call_gpt54(question):
    prompt = PROMPT_V2.format(question=question)
    payload = {
        "model": "gpt-5.4-2026-03-05",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_completion_tokens": 8192,
    }
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                      json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    msg = choice["message"]
    usage = data.get("usage", {})
    return {
        "answer": msg.get("content") or "",
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "finish_reason": choice.get("finish_reason"),
    }


def make_runner(label, call_fn, out_path, num_workers=5):
    save_lock = threading.Lock()
    all_results = []
    done_count = [0]
    total_pending = [0]
    t0 = [time.time()]

    def process(q):
        try:
            t_start = time.time()
            data = call_fn(q["question"])
            elapsed = time.time() - t_start
            ans = data.get("answer", "")
            return {
                "id": q["id"],
                "question_type": q.get("question_type"),
                "category": q.get("category"),
                "difficulty": q.get("difficulty"),
                "question": q["question"],
                "expected_papers": q.get("expected_papers", ""),
                "answer": ans,
                "client_latency_s": round(elapsed, 2),
                "char_count": len(ans),
                "word_count": len(ans.split()),
                **{k: v for k, v in data.items() if k != "answer"},
            }
        except Exception as e:
            return {"id": q["id"], "question_type": q.get("question_type"),
                    "question": q["question"], "answer": "", "error": str(e)}

    def saver(entry):
        with save_lock:
            all_results.append(entry)
            done_count[0] += 1
            i = done_count[0]
            eta = (time.time() - t0[0]) / i * (total_pending[0] - i)
            wc = entry.get('word_count', 0)
            err = ' ERR' if 'error' in entry else ''
            print(f"  [{label}] [{i:3d}/{total_pending[0]}] Q{entry['id']:3d} -> {wc:3d}w{err} | ETA {eta/60:.1f}min", flush=True)
            sorted_out = sorted(all_results, key=lambda x: x['id'])
            out_path.write_text(json.dumps(sorted_out, ensure_ascii=False, indent=2))

    def run(questions):
        nonlocal all_results
        if out_path.exists():
            all_results = json.loads(out_path.read_text())
        done_ids = {r["id"] for r in all_results if r.get("answer") and not r.get("error")}
        pending = [q for q in questions if q["id"] not in done_ids]
        total_pending[0] = len(pending)
        done_count[0] = 0
        t0[0] = time.time()
        print(f"[{label}] Already done: {len(done_ids)}, pending: {len(pending)}, workers: {num_workers}")

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(process, q) for q in pending]
            for fut in as_completed(futures):
                saver(fut.result())

        sorted_out = sorted(all_results, key=lambda x: x['id'])
        out_path.write_text(json.dumps(sorted_out, ensure_ascii=False, indent=2))
        errs = sum(1 for r in all_results if 'error' in r)
        print(f"[{label}] DONE. Total: {len(sorted_out)}, errors: {errs}")

    return run


def main():
    questions = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(questions)} questions\n")
    print(f"PROMPT v2 (with DOI/ISBN request):")
    print("-"*70)
    print(PROMPT_V2.format(question="<question>"))
    print("-"*70 + "\n")

    import sys
    # Usage: python 48_baselines_v2_with_doi.py [model] [N]
    # model: all (default), gpt54, gemini, grok, glm
    # N: number of questions (default: all)
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_questions = int(sys.argv[2]) if len(sys.argv) > 2 else len(questions)
    subset = questions[:n_questions]
    print(f"Target: {target}, N={len(subset)}\n")

    runner_map = {
        "gemini": make_runner("GEM_v2", call_gemini, DRY / "responses_350_gemini_v2.json", num_workers=5),
        "grok":   make_runner("XAI_v2", call_grok,   DRY / "responses_350_grok_v2.json",   num_workers=5),
        "glm":    make_runner("GLM_v2", call_glm,    DRY / "responses_350_glm_v2.json",    num_workers=5),
        "gpt54":  make_runner("GPT54",  call_gpt54,  DRY / "responses_350_gpt54_v2.json",  num_workers=5),
    }

    if target == "all":
        runners = list(runner_map.values())
    else:
        runners = [runner_map[target]]

    threads = [threading.Thread(target=r, args=(subset,)) for r in runners]
    for t in threads: t.start()
    for t in threads: t.join()

    print("\nDone.")


if __name__ == "__main__":
    main()
