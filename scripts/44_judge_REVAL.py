"""F3a+F3b: REVAL multi-judge cross-evaluation.

Implements paper rubric (Tabela tab:rubric):
- 6 dimensions, 1-5 scale, weighted
- Correção Científica 25%
- Verificab. Citações 25%
- Especificidade 20%
- Completude 15%
- Assertividade 10%
- Aplicação Prática 5%

Multi-judge graph (4 judges, 4 generators, self-excluded = 13 edges):
- Atla Selene Mini → judges all 4 (Ours, GLM, Gemini, Grok) — 4 edges
- GLM-5.1 → judges 3 (Ours, Gemini, Grok) — 3 edges
- Gemini Pro → judges 3 (Ours, GLM, Grok) — 3 edges
- Grok R → judges 3 (Ours, GLM, Gemini) — 3 edges
Total: 13 edges × 350 = 4550 judge calls

Output: scores_REVAL.json with all (judge, generator, qid, scores) tuples
"""
# NOTE (release): this script is published as run. It reads and writes the working
# directory layout used during the experiments ("responses_350_<system>.json",
# "scores_*.json"); see the file-name mapping in README.md to match these against
# the released responses/, scores/ and results/ folders. API keys are read from a
# local .env; no credentials are included here.
import json
import os
import re
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
OUT = DRY / "scores_REVAL.json"

HF_TOKEN = os.environ["HF_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
XAI_API_KEY = os.environ["XAI_API_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Generator file mapping
GENERATORS = {
    "Ours":      DRY / "responses_350_RAG.json",
    "GLM":       DRY / "responses_350_glm_v2.json",
    "Gemini":    DRY / "responses_350_gemini_v2.json",
    "Grok":      DRY / "responses_350_grok_v2.json",
    "GemSearch": DRY / "responses_350_gemsearch.json",
    "GPT54":     DRY / "responses_350_gpt54_v2.json",
}

# REVAL paper rubric (Tabela 2)
DIMENSIONS = [
    ("correcao_cientifica", "Correção Científica", 0.25),
    ("verificab_citacoes",  "Verificab. Citações",  0.25),
    ("especificidade",      "Especificidade",       0.20),
    ("completude",          "Completude",           0.15),
    ("assertividade",       "Assertividade",        0.10),
    ("aplicacao_pratica",   "Aplicação Prática",    0.05),
]

JUDGE_PROMPT = """Você é um avaliador especialista de respostas científicas em ciência do exercício e nutrição esportiva.

PERGUNTA:
{question}

RESPOSTA AVALIADA:
{answer}

Avalie a RESPOSTA acima nas 6 dimensões abaixo, atribuindo um score INTEIRO de 1 a 5 para cada uma:

1. **correcao_cientifica** (1-5): Precisão factual. As afirmações estão de acordo com a literatura científica atual?
   - 5 = totalmente correto, alinhado com evidência
   - 3 = parcialmente correto, alguns erros
   - 1 = incorreto ou contradiz a ciência

2. **verificab_citacoes** (1-5): As fontes citadas são reais e verificáveis (DOI, ISBN, autor, ano)?
   - 5 = todas as citações são reais e bem identificadas
   - 3 = algumas citações verificáveis, outras vagas
   - 1 = sem citações ou citações inventadas

3. **especificidade** (1-5): A resposta inclui valores numéricos, doses, ranges, percentuais quantitativos?
   - 5 = altamente específica, com números e ranges
   - 3 = parcialmente específica
   - 1 = vaga, sem números

4. **completude** (1-5): A resposta cobre os pontos-chave da pergunta?
   - 5 = cobre todos os aspectos relevantes
   - 3 = cobre o básico, omite nuances
   - 1 = superficial ou incompleta

5. **assertividade** (1-5): A resposta é direta e não evasiva?
   - 5 = direta, sem evasão ("depende", "talvez", "mais pesquisa necessária")
   - 3 = parcialmente direta
   - 1 = evasiva ou hedge demais

6. **aplicacao_pratica** (1-5): A resposta inclui orientação prática acionável?
   - 5 = inclui dicas práticas claras
   - 3 = pouca orientação prática
   - 1 = só teoria, sem aplicação

Retorne APENAS um objeto JSON puro com as 6 chaves, sem markdown, sem explicação, sem code fences. Exemplo:
{{"correcao_cientifica": 4, "verificab_citacoes": 5, "especificidade": 3, "completude": 4, "assertividade": 5, "aplicacao_pratica": 4}}"""


def parse_scores(text):
    """Extract JSON object from judge response. Tries multiple strategies."""
    if not text: return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r'^```(?:json)?\s*', '', s)
        s = re.sub(r'\s*```$', '', s)
    # Strategy 1: outermost {...}
    i = s.find("{"); j = s.rfind("}")
    candidates = []
    if i >= 0 and j > i:
        candidates.append(s[i:j+1])
    # Strategy 2: regex match for any {"correcao_cientifica":...} block
    rx = re.compile(r'\{[^{}]*"correcao_cientifica"[^{}]*\}', re.DOTALL)
    candidates.extend(rx.findall(s))
    # Try each candidate, return first valid one
    for cand in candidates:
        try:
            d = json.loads(cand)
            out = {}
            ok = True
            for key, _, _ in DIMENSIONS:
                v = d.get(key)
                if v is None:
                    for alt in [key.replace("_", " "), key.replace("_", "-")]:
                        if alt in d: v = d[alt]; break
                if v is None:
                    ok = False
                    break
                out[key] = float(v)
            if ok:
                return out
        except Exception:
            continue
    return None


def reval_score(scores):
    """Weighted REVAL score (1-5)."""
    return sum(scores[k] * w for k, _, w in DIMENSIONS)


# ==== Judges ====

def judge_atla(prompt):
    for attempt in range(5):
        r = requests.post("https://router.huggingface.co/v1/chat/completions",
                          headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                          json={"model": "AtlaAI/Selene-1-Mini-Llama-3.1-8B:featherless-ai",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 8192, "temperature": 0.0}, timeout=120)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt * 5)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    r.raise_for_status()


def judge_glm(prompt):
    # 2026-07-11: together dropped GLM-5.1 serverless -> deepinfra (same weights,
    # different serving provider; temp 0.0 unchanged)
    for attempt in range(5):
        r = requests.post("https://router.huggingface.co/v1/chat/completions",
                          headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                          json={"model": "zai-org/GLM-5.1:deepinfra",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 8192, "temperature": 0.0}, timeout=600)
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt * 5
            time.sleep(wait)
            continue
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning") or ""
        return text
    r.raise_for_status()  # last attempt raises


def judge_gemini(prompt):
    # Gemini 3.1 Pro is also reasoning — generous max_tokens
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-pro-preview:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")


def judge_grok(prompt):
    payload = {
        "model": "grok-4.20-0309-reasoning",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    r = requests.post("https://api.x.ai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                      json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def judge_gpt54(prompt):
    payload = {
        "model": "gpt-5.4-2026-03-05",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_completion_tokens": 8192,
    }
    for attempt in range(6):
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                          json=payload, timeout=300)
        if r.status_code == 429:
            # fresh keys sit in a low rate tier; honor Retry-After when present
            wait = float(r.headers.get("retry-after") or min(2 ** attempt * 10, 120))
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    r.raise_for_status()


JUDGE_FNS = {
    "Atla":   judge_atla,
    "GLM":    judge_glm,
    "Gemini": judge_gemini,
    "Grok":   judge_grok,
    "GPT54":  judge_gpt54,
}

# Edges: judge_X never judges itself (Gemini also excluded from GemSearch — same base model)
EDGES = []
for judge_name in JUDGE_FNS:
    for gen_name in GENERATORS:
        if judge_name == gen_name:
            continue  # self-exclusion
        if judge_name == "Gemini" and gen_name == "GemSearch":
            continue  # same base model
        EDGES.append((judge_name, gen_name))


# ==== Runner ====

save_lock = threading.Lock()
all_scores = []
done_count = [0]
total_pending = [0]
t0_global = [time.time()]


def run_one(judge_name, gen_name, qid, question, answer):
    try:
        prompt = JUDGE_PROMPT.format(question=question, answer=answer)
        text = JUDGE_FNS[judge_name](prompt)
        scores = parse_scores(text)
        if scores is None:
            return {"judge": judge_name, "generator": gen_name, "qid": qid,
                    "error": "parse_fail", "raw": text[:300]}
        return {"judge": judge_name, "generator": gen_name, "qid": qid,
                "scores": scores, "reval": round(reval_score(scores), 4)}
    except Exception as e:
        return {"judge": judge_name, "generator": gen_name, "qid": qid,
                "error": f"{type(e).__name__}: {str(e)[:120]}"}


def saver(entry):
    with save_lock:
        all_scores.append(entry)
        done_count[0] += 1
        i = done_count[0]
        eta = (time.time() - t0_global[0]) / i * (total_pending[0] - i)
        err = ' ERR' if 'error' in entry else f" reval={entry.get('reval','?')}"
        print(f"  [{i:5d}/{total_pending[0]}] {entry['judge']:6s}→{entry['generator']:6s} Q{entry['qid']:3d}{err} | ETA {eta/60:.1f}min", flush=True)
        if i % 25 == 0:
            OUT.write_text(json.dumps(all_scores, ensure_ascii=False, indent=2))


def main():
    # Load all generator responses
    gen_responses = {}
    for name, path in GENERATORS.items():
        if not path.exists():
            print(f"WARN: {name} responses not yet ready ({path.name}) — will skip its edges")
            gen_responses[name] = None
        else:
            gen_responses[name] = {r["id"]: r for r in json.loads(path.read_text())}
            print(f"Loaded {name}: {len(gen_responses[name])} responses")

    # Resume support: skip already-done edges
    global all_scores
    if OUT.exists():
        all_scores = json.loads(OUT.read_text())
    done_set = {(s["judge"], s["generator"], s["qid"]) for s in all_scores
                if s.get("scores") and not s.get("error")}
    print(f"Already scored: {len(done_set)}")

    # Build all judge calls
    pending = []
    for judge_name, gen_name in EDGES:
        gen_data = gen_responses.get(gen_name)
        if gen_data is None:
            continue  # generator not ready, skip its edges this run
        for qid, resp in gen_data.items():
            if not resp.get("answer"):
                continue
            if (judge_name, gen_name, qid) in done_set:
                continue
            pending.append((judge_name, gen_name, qid, resp["question"], resp["answer"]))

    total_pending[0] = len(pending)
    done_count[0] = 0
    t0_global[0] = time.time()
    print(f"\nTotal pending judge calls: {len(pending)}")
    print(f"Edges (judge,gen) used: {sorted(set((j,g) for j,g,_,_,_ in pending))}")
    print(f"Workers: 8\n")

    with ThreadPoolExecutor(max_workers=1) as ex:
        futures = [ex.submit(run_one, j, g, qid, q, a) for j, g, qid, q, a in pending]
        for fut in as_completed(futures):
            saver(fut.result())

    OUT.write_text(json.dumps(all_scores, ensure_ascii=False, indent=2))
    errs = sum(1 for s in all_scores if 'error' in s)
    print(f"\nDONE. Total: {len(all_scores)}, errors: {errs}")


if __name__ == "__main__":
    main()
