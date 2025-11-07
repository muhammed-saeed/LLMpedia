
```markdown
# 🧠 GPT-KB — Multi-LLM Knowledge Graph Crawler (Batch + Concurrency)

**GPT-KB** builds a structured **knowledge graph** from LLMs by:
1) eliciting factual triples `(subject, predicate, object)`,
2) expanding via NER into new subjects, and
3) persisting everything in SQLite + JSONL for auditability.

It supports multiple providers (OpenAI, DeepSeek, Claude, Replicate) and runs in either:
- **Batch mode** (OpenAI-optimized), or
- **Concurrent mode** (DeepSeek / Claude / Replicate and any OpenAI-compatible base URL).

---

## ✨ Highlights

- **Hybrid Execution**
  - **Batch** for OpenAI GPT-4o / GPT-5 families.
  - **Concurrent** for DeepSeek, Claude, Replicate (and other OpenAI-compat).
- **Robust JSON Handling**
  - Schema mode where available, plus **auto-salvage** of malformed JSON.
  - DeepSeek fix: auto-injects the word **“json”** when required by the API.
- **Calibration Mode**
  - Optional confidence filtering for triples in `--elicitation-strategy calibrate` with `--conf-threshold`.
- **Resumable BFS**
  - Queue persisted in SQLite with `pending → working → done` states; safe to `--resume`.
- **Structured Logs**
  - `facts.jsonl`, `queue.jsonl`, `ner_decisions.jsonl`, `errors.log`, and durable SQLite stores.

---

## 🗂️ Repository Layout

```

GPTKB_Hallucinations/
├── crawler_runner.py                    # ✅ Unified runner (OpenAI batch + non-OpenAI concurrency)
├── bench_runner_concurrent.py           # Optional benchmarking harness
├── db_models.py                         # SQLite schema + helpers
├── llm/
│   ├── factory.py                       # Builds clients from settings
│   ├── deepseek_client.py               # DeepSeek adapter
│   ├── replicate_client.py              # Replicate adapter(s)
│   └── ...                              # Other providers
├── prompter_parser.py                   # Loads Jinja prompt templates
├── prompts/                             # Elicitation + NER prompt templates
├── settings.py                          # Model registry + JSON schemas
├── requirements.txt
└── runs*/                               # Outputs per run

````

---

## ⚙️ Setup

### 1) Environment

Create a `.env` or export:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# DeepSeek (OpenAI-compatible)
export DEEPSEEK_API_KEY=...
# Optional: treat OpenAI client as DeepSeek by overriding base URL
# export OPENAI_BASE_URL=https://api.deepseek.com/v1

# Replicate (Gemini/Grok/Llama/Mistral on Replicate)
export REPLICATE_API_TOKEN=...
````

> **Note:** DeepSeek’s JSON mode sometimes requires the word **“json”** in the prompt.
> The runner automatically injects a tiny system hint when needed.

### 2) Install

```bash
pip install -r requirements.txt
```

### 3) Smoke test

```bash
python diag.py
```

---

## 🧱 Prompt Architecture

Prompts are Jinja2 templates, organized by **domain** and **strategy**.

```
prompts/
├── general/                       # 🌍 Open-domain (default)
│   ├── baseline/
│   ├── calibrate/
│   ├── icl/
│   └── dont_know/
└── topics/                        # 🎯 Topic-anchored runs
    ├── baseline/
    ├── calibrate/
    ├── icl/
    └── dont_know/
```

* Use `--domain general` for open-domain exploration.
* Use `--domain topic` to anchor all hops to `--seed` (available as `root_subject` in templates).

Templates are resolved like:

```
prompts/{domain}/{strategy}/{task}.j2
# task ∈ {elicitation, ner}
```

---

## 🚀 Running the Unified Crawler

The **same script** adapts to your provider:

* If **OpenAI**, it uses **batch** (when available).
* If **DeepSeek / Claude / Replicate**, it uses **concurrency**.

### A) OpenAI (Batch)

> From experience, **batch size = 10** is very reliable.

```bash
python crawler_runner.py \
  --seed "Grace Hopper" \
  --domain general \
  --output-dir runs/openai_batch_hopper \
  --elicit-model-key gpt4o-mini \
  --ner-model-key gpt4o-mini \
  --batch-size 10 \
  --max-depth 2 \
  --max-subjects 15
```

### B) DeepSeek (Concurrent)

```bash
python crawler_runner.py \
  --seed "Vannevar Bush" \
  --domain general \
  --output-dir runs/deepseek_concurrent_bush \
  --elicit-model-key deepseek \
  --ner-model-key deepseek \
  --concurrency 12 \
  --ner-batch-size 20 \
  --max-depth 2 \
  --max-subjects 20 \
  --debug
```

### C) Claude or Replicate (Concurrent)

```bash
python crawler_runner.py \
  --seed "Umar ibn al-Khattab" \
  --domain general \
  --output-dir runs/claude_concurrent \
  --elicit-model-key claude35h \
  --ner-model-key claude35h \
  --concurrency 12 \
  --max-depth 2 \
  --max-subjects 15
```

### D) Topic-Anchored Crawl (any provider)

```bash
python crawler_runner.py \
  --seed "Game of Thrones" \
  --domain topic \
  --output-dir runs/topic_got \
  --elicit-model-key deepseek \
  --ner-model-key deepseek \
  --concurrency 10 \
  --max-depth 2 \
  --max-subjects 15
```

> With `--domain topic` the seed is injected as `root_subject`, so the crawler stays on-topic.

---

## 🧮 Calibration Mode

Keep only higher-confidence triples:

```bash
python crawler_runner.py \
  --seed "Grace Hopper" \
  --output-dir runs/calibrated \
  --elicit-model-key gpt4o-mini \
  --ner-model-key gpt4o-mini \
  --elicitation-strategy calibrate \
  --conf-threshold 0.7 \
  --batch-size 10
```

---

## 🧠 GPT-5 (Responses API)

When the selected model uses OpenAI’s **Responses API** (GPT-5 family), the runner will **not** pass legacy sampling params or `response_format`, and will honor:

* `--reasoning-effort minimal|low|medium|high`
* `--verbosity low|medium|high`

Example:

```bash
python crawler_runner.py \
  --seed "Ada Lovelace" \
  --output-dir runs/gpt5 \
  --elicit-model-key gpt-5-nano \
  --ner-model-key gpt-5-nano \
  --reasoning-effort minimal \
  --verbosity low \
  --batch-size 10
```

---

## 📊 Outputs

Each run writes:

```
<output-dir>/
├── queue.sqlite           # BFS queue (+ status + retries)
├── facts.sqlite           # triples_accepted / triples_sink
├── queue.jsonl            # appended on every enqueue
├── facts.jsonl            # appended on every accepted triple
├── ner_decisions.jsonl    # NER classification traces
├── errors.log             # stack traces + parse failures
├── facts.json             # final accepted + sink snapshot
├── queue.json             # final queue snapshot
└── lowconf.json           # below-threshold facts (calibrate mode)
```

---

## 🧩 CLI Reference

| Flag                                       | Description                                 |
| ------------------------------------------ | ------------------------------------------- |
| `--seed`                                   | Starting subject entity                     |
| `--domain`                                 | `general` (default) or `topic`              |
| `--output-dir`                             | Results folder                              |
| `--max-depth`, `--max-subjects`            | Crawl limits                                |
| `--elicitation-strategy`, `--ner-strategy` | `baseline`, `icl`, `dont_know`, `calibrate` |
| `--conf-threshold`                         | Confidence cutoff in calibrate mode         |
| `--batch-size`                             | OpenAI batch size (tip: 10)                 |
| `--concurrency`                            | Parallel workers for non-OpenAI providers   |
| `--ner-batch-size`                         | Phrases per NER request                     |
| `--max-tokens`, `--temperature`, `--top-p` | Model sampling knobs (non-Responses models) |
| `--reasoning-effort`, `--verbosity`        | GPT-5 Responses API controls                |
| `--resume`, `--reset-working`              | Safe resume after interruption              |
| `--debug`                                  | Print prompts to stdout                     |

---

## ✅ Best Practices

* **OpenAI batch size = 10** → best stability vs. speed.
* **Calibrate early** (0.6–0.8) for higher precision.
* **Depth**: 1–2 for quick scans; 3+ for deep graphs.
* **Resume safely** with `--resume` after a crash/interrupt.

---

## 🧪 Troubleshooting

| Symptom                                                       | Fix                                                                                     |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `DeepSeek API error: Prompt must contain the word 'json' ...` | Runner auto-adds a small “Output ONLY JSON …” system line; keep `--debug` to verify.    |
| `Responses.create() got unexpected keyword 'response_format'` | You picked a GPT-5 model; the runner already strips it. Ensure model key is correct.    |
| `sqlite3.OperationalError: no such table: ...`                | The runner creates tables via `open_*_db`. Delete the run folder and start a fresh run. |
| Many `empty_or_unparseable_output` in `facts_sink`            | Reduce temperature; increase `--max-tokens`; keep batch size ≤ 10; inspect prompts.     |
| Slow or stuck                                                 | Lower `--concurrency` (non-OpenAI) or `--batch-size` (OpenAI).                          |

---

## 📚 Model Hints (from `settings.MODELS`)

* **OpenAI**: `gpt4o-mini` (batch-friendly; great $/quality).
* **DeepSeek**: `deepseek` or `deepseek-reasoner` (use **concurrency**).
* **Replicate**: pick Llama/Mistral family keys you registered (use **concurrency**).
* **Claude via Replicate**: `claude35h` (concurrency).

> The runner auto-detects whether to use **batch** (OpenAI) or **concurrency** (others).

---

## 🧵 Bench Harness (optional)

```bash
python bench_runner_concurrent.py \
  --root-out runs/AllParallel \
  --domains topic \
  --seeds "ancient city of Babylon,The Big Bang Theory,DAX 40 Index" \
  --models "deepseek,granite8b,gpt4o-mini" \
  --strategies "baseline,calibrate,icl,dont_know" \
  --profiles "det,medium,wild" \
  --max-depth 1 \
  --max-subjects 10 \
  --default-concurrency 20 \
  --openai-batch-size 10 \
  --max-procs 6 \
  --skip-existing \
  --verbose >> runs/Bench/LogTest.txt
```

---

## 📄 License

MIT — use, modify, and extend freely.

```