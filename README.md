```markdown
# LLMPedia

**Autonomous Knowledge Graph Generator using Large Language Models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI](https://img.shields.io/badge/OpenAI-Batch%20API%20Ready-green.svg)](https://platform.openai.com/docs/guides/batch)

LLMPedia automatically generates interconnected encyclopedia-style articles by recursively expanding from a seed topic. It uses intelligent entity extraction, semantic deduplication, and parallel processing to build comprehensive knowledge bases — then provides evaluation tools and a static site generator to analyze and publish the results.

```
Seed: "Vannevar Bush"
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LLMPedia Pipeline                            │
│                                                                     │
│  Self-RAG → Outline → Article Generation → NER → Similarity → Queue │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Post-Processing Tools                           │
│                                                                     │
│  Funnel Analysis → Image Download → Static Site Generation          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
         │
         ▼
    50,000+ interconnected articles with:
    • Per-entity pipeline analytics
    • Wikipedia/Wikidata images with attribution
    • Searchable static website
```

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Execution Modes](#execution-modes)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
  - [Track 1: Topic-Focused](#track-1-topic-focused)
  - [Track 2: Cross-Model](#track-2-cross-model)
  - [Key Flags Reference](#key-flags-reference)
  - [Wikipedia Coverage](#wikipedia-coverage-how-it-works)
- [Web & Post-Processing](#web--post-processing)
- [Output Structure](#output-structure)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Features

| Feature | Description |
|---------|-------------|
| **Recursive Expansion** | Automatically discovers and generates articles for related entities extracted from wikilinks and categories |
| **Self-RAG Context** | Retrieval-Augmented Generation gathers factual context before writing to reduce hallucination |
| **Semantic Deduplication** | Prevents duplicate articles using embedding similarity and LLM-based duplicate detection |
| **Named Entity Recognition** | Filters extracted candidates to valid named entities only using confidence-calibrated classifiers |
| **Parallel Processing** | Multi-worker architecture with configurable concurrency for high throughput |
| **Batch API Support** | 50% cost reduction using OpenAI Batch API for large-scale runs |
| **Buffered I/O** | Optimized disk writes with configurable flush intervals for 10–50× performance improvement |
| **Graceful Shutdown** | Ctrl+C safely flushes all buffers and persists queue state before exit |
| **Resume Support** | Continue interrupted runs without data loss using JSON-backed queues |
| **Funnel Analysis** | Per-entity pipeline attrition tracking across all processing stages |
| **Evaluation Suite** | Factuality checking, stylistic analysis, and cross-model/persona comparisons |
| **Image Index** | Automated Wikipedia/Wikidata image download with license attribution |
| **Static Site** | Searchable encyclopedia website with genealogy trees, images, and funnel statistics |

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LLMPedia Architecture                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    ┌──────────────┐     ┌────────────────────────────────────────────────────┐ │
│    │              │     │                  Worker Pool                       │ │
│    │  Dispatcher  │────►│  ┌────────┐ ┌────────┐ ┌────────┐ ┌─────┐ ┌─────┐ │ │
│    │              │     │  │SelfRAG │ │Outline │ │ Elicit │ │ NER │ │ Sim │ │ │
│    └──────────────┘     │  │Workers │ │Workers │ │Workers │ │ Wkr │ │ Wkr │ │ │
│           │             │  └───┬────┘ └───┬────┘ └───┬────┘ └──┬──┘ └──┬──┘ │ │
│           │             └──────┼──────────┼──────────┼─────────┼───────┼────┘ │
│           │                    │          │          │         │       │      │
│           ▼                    ▼          ▼          ▼         ▼       ▼      │
│    ┌──────────────┐     ┌──────────────────────────────────────────────────┐  │
│    │    Main      │     │              Buffered Writers                    │  │
│    │    Queue     │◄───►│  articles.jsonl │ embeddings.jsonl │ *.jsonl    │  │
│    │  (JsonQueue) │     └──────────────────────────────────────────────────┘  │
│    └──────────────┘                                                           │
│           │                     ┌──────────────────────────────────────┐      │
│           │                     │         Similarity Engine            │      │
│           └────────────────────►│  • Embedding generation              │      │
│                                 │  • Cosine similarity search          │      │
│                                 │  • LLM-based duplicate detection     │      │
│                                 └──────────────────────────────────────┘      │
│                                                                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                          Post-Processing Pipeline                              │
│                                                                                │
│  ┌───────────────┐    ┌──────────────────┐    ┌──────────────────────────┐    │
│  │    Funnel      │    │   Image Index     │    │     Site Builder         │    │
│  │   Analysis     │───►│   (Wikipedia/     │───►│   (Static HTML +        │    │
│  │ (evaluation/)  │    │    Wikidata)       │    │    search + genealogy)  │    │
│  └───────────────┘    └──────────────────┘    └──────────────────────────┘    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/yourorg/llmpedia.git
cd llmpedia
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Requirements

- Python 3.8+
- OpenAI API key (for GPT models)
- Optional: Local embedding models via `sentence-transformers`

### Environment Configuration

Create `.env`:

```bash
OPENAI_API_KEY=sk-...
# Optional: For custom endpoints
OPENAI_BASE_URL=https://api.openai.com/v1
```

## Quick Start

### End-to-End: Generate → Analyze → Publish

```bash
# 1. Generate articles (Online mode, 1000 subjects)
python llmpedia.py \
  --seed "Albert Einstein" \
  --mode online \
  --model-key gpt-4.1-mini \
  --max-subjects 1000 \
  --use-ner true \
  --use-similarity true \
  --elicit-workers 10 \
  --output-dir runs/einstein

# 2. Analyze the pipeline funnel
python evaluation/funnel_analysis.py runs/einstein

# 3. Download article images
python web/build_image_index.py runs/einstein --workers 40

# 4. Build the website
python web/build_llmpedia_site.py runs/einstein --workers 16

# 5. Open the generated site
open runs/einstein/site/index.html
```

### Quick Test Run (50 articles)

```bash
python llmpedia.py \
  --mode online \
  --model-key gpt-4.1-mini \
  --seed "Albert Einstein" \
  --max-subjects 50 \
  --max-depth 2 \
  --use-ner true \
  --use-similarity true \
  --elicit-workers 5 \
  --output-dir runs/test
```

## Pipeline Stages

### Complete Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FULL PROCESSING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  STAGE 0: SELF-RAG (Optional)                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ Input:  Subject name + Domain context                                  │    │
│  │ Action: Retrieval-Augmented Generation to gather background context    │    │
│  │ Output: Context dictionary with retrieved information                  │    │
│  │ 🎯 PURPOSE: Reduces hallucination by grounding in real facts          │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                         │
│                                       ▼                                         │
│  STAGE 1: OUTLINE (Optional)                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ Input:  Subject + Root topic + Self-RAG context (if available)        │    │
│  │ Action: Generate section structure for article                         │    │
│  │ Output: Outline text with level-2 headings                             │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                         │
│                                       ▼                                         │
│  STAGE 2: ELICIT (Required)                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ Input:  Subject + Outline + Self-RAG context                           │    │
│  │ Action: Generate full Wikipedia-style article in Wikitext format       │    │
│  │ Output: Article text + Extracted candidates (links + categories)       │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                         │
│                                       ▼                                         │
│  STAGE 3: PRE-NER DEDUPLICATION (Automatic)                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ Input:  Raw candidates (e.g., 150 phrases)                             │    │
│  │ Actions:                                                               │    │
│  │   • Canonical key normalization ("MIT" = "mit" = "M.I.T.")            │    │
│  │   • Global seen-set check (already processed?)                         │    │
│  │   • Plural/singular variant detection                                  │    │
│  │   • Batch-level deduplication                                          │    │
│  │ Output: Deduplicated candidates (e.g., 45 phrases)                     │    │
│  │ 💰 COST SAVINGS: 70–90% reduction in NER API calls                    │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                         │
│                                       ▼                                         │
│  STAGE 4: NER FILTER (Optional)                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ Input:  Deduplicated candidates                                        │    │
│  │ Action: LLM classifies each phrase as Named Entity or not              │    │
│  │ Output: Validated named entities only                                  │    │
│  │ Confidence semantics:                                                  │    │
│  │   • If strategy contains 'calib': enforce ner-conf-threshold          │    │
│  │   • Otherwise: IGNORE confidence entirely (boolean only)              │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                         │
│                                       ▼                                         │
│  STAGE 5: SIMILARITY FILTER (Optional)                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ Input:  NER-validated candidates                                       │    │
│  │ Actions:                                                               │    │
│  │   1. Generate embedding for each candidate                             │    │
│  │   2. Compare against existing embeddings database                      │    │
│  │   3. Within-batch duplicate detection                                  │    │
│  │   4. If similarity ≥ threshold: LLM decides if truly duplicate        │    │
│  │ Output: Unique candidates only                                         │    │
│  │ Example: "MIT" vs "Massachusetts Institute of Technology" → DUPLICATE  │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                         │
│                                       ▼                                         │
│  STAGE 6: ENQUEUE                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ Input:  Final candidates                                               │    │
│  │ Action: Add to main queue with parent pointers (hop+1)                 │    │
│  │ Output: New subjects ready for processing                              │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Stage Summary

| Stage | Name | Required | Purpose | Output File |
|-------|------|----------|---------|-------------|
| 0 | **Self-RAG** | Optional | Gather factual context via retrieval | `self_rag_log.jsonl` |
| 1 | **Outline** | Optional | Generate article structure | `outlines.jsonl` |
| 2 | **Elicit** | ✅ Required | Generate article + extract candidates | `articles.jsonl` |
| 3 | **Pre-NER Dedup** | Automatic | Canonical deduplication | `ner_decisions.jsonl` |
| 4 | **NER** | Optional | Validate named entities | `ner_decisions.jsonl` |
| 5 | **Similarity** | Optional | Semantic deduplication | `similarity_decisions.jsonl` |
| 6 | **Enqueue** | Automatic | Add new subjects to queue | `queue.json` |

## Execution Modes

### Online Parallel Mode (`--mode online`)

Direct API calls with parallel workers. Best for:
- Real-time progress monitoring
- Non-OpenAI models (local/custom endpoints)
- Interactive exploration and debugging
- Smaller runs (< 5,000 subjects)

```bash
python llmpedia.py \
  --mode online \
  --concurrency 20 \
  --elicit-workers 16 \
  --ner-workers 2 \
  --sim-workers 2 \
  ...
```

### Batch Mode (`--mode batch`)

Uses OpenAI Batch API for 50% cost reduction. Best for:
- Large-scale runs (10,000+ subjects)
- Cost-sensitive deployments
- Overnight processing

```bash
python llmpedia.py \
  --mode batch \
  --batch-size 1000 \
  --concurrency 100 \
  --batch-poll-interval 30 \
  ...
```

**Important**: Batch mode requires all models (elicit, ner, self-rag) to use OpenAI provider.

## Configuration

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | string | **required** | Starting topic for knowledge graph expansion |
| `--mode` | choice | `online` | Execution mode: `online` or `batch` |
| `--domain` | choice | `general` | Prompt domain: `topic` (rooted) or `general` (standalone) |
| `--output-dir` | path | auto-generated | Directory for all output files |
| `--max-subjects` | int | 0 (unlimited) | Maximum articles to generate |
| `--max-depth` | int | 0 (unlimited) | Maximum hop distance from seed |

### Model Configuration (Cascading)

LLMPedia uses a cascading configuration system:

```
--model-key (global default)
    ├── --elicit-model-key (inherits from global)
    ├── --ner-model-key (inherits from global)
    └── --self-rag-model-key (inherits from global)
```

| Argument | Description |
|----------|-------------|
| `--model-key` | Default model for all stages |
| `--elicit-model-key` | Override for article generation |
| `--ner-model-key` | Override for NER classification |
| `--self-rag-model-key` | Override for Self-RAG retrieval |
| `--similarity-filter-model-key` | Override for similarity decisions |

### Worker Allocation

Recommended worker distribution (should not exceed `--concurrency`):

| Stage | % of workers | Notes |
|-------|--------------|-------|
| Elicit | 50–70% | Highest throughput need |
| NER | 15–25% | If enabled |
| Outline | 10–15% | If enabled |
| Similarity | 5–10% | I/O bound, few needed |
| Self-RAG | 5–15% | If enabled |

### NER Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-ner` | `true` | Enable NER validation |
| `--ner-mode` | inherits `--mode` | `online` or `batch` |
| `--ner-strategy` | `baseline` | `baseline` or `calib` (confidence-calibrated) |
| `--ner-chunk-size` | 25 | Candidates per NER API call |
| `--ner-conf-threshold` | 0.7 | Confidence threshold (only used if `calib` in strategy) |

**Important**: Confidence thresholds are **only** applied when `--ner-strategy` contains the substring `calib`. Otherwise, NER operates as a strict binary classifier.

### Similarity Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-similarity` | `true` | Enable semantic deduplication |
| `--similarity-provider` | `openai` | `openai` or `local` (sentence-transformers) |
| `--similarity-generation-model` | `text-embedding-3-small` | OpenAI embeddings model |
| `--similarity-threshold` | 0.92 | Cosine similarity threshold for duplicate detection |
| `--similarity-action` | `llm` | `reject` (auto-reject above threshold) or `llm` (LLM confirmation) |
| `--similarity-top-k` | 5 | Top-k similar items to consider |

## Evaluation

LLMPedia includes comprehensive evaluation tools for analyzing generated content quality, factuality, and stylistic variations across models and personas.

### Track 1: Topic-Focused

**Script**: `evaluation/run_track1_topic.py`

Topic-focused evaluation analyzing 3 models × 3 topics × 3 personas with cross-analysis.

**Directory Structure:**
```
topic_runs/
  gpt-5-mini/ancient_babylon/scientific_neutral/
  gpt-5-mini/ancient_babylon/left_leaning/
  gpt-5-mini/ancient_babylon/conservative/
  gpt-5-mini/us_civil_rights_movement/scientific_neutral/
  ...
  scads-llama-3.3-70b/ancient_babylon/scientific_neutral/
  ...
  scads-DeepSeek-V3.2/ancient_babylon/scientific_neutral/
  ...
```

#### Analyze Existing Results (FREE, no API calls)

```bash
# All 27 runs at once (3 models × 3 topics × 3 personas)
python evaluation/run_track1_topic.py \
    --analyze-only \
    --root-dir /home/samu170h/LLMPedia/openLLMPedia/topic_runs

# Single topic (finds sibling personas automatically)
python evaluation/run_track1_topic.py \
    --analyze-only \
    --run-dir /home/samu170h/LLMPedia/openLLMPedia/topic_runs/gpt-5-mini/ancient_babylon/conservative
```

Output → `topic_runs/analysis/` (12 CSV files)

#### Run Evaluation + Auto-Analyze (costs API money)

```bash
# Single run (small test — ~$0.05)
python evaluation/run_track1_topic.py \
    --run-dir /home/samu170h/LLMPedia/openLLMPedia/topic_runs/gpt-5-mini/ancient_babylon/conservative \
    --fact-model-key gpt-4.1-nano \
    --evidence wikipedia,web \
    --compute-similarity --compute-stylistic \
    --sample-n 10 --sample-min 5 \
    --concurrency 10

# All 27 runs, 50 articles each (~$5-10)
python evaluation/run_track1_topic.py \
    --root-dir /home/samu170h/LLMPedia/openLLMPedia/topic_runs \
    --fact-model-key gpt-4.1-nano \
    --evidence wikipedia,web \
    --compute-similarity --compute-stylistic \
    --sample-n 50 --sample-min 10 --sample-max 100 \
    --concurrency 50

# Full evaluation, all articles, with BERTScore
python evaluation/run_track1_topic.py \
    --root-dir /home/samu170h/LLMPedia/openLLMPedia/topic_runs \
    --fact-model-key gpt-4.1-nano \
    --evidence wikipedia,web \
    --compute-similarity --compute-bertscore --compute-stylistic \
    --sample-max 0 \
    --concurrency 50
```

#### Track 1 Output Files

All in `topic_runs/analysis/`:

| File | Paper section |
|------|--------------|
| `wikipedia_coverage_summary.csv` | Table 2 — Subj. ∩ Wiki |
| `wikipedia_coverage_detail.csv` | Per-subject found/not-found |
| `factuality_summary.csv` | Per-run: precision (wiki-found only), T/F/U rates |
| `factuality_by_topic_model.csv` | Table 4 — averaged across personas |
| `wikilink_stats.csv` | §3.4 — wikilink counts, diversity |
| `entity_overlap_cross_model.csv` | Table 4 — Ex-J, Ca-J |
| `entity_overlap_cross_persona.csv` | §3.6 — persona entity shift |
| `cross_model_text_similarity.csv` | §5.2 — lexical similarity between models |
| `cross_persona_text_similarity.csv` | §5.1 — persona framing similarity |
| `stylistic_summary.csv` | §5.1 — TTR, func words, punctuation |
| `ngram_analysis.csv` | §5.3 — corpus vocabulary diversity |
| `persona_effect_analysis.csv` | §3.6 — precision delta + entity shift |
| `paper_table4_topic_results.csv` | Ready for LaTeX Table 4 |
| `paper_table5_funnel.csv` | Entity sanitization funnel |
| `subject_overlap_cross_model.csv` | §5.2 — subject-level comparison |

### Track 2: Cross-Model

**Script**: `evaluation/run_track2_crossmodel.py`

Cross-model comparison for general-domain runs (100K+ articles) using neutral persona only.

**Input directories:**
```
/home/samu170h/LLMPedia/openLLMPedia/llama3.3-70b_100K/     (articles.jsonl + run_meta.json)
/home/samu170h/LLMPedia/openLLMPedia/deepseekV3.2_100K/     (articles.jsonl + run_meta.json)
/home/samu170h/LLMPedia/openLLMPedia/gpt_5_mini_1M/         (articles.jsonl + run_meta.json)
```

#### Structural Only (FREE, instant)

```bash
python evaluation/run_track2_crossmodel.py \
    --llama-dir /home/samu170h/LLMPedia/openLLMPedia/llama3.3-70b_100K \
    --deepseek-dir /home/samu170h/LLMPedia/openLLMPedia/deepseekV3.2_100K \
    --gpt-dir /home/samu170h/LLMPedia/openLLMPedia/gpt_5_mini_1M \
    --max-subjects 100000 \
    --sample-n 1000 --sample-min 100 \
    --output-dir /home/samu170h/LLMPedia/openLLMPedia/cross_model_results
```

#### With Similarity + Factuality (costs API money)

```bash
python evaluation/run_track2_crossmodel.py \
    --llama-dir /home/samu170h/LLMPedia/openLLMPedia/llama3.3-70b_100K \
    --deepseek-dir /home/samu170h/LLMPedia/openLLMPedia/deepseekV3.2_100K \
    --gpt-dir /home/samu170h/LLMPedia/openLLMPedia/gpt_5_mini_1M \
    --max-subjects 100000 \
    --sample-n 500 --sample-min 100 \
    --compute-similarity --compute-factuality \
    --fact-model-key gpt-4.1-nano \
    --evidence-sources wikipedia,web \
    --max-claims 10 --max-retries 2 \
    --concurrency 50 \
    --output-dir /home/samu170h/LLMPedia/openLLMPedia/cross_model_results
```

#### Clean and Re-run

```bash
python evaluation/run_track2_crossmodel.py \
    --llama-dir /path/to/llama \
    --deepseek-dir /path/to/deepseek \
    --gpt-dir /path/to/gpt \
    --clean-audit \
    --max-subjects 100000 --sample-n 1000 \
    --compute-similarity --compute-factuality \
    --concurrency 50 \
    --output-dir /path/to/output
```

#### Track 2 Output Files

All in `cross_model_results/`:

| File | What |
|------|------|
| `subject_overlap.csv` | Per-subject: which models have it |
| `entity_overlap_cross_model.csv` | Exact + canonical Jaccard on wikilinks |
| `wikilink_stats.csv` | Per-model: mean/std links, diversity, word count |
| `cross_model_text_similarity.csv` | Jaccard + n-gram between models' articles |
| `wikipedia_coverage.csv` | Found/not-found per model |
| `factuality_summary.csv` | Precision, hallucination (wiki-found filtered) |
| `sampled_subjects.csv` | The intersection subjects used |
| `cross_model_report.json` | Full report with all stats |
| `figures/*.pdf` | ACL-quality figures (Venn, heatmap, bars) |

### Key Flags Reference

| Flag | Track 1 | Track 2 | What |
|------|---------|---------|------|
| `--analyze-only` | ✅ | ❌ | Skip evaluation, just analyze existing results |
| `--skip-analysis` | ✅ | ❌ | Skip cross-analysis after evaluation |
| `--skip-cross-sim` | ✅ | ❌ | Skip slow cross-model/persona text similarity |
| `--clean-audit` | ✅ | ✅ | Delete previous outputs before running |
| `--sample-n N` | ✅ | ✅ | Number of articles/subjects to sample |
| `--sample-min N` | ✅ | ✅ | Skip if fewer than N available |
| `--sample-max N` | ✅ | ✅ | Cap sample size |
| `--compute-similarity` | ✅ | ✅ | TF-IDF, Jaccard, semantic vs Wikipedia |
| `--compute-bertscore` | ✅ | ✅ | BERTScore F1 (slow, needs bert-score) |
| `--compute-stylistic` | ✅ | ✅ | TTR, sentence length, function words |
| `--compute-factuality` | ❌ (always on) | ✅ | LLM-based claim verification |
| `--concurrency N` | ✅ | ✅ | Parallel workers |
| `--max-retries N` | ✅ | ✅ | Retry on JSON parse failure |

### Wikipedia Coverage: How it Works

Subjects **not in Wikipedia** are excluded from `wiki_precision` and `wiki_false_rate`. They appear separately as `n_wiki_not_found` and `wiki_coverage_rate`.

This means:
- "Edgar J. Banks has no Wikipedia page" → counted in `n_wiki_not_found`, NOT as hallucination
- "Hammurabi's article says he was born in 1850 BCE" → refuted by Wikipedia → counts in `wiki_false_rate`
- Web-based metrics (`web_precision`) use ALL subjects regardless

This is a **positive finding** for the paper: LLMpedia generates articles about subjects beyond Wikipedia's coverage.

## Web & Post-Processing

### Funnel Analysis

Tracks every entity through the pipeline and measures where candidates are generated, filtered, and enqueued.

```bash
python evaluation/funnel_analysis.py runs/my_run
```

Output → `runs/my_run/funnel_analysis/`:
- `funnel_per_entity.csv` — One row per entity: all stage counts, rates, and explanations
- `funnel_summary_by_hop.csv` — Aggregate statistics per hop level
- `funnel_children.csv` — Parent→child mapping
- `full_report.txt` — Human-readable report with bar charts
- `figures/` — Matplotlib visualizations

### Image Index Builder

Downloads article images from Wikipedia and Wikidata with full license attribution.

```bash
python web/build_image_index.py runs/my_run \
    --workers 40 \
    --thumb 500 \
    --force  # Re-download all
```

### Site Builder

Generates a searchable static encyclopedia website.

```bash
python web/build_llmpedia_site.py runs/my_run \
    --workers 16 \
    --clean  # Delete existing site/ and rebuild
```

**Prerequisites**: Must run `funnel_analysis.py` first.

## Output Structure

```
run_dir/
├── queue.json                      # Current queue state (all subjects)
├── queue.jsonl                     # Queue event log
├── seen_canon_keys.json            # Canonical keys for deduplication
├── run_meta.json                   # Run configuration and metadata
├── errors.log                      # Error log with stack traces
│
├── articles.jsonl                  # Generated articles (full records)
├── articles_wikitext.jsonl         # Article wikitext only
├── articles_meta.jsonl             # Article metadata + extraction stats
├── outlines.jsonl                  # Generated outlines
├── self_rag_log.jsonl              # Self-RAG context (if enabled)
│
├── ner_decisions.jsonl             # NER classification decisions
├── ner_lowconf.jsonl               # NER rejections (low confidence)
├── elicit_lowconf.jsonl            # Elicitation confidence rejections
├── plural_s_dedup.jsonl            # Plural/singular dedup log
│
├── embeddings.jsonl                # Vector embeddings for similarity
├── similarity_decisions.jsonl      # Similarity filter decisions
├── reject_similarity.jsonl         # Similarity rejections
│
├── gptkb_image_cache.json          # Image cache (Wikipedia/Wikidata)
├── image_store/                    # Downloaded images with attribution
│
├── funnel_analysis/                # Funnel outputs
│   ├── funnel_per_entity.csv
│   ├── funnel_summary_by_hop.csv
│   └── figures/
│
├── site/                           # Generated website
│   ├── index.html
│   ├── style.css
│   ├── search_index.js
│   └── *.html
│
└── parallelqueue/                  # Stage queue state (payload-based)
    └── payloads/
```

## Performance Tuning

### Recommended Configurations

#### Batch Mode (Cost-Optimized, 50K subjects)
```bash
python llmpedia.py \
  --mode batch \
  --model-key gpt-4.1-mini \
  --seed "Vannevar Bush" \
  --max-subjects 50000 \
  --self-rag false \
  --use-ner true \
  --ner-mode batch \
  --ner-chunk-size 100 \
  --batch-size 1000 \
  --concurrency 100 \
  --outline-workers 3 \
  --elicit-workers 4 \
  --ner-workers 3 \
  --sim-workers 3 \
  --use-similarity true \
  --output-dir runs/batch_50k \
  --resume \
  --reset-working
```

#### High-Quality Mode (With Self-RAG, 1K subjects)
```bash
python llmpedia.py \
  --mode online \
  --model-key gpt-4.1 \
  --seed "Albert Einstein" \
  --max-subjects 1000 \
  --self-rag true \
  --self-rag-model-key gpt-4.1-mini \
  --two-stage-elicitation true \
  --use-ner true \
  --use-similarity true \
  --concurrency 20 \
  --selfrag-workers 4 \
  --outline-workers 4 \
  --elicit-workers 8 \
  --output-dir runs/high_quality
```

#### Fixed Topic List (No Expansion)
```bash
python llmpedia.py \
  --mode online \
  --preload-topics topics.txt \
  --preload-only true \
  --use-ner false \
  --use-similarity false \
  --max-depth 0 \
  --output-dir runs/fixed_list
```

### Memory Optimization

The similarity engine uses a pre-allocated embedding matrix with doubling strategy:

- **Memory per embedding**: ~6KB (1536 dims × 4 bytes)
- **100K embeddings**: ~600MB
- **1M embeddings**: ~6GB

For very large runs (> 500K subjects), consider:
- Using `--similarity-provider local` with smaller models (e.g., `all-MiniLM-L6-v2`, 384 dims)
- Increasing `--similarity-threshold` to reduce LLM filter calls
- Running similarity in `batch` mode to reduce memory pressure

## Troubleshooting

### Common Issues

#### Pipeline Stuck (No Progress)

If workers are idle but queue shows pending items:
```bash
# Reset stuck 'working' items to 'pending'
python llmpedia.py ... --resume --reset-working
```

#### Missing Funnel Data for Site Builder

```
❌ ERROR: Funnel data not found at:
   /path/to/run_dir/funnel_analysis/funnel_per_entity.csv
```

**Solution**:
```bash
python evaluation/funnel_analysis.py /path/to/run_dir
```

#### No Images on Website

```bash
# Download images
python web/build_image_index.py /path/to/run_dir --workers 40

# Rebuild site with images
python web/build_llmpedia_site.py /path/to/run_dir --clean
```

#### Batch Mode Failures

If batch jobs fail with `provider must be 'openai'`:
- Ensure all model keys (elicit, ner, self-rag) use OpenAI provider
- Check that `--ner-mode` and `--similarity-mode` match `--mode batch`

### Diagnostic Commands

```bash
cd your-output-dir

# Check queue state
jq -c '{done: [.[] | select(.status=="done")] | length,
        working: [.[] | select(.status=="working")] | length,
        pending: [.[] | select(.status=="pending")] | length}' queue.json

# Count articles
wc -l articles.jsonl

# Check funnel summary
head -1 funnel_analysis/funnel_summary_by_hop.csv && \
cat funnel_analysis/funnel_summary_by_hop.csv

# Monitor progress (if running)
watch -n 5 'jq -c "{d: [.[] | select(.status==\"done\")] | length, \
  w: [.[] | select(.status==\"working\")] | length, \
  p: [.[] | select(.status==\"pending\")] | length}" queue.json'
```

## Citation

If you use LLMPedia in your research, please cite:

```bibtex
@software{llmpedia2024,
  title = {LLMPedia: Autonomous Knowledge Graph Generator},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourorg/llmpedia}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- OpenAI for GPT models and Batch API
```