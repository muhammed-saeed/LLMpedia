# LLMpedia Evaluation — Quick Reference

## Files

| File | What it does |
|------|-------------|
| `run_track1_topic.py` | Topic-focused evaluation + cross-analysis (3 models × 3 topics × 3 personas) |
| `run_track2_crossmodel.py` | Cross-model comparison (general-domain, neutral persona only) |

---

## Track 1: Topic-Focused (models × topics × personas)

**Directory structure:**
```
/home/samu170h/LLMPedia/openLLMPedia/topic_runs/
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

### Analyze existing results (FREE, instant, no API calls)

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

### Run evaluation + auto-analyze (costs API money)

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

### Clean and re-run

```bash
python evaluation/run_track1_topic.py \
    --root-dir /home/samu170h/LLMPedia/openLLMPedia/topic_runs \
    --clean-audit \
    --fact-model-key gpt-4.1-nano \
    --evidence wikipedia,web \
    --compute-similarity --compute-stylistic \
    --sample-max 100 --concurrency 50
```

### Track 1 output files

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

---

## Track 2: Cross-Model (general-domain, 100K+ articles)

**Input directories:**
```
/home/samu170h/LLMPedia/openLLMPedia/llama3.3-70b_100K/     (articles.jsonl + run_meta.json)
/home/samu170h/LLMPedia/openLLMPedia/deepseekV3.2_100K/     (articles.jsonl + run_meta.json)
/home/samu170h/LLMPedia/openLLMPedia/gpt_5_mini_1M/         (articles.jsonl + run_meta.json)
```

### Structural only (FREE, instant)

```bash
python evaluation/run_track2_crossmodel.py \
    --llama-dir /home/samu170h/LLMPedia/openLLMPedia/llama3.3-70b_100K \
    --deepseek-dir /home/samu170h/LLMPedia/openLLMPedia/deepseekV3.2_100K \
    --gpt-dir /home/samu170h/LLMPedia/openLLMPedia/gpt_5_mini_1M \
    --max-subjects 100000 \
    --sample-n 1000 --sample-min 100 \
    --output-dir /home/samu170h/LLMPedia/openLLMPedia/cross_model_results
```

### With similarity + factuality (costs API money)

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

### Clean and re-run

```bash
python evaluation/run_track2_crossmodel.py \
    --llama-dir /home/samu170h/LLMPedia/openLLMPedia/llama3.3-70b_100K \
    --deepseek-dir /home/samu170h/LLMPedia/openLLMPedia/deepseekV3.2_100K \
    --gpt-dir /home/samu170h/LLMPedia/openLLMPedia/gpt_5_mini_1M \
    --clean-audit \
    --max-subjects 100000 --sample-n 1000 \
    --compute-similarity --compute-factuality \
    --concurrency 50 \
    --output-dir /home/samu170h/LLMPedia/openLLMPedia/cross_model_results
```

### Track 2 output files

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

---

## Key flags reference

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

---

## Wikipedia coverage: how it works

Subjects **not in Wikipedia** are excluded from `wiki_precision` and `wiki_false_rate`.
They appear separately as `n_wiki_not_found` and `wiki_coverage_rate`.

This means:
- "Edgar J. Banks has no Wikipedia page" → counted in `n_wiki_not_found`, NOT as hallucination
- "Hammurabi's article says he was born in 1850 BCE" → refuted by Wikipedia → counts in `wiki_false_rate`
- Web-based metrics (`web_precision`) use ALL subjects regardless

This is a **positive finding** for the paper: LLMpedia generates articles about subjects beyond Wikipedia's coverage.