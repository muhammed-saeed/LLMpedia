```
# ══════════════════════════════════════════
# Topic 1: Ancient Babylon
# ══════════════════════════════════════════

# --- gpt-5-mini ---
python3 llmpedia.py --seed "Ancient Babylon" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode batch --model-key gpt-5-mini --reasoning-effort minimal --text-verbosity low --elicit-workers 1 --outline-workers 1 --selfrag-workers 0 --ner-workers 1 --sim-workers 1 --concurrency 6 --batch-size 1000 --batch-poll-interval 30 --ner-mode batch --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona scientific_neutral \
  --output-dir openLLMPedia/topic_runs/gpt-5-mini/ancient_babylon/scientific_neutral

python3 llmpedia.py --seed "Ancient Babylon" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode batch --model-key gpt-5-mini --reasoning-effort minimal --text-verbosity low --elicit-workers 1 --outline-workers 1 --selfrag-workers 0 --ner-workers 1 --sim-workers 1 --concurrency 6 --batch-size 1000 --batch-poll-interval 30 --ner-mode batch --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona left_leaning \
  --output-dir openLLMPedia/topic_runs/gpt-5-mini/ancient_babylon/left_leaning

python3 llmpedia.py --seed "Ancient Babylon" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode batch --model-key gpt-5-mini --reasoning-effort minimal --text-verbosity low --elicit-workers 1 --outline-workers 1 --selfrag-workers 0 --ner-workers 1 --sim-workers 1 --concurrency 6 --batch-size 1000 --batch-poll-interval 30 --ner-mode batch --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona conservative \
  --output-dir openLLMPedia/topic_runs/gpt-5-mini/ancient_babylon/conservative

# --- scads-llama-3.3-70b ---
python3 llmpedia.py --seed "Ancient Babylon" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-llama-3.3-70b --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona scientific_neutral \
  --output-dir openLLMPedia/topic_runs/scads-llama-3.3-70b/ancient_babylon/scientific_neutral

python3 llmpedia.py --seed "Ancient Babylon" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-llama-3.3-70b --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona left_leaning \
  --output-dir openLLMPedia/topic_runs/scads-llama-3.3-70b/ancient_babylon/left_leaning

python3 llmpedia.py --seed "Ancient Babylon" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-llama-3.3-70b --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona conservative \
  --output-dir openLLMPedia/topic_runs/scads-llama-3.3-70b/ancient_babylon/conservative

# --- scads-DeepSeek-V3.2 ---
python3 llmpedia.py --seed "Ancient Babylon" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-DeepSeek-V3.2 --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona scientific_neutral \
  --output-dir openLLMPedia/topic_runs/scads-DeepSeek-V3.2/ancient_babylon/scientific_neutral

python3 llmpedia.py --seed "Ancient Babylon" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-DeepSeek-V3.2 --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona left_leaning \
  --output-dir openLLMPedia/topic_runs/scads-DeepSeek-V3.2/ancient_babylon/left_leaning

python3 llmpedia.py --seed "Ancient Babylon" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-DeepSeek-V3.2 --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona conservative \
  --output-dir openLLMPedia/topic_runs/scads-DeepSeek-V3.2/ancient_babylon/conservative


# ══════════════════════════════════════════
# Topic 2: US Civil Rights Movement
# ══════════════════════════════════════════

# --- gpt-5-mini ---
python3 llmpedia.py --seed "US Civil Rights Movement" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode batch --model-key gpt-5-mini --reasoning-effort minimal --text-verbosity low --elicit-workers 1 --outline-workers 1 --selfrag-workers 0 --ner-workers 1 --sim-workers 1 --concurrency 6 --batch-size 1000 --batch-poll-interval 30 --ner-mode batch --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona scientific_neutral \
  --output-dir openLLMPedia/topic_runs/gpt-5-mini/us_civil_rights_movement/scientific_neutral

python3 llmpedia.py --seed "US Civil Rights Movement" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode batch --model-key gpt-5-mini --reasoning-effort minimal --text-verbosity low --elicit-workers 1 --outline-workers 1 --selfrag-workers 0 --ner-workers 1 --sim-workers 1 --concurrency 6 --batch-size 1000 --batch-poll-interval 30 --ner-mode batch --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona left_leaning \
  --output-dir openLLMPedia/topic_runs/gpt-5-mini/us_civil_rights_movement/left_leaning

python3 llmpedia.py --seed "US Civil Rights Movement" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode batch --model-key gpt-5-mini --reasoning-effort minimal --text-verbosity low --elicit-workers 1 --outline-workers 1 --selfrag-workers 0 --ner-workers 1 --sim-workers 1 --concurrency 6 --batch-size 1000 --batch-poll-interval 30 --ner-mode batch --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona conservative \
  --output-dir openLLMPedia/topic_runs/gpt-5-mini/us_civil_rights_movement/conservative

# --- scads-llama-3.3-70b ---
python3 llmpedia.py --seed "US Civil Rights Movement" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-llama-3.3-70b --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona scientific_neutral \
  --output-dir openLLMPedia/topic_runs/scads-llama-3.3-70b/us_civil_rights_movement/scientific_neutral

python3 llmpedia.py --seed "US Civil Rights Movement" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-llama-3.3-70b --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona left_leaning \
  --output-dir openLLMPedia/topic_runs/scads-llama-3.3-70b/us_civil_rights_movement/left_leaning

python3 llmpedia.py --seed "US Civil Rights Movement" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-llama-3.3-70b --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona conservative \
  --output-dir openLLMPedia/topic_runs/scads-llama-3.3-70b/us_civil_rights_movement/conservative

# --- scads-DeepSeek-V3.2 ---
python3 llmpedia.py --seed "US Civil Rights Movement" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-DeepSeek-V3.2 --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona scientific_neutral \
  --output-dir openLLMPedia/topic_runs/scads-DeepSeek-V3.2/us_civil_rights_movement/scientific_neutral

python3 llmpedia.py --seed "US Civil Rights Movement" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-DeepSeek-V3.2 --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona left_leaning \
  --output-dir openLLMPedia/topic_runs/scads-DeepSeek-V3.2/us_civil_rights_movement/left_leaning

python3 llmpedia.py --seed "US Civil Rights Movement" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-DeepSeek-V3.2 --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona conservative \
  --output-dir openLLMPedia/topic_runs/scads-DeepSeek-V3.2/us_civil_rights_movement/conservative


# ══════════════════════════════════════════
# Topic 3: Dutch Colonization in Southeast Asia
# ══════════════════════════════════════════

# --- gpt-5-mini ---
python3 llmpedia.py --seed "Dutch Colonization in Southeast Asia" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode batch --model-key gpt-5-mini --reasoning-effort minimal --text-verbosity low --elicit-workers 1 --outline-workers 1 --selfrag-workers 0 --ner-workers 1 --sim-workers 1 --concurrency 6 --batch-size 1000 --batch-poll-interval 30 --ner-mode batch --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona scientific_neutral \
  --output-dir openLLMPedia/topic_runs/gpt-5-mini/dutch_colonization_se_asia/scientific_neutral

python3 llmpedia.py --seed "Dutch Colonization in Southeast Asia" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode batch --model-key gpt-5-mini --reasoning-effort minimal --text-verbosity low --elicit-workers 1 --outline-workers 1 --selfrag-workers 0 --ner-workers 1 --sim-workers 1 --concurrency 6 --batch-size 1000 --batch-poll-interval 30 --ner-mode batch --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona left_leaning \
  --output-dir openLLMPedia/topic_runs/gpt-5-mini/dutch_colonization_se_asia/left_leaning

python3 llmpedia.py --seed "Dutch Colonization in Southeast Asia" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode batch --model-key gpt-5-mini --reasoning-effort minimal --text-verbosity low --elicit-workers 1 --outline-workers 1 --selfrag-workers 0 --ner-workers 1 --sim-workers 1 --concurrency 6 --batch-size 1000 --batch-poll-interval 30 --ner-mode batch --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona conservative \
  --output-dir openLLMPedia/topic_runs/gpt-5-mini/dutch_colonization_se_asia/conservative

# --- scads-llama-3.3-70b ---
python3 llmpedia.py --seed "Dutch Colonization in Southeast Asia" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-llama-3.3-70b --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona scientific_neutral \
  --output-dir openLLMPedia/topic_runs/scads-llama-3.3-70b/dutch_colonization_se_asia/scientific_neutral

python3 llmpedia.py --seed "Dutch Colonization in Southeast Asia" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-llama-3.3-70b --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona left_leaning \
  --output-dir openLLMPedia/topic_runs/scads-llama-3.3-70b/dutch_colonization_se_asia/left_leaning

python3 llmpedia.py --seed "Dutch Colonization in Southeast Asia" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-llama-3.3-70b --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona conservative \
  --output-dir openLLMPedia/topic_runs/scads-llama-3.3-70b/dutch_colonization_se_asia/conservative

# --- scads-DeepSeek-V3.2 ---
python3 llmpedia.py --seed "Dutch Colonization in Southeast Asia" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-DeepSeek-V3.2 --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona scientific_neutral \
  --output-dir openLLMPedia/topic_runs/scads-DeepSeek-V3.2/dutch_colonization_se_asia/scientific_neutral

python3 llmpedia.py --seed "Dutch Colonization in Southeast Asia" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-DeepSeek-V3.2 --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona left_leaning \
  --output-dir openLLMPedia/topic_runs/scads-DeepSeek-V3.2/dutch_colonization_se_asia/left_leaning

python3 llmpedia.py --seed "Dutch Colonization in Southeast Asia" --domain topic --max-depth 0 --max-subjects 1000 --use-ner true --progress-metrics --use-similarity true --resume --reset-working --timeout 450 --mode online --model-key scads-DeepSeek-V3.2 --text-verbosity low --elicit-workers 30 --outline-workers 10 --selfrag-workers 0 --ner-workers 5 --sim-workers 5 --concurrency 50 --ner-mode online --elicitation-strategy baseline --ner-strategy baseline --self-rag false --ner-chunk-size 100 --similarity-mode online \
  --persona conservative \
  --output-dir openLLMPedia/topic_runs/scads-DeepSeek-V3.2/dutch_colonization_se_asia/conservative
```