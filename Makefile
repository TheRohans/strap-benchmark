PYTHON         ?= /Users/robrohan/miniconda3/envs/strap_benchmark/bin/python

# ── strap server ──────────────────────────────────────────────────────────────
STRAP_HOST     ?= 127.0.0.1
STRAP_PORT     ?= 6667
HASH           := \#
STRAP_CHANNEL  ?= $(HASH)main
STRAP_PASSWORD ?= password

# ── qwen / ollama baseline ────────────────────────────────────────────────────
QWEN_BASE_URL  ?= http://192.168.1.25:8080/v1
QWEN_MODEL     ?= qwen3.6:27b

# ── llm judge (required for math benchmark only) ──────────────────────────────
# Point at LM Studio (default port 1234) or any OpenAI-compatible server
JUDGE_BASE_URL ?= http://localhost:1234/v1
JUDGE_MODEL    ?= qwen3.6:27b
JUDGE_API_KEY  ?= ollama

# ── benchmark parameters ──────────────────────────────────────────────────────
MAX_SAMPLES     ?= 50
MAX_LEVEL       ?= 3                    # math difficulty 1-5
CONTEXT_LENGTHS ?= 1000,2000,4000,8000  # ruler token counts
NEEDLE_DEPTHS   ?= 0.1,0.5,0.9         # ruler needle positions
RULER_N         ?= 3                    # tasks per (length, depth) config

LOG_LEVEL      ?= WARNING

STRAP_TIMEOUT  ?= 300   # seconds to wait for a response before giving up

# ── env var bundles ───────────────────────────────────────────────────────────
_STRAP_ENV  = STRAP_HOST=$(STRAP_HOST) STRAP_PORT=$(STRAP_PORT) STRAP_CHANNEL="$(STRAP_CHANNEL)" STRAP_TIMEOUT=$(STRAP_TIMEOUT)
_QWEN_ENV   = QWEN_BASE_URL=$(QWEN_BASE_URL) QWEN_MODEL=$(QWEN_MODEL)
_JUDGE_ENV  = JUDGE_BASE_URL=$(JUDGE_BASE_URL) JUDGE_MODEL=$(JUDGE_MODEL) JUDGE_API_KEY=$(JUDGE_API_KEY)

# ── targets ───────────────────────────────────────────────────────────────────
.PHONY: help \
        gsm8k-strap gsm8k-qwen gsm8k \
        math-strap  math-qwen  math  \
        ruler-strap ruler-qwen ruler \
        all-strap   all-qwen   all

help:
	@echo ""
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "Targets:"
	@echo "  gsm8k-strap    GSM8K grade-school math      → strap"
	@echo "  gsm8k-qwen     GSM8K grade-school math      → base_qwen (ollama)"
	@echo "  gsm8k          GSM8K grade-school math      → both"
	@echo "  math-strap     MATH-500 competition math    → strap     (needs judge LLM)"
	@echo "  math-qwen      MATH-500 competition math    → base_qwen (needs judge LLM)"
	@echo "  math           MATH-500 competition math    → both      (needs judge LLM)"
	@echo "  ruler-strap    RULER needle-in-haystack     → strap"
	@echo "  ruler-qwen     RULER needle-in-haystack     → base_qwen"
	@echo "  ruler          RULER needle-in-haystack     → both"
	@echo "  all-strap      All three benchmarks         → strap"
	@echo "  all-qwen       All three benchmarks         → base_qwen"
	@echo "  all            All three benchmarks         → both"
	@echo ""
	@echo "Key variables (current values):"
	@echo "  STRAP_HOST=$(STRAP_HOST)  STRAP_PORT=$(STRAP_PORT)  STRAP_CHANNEL=$(STRAP_CHANNEL)"
	@echo "  QWEN_BASE_URL=$(QWEN_BASE_URL)  QWEN_MODEL=$(QWEN_MODEL)"
	@echo "  JUDGE_BASE_URL=$(JUDGE_BASE_URL)  JUDGE_MODEL=$(JUDGE_MODEL)"
	@echo "  MAX_SAMPLES=$(MAX_SAMPLES)  MAX_LEVEL=$(MAX_LEVEL)"
	@echo "  CONTEXT_LENGTHS=$(CONTEXT_LENGTHS)  NEEDLE_DEPTHS=$(NEEDLE_DEPTHS)  RULER_N=$(RULER_N)"
	@echo ""

# ── gsm8k ─────────────────────────────────────────────────────────────────────

gsm8k-strap:
	$(_STRAP_ENV) $(PYTHON) run.py \
		--dataset gsm8k --runner strap \
		--max-samples $(MAX_SAMPLES) \
		--log-level $(LOG_LEVEL)

gsm8k-qwen:
	$(_QWEN_ENV) $(PYTHON) run.py \
		--dataset gsm8k --runner base_qwen \
		--max-samples $(MAX_SAMPLES) \
		--log-level $(LOG_LEVEL)

# ── math-500 (judge LLM required) ─────────────────────────────────────────────

math-strap:
	$(_STRAP_ENV) $(_JUDGE_ENV) $(PYTHON) run.py \
		--dataset math --runner strap \
		--max-samples $(MAX_SAMPLES) \
		--max-level $(MAX_LEVEL) \
		--log-level $(LOG_LEVEL)

math-qwen:
	$(_QWEN_ENV) $(_JUDGE_ENV) $(PYTHON) run.py \
		--dataset math --runner base_qwen \
		--max-samples $(MAX_SAMPLES) \
		--max-level $(MAX_LEVEL) \
		--log-level $(LOG_LEVEL)

# ── ruler niah ────────────────────────────────────────────────────────────────

ruler-strap:
	$(_STRAP_ENV) $(PYTHON) run.py \
		--dataset ruler --runner strap \
		--context-lengths $(CONTEXT_LENGTHS) \
		--needle-depths $(NEEDLE_DEPTHS) \
		--ruler-n $(RULER_N) \
		--log-level $(LOG_LEVEL)

ruler-qwen:
	$(_QWEN_ENV) $(PYTHON) run.py \
		--dataset ruler --runner base_qwen \
		--context-lengths $(CONTEXT_LENGTHS) \
		--needle-depths $(NEEDLE_DEPTHS) \
		--ruler-n $(RULER_N) \
		--log-level $(LOG_LEVEL)

# ── combined ──────────────────────────────────────────────────────────────────

gsm8k: gsm8k-strap gsm8k-qwen

math: math-strap math-qwen

ruler: ruler-strap ruler-qwen

all-strap: gsm8k-strap math-strap ruler-strap

all-qwen: gsm8k-qwen math-qwen ruler-qwen

all: all-strap all-qwen
