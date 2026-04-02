# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

life-core is the FastAPI Python backend for FineFab — LLM router, RAG pipeline, multi-tier cache, and service orchestration. Port 8000.

## Commands

```bash
source .venv/bin/activate
PYTHONPATH=$PWD:$PYTHONPATH pytest tests/ -v              # all tests
PYTHONPATH=$PWD:$PYTHONPATH pytest tests/test_router.py -v # single file
PYTHONPATH=$PWD:$PYTHONPATH pytest -k "test_name" -v       # single test
```

## Architecture

```text
life_core/
  api.py                FastAPI app, lifespan, routes (/chat, /health, /models)
  router/               LLM multi-provider router with fallback
    router.py           Router with health checks and fallback logic
    providers/          LiteLLM unified provider (routes to vLLM, Ollama, cloud)
  services/chat.py      ChatService — cache + RAG + LLM orchestration
  rag/pipeline.py       RAG: chunking, embeddings (Ollama/sentence-transformers), Qdrant
  cache.py              MultiTierCache: L1 in-memory LRU + L2 Redis
  infra_api.py          /infra/* endpoints (containers, storage, network health)
  models_api.py         /models/catalog with domain metadata
  audit_api.py          /audit/status and /audit/report
  telemetry.py          OpenTelemetry init
  langfuse_tracing.py   Langfuse callbacks and scoring
```

## Key env vars

`VLLM_BASE_URL`, `VLLM_MODELS`, `OLLAMA_URL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `REDIS_URL`, `LANGFUSE_HOST`
