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
  api.py                FastAPI app, lifespan, routes (/chat, /health, /models, /search, /alerts)
  router/               LLM multi-provider router with fallback
    router.py           Router with health checks and fallback logic
    providers/          LiteLLM unified provider (routes to vLLM, Ollama, cloud)
  services/chat.py      ChatService — cache + RAG + LLM orchestration
  rag/pipeline.py       RAG: chunking, embeddings (Ollama), Qdrant — search_multi() cross-collections
  cache.py              MultiTierCache: L1 in-memory LRU + L2 Redis
  infra_api.py          /infra/* endpoints (containers, storage, network health)
  models_api.py         /models/catalog with domain metadata
  audit_api.py          /audit/status and /audit/report
  telemetry.py          OpenTelemetry init
  langfuse_tracing.py   Langfuse callbacks and scoring
```

## Key env vars

| Variable | Description |
|----------|-------------|
| `VLLM_BASE_URL` | SSH tunnel vers KXKM-AI, port 11436 |
| `VLLM_MODELS` | `openai/qwen-14b-awq` |
| `OLLAMA_URL` | Ollama Tower — LLM (résumés, qwen3:4b) |
| `OLLAMA_EMBED_URL` | Ollama Tower — embeddings uniquement (nomic-embed-text). Séparé de `OLLAMA_URL` pour permettre des endpoints distincts. |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Cloud fallback |
| `REDIS_URL` | Cache L2 |
| `LANGFUSE_HOST` | Tracing |

## Nouveautés

- `search_multi()` dans `rag/pipeline.py` — recherche parallèle sur les 4 collections Qdrant (`nextcloud_docs`, `outline_wiki`, `github_repos`, `life_chunks`) avec filtre par collection configurable
- `/search` endpoint — proxy vers search_multi, utilisé par rag-web
- `/alerts` endpoint — lit les alertes veille générées par nc-rag-indexer dans `github_repos`
- Batch embedding — envoi groupé à Ollama pour réduire la latence d'indexation
