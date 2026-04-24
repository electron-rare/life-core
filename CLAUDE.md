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

## /v1/embeddings (V1.8 Wave B, axis 10)

OpenAI-compat embeddings endpoint. Accepts `input` string or list of
strings, returns `data[].embedding` float vectors and
`usage.prompt_tokens`. Single backend in V1.8: Tower TEI via `EMBED_URL`
(default `http://host.docker.internal:11437`) with
sentence-transformers fallback. Multi-backend routing + tiktoken-based
usage is V1.9.

```bash
curl -s https://life.saillant.cc/v1/embeddings \
  -H "Authorization: Bearer $LIFE_INTERNAL_BEARER" \
  -H "Content-Type: application/json" \
  -d '{"input": "STM32F103C8T6 bluepill", "model": "tei/bge-small"}' \
  | jq '.data[0].embedding | length'
```

### Sprint 1 additions (2026-04-24)

| Module | Role |
|--------|------|
| `life_core.agents` | `POST /agents/{role}/run` inner HITL orchestrator + `/agents/{role}/decide/{id}` + `/agents/runs/{id}`. AgentEnvelope {job_id, result: AgentResult} two-level shape matching engine's LifeCoreClient. |
| `life_core.generators` | spec/kicad/firmware/spice LLM generators inheriting BaseGenerator. Jinja2 prompts in `life_core/llm/prompts/`. |
| `life_core.evaluations` | 4 comparators (spec_coverage, hardware_diff, firmware_behavior, simulation_diff) + harness + `/evaluations/run` router. |
| `life_core.traceability` | inner DAG service + `/traceability/graph?deliverable_slug=` router. |
| `life_core.artifacts` | content-hashed versioned storage (write/read) + ArtifactRef. |
| `life_core.tools` | kicad-cli / PlatformIO / ngspice (via spice-life) / emc (stub) wrappers. |
| `life_core.inner_trace` | SQLAlchemy models for 5 tables (agent_run, artifact, generation_run, relation, evaluation) under schema `inner_trace`. Alembic revision `2026042401_inner_trace_init`. |

Inner state machine and data model: see
`docs/superpowers/specs/2026-04-24-llm-workflow-sensor-node-pilot-design.md`.

ADR-006 decided `kiutils` for kicad_generator (SKiDL 1.2.3 doesn't support KiCad 8). See `docs/superpowers/decisions/2026-04-24-adr-006-kicad-generator-approach.md`.

