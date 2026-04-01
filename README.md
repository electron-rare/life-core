# life-core

AI backend engine for FineFab -- LLM router, RAG pipeline, caching, and agent orchestration.

Part of the [FineFab](https://github.com/L-electron-Rare) platform.

## What it does

- Routes LLM requests across multiple providers with automatic fallback and circuit breakers
- Runs a multi-stage RAG pipeline (chunking, embedding, retrieval, reranking)
- Manages a multi-tier cache (in-memory + Redis) for fast repeated inference
- Orchestrates AI agents via a state-graph execution engine
- Exposes stable backend services consumed by `life-reborn` and other platform components

## Tech stack

Python 3.12+ / FastAPI / Pydantic / pytest

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Related repos

| Repo | Role |
|------|------|
| [life-reborn](https://github.com/L-electron-Rare/life-reborn) | API gateway (auth, rate limiting, OpenAPI) |
| [life-web](https://github.com/L-electron-Rare/life-web) | Operator cockpit UI |
| [life-spec](https://github.com/L-electron-Rare/life-spec) | Functional specifications and BMAD gates |
| [finefab-shared](https://github.com/L-electron-Rare/finefab-shared) | Shared contracts and types |

## License

[MIT](LICENSE)
