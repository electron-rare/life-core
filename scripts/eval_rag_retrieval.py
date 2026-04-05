#!/usr/bin/env python3
"""Replay a retrieval dataset against life-core /rag/search and emit JSON plus optional Langfuse runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from datetime import UTC, datetime


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        row.setdefault("id", _case_id(row))
        rows.append(row)
    return rows


def _case_id(case: dict[str, Any]) -> str:
    if case.get("id"):
        return str(case["id"])

    stable_payload = {
        "query": case.get("query", ""),
        "top_k": case.get("top_k"),
        "expected_document_ids": case.get("expected_document_ids", []),
        "expected_terms": case.get("expected_terms", []),
    }
    raw = json.dumps(stable_payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _dataset_item_id(case: dict[str, Any]) -> str:
    return f"rag-eval-{_case_id(case)}"


def _search(base_url: str, query: str, top_k: int, mode: str | None = None) -> dict[str, Any]:
    params = {"q": query, "top_k": top_k}
    if mode:
        params["mode"] = mode
    qs = urlencode(params)
    url = f"{base_url.rstrip('/')}/rag/search?{qs}"
    request = Request(url, method="GET")
    with urlopen(request, timeout=30) as response:  # noqa: S310 - explicit operator-controlled URL
        return json.loads(response.read().decode("utf-8"))


def _match_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    results = response.get("results", [])
    expected_document_ids = set(case.get("expected_document_ids", []))
    expected_terms = [term.lower() for term in case.get("expected_terms", [])]

    matched_document = any(result.get("document_id") in expected_document_ids for result in results)
    matched_term = any(
        term in (result.get("content", "").lower())
        for term in expected_terms
        for result in results
    )

    top_score = float(results[0].get("score", 0.0)) if results else 0.0
    success = matched_document or matched_term or (not expected_document_ids and not expected_terms and bool(results))
    return {
        "case_id": _case_id(case),
        "query": case["query"],
        "success": success,
        "top_score": top_score,
        "result_count": len(results),
        "matched_document": matched_document,
        "matched_term": matched_term,
        "mode": response.get("mode", "unknown"),
        "results": results,
    }


def _summarize(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    top_scores = [case["top_score"] for case in outcomes]
    successes = [case for case in outcomes if case["success"]]
    failures = [case for case in outcomes if not case["success"]]
    return {
        "total_cases": len(outcomes),
        "successful_cases": len(successes),
        "failed_cases": len(failures),
        "hit_rate": (len(successes) / len(outcomes)) if outcomes else 0.0,
        "mean_top_score": mean(top_scores) if top_scores else 0.0,
    }


def _require_langfuse_client():
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    host = os.environ.get("LANGFUSE_HOST", "").strip()
    if not all([public_key, secret_key, host]):
        raise RuntimeError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST are required")

    from langfuse import Langfuse

    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def _ensure_langfuse_dataset(client, dataset_name: str, cases: list[dict[str, Any]], base_url: str) -> dict[str, Any]:
    try:
        dataset = client.get_dataset(dataset_name)
    except Exception:
        client.create_dataset(
            name=dataset_name,
            description="Factory 4 Life RAG retrieval eval dataset",
            metadata={"source": "life-core/scripts/eval_rag_retrieval.py", "base_url": base_url},
        )
        dataset = client.get_dataset(dataset_name)

    existing_items = {item.id: item for item in dataset.items}
    for case in cases:
        item_id = _dataset_item_id(case)
        if item_id in existing_items:
            continue
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item_id,
            input={"query": case["query"], "top_k": int(case.get("top_k", 5))},
            expected_output={
                "expected_document_ids": case.get("expected_document_ids", []),
                "expected_terms": case.get("expected_terms", []),
            },
            metadata={"case_id": _case_id(case), "source": "life-core-rag-eval"},
        )

    return {item.id: item for item in client.get_dataset(dataset_name).items}


def _publish_langfuse_run(
    *,
    client,
    dataset_name: str,
    dataset_items: dict[str, Any],
    cases: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    run_name: str,
    run_description: str | None,
    run_metadata: dict[str, Any],
) -> None:
    for case, outcome in zip(cases, outcomes):
        item = dataset_items[_dataset_item_id(case)]
        trace = client.trace(
            name="rag-retrieval-eval",
            input={"query": case["query"], "top_k": int(case.get("top_k", 5)), "mode": outcome["mode"]},
            output={"results": outcome.get("results", [])},
            metadata={
                **run_metadata,
                "dataset_name": dataset_name,
                "case_id": outcome["case_id"],
                "success": outcome["success"],
                "matched_document": outcome["matched_document"],
                "matched_term": outcome["matched_term"],
                "result_count": outcome["result_count"],
            },
        )
        client.score(
            trace_id=trace.id,
            name="retrieval-success",
            value=1.0 if outcome["success"] else 0.0,
            comment=f"query={case['query']}",
        )
        client.score(
            trace_id=trace.id,
            name="retrieval-top-score",
            value=float(outcome["top_score"]),
        )
        item.link(
            trace,
            run_name=run_name,
            run_metadata=run_metadata,
            run_description=run_description,
        )

    client.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset file")
    parser.add_argument("--base-url", default="http://localhost:8000", help="life-core base URL")
    parser.add_argument("--top-k", type=int, default=5, help="top_k value for /rag/search")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["hybrid"],
        choices=["dense", "hybrid"],
        help="Retrieval modes to evaluate. Example: --modes dense hybrid",
    )
    parser.add_argument("--output", help="Optional path for the JSON report")
    parser.add_argument("--langfuse-dataset", help="Optional Langfuse dataset name to sync and evaluate")
    parser.add_argument(
        "--langfuse-run-prefix",
        default="factory4life-rag-eval",
        help="Prefix used to name Langfuse dataset runs",
    )
    parser.add_argument(
        "--langfuse-run-description",
        default="Factory 4 Life RAG retrieval evaluation replay",
        help="Description stored on Langfuse dataset runs",
    )
    args = parser.parse_args()

    dataset = _load_dataset(Path(args.dataset))
    report: dict[str, Any] = {
        "base_url": args.base_url,
        "dataset": str(args.dataset),
        "cases": {},
        "summary": {},
    }

    langfuse_client = None
    dataset_items: dict[str, Any] = {}
    if args.langfuse_dataset:
        langfuse_client = _require_langfuse_client()
        dataset_items = _ensure_langfuse_dataset(
            langfuse_client,
            dataset_name=args.langfuse_dataset,
            cases=dataset,
            base_url=args.base_url,
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for mode in args.modes:
        outcomes: list[dict[str, Any]] = []
        for case in dataset:
            try:
                response = _search(
                    args.base_url,
                    case["query"],
                    int(case.get("top_k", args.top_k)),
                    mode=mode,
                )
                outcome = _match_case(case, response)
            except (HTTPError, URLError, TimeoutError) as exc:
                outcome = {
                    "case_id": _case_id(case),
                    "query": case["query"],
                    "success": False,
                    "top_score": 0.0,
                    "result_count": 0,
                    "matched_document": False,
                    "matched_term": False,
                    "mode": mode,
                    "results": [],
                    "error": str(exc),
                }
            outcomes.append(outcome)

        report["cases"][mode] = outcomes
        report["summary"][mode] = _summarize(outcomes)

        if langfuse_client and args.langfuse_dataset:
            run_name = f"{args.langfuse_run_prefix}-{mode}-{timestamp}"
            _publish_langfuse_run(
                client=langfuse_client,
                dataset_name=args.langfuse_dataset,
                dataset_items=dataset_items,
                cases=dataset,
                outcomes=outcomes,
                run_name=run_name,
                run_description=args.langfuse_run_description,
                run_metadata={
                    "mode": mode,
                    "base_url": args.base_url,
                    "dataset_path": str(args.dataset),
                    "summary": report["summary"][mode],
                },
            )

    output = json.dumps(report, indent=2, ensure_ascii=True)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
