"""Export Studio threads from the local LangGraph API for blog comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langgraph_sdk import get_client

GRAPH_IDS = (
    "self-rag-assistant",
    "query-decomposition-assistant",
    "hyde-assistant",
)

GRAPH_LABELS = {
    "self-rag-assistant": "Self-RAG",
    "query-decomposition-assistant": "Query decomposition",
    "hyde-assistant": "HyDE",
}


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _duration_ms(start: str | None, end: str | None) -> int | None:
    a, b = _parse_dt(start), _parse_dt(end)
    if a is None or b is None:
        return None
    return int((b - a).total_seconds() * 1000)


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(_text(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _msg_type(message: dict[str, Any]) -> str:
    return str(message.get("type") or message.get("role") or "").lower()


def _turns(messages: list[Any]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    pending: str | None = None
    for raw in messages:
        if not isinstance(raw, dict):
            continue
        kind = _msg_type(raw)
        text = _text(raw.get("content"))
        if not text:
            continue
        if kind in {"human", "user"}:
            pending = text
        elif kind in {"ai", "assistant"} and pending is not None:
            turns.append({"question": pending, "answer": text})
            pending = None
    return turns


def _doc_stats(docs: list[Any]) -> dict[str, Any]:
    items = [d for d in docs if isinstance(d, dict)]
    relevant = [d for d in items if d.get("is_rel") == "relevant"]
    supported = [
        d for d in relevant if d.get("is_sup") in {"fully", "partially"}
    ]
    sources = []
    seen: set[str] = set()
    for doc in relevant:
        src = str(doc.get("source") or "").strip()
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    sub_queries = sorted(
        {
            str(d.get("sub_query")).strip()
            for d in items
            if d.get("sub_query")
        }
    )
    return {
        "n_docs": len(items),
        "n_relevant": len(relevant),
        "n_supported": len(supported),
        "sources": sources,
        "sub_queries_on_docs": sub_queries,
    }


def _norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def summarize_thread(thread: dict[str, Any], state: dict[str, Any], runs: list[Any]) -> dict[str, Any]:
    values = state.get("values") if isinstance(state.get("values"), dict) else {}
    messages = values.get("messages") or []
    docs = values.get("documents") or []
    turns = _turns(messages if isinstance(messages, list) else [])
    stats = _doc_stats(docs if isinstance(docs, list) else [])
    run_rows = []
    for run in runs:
        run_rows.append(
            {
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "created_at": run.get("created_at"),
                "updated_at": run.get("updated_at"),
                "duration_ms": _duration_ms(run.get("created_at"), run.get("updated_at")),
            }
        )
    durations = [r["duration_ms"] for r in run_rows if r["duration_ms"] is not None]
    last_q = turns[-1]["question"] if turns else _text(values.get("question"))
    last_a = turns[-1]["answer"] if turns else _text(values.get("generation"))
    return {
        "thread_id": thread.get("thread_id"),
        "graph_id": (thread.get("metadata") or {}).get("graph_id"),
        "status": thread.get("status"),
        "created_at": thread.get("created_at"),
        "updated_at": thread.get("updated_at"),
        "question": last_q,
        "answer": last_a,
        "all_turns": turns,
        "generation": _text(values.get("generation")),
        "utility": values.get("utility"),
        "iterations": values.get("iterations"),
        "retrieve_decision": values.get("retrieve_decision"),
        "hypothetical": _text(values.get("hypothetical")),
        "sub_queries": values.get("sub_queries") or [],
        **stats,
        "n_messages": len(messages) if isinstance(messages, list) else 0,
        "n_runs": len(run_rows),
        "n_success_runs": sum(1 for r in run_rows if r["status"] == "success"),
        "total_run_ms": sum(durations) if durations else None,
        "last_run_ms": durations[0] if durations else None,
        "runs": run_rows,
    }


async def _all_threads(client: Any, graph_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    limit = 100
    while True:
        page = await client.threads.search(
            metadata={"graph_id": graph_id},
            limit=limit,
            offset=offset,
            sort_by="updated_at",
            sort_order="desc",
        )
        out.extend(page)
        if len(page) < limit:
            break
        offset += limit
    return out


async def export(url: str, graph_ids: tuple[str, ...]) -> dict[str, Any]:
    client = get_client(url=url)
    by_graph: dict[str, list[dict[str, Any]]] = {}
    for graph_id in graph_ids:
        threads = await _all_threads(client, graph_id)
        rows: list[dict[str, Any]] = []
        for thread in threads:
            tid = thread["thread_id"]
            state = await client.threads.get_state(tid)
            runs = await client.runs.list(tid, limit=50)
            rows.append(summarize_thread(thread, state, runs))
        by_graph[graph_id] = rows

    questions: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for graph_id, rows in by_graph.items():
        for row in rows:
            q = row.get("question") or ""
            if q:
                questions[_norm_question(q)][graph_id].append(row)

    matched = []
    for key, graphs in questions.items():
        if len(graphs) >= 2:
            sample = next(iter(next(iter(graphs.values()))))
            matched.append(
                {
                    "question": sample.get("question"),
                    "graphs": {
                        gid: [
                            {
                                "thread_id": r["thread_id"],
                                "answer": r["answer"],
                                "utility": r["utility"],
                                "n_docs": r["n_docs"],
                                "n_relevant": r["n_relevant"],
                                "n_supported": r["n_supported"],
                                "sources": r["sources"],
                                "sub_queries": r["sub_queries"],
                                "last_run_ms": r["last_run_ms"],
                                "status": r["status"],
                            }
                            for r in rows
                        ]
                        for gid, rows in graphs.items()
                    },
                }
            )

    def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
        utils = [r["utility"] for r in rows if isinstance(r.get("utility"), (int, float))]
        durs = [r["last_run_ms"] for r in rows if r.get("last_run_ms") is not None]
        rel = [r["n_relevant"] for r in rows]
        return {
            "n_threads": len(rows),
            "n_idle": sum(1 for r in rows if r["status"] == "idle"),
            "n_error": sum(1 for r in rows if r["status"] == "error"),
            "n_interrupted": sum(1 for r in rows if r["status"] == "interrupted"),
            "mean_utility": round(sum(utils) / len(utils), 2) if utils else None,
            "mean_last_run_s": round((sum(durs) / len(durs)) / 1000, 1) if durs else None,
            "mean_relevant_docs": round(sum(rel) / len(rel), 2) if rel else None,
        }

    totals = {}
    idle_totals = {}
    for graph_id, rows in by_graph.items():
        label = GRAPH_LABELS.get(graph_id, graph_id)
        totals[graph_id] = {"label": label, **_totals(rows)}
        idle_totals[graph_id] = {
            "label": label,
            **_totals([r for r in rows if r["status"] == "idle"]),
        }

    return {
        "source": url,
        "graphs": by_graph,
        "totals": totals,
        "idle_totals": idle_totals,
        "matched_questions": matched,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# RAG graph thread comparison",
        "",
        f"Source: `{payload['source']}`",
        "",
        "## Totals (all threads)",
        "",
        "| Graph | Threads | Idle | Error | Interrupted | Mean utility | Mean last run (s) | Mean relevant docs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tot in payload["totals"].values():
        lines.append(
            "| {label} | {n_threads} | {n_idle} | {n_error} | {n_interrupted} | {mean_utility} | {mean_last_run_s} | {mean_relevant_docs} |".format(
                **{k: tot.get(k) if tot.get(k) is not None else "—" for k in tot}
            )
        )
    lines += [
        "",
        "## Totals (idle / completed only)",
        "",
        "| Graph | Threads | Mean utility | Mean last run (s) | Mean relevant docs |",
        "|---|---:|---:|---:|---:|",
    ]
    for tot in payload.get("idle_totals", {}).values():
        lines.append(
            "| {label} | {n_threads} | {mean_utility} | {mean_last_run_s} | {mean_relevant_docs} |".format(
                **{k: tot.get(k) if tot.get(k) is not None else "—" for k in tot}
            )
        )
    lines += ["", "## Shared questions", ""]
    if not payload["matched_questions"]:
        lines.append("No overlapping questions across graphs.")
    for item in payload["matched_questions"]:
        lines += [f"### {item['question']}", ""]
        for graph_id, rows in item["graphs"].items():
            label = GRAPH_LABELS.get(graph_id, graph_id)
            for row in rows:
                lines += [
                    f"**{label}** (`{row['thread_id'][:8]}…`, utility={row['utility']}, "
                    f"relevant={row['n_relevant']}/{row['n_docs']}, "
                    f"{(row['last_run_ms'] or 0) / 1000:.1f}s)",
                    "",
                    row["answer"] or "_(no answer)_",
                    "",
                ]
                if row.get("sub_queries"):
                    lines.append("Sub-queries: " + "; ".join(row["sub_queries"]))
                    lines.append("")
                if row.get("sources"):
                    lines.append("Sources: " + ", ".join(row["sources"]))
                    lines.append("")
    lines += ["", "## All threads", ""]
    for graph_id, rows in payload["graphs"].items():
        lines += [f"### {GRAPH_LABELS.get(graph_id, graph_id)}", ""]
        for row in rows:
            q = (row.get("question") or "(no question)")[:120]
            lines.append(
                f"- `{row['thread_id']}` · {row['status']} · utility={row.get('utility')} · "
                f"{q}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


async def _amain() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:2024")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("exports/thread-comparison"),
    )
    args = parser.parse_args()
    payload = await export(args.url, GRAPH_IDS)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "threads.json"
    md_path = args.out / "comparison.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    md_path.write_text(render_markdown(payload))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    for graph_id, tot in payload["totals"].items():
        print(f"{tot['label']}: {tot['n_threads']} threads")


if __name__ == "__main__":
    asyncio.run(_amain())
