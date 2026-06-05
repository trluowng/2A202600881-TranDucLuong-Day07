from __future__ import annotations

import html
import re
from typing import Any

import streamlit as st

from phase2_benchmark import BENCHMARK, HashingKeywordEmbedder, chunk_docs, load_vinmec_docs
from src import (
    ChunkingStrategyComparator,
    EmbeddingStore,
    FixedSizeChunker,
    LOCAL_EMBEDDING_MODEL,
    LocalEmbedder,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)
from src.models import Document


st.set_page_config(
    page_title="Day 7 Lab - Retrieval Workbench",
    page_icon="D7",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --surface: #ffffff;
        --muted: #667085;
        --border: #e4e7ec;
        --ink: #182230;
        --accent: #0b6bcb;
        --soft: #f6f8fb;
    }
    .main .block-container {
        padding-top: 1.25rem;
        max-width: 1280px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .workbench-title {
        font-size: 1.65rem;
        font-weight: 750;
        color: var(--ink);
        margin: 0 0 0.15rem 0;
    }
    .workbench-subtitle {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }
    .result-card {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.7rem 0;
        background: var(--surface);
    }
    .result-head {
        display: flex;
        justify-content: space-between;
        align-items: start;
        gap: 1rem;
        margin-bottom: 0.45rem;
    }
    .source {
        font-weight: 700;
        color: var(--ink);
        overflow-wrap: anywhere;
    }
    .score {
        color: var(--accent);
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }
    .excerpt {
        color: #344054;
        line-height: 1.55;
        margin-top: 0.35rem;
    }
    .chip {
        display: inline-block;
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.12rem 0.45rem;
        margin: 0.15rem 0.2rem 0 0;
        color: #344054;
        background: var(--soft);
        font-size: 0.78rem;
    }
    .answer-box {
        border-left: 4px solid var(--accent);
        background: #f5faff;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        color: var(--ink);
        line-height: 1.55;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_text(text: str, limit: int | None = None) -> str:
    cleaned = " ".join(text.split())
    if limit is not None and len(cleaned) > limit:
        return cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def doc_title(doc: Document) -> str:
    for line in doc.content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return doc.id


def token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", text.lower())
        if len(token) > 1 and token not in HashingKeywordEmbedder.STOPWORDS
    }


@st.cache_data(show_spinner=False)
def cached_docs() -> list[Document]:
    return load_vinmec_docs()


@st.cache_resource(show_spinner=False)
def cached_embedder(kind: str):
    if kind == "Local MiniLM":
        try:
            return LocalEmbedder(LOCAL_EMBEDDING_MODEL)
        except Exception:
            return HashingKeywordEmbedder()
    if kind == "Mock":
        return _mock_embed
    return HashingKeywordEmbedder()


def make_chunker(strategy: str, chunk_size: int, overlap: int, sentences_per_chunk: int):
    if strategy == "Fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "Recursive":
        return RecursiveChunker(chunk_size=chunk_size)
    return SentenceChunker(max_sentences_per_chunk=sentences_per_chunk)


def build_store(docs: list[Document], chunker, embedder) -> tuple[EmbeddingStore, list[Document]]:
    chunks = chunk_docs(docs, chunker)
    store = EmbeddingStore(collection_name="ui_workbench", embedding_fn=embedder)
    store.add_documents(chunks)
    return store, chunks


def split_disease_types(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def record_matches(record: dict[str, Any], filters: dict[str, str]) -> bool:
    metadata = record["metadata"]
    for key, value in filters.items():
        if not value or value == "All":
            continue
        if key == "disease_type":
            if value not in split_disease_types(str(metadata.get(key, ""))):
                return False
        elif metadata.get(key) != value:
            return False
    return True


def search_records(
    store: EmbeddingStore,
    query: str,
    top_k: int,
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    filters = filters or {}
    active_filters = {key: value for key, value in filters.items() if value and value != "All"}
    if not active_filters:
        return store.search(query, top_k=top_k)

    records = [record for record in store._store if record_matches(record, active_filters)]
    return store._search_records(query, records, top_k=top_k)


def pick_sentences(query: str, content: str, limit: int = 2) -> list[str]:
    query_tokens = token_set(query)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", content) if part.strip()]
    if not sentences:
        return [normalize_text(content, 260)]
    ranked = sorted(
        sentences,
        key=lambda sentence: len(token_set(sentence) & query_tokens),
        reverse=True,
    )
    picked = [sentence for sentence in ranked[:limit] if sentence]
    return picked or sentences[:limit]


def build_answer(query: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return "Không tìm thấy chunk phù hợp với bộ lọc hiện tại."
    snippets = []
    for result in results[:2]:
        source = result["metadata"].get("doc_id", result["metadata"].get("source", "unknown"))
        sentence = " ".join(pick_sentences(query, result["content"], limit=2))
        snippets.append(f"{sentence} (Nguồn: {source})")
    return " ".join(snippets)


def render_result(result: dict[str, Any], rank: int) -> None:
    metadata = result["metadata"]
    source = metadata.get("doc_id", metadata.get("source", "unknown"))
    chips = [
        f"score={result['score']:.4f}",
        f"type={metadata.get('type', '-')}",
        f"audience={metadata.get('audience', '-')}",
        f"disease={metadata.get('disease_type', '-')}",
        f"chunk={metadata.get('chunk_index', '-')}",
    ]
    chip_html = "".join(f"<span class='chip'>{html.escape(chip)}</span>" for chip in chips)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-head">
                <div class="source">#{rank} {html.escape(source)}</div>
                <div class="score">{result['score']:.4f}</div>
            </div>
            <div>{chip_html}</div>
            <div class="excerpt">{html.escape(normalize_text(result['content'], 850))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def benchmark_score(results: list[dict[str, Any]], gold_sources: list[str]) -> tuple[int, bool]:
    if not results:
        return 0, False
    gold = set(gold_sources)
    retrieved = [result["metadata"].get("doc_id", "") for result in results]
    hit = bool(set(retrieved) & gold)
    score = 2 if hit and retrieved[0] in gold else (1 if hit else 0)
    return score, hit


def run_benchmark_rows(docs: list[Document], embedder, strategy_name: str, chunker) -> tuple[int, list[dict[str, Any]]]:
    store, chunks = build_store(docs, chunker, embedder)
    rows: list[dict[str, Any]] = []
    total = 0
    for item in BENCHMARK:
        filters = item.get("metadata_filter", {})
        results = search_records(store, item["query"], top_k=3, filters=filters)
        score, hit = benchmark_score(results, item["gold_sources"])
        total += score
        rows.append(
            {
                "id": item["id"],
                "query": item["query"],
                "filter": filters or {},
                "gold": ", ".join(item["gold_sources"]),
                "top_3": ", ".join(result["metadata"].get("doc_id", "?") for result in results),
                "score": f"{score}/2",
                "hit": "Yes" if hit else "No",
                "chunks": len(chunks),
            }
        )
    return total, rows


def inventory_rows(docs: list[Document]) -> list[dict[str, Any]]:
    rows = []
    for doc in docs:
        rows.append(
            {
                "title": doc_title(doc),
                "doc_id": doc.id,
                "chars": len(doc.content),
                "type": doc.metadata.get("type", ""),
                "audience": doc.metadata.get("audience", ""),
                "disease_type": doc.metadata.get("disease_type", ""),
                "source": doc.metadata.get("source", ""),
            }
        )
    return rows


docs = cached_docs()

st.markdown("<div class='workbench-title'>Day 7 Retrieval Workbench</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='workbench-subtitle'>Vector search, metadata filtering, benchmark comparison, and chunk inspection for the Vinmec fever dataset.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    strategy = st.radio("Strategy", ["Sentence", "Fixed", "Recursive"], horizontal=True)
    embedding_kind = st.selectbox("Embedding", ["Keyword", "Local MiniLM", "Mock"], index=0)
    top_k = st.slider("Top K", min_value=1, max_value=8, value=3)

    st.divider()
    chunk_size = st.slider("Chunk size", min_value=150, max_value=900, value=300, step=50)
    overlap = st.slider("Fixed overlap", min_value=0, max_value=200, value=50, step=10)
    sentences_per_chunk = st.slider("Sentences/chunk", min_value=1, max_value=6, value=3)

    st.divider()
    type_values = ["All"] + sorted({doc.metadata.get("type", "") for doc in docs if doc.metadata.get("type")})
    audience_values = ["All"] + sorted({doc.metadata.get("audience", "") for doc in docs if doc.metadata.get("audience")})
    disease_values = ["All"] + sorted(
        {
            disease
            for doc in docs
            for disease in split_disease_types(str(doc.metadata.get("disease_type", "")))
        }
    )
    selected_type = st.selectbox("Type filter", type_values)
    selected_audience = st.selectbox("Audience filter", audience_values)
    selected_disease = st.selectbox("Disease filter", disease_values)


embedder = cached_embedder(embedding_kind)
chunker = make_chunker(strategy, chunk_size, overlap, sentences_per_chunk)
store, chunks = build_store(docs, chunker, embedder)
backend_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)

metric_cols = st.columns(4)
metric_cols[0].metric("Documents", len(docs))
metric_cols[1].metric("Chunks", len(chunks))
metric_cols[2].metric("Strategy", strategy)
metric_cols[3].metric("Embedding", backend_name)

filters = {
    "type": selected_type,
    "audience": selected_audience,
    "disease_type": selected_disease,
}

search_tab, benchmark_tab, documents_tab, chunking_tab = st.tabs(
    ["Search", "Benchmark", "Documents", "Chunking"]
)

with search_tab:
    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        default_query = BENCHMARK[0]["query"]
        query = st.text_area("Query", value=default_query, height=90)
        run_search = st.button("Search", type="primary", use_container_width=True)
        if query.strip() and (run_search or "last_query" not in st.session_state):
            st.session_state["last_query"] = query.strip()
            st.session_state["last_results"] = search_records(store, query.strip(), top_k=top_k, filters=filters)

        results = st.session_state.get("last_results", [])
        if results:
            st.markdown("#### Agent Answer")
            st.markdown(
                f"<div class='answer-box'>{html.escape(build_answer(st.session_state.get('last_query', query), results))}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("#### Retrieved Chunks")
            for index, result in enumerate(results, start=1):
                render_result(result, index)
        else:
            st.info("No results for the current query and filters.")

    with right:
        st.markdown("#### Benchmark Queries")
        for item in BENCHMARK:
            filters_text = item.get("metadata_filter") or {}
            with st.expander(item["id"] + " · " + item["query"], expanded=False):
                st.write(item["gold_answer"])
                st.caption(f"Gold: {', '.join(item['gold_sources'])}")
                if filters_text:
                    st.caption(f"Filter: {filters_text}")

with benchmark_tab:
    st.markdown("#### Strategy Scores")
    run_all = st.button("Run Benchmark", type="primary")
    if run_all or "benchmark_rows" not in st.session_state:
        strategies = {
            "fixed": FixedSizeChunker(chunk_size=chunk_size, overlap=overlap),
            "sentence": SentenceChunker(max_sentences_per_chunk=sentences_per_chunk),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        summary = []
        detail: dict[str, list[dict[str, Any]]] = {}
        for name, bench_chunker in strategies.items():
            total, rows = run_benchmark_rows(docs, embedder, name, bench_chunker)
            summary.append({"strategy": name, "score": total, "chunks": rows[0]["chunks"] if rows else 0})
            detail[name] = rows
        st.session_state["benchmark_summary"] = summary
        st.session_state["benchmark_rows"] = detail

    summary_rows = st.session_state.get("benchmark_summary", [])
    if summary_rows:
        st.dataframe(summary_rows, width="stretch", hide_index=True)
        st.bar_chart(summary_rows, x="strategy", y="score")
        selected_strategy = st.selectbox("Benchmark detail", [row["strategy"] for row in summary_rows])
        st.dataframe(
            st.session_state["benchmark_rows"][selected_strategy],
            width="stretch",
            hide_index=True,
        )

with documents_tab:
    st.markdown("#### Document Inventory")
    st.dataframe(inventory_rows(docs), width="stretch", hide_index=True)
    doc_options = {doc_title(doc): doc for doc in docs}
    selected_title = st.selectbox("Document", list(doc_options))
    selected_doc = doc_options[selected_title]
    preview_cols = st.columns([0.35, 0.65], gap="large")
    with preview_cols[0]:
        st.json(selected_doc.metadata)
    with preview_cols[1]:
        st.markdown(selected_doc.content[:3000])

with chunking_tab:
    st.markdown("#### Chunking Comparison")
    selected_title_for_chunking = st.selectbox("Compare document", list(doc_options), key="compare_doc")
    compare_doc = doc_options[selected_title_for_chunking]
    comparison = ChunkingStrategyComparator().compare(compare_doc.content, chunk_size=chunk_size)
    comparison_rows = [
        {
            "strategy": name,
            "count": stats["count"],
            "avg_length": round(stats["avg_length"], 1),
            "first_chunk": normalize_text(stats["chunks"][0], 220) if stats["chunks"] else "",
        }
        for name, stats in comparison.items()
    ]
    st.dataframe(comparison_rows, width="stretch", hide_index=True)
    selected_preview_strategy = st.selectbox("Preview chunks", list(comparison))
    preview_chunks = comparison[selected_preview_strategy]["chunks"][:8]
    for index, chunk in enumerate(preview_chunks, start=1):
        with st.expander(f"Chunk {index} · {len(chunk)} chars", expanded=index <= 2):
            st.write(chunk)

st.caption("Educational retrieval demo. Medical content should be verified with the cited source and a qualified clinician.")
