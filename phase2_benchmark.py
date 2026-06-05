"""
Phase 2 — Benchmark Script
Chay 5 benchmark queries tren bo tai lieu Vinmec (chu de: Sot).

Usage:
    python phase2_benchmark.py [fixed|sentence|recursive]

Default: chay ca 3 strategy va so sanh.
"""

from __future__ import annotations

import io
import hashlib
import math
import re
import sys
from pathlib import Path

from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import LOCAL_EMBEDDING_MODEL, LocalEmbedder
from src.models import Document
from src.store import EmbeddingStore


class HashingKeywordEmbedder:
    """Fallback lexical embedder for Vietnamese benchmark runs without downloads."""

    STOPWORDS = {
        "và", "là", "của", "có", "cho", "khi", "thì", "để", "với", "một",
        "những", "các", "người", "bệnh", "nên", "cần", "như", "nào", "gì",
        "ở", "do", "từ", "trong", "ngoài", "không", "được", "phải",
    }

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self._backend_name = "hashing keyword fallback"

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = [
            token
            for token in re.findall(r"[0-9A-Za-zÀ-ỹĐđ]+", text.lower())
            if len(token) > 1 and token not in self.STOPWORDS
        ]
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.dim
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

def get_embedder():
    try:
        embedder = LocalEmbedder(LOCAL_EMBEDDING_MODEL)
        print(f"  Embedding: {embedder._backend_name}")
        return embedder
    except Exception:
        embedder = HashingKeywordEmbedder()
        print(f"  Embedding: {embedder._backend_name} (cai sentence-transformers de dung model that)")
        return embedder

# ── 5 Benchmark Queries + Gold Answers ──────────────────────────────────────
BENCHMARK = [
    {
        "id": "Q1",
        "query": "Làm thế nào để phân biệt sốt xuất huyết và sốt virus?",
        "gold_sources": [
            "huong-dan-phan-biet-sot-virus-voi-sot-xuat-huyet-vi",
            "phan-biet-sot-thuong-sot-virus-va-sot-xuat-huyet-vi",
        ],
        "gold_answer": "Sốt xuất huyết thường kèm phát ban, đau nhức xương khớp mạnh, có thể xuất huyết dưới da; sốt virus thường kèm triệu chứng hô hấp (ho, sổ mũi), không có xuất huyết.",
    },
    {
        "id": "Q2",
        "query": "Trẻ em sốt đến bao nhiêu độ thì cần uống thuốc hạ sốt?",
        "gold_sources": [
            "tre-sot-den-dau-moi-phai-uong-thuoc-ha-sot-vi",
        ],
        "gold_answer": "Phụ huynh chỉ nên dùng thuốc hạ sốt cho trẻ khi sốt từ 38,5°C trở lên; trẻ từ 38°C được xem là thật sự bị sốt và cần theo dõi/xử lý phù hợp.",
        "metadata_filter": {"audience": "tre_em"},
    },
    {
        "id": "Q3",
        "query": "Triệu chứng của sốt xuất huyết nặng là gì?",
        "gold_sources": [
            "sot-xuat-huyet-va-sot-xuat-huyet-nang",
        ],
        "gold_answer": "Sốt xuất huyết nặng gồm: xuất huyết nghiêm trọng, suy tạng (gan, thận), thoát huyết tương gây sốc, hạ tiểu cầu nặng.",
    },
    {
        "id": "Q4",
        "query": "Sốt rét và sốt xuất huyết khác nhau như thế nào?",
        "gold_sources": [
            "phan-biet-sot-ret-va-sot-xuat-huyet-vi",
        ],
        "gold_answer": "Sốt rét do ký sinh trùng Plasmodium, lây qua muỗi Anopheles, sốt theo chu kỳ kèm rét run; sốt xuất huyết do virus Dengue, lây qua muỗi Aedes, sốt liên tục kèm phát ban.",
    },
    {
        "id": "Q5",
        "query": "Sốt phát ban khác sốt xuất huyết như thế nào?",
        "gold_sources": [
            "sot-phat-ban-khac-sot-xuat-huyet-nhu-nao-vi",
        ],
        "gold_answer": "Khi căng vùng da có ban, ban sốt phát ban sẽ mất đi rồi hồi phục màu đỏ khi buông ra; nốt xuất huyết do sốt xuất huyết không mất sau khi căng da.",
        "metadata_filter": {"type": "phan_biet"},
    },
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Tach YAML frontmatter va noi dung chinh."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    fm_block = text[3:end].strip()
    body = text[end + 3:].strip()
    meta = {}
    for line in fm_block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body


def load_vinmec_docs(data_dir: Path = Path("data")) -> list[Document]:
    docs = []
    for path in sorted(data_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        if not meta:
            continue  # bo qua file khong co frontmatter (data mau goc)
        meta["source"] = path.stem
        docs.append(Document(id=path.stem, content=body, metadata=meta))
    return docs


def chunk_docs(docs: list[Document], chunker, chunk_size: int = 300) -> list[Document]:
    chunks = []
    for doc in docs:
        parts = chunker.chunk(doc.content)
        for i, part in enumerate(parts):
            chunks.append(Document(
                id=f"{doc.id}__chunk{i}",
                content=part,
                metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": str(i)},
            ))
    return chunks


def run_benchmark(strategy_name: str, chunker, chunk_size: int = 300):
    print(f"\n{'='*60}")
    print(f"  STRATEGY: {strategy_name.upper()}  (chunk_size={chunk_size})")
    print(f"{'='*60}")

    docs = load_vinmec_docs()
    chunks = chunk_docs(docs, chunker, chunk_size)
    print(f"  Tai lieu: {len(docs)} | Chunks: {len(chunks)}\n")

    embedder = get_embedder()
    store = EmbeddingStore(collection_name=f"bench_{strategy_name}", embedding_fn=embedder)
    store.add_documents(chunks)

    total_score = 0
    for bq in BENCHMARK:
        metadata_filter = bq.get("metadata_filter")
        if metadata_filter:
            results = store.search_with_filter(bq["query"], top_k=3, metadata_filter=metadata_filter)
        else:
            results = store.search(bq["query"], top_k=3)
        retrieved_sources = {r["metadata"].get("doc_id", "") for r in results}
        gold = set(bq["gold_sources"])
        hit = bool(retrieved_sources & gold)
        score = 2 if hit and results[0]["metadata"].get("doc_id") in gold else (1 if hit else 0)
        total_score += score

        print(f"  [{bq['id']}] {bq['query'][:55]}...")
        if metadata_filter:
            print(f"  Filter: {metadata_filter}")
        print(f"  Gold  : {', '.join(gold)}")
        print(f"  Top-3 : {', '.join(r['metadata'].get('doc_id','?')[:40] for r in results)}")
        print(f"  Score : {score}/2  {'HIT' if hit else 'MISS'}")
        print()

    print(f"  TOTAL: {total_score}/10")
    return total_score


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    strategy = sys.argv[1] if len(sys.argv) > 1 else "all"
    chunk_size = 300

    strategies = {
        "fixed":     FixedSizeChunker(chunk_size=chunk_size, overlap=50),
        "sentence":  SentenceChunker(max_sentences_per_chunk=3),
        "recursive": RecursiveChunker(chunk_size=chunk_size),
    }

    if strategy == "all":
        scores = {}
        for name, chunker in strategies.items():
            scores[name] = run_benchmark(name, chunker, chunk_size)
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)
        for name, score in scores.items():
            bar = "█" * score + "░" * (10 - score)
            print(f"  {name:<12} {bar}  {score}/10")
    else:
        if strategy not in strategies:
            print(f"Unknown strategy: {strategy}. Chon: fixed | sentence | recursive | all")
            sys.exit(1)
        run_benchmark(strategy, strategies[strategy], chunk_size)


if __name__ == "__main__":
    main()
