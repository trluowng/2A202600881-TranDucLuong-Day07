"""
Convert Vinmec PDF articles to clean Markdown files for the RAG lab.

Usage:
    python pdf_to_md.py bai1.pdf
    python pdf_to_md.py bai1.pdf bai2.pdf bai3.pdf
    python pdf_to_md.py *.pdf

Output: data/<filename>.md
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --- Noise patterns to strip (Vinmec promotional boilerplate) ---
NOISE_PATTERNS = [
    r"Để đặt lịch khám.*",
    r"Quý khách vui lòng.*",
    r"Bệnh viện Đa khoa Quốc tế Vinmec có các gói.*",
    r"Gói khám sức khỏe tổng quát.*",
    r"Kết quả khám của người bệnh.*",
    r"TẠI ĐÂY.*",
    r"HOTLINE.*",
    r"MyVinmec.*",
    r"Tải và đặt lịch.*",
    r"Ngoài việc xây dựng chế độ ăn khoa học.*khám sức khỏe.*",
]

# Heading patterns: "1. Tiêu đề", "2.1 Tiêu đề"
HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+(.+)$")


def extract_text(pdf_path: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
    except ImportError:
        pass

    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        return "\n\n".join(pages)
    except ImportError:
        pass

    raise RuntimeError("Cần cài một trong hai: pip install pdfplumber  hoặc  pip install pymupdf")


def clean_text(raw: str) -> str:
    lines = raw.splitlines()
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            cleaned.append("")
            continue

        # Bỏ dòng noise
        is_noise = False
        for pattern in NOISE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE | re.DOTALL):
                is_noise = True
                break
        if is_noise:
            continue

        # Bỏ dòng toàn số trang hoặc ký tự rác
        if re.fullmatch(r"[\d\s\.\-–]+", line) and len(line) < 10:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def to_markdown(text: str, title: str = "") -> str:
    lines = text.splitlines()
    md_lines = []

    if title:
        md_lines.append(f"# {title}\n")

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect heading: "1. Tiêu đề" hoặc "2.1 Tiêu đề"
        m = HEADING_RE.match(line)
        if m:
            num = m.group(1)
            heading_text = m.group(2)
            depth = num.count(".") + 1        # "1" → ##, "2.1" → ###
            prefix = "#" * (depth + 1)
            md_lines.append(f"\n{prefix} {num}. {heading_text}\n")
            i += 1
            continue

        # Bullet: dòng bắt đầu bằng "•" hoặc "-"
        if line.startswith(("•", "-", "–")):
            bullet_text = line.lstrip("•-–").strip()
            md_lines.append(f"- {bullet_text}")
            i += 1
            continue

        # Dòng thường
        if line:
            md_lines.append(line)
        else:
            md_lines.append("")

        i += 1

    # Gộp nhiều dòng trống liên tiếp thành 1
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(md_lines))
    return result.strip()


def convert(pdf_path: Path, out_dir: Path) -> Path:
    print(f"Đang xử lý: {pdf_path.name} ...")

    raw = extract_text(pdf_path)
    cleaned = clean_text(raw)

    # Tìm tiêu đề: dòng đầu tiên không rỗng
    first_lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    title = first_lines[0] if first_lines else pdf_path.stem

    # Bỏ dòng tiêu đề khỏi body (đã đưa lên heading)
    body = cleaned.replace(title, "", 1).strip()

    md = to_markdown(body, title=title)

    out_path = out_dir / (pdf_path.stem + ".md")
    out_path.write_text(md, encoding="utf-8")
    print(f"  → Saved: {out_path}  ({len(md)} ký tự)")
    return out_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    pdf_files = [Path(p) for p in sys.argv[1:]]
    missing = [p for p in pdf_files if not p.exists()]
    if missing:
        for m in missing:
            print(f"Không tìm thấy file: {m}")
        sys.exit(1)

    for pdf_path in pdf_files:
        out = convert(pdf_path, out_dir)
        print(f"  Preview (200 ký tự đầu):")
        print("  " + out.read_text(encoding="utf-8")[:200].replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()
