"""
Scrape Vinmec articles to clean Markdown files.

Usage:
    python scrape_vinmec.py <url> [<url2> ...]

Example:
    python scrape_vinmec.py https://www.vinmec.com/vie/bai-viet/dinh-duong-trong-benh-dai-thao-duong-vi

Output: data/<slug>.md
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

NOISE_PATTERNS = [
    r"để đặt lịch khám",
    r"quý khách vui lòng",
    r"gói khám sức khỏe",
    r"hotline",
    r"myvinmec",
    r"tải và đặt lịch",
    r"bài viết liên quan",
    r"xem thêm",
    r"đặt lịch trực tiếp",
    r"chính sách giá",
    r"kết quả khám của người bệnh",
    r"dịch vụ khách hàng vượt trội",
    r"chuyên khoa khác ngay tại bệnh viện",
]


def is_noise(text: str) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t) for p in NOISE_PATTERNS)


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] or "article"


SKIP_TAGS = {"figure", "script", "style", "section", "nav", "form"}


def entry_to_md(entry: Tag) -> str:
    lines = []
    for el in entry.children:
        if not isinstance(el, Tag):
            continue
        if el.name in SKIP_TAGS:
            continue

        text = el.get_text(separator=" ", strip=True)
        if not text or is_noise(text):
            continue

        if el.name == "h2":
            lines.append(f"\n## {text}\n")
        elif el.name == "h3":
            lines.append(f"\n### {text}\n")
        elif el.name == "h4":
            lines.append(f"\n#### {text}\n")
        elif el.name == "p":
            lines.append(f"{text}\n")
        elif el.name in ("ul", "ol"):
            for li in el.find_all("li", recursive=False):
                li_text = li.get_text(separator=" ", strip=True)
                if li_text and not is_noise(li_text):
                    lines.append(f"- {li_text}")
            lines.append("")

    return "\n".join(lines)


def scrape(url: str) -> tuple[str, str]:
    print(f"Scraping: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else slug_from_url(url)

    # Content: Vinmec dung div.entry cho noi dung bai viet
    entry = soup.find("div", class_="entry")
    if not entry:
        raise RuntimeError("Khong tim thay div.entry")

    body = entry_to_md(entry)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    md = f"# {title}\n\n{body}"
    return title, md


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    for url in sys.argv[1:]:
        try:
            title, md = scrape(url)
            slug = slug_from_url(url)
            out_path = out_dir / f"{slug}.md"
            out_path.write_text(md, encoding="utf-8")
            print(f"  Saved: {out_path}  ({len(md)} ky tu)")
            print(f"  Preview:\n")
            print("  " + md[:400].replace("\n", "\n  "))
            print()
        except Exception as e:
            print(f"  LOI: {e}")


if __name__ == "__main__":
    main()
