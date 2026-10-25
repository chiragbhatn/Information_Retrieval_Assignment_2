"""Convert the raw Hugging Face CSV into the project's corpus files.

Source: https://huggingface.co/datasets/gopalkalpande/bbc-news-summary (CC0)
Original corpus: Greene & Cunningham (2006), BBC News, 2004-2005.

Outputs (document text and metadata are kept in separate files, joined on doc_id):
    data/bbc_documents.csv  -> doc_id, text
    data/bbc_metadata.csv   -> doc_id, title, category, source, summary

The raw noise ('..' paragraph separators, inconsistent casing, numbers and
currency symbols, duplicate and near-duplicate articles) is deliberately kept so the app's preprocessing and duplicate detection have
something real to work on.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bbc-news-summary.csv"
OUT_DOCS = ROOT / "data" / "bbc_documents.csv"
OUT_META = ROOT / "data" / "bbc_metadata.csv"

CODES = {"business": "BUS", "entertainment": "ENT", "politics": "POL", "sport": "SPO", "tech": "TEC"}


def main() -> None:
    raw = pd.read_csv(RAW, encoding="utf-8")
    counters: dict[str, int] = {}
    docs, meta = [], []
    for row in raw.itertuples(index=False):
        category = str(row.File_path).strip().lower()
        counters[category] = counters.get(category, 0) + 1
        doc_id = f"{CODES[category]}-{counters[category]:03d}"
        # Each article is "<headline>..<body>"; the headline becomes metadata.
        title, _, body = str(row.Articles).partition("..")
        docs.append({"doc_id": doc_id, "text": body.strip()})
        meta.append({
            "doc_id": doc_id,
            "title": title.strip(),
            "category": category,
            "source": "BBC News",
            "summary": str(row.Summaries).strip(),
        })
    pd.DataFrame(docs).to_csv(OUT_DOCS, index=False, encoding="utf-8")
    pd.DataFrame(meta).to_csv(OUT_META, index=False, encoding="utf-8")
    print(f"Wrote {len(docs)} documents -> {OUT_DOCS.name}, {OUT_META.name}")
    print(pd.Series(counters).to_string())


if __name__ == "__main__":
    main()
