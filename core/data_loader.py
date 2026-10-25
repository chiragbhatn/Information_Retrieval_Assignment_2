"""Loading a document collection and keeping text and metadata separate.

A Corpus holds two aligned structures indexed by doc_id:
    texts : pd.Series     -> the main document text (content)
    meta  : pd.DataFrame  -> title, category label and other metadata
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLED_DOCS = ROOT / "data" / "bbc_documents.csv"
BUNDLED_META = ROOT / "data" / "bbc_metadata.csv"


@dataclass
class Corpus:
    texts: pd.Series
    meta: pd.DataFrame
    label_col: str = "category"
    title_col: str = "title"
    name: str = "corpus"

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def doc_ids(self) -> list[str]:
        return list(self.texts.index)

    @property
    def labels(self) -> list[str]:
        return self.meta[self.label_col].astype(str).tolist()

    @property
    def titles(self) -> list[str]:
        if self.title_col in self.meta.columns:
            return self.meta[self.title_col].fillna("").astype(str).tolist()
        return [""] * len(self)

    @property
    def fingerprint(self) -> str:
        """Cheap identity used to know when cached results are stale."""
        return f"{self.name}|{len(self)}|{hash(tuple(self.texts.index))}|{int(self.texts.str.len().sum())}"

    def subset(self, doc_ids: list[str], name: str | None = None) -> "Corpus":
        return Corpus(self.texts.loc[doc_ids], self.meta.loc[doc_ids], self.label_col,
                      self.title_col, name or self.name)

    def stratified_sample(self, n: int, seed: int = 42) -> "Corpus":
        """Keep roughly n documents with the same class proportions."""
        if n >= len(self):
            return self
        frac = n / len(self)
        parts = []
        for _, grp in self.meta.groupby(self.label_col, sort=True):
            k = max(1, round(len(grp) * frac))
            parts.append(grp.sample(n=min(k, len(grp)), random_state=seed))
        chosen = set(pd.concat(parts).index)
        ordered = [d for d in self.texts.index if d in chosen]
        return self.subset(ordered, name=f"{self.name} (sample {len(ordered)})")


def _finalise(df: pd.DataFrame, id_col: str, text_col: str, label_col: str,
              meta_cols: list[str], title_col: str | None, name: str) -> Corpus:
    df = df.copy()
    df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")]
    if id_col == "<auto>":
        df["doc_id"] = [f"DOC-{i:05d}" for i in range(1, len(df) + 1)]
        id_col = "doc_id"
    df[id_col] = df[id_col].astype(str)
    if df[id_col].duplicated().any():
        # Make IDs unique instead of silently dropping documents.
        df[id_col] = df[id_col] + "_" + df.groupby(id_col).cumcount().astype(str)
    df = df.set_index(id_col)
    df.index.name = "doc_id"
    texts = df[text_col].astype(str)
    keep = [c for c in dict.fromkeys([title_col, label_col, *meta_cols]) if c and c in df.columns and c != text_col]
    meta = df[keep].copy()
    meta[label_col] = meta[label_col].astype(str)
    return Corpus(texts, meta, label_col=label_col, title_col=title_col or "", name=name)


def load_bundled() -> Corpus:
    """BBC News corpus shipped with the app (documents and metadata in separate files)."""
    docs = pd.read_csv(BUNDLED_DOCS, encoding="utf-8")
    meta = pd.read_csv(BUNDLED_META, encoding="utf-8")
    df = docs.merge(meta, on="doc_id", how="inner", validate="one_to_one")
    return _finalise(df, "doc_id", "text", "category", ["source", "summary"], "title", "BBC News")


def read_table(uploaded, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV, JSON array or JSON-lines file into a DataFrame."""
    raw = uploaded.read() if hasattr(uploaded, "read") else Path(uploaded).read_bytes()
    lower = filename.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8", encoding_errors="replace")
    text = raw.decode("utf-8", errors="replace").strip()
    if lower.endswith(".jsonl") or (text and text[0] == "{" and "\n{" in text):
        return pd.DataFrame([json.loads(line) for line in text.splitlines() if line.strip()])
    data = json.loads(text)
    if isinstance(data, dict):  # {"documents": [...]} style
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    return pd.DataFrame(data)


def from_table(df: pd.DataFrame, id_col: str, text_col: str, label_col: str,
               title_col: str | None, meta_cols: list[str], name: str) -> Corpus:
    return _finalise(df, id_col, text_col, label_col, meta_cols, title_col, name)


def from_zip(uploaded, name: str = "Uploaded folder") -> Corpus:
    """A zip of .txt files; the parent folder name is the category label.

    e.g. business/001.txt -> doc_id 'business/001', category 'business',
    title = first non-empty line of the file.
    """
    raw = uploaded.read() if hasattr(uploaded, "read") else Path(uploaded).read_bytes()
    rows = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for info in zf.infolist():
            path = Path(info.filename)
            if info.is_dir() or path.suffix.lower() != ".txt" or path.name.startswith("."):
                continue
            content = zf.read(info).decode("utf-8", errors="replace")
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            if not lines:
                continue
            label = path.parent.name or "unlabelled"
            rows.append({
                "doc_id": f"{label}/{path.stem}",
                "title": lines[0],
                "text": "\n".join(lines[1:]) if len(lines) > 1 else lines[0],
                "category": label,
                "source": info.filename,
            })
    if not rows:
        raise ValueError("No .txt files found in the zip.")
    return _finalise(pd.DataFrame(rows), "doc_id", "text", "category", ["source"], "title", name)


def corpus_stats(corpus: Corpus) -> dict:
    lengths = corpus.texts.str.split().str.len()
    chars = corpus.texts.str.len()
    return {
        "documents": len(corpus),
        "categories": corpus.meta[corpus.label_col].nunique(),
        "raw_text_mb": round(chars.sum() / 1e6, 2),
        "avg_words": round(lengths.mean(), 1),
        "median_words": int(lengths.median()),
        "min_words": int(lengths.min()),
        "max_words": int(lengths.max()),
        "total_words": int(lengths.sum()),
    }
