"""Configurable text preprocessing pipeline.

Steps (each can be switched on/off from the UI):
    clean -> tokenize -> stop-word removal -> stemming / lemmatization -> n-grams
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .resources import ENGLISH_STOPWORDS, NEWS_STOPWORDS


@dataclass(frozen=True)
class PreprocessConfig:
    include_title: bool = True
    fix_encoding: bool = True
    strip_markup: bool = True
    lowercase: bool = True
    remove_numbers: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    news_stopwords: bool = False
    custom_stopwords: tuple[str, ...] = ()
    normalizer: str = "lemmatize"  # "none" | "stem" | "lemmatize"
    min_token_len: int = 2
    ngram_max: int = 1  # 1 = unigrams, 2 = uni+bigrams, 3 = up to trigrams

    def as_dict(self) -> dict:
        return asdict(self)

    def short_label(self) -> str:
        parts = ["lower" if self.lowercase else "cased"]
        if self.remove_stopwords:
            parts.append("stop" + ("+news" if self.news_stopwords else ""))
        if self.normalizer != "none":
            parts.append(self.normalizer)
        if self.ngram_max > 1:
            parts.append(f"1-{self.ngram_max}gram")
        return " | ".join(parts)


# ---------------------------------------------------------------- cleaning
_PARA_SEP = re.compile(r"\.{2,}")
_TAG = re.compile(r"<[^>]+>")
_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_SPACES = re.compile(r"\s+")
# letters-only words | numbers like 1,200.50 | any single symbol
_TOKEN = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)*|[^\w\s]")


# UTF-8 text wrongly decoded as Windows-1252 ("mojibake"), plus curly quotes.
_ENCODING_FIXES = {
    "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€“": "-", "â€”": "-", "Â£": "£", "Â ": " ",
    "’": "'", "‘": "'", "“": '"', "”": '"', "�": " ",
}


def clean_text(text: str, cfg: PreprocessConfig) -> str:
    if cfg.fix_encoding:
        for bad, good in _ENCODING_FIXES.items():
            if bad in text:
                text = text.replace(bad, good)
    if cfg.strip_markup:
        text = _URL.sub(" ", text)
        text = _EMAIL.sub(" ", text)
        text = _TAG.sub(" ", text)
        text = _ENTITY.sub(" ", text)
    text = _PARA_SEP.sub(". ", text)
    text = _SPACES.sub(" ", text).strip()
    if cfg.lowercase:
        text = text.lower()
    return text


def tokenize(text: str, cfg: PreprocessConfig) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text):
        first = tok[0]
        if first.isdigit():
            if cfg.remove_numbers:
                continue
            tok = tok.replace(",", "")
        elif not first.isalpha():
            if cfg.remove_punctuation:
                continue
        elif len(tok) < cfg.min_token_len:
            continue
        tokens.append(tok)
    return tokens


def stopword_set(cfg: PreprocessConfig) -> frozenset[str]:
    words = set(ENGLISH_STOPWORDS)
    if cfg.news_stopwords:
        words |= NEWS_STOPWORDS
    words |= {w.strip().lower() for w in cfg.custom_stopwords if w.strip()}
    return frozenset(words)


# ---------------------------------------------------------------- stemming / lemmatization
_STEM_CACHE: dict[str, str] = {}
_LEMMA_CACHE: dict[str, str] = {}
_stemmer = None
_lemmatizer = None
LEMMATIZER_BACKEND = "not loaded"


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        from nltk.stem import PorterStemmer
        _stemmer = PorterStemmer()
    return _stemmer


def _rule_lemma(word: str) -> str:
    """Small fallback used only if WordNet cannot be loaded."""
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("xes", "x"), ("ches", "ch"), ("shes", "sh")):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)] + repl
    if word.endswith("s") and not word.endswith(("ss", "us", "is")) and len(word) > 3:
        return word[:-1]
    return word


def _get_lemmatizer():
    global _lemmatizer, LEMMATIZER_BACKEND
    if _lemmatizer is not None:
        return _lemmatizer
    try:
        import nltk
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer
        try:
            wordnet.ensure_loaded()
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            wordnet.ensure_loaded()
        wnl = WordNetLemmatizer()

        def lemma(word: str) -> str:
            # No POS tagger: try noun first, then verb.
            noun = wnl.lemmatize(word, "n")
            return noun if noun != word else wnl.lemmatize(word, "v")

        lemma("tests")
        _lemmatizer = lemma
        LEMMATIZER_BACKEND = "NLTK WordNet"
    except Exception:  # offline / missing data
        _lemmatizer = _rule_lemma
        LEMMATIZER_BACKEND = "rule-based fallback"
    return _lemmatizer


def normalize_token(token: str, method: str) -> str:
    if method == "stem":
        out = _STEM_CACHE.get(token)
        if out is None:
            out = _STEM_CACHE[token] = _get_stemmer().stem(token)
        return out
    if method == "lemmatize":
        out = _LEMMA_CACHE.get(token)
        if out is None:
            out = _LEMMA_CACHE[token] = _get_lemmatizer()(token)
        return out
    return token


def add_ngrams(tokens: list[str], n_max: int) -> list[str]:
    """Return unigrams plus contiguous n-grams up to n_max, joined with '_'."""
    out = list(tokens)
    for n in range(2, n_max + 1):
        out.extend("_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return out


# ---------------------------------------------------------------- full pipeline
def preprocess(text: str, cfg: PreprocessConfig, stops: frozenset[str] | None = None) -> list[str]:
    stops = stopword_set(cfg) if stops is None else stops
    tokens = tokenize(clean_text(text, cfg), cfg)
    if cfg.remove_stopwords:
        tokens = [t for t in tokens if t.lower() not in stops]
    if cfg.normalizer != "none":
        tokens = [normalize_token(t, cfg.normalizer) for t in tokens]
    if cfg.ngram_max > 1:
        tokens = add_ngrams(tokens, cfg.ngram_max)
    return tokens


def preprocess_with_trace(text: str, cfg: PreprocessConfig) -> list[tuple[str, object]]:
    """Same as preprocess() but returns the output of every step (for the UI)."""
    steps: list[tuple[str, object]] = [("Raw text", text)]
    cleaned = clean_text(text, cfg)
    steps.append(("Cleaned" + (" + lowercased" if cfg.lowercase else ""), cleaned))
    tokens = tokenize(cleaned, cfg)
    steps.append(("Tokens", tokens))
    if cfg.remove_stopwords:
        stops = stopword_set(cfg)
        tokens = [t for t in tokens if t.lower() not in stops]
        steps.append(("After stop-word removal", tokens))
    if cfg.normalizer != "none":
        tokens = [normalize_token(t, cfg.normalizer) for t in tokens]
        steps.append(("After " + ("stemming" if cfg.normalizer == "stem" else "lemmatization"), tokens))
    if cfg.ngram_max > 1:
        tokens = add_ngrams(tokens, cfg.ngram_max)
        steps.append((f"With n-grams (n <= {cfg.ngram_max})", tokens))
    return steps


def document_text(title: str, body: str, cfg: PreprocessConfig) -> str:
    return f"{title}. {body}" if cfg.include_title and title else body


def preprocess_corpus(titles: list[str], bodies: list[str], cfg: PreprocessConfig) -> list[list[str]]:
    stops = stopword_set(cfg)
    return [preprocess(document_text(t, b, cfg), cfg, stops) for t, b in zip(titles, bodies)]


def raw_tokens(text: str) -> list[str]:
    """'Before preprocessing' view: plain whitespace split, nothing else."""
    return text.split()
