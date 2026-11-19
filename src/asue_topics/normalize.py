"""Vereinheitlichung der Wörter für die Topic-Darstellung.

Wirkt nur auf den Text, aus dem die Topic-Wörter gebildet werden (Spalte text_repr).
Die Embeddings sehen weiterhin den Originalsatz.

1. Schreibvarianten (datengetrieben, ohne feste Wörterliste)
   Für jedes Wort werden historische bzw. maschinenschriftliche Varianten gebildet:
       ss <-> ß        weiss / weiß, Strasse / Straße, dass / daß
       ue/ae/oe -> ü/ä/ö   moechte / möchte (Telegramme, Schreibmaschine ohne Umlaute)
       th -> t         Theil / Teil, Rath / Rat, Thür / Tür
       c -> k/z        Correctur / Korrektur, Concert / Konzert, circa / zirka
   Ein Wort wird nur dann ersetzt, wenn die Variante IM KORPUS SELBST häufiger vorkommt.
   So entstehen keine erfundenen Formen. Alle Ersetzungen werden in normalization_map.csv
   dokumentiert und sollten durchgesehen werden (Liste ausschließen: NormalizeConfig.exclude).

   Zum Schluss (final_ss) wird ß einheitlich als ss geschrieben, damit z. B. „groß"/„gross" nach der
   Lemmatisierung sicher zusammenfallen. Das betrifft nur die Darstellung der Topics.

2. Lemmatisierung (optional, spaCy)
   Vertrag / Vertrages / Verträge -> Vertrag.
   Benötigt:  pip install spacy  und  python -m spacy download de_core_news_md
   Ist spaCy nicht installiert, wird dieser Schritt mit einem Hinweis übersprungen.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import pandas as pd

WORD = re.compile(r"[^\W\d_]+")


@dataclass
class NormalizeConfig:
    orthography: bool = True
    lemmatize: bool = True
    spacy_model: str = "de_core_news_md"
    min_len: int = 3                       # kürzere Wörter nicht normalisieren
    exclude: set = field(default_factory=set)   # Wörter (klein), die nie ersetzt werden sollen
    final_ss: bool = True                  # zum Schluss ß -> ss: eine einheitliche Schreibung für die Topic-Wörter


# ---------------------------------------------------------------------------
# 1. Schreibvarianten
# ---------------------------------------------------------------------------

def _c_modern(w: str) -> str:
    """c -> z vor e/i/ä/y, sonst -> k; ch, ck, sc bleiben."""
    def rep(m):
        nxt = m.string[m.end():m.end() + 1]
        return "z" if nxt in "eiäy" else "k"
    return re.sub(r"(?<!s)c(?![hk])", rep, w)


RULES = {
    "ss>ß": lambda w: w.replace("ss", "ß"),
    "ß>ss": lambda w: w.replace("ß", "ss"),
    "umlaut": lambda w: w.replace("ue", "ü").replace("ae", "ä").replace("oe", "ö"),
    "th>t": lambda w: w.replace("th", "t"),
    "c>k/z": _c_modern,
}


def _variants(w: str) -> set[str]:
    out = set()
    names = list(RULES)
    for r in (1, 2):
        for combo in combinations(names, r):
            if "ss>ß" in combo and "ß>ss" in combo:
                continue
            v = w
            for n in combo:
                v = RULES[n](v)
            if v != w:
                out.add(v)
    return out


def build_orthography_map(texts, cfg: NormalizeConfig) -> pd.DataFrame:
    freq = Counter(t for text in texts for t in WORD.findall(text.lower()))
    rows = []
    for w, f in freq.items():
        if len(w) < cfg.min_len or w in cfg.exclude:
            continue
        best, best_f = None, f
        for v in _variants(w):
            fv = freq.get(v, 0)
            if fv > best_f:
                best, best_f = v, fv
        if best:
            rows.append({"wort": w, "häufigkeit": f, "ersetzt_durch": best, "häufigkeit_ziel": best_f})
    m = pd.DataFrame(rows, columns=["wort", "häufigkeit", "ersetzt_durch", "häufigkeit_ziel"])
    # Ketten auflösen (a -> b -> c  =>  a -> c)
    d = dict(zip(m["wort"], m["ersetzt_durch"]))
    for k in list(d):
        seen = {k}
        while d[k] in d and d[k] not in seen:
            seen.add(d[k])
            d[k] = d[d[k]]
    m["ersetzt_durch"] = m["wort"].map(d)
    return m.sort_values("häufigkeit", ascending=False).reset_index(drop=True)


def apply_map(text: str, mapping: dict[str, str]) -> str:
    def rep(m):
        w = m.group(0)
        t = mapping.get(w.lower())
        if not t:
            return w
        return t[:1].upper() + t[1:] if w[:1].isupper() else t
    return WORD.sub(rep, text)


# ---------------------------------------------------------------------------
# 2. Lemmatisierung
# ---------------------------------------------------------------------------

def lemmatize(texts: list[str], model: str) -> list[str] | None:
    try:
        import spacy
    except ImportError:
        print("Hinweis: spaCy nicht installiert – Lemmatisierung übersprungen "
              "(pip install spacy; python -m spacy download de_core_news_md).")
        return None
    try:
        nlp = spacy.load(model, disable=["parser", "ner"])
    except OSError:
        print(f"Hinweis: spaCy-Modell '{model}' fehlt – python -m spacy download {model}. "
              "Lemmatisierung übersprungen.")
        return None
    out = []
    for doc in nlp.pipe(texts, batch_size=256):
        out.append(" ".join(t.lemma_ for t in doc if not (t.is_punct or t.is_space)))
    return out


# ---------------------------------------------------------------------------
# Gesamt
# ---------------------------------------------------------------------------

def prepare(units: pd.DataFrame, cfg: NormalizeConfig, out_dir: str | Path,
            source_column: str = "text_noent") -> pd.DataFrame:
    """Fügt units die Spalte text_repr hinzu und schreibt normalization_map.csv."""
    out_dir = Path(out_dir)
    u = units.copy()
    texts = u[source_column].fillna("").astype(str).tolist()
    if cfg.orthography:
        m = build_orthography_map(texts, cfg)
        m.to_csv(out_dir / "normalization_map.csv", index=False)
        mapping = dict(zip(m["wort"], m["ersetzt_durch"]))
        texts = [apply_map(t, mapping) for t in texts]
        print(f"Schreibvarianten: {len(m)} Wortformen vereinheitlicht -> {out_dir / 'normalization_map.csv'}")
    if cfg.lemmatize:
        lem = lemmatize(texts, cfg.spacy_model)
        if lem is not None:
            texts = lem
            print("Lemmatisierung: fertig")
    if cfg.final_ss:
        texts = [t.replace("ß", "ss").replace("ẞ", "SS") for t in texts]
    u["text_repr"] = texts
    return u
