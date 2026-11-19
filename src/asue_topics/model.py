"""Topic Modeling auf units.csv.

Zwei Verfahren:
    fit_bertopic  Hauptverfahren (wie v1), aber korrekt konfiguriert:
                  - Embeddings aus VOLLSTÄNDIGEN Sätzen (keine Stoppwortentfernung, keine Kleinschreibung)
                  - Stoppwörter nur im CountVectorizer, d. h. nur für die Topic-Darstellung
                  - feste Seeds (UMAP), damit Ergebnisse reproduzierbar sind
                  - sehr kurze Einheiten (z. B. „1 u.", „ii.") werden vorab gefiltert
                  - Topic-Wörter aus dem Text OHNE ausgezeichnete Namen (TEI persName/placeName/orgName),
                    damit Topics nicht aus Personen- und Ortsnamen bestehen; die Embeddings sehen den vollen Satz
    fit_nmf       leichte Vergleichsbasis mit scikit-learn (TF-IDF + NMF), ohne Sprachmodell.
                  Nützlich, um zu zeigen, wie stark das Ergebnis vom Verfahren abhängt.

Ausgabe (output/<prefix>_*.csv):
    unit_topics.csv   letter_id, unit_no, text, topic
    topic_info.csv    topic, count, top_words (+ repräsentative Sätze bei BERTopic)
    topic_labels.csv  Vorlage zum manuellen Benennen der Topics (wird NIE überschrieben)
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import NormalizeConfig, prepare as normalize_prepare

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    min_words: int = 4                       # kürzere Einheiten werden nicht modelliert
    repr_column: str = "text_repr"           # Text für die Topic-Darstellung: ohne Namen, vereinheitlicht, lemmatisiert
    normalize: NormalizeConfig | None = field(default_factory=NormalizeConfig)   # None = keine Normalisierung
    seed: int = 42
    # BERTopic
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    min_cluster_size: int = 15               # HDBSCAN: kleinste Topic-Größe
    min_samples: int | None = None           # HDBSCAN: kleiner = weniger Outlier
    nr_topics: int | str | None = None       # z. B. 30 oder "auto"; None = keine Reduktion
    reduce_outliers: bool = False            # Outlier (-1) nachträglich nächstem Topic zuordnen
    merge_topics: list = field(default_factory=list)   # z. B. [[4, 11]] – nach Ansicht eines Laufs gleiche Topics zusammenlegen
    # Topic-Darstellung
    ngram_range: tuple = (1, 2)
    min_df: int = 3
    stopword_files: list = field(default_factory=lambda: [
        ROOT / "resources" / "stopwords-de.txt",
        ROOT / "resources" / "stopwords-domain.txt",
    ])
    extra_stopwords: list = field(default_factory=list)
    # NMF
    nmf_topics: int = 20


def load_stopwords(cfg: ModelConfig) -> list[str]:
    words = set(w.lower() for w in cfg.extra_stopwords)
    for f in cfg.stopword_files:
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                words.add(line)
                words.add(line.replace("ß", "ss"))   # passend zur Normalisierung ß -> ss
    return sorted(words)


def load_units(out_dir: str | Path, cfg: ModelConfig) -> pd.DataFrame:
    df = pd.read_csv(Path(out_dir) / "units.csv", dtype={"text": str, "text_noent": str})
    df["text"] = df["text"].fillna("").astype(str)
    if "text_noent" not in df:
        df["text_noent"] = df["text"]
    df["text_noent"] = df["text_noent"].fillna("").astype(str)
    n0 = len(df)
    df = df[df["n_words"] >= cfg.min_words]
    # Einheiten ohne Buchstaben (Daten, Nummerierungen) entfernen – v1-Problem „float"
    df = df[df["text"].str.contains(r"[^\W\d_]{2,}", regex=True)]
    df = df.reset_index(drop=True)
    print(f"{len(df)} von {n0} Einheiten mit >= {cfg.min_words} Wörtern")
    if cfg.normalize is not None:
        df = normalize_prepare(df, cfg.normalize, out_dir, source_column="text_noent")
    else:
        df["text_repr"] = df["text_noent"]
    return df


TOKEN_PATTERN = r"(?u)\b[^\W\d_]{3,}\b"   # nur Wörter aus Buchstaben, mind. 3 Zeichen


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------

def compute_embeddings(docs: list[str], cfg: ModelConfig, cache_dir: str | Path):
    """Embeddings berechnen bzw. aus dem Cache laden (Berechnung dauert am längsten)."""
    from sentence_transformers import SentenceTransformer

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(("\n".join(docs) + cfg.embedding_model).encode("utf-8")).hexdigest()[:12]
    cache = cache_dir / f"embeddings_{h}.npy"
    model = SentenceTransformer(cfg.embedding_model)
    if cache.exists():
        print(f"Embeddings aus Cache: {cache.name}")
        return model, np.load(cache)
    emb = model.encode(docs, show_progress_bar=True, batch_size=32)
    np.save(cache, emb)
    return model, emb


def fit_bertopic(units: pd.DataFrame, cfg: ModelConfig, cache_dir: str | Path):
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    # Embeddings aus dem vollständigen Originalsatz, Topic-Wörter aus dem Text ohne Namen
    docs_embed = units["text"].tolist()
    docs = units[cfg.repr_column].tolist()
    emb_model, embeddings = compute_embeddings(docs_embed, cfg, cache_dir)

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                      random_state=cfg.seed)
    hdbscan_model = HDBSCAN(min_cluster_size=cfg.min_cluster_size, min_samples=cfg.min_samples,
                            metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    # Bei Substantiv-Listen (POS-Filter) keine Wortpaare bilden – sie ergäben nur „vertrag vertrag"
    pos_only = cfg.normalize is not None and cfg.normalize.lemmatize and cfg.normalize.pos_keep
    ngram = (1, 1) if pos_only else cfg.ngram_range
    vectorizer = CountVectorizer(stop_words=load_stopwords(cfg), ngram_range=ngram,
                                 min_df=cfg.min_df, token_pattern=TOKEN_PATTERN, lowercase=True)

    topic_model = BERTopic(
        embedding_model=emb_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=KeyBERTInspired(),
        nr_topics=cfg.nr_topics,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings)

    if cfg.merge_topics:
        topic_model.merge_topics(docs, cfg.merge_topics)
        topics = topic_model.topics_

    if cfg.reduce_outliers:
        topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
        topic_model.update_topics(docs, topics=topics, vectorizer_model=vectorizer,
                                  representation_model=KeyBERTInspired())

    result = units.copy()
    result["topic"] = topics
    info = topic_model.get_topic_info()
    rep = representative_units(units, np.asarray(topics), embeddings)
    info = pd.DataFrame({
        "topic": info["Topic"],
        "count": info["Count"],
        "top_words": info["Representation"].apply(lambda ws: ", ".join(ws[:10])),
        "representative_units": info["Topic"].map(rep).fillna(""),
    })
    return topic_model, result, info, embeddings


def representative_units(units: pd.DataFrame, topics: np.ndarray, embeddings: np.ndarray, n: int = 4) -> dict:
    """Je Topic die n Sätze (Originaltext), die dem Mittelpunkt des Topics am nächsten liegen."""
    emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    texts = units["text"].to_numpy()
    out = {}
    for t in np.unique(topics):
        idx = np.where(topics == t)[0]
        if t == -1:
            idx = idx[:2000]                       # Outlier: nur Stichprobe, Mittelpunkt wenig aussagekräftig
        c = emb[idx].mean(axis=0)
        sims = emb[idx] @ (c / np.linalg.norm(c))
        chosen, seen = [], set()
        for i in idx[np.argsort(-sims)]:
            if texts[i] not in seen:
                seen.add(texts[i])
                chosen.append(texts[i])
            if len(chosen) == n:
                break
        out[int(t)] = " || ".join(chosen)
    return out


def save_bertopic(topic_model, path: str | Path, cfg: ModelConfig):
    """Speichern als safetensors (statt pickle wie in v1: versionsunabhängiger)."""
    topic_model.save(str(path), serialization="safetensors", save_ctfidf=True,
                     save_embedding_model=cfg.embedding_model)


# ---------------------------------------------------------------------------
# NMF (Vergleichsbasis, nur scikit-learn)
# ---------------------------------------------------------------------------

def fit_nmf(units: pd.DataFrame, cfg: ModelConfig, n_topics: int | None = None):
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_topics = n_topics or cfg.nmf_topics
    vec = TfidfVectorizer(stop_words=load_stopwords(cfg), ngram_range=cfg.ngram_range,
                          min_df=min(cfg.min_df, max(1, len(units) // 50)),
                          token_pattern=TOKEN_PATTERN, lowercase=True, sublinear_tf=True)
    X = vec.fit_transform(units[cfg.repr_column])
    n_topics = min(n_topics, X.shape[1] - 1, X.shape[0] - 1)
    nmf = NMF(n_components=n_topics, random_state=cfg.seed, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(X)
    topics = np.where(W.max(axis=1) > 0, W.argmax(axis=1), -1)
    vocab = np.array(vec.get_feature_names_out())
    rows = []
    for k, comp in enumerate(nmf.components_):
        rows.append({"topic": k, "count": int((topics == k).sum()),
                     "top_words": ", ".join(vocab[comp.argsort()[::-1][:10]])})
    rows.append({"topic": -1, "count": int((topics == -1).sum()), "top_words": "(ohne Zuordnung)"})
    info = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    result = units.copy()
    result["topic"] = topics
    return (nmf, vec), result, info


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(result: pd.DataFrame, info: pd.DataFrame, out_dir: str | Path, prefix: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["letter_id", "unit_no", "text", "text_repr", "topic"] if c in result]
    result[cols].to_csv(out_dir / f"{prefix}_unit_topics.csv", index=False)
    info.to_csv(out_dir / f"{prefix}_topic_info.csv", index=False)
    labels = out_dir / f"{prefix}_topic_labels.csv"
    if not labels.exists():   # manuelle Benennungen nie überschreiben
        tmpl = info[["topic", "count", "top_words"]].copy()
        tmpl["label"] = np.where(tmpl["topic"] == -1, "Outlier", "")
        tmpl.to_csv(labels, index=False)
        print(f"Vorlage zum Benennen: {labels}")
    n_out = int((result["topic"] == -1).sum())
    print(f"{prefix}: {info[info.topic != -1].shape[0]} Topics, Outlier: {n_out} "
          f"({n_out / max(len(result), 1):.0%})")


def load_labels(out_dir: str | Path, prefix: str) -> dict[int, str]:
    """Topic-Nummer -> Label. Fehlt ein Label, wird 'T<n>: ersten 3 Wörter' verwendet."""
    df = pd.read_csv(Path(out_dir) / f"{prefix}_topic_labels.csv")
    labels = {}
    for _, r in df.iterrows():
        lab = r.get("label")
        if isinstance(lab, str) and lab.strip():
            labels[int(r["topic"])] = lab.strip()
        else:
            words = re.split(r",\s*", str(r["top_words"]))[:3]
            labels[int(r["topic"])] = f"T{int(r['topic'])}: " + ", ".join(words)
    return labels
