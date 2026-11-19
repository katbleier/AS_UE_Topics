"""Von Satz-Topics zu Beziehungsprofilen.

Schritte:
    1. letter_topics   Anteil jedes Topics an den Sätzen eines Briefs
    2. add_partner     Richtung (von/an Schönberg) und Korrespondenzpartner:in aus correspDesc
    3. profile         mittlere Topic-Anteile je Partner:in, Zeitraum, Richtung …
    4. Abbildungen     Heatmap Partner × Topic, Topics im Zeitverlauf, Partner–Topic-Netzwerk
    5. Export          GEXF-Dateien für Gephi (Korrespondenznetz mit Topic-Attributen, Partner–Topic-Netz)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PERIODS = [(1902, 1911, "1902–1911 Dreililien/frühe UE"),
                   (1912, 1918, "1912–1918"),
                   (1919, 1925, "1919–1925"),
                   (1926, 1933, "1926–1933 Berlin"),
                   (1934, 1951, "1934–1951 USA")]


# ---------------------------------------------------------------------------
# 1. Brief-Topic-Matrix
# ---------------------------------------------------------------------------

def letter_topics(unit_topics: pd.DataFrame, include_outliers: bool = False) -> pd.DataFrame:
    df = unit_topics if include_outliers else unit_topics[unit_topics["topic"] != -1]
    m = pd.crosstab(df["letter_id"], df["topic"], normalize="index")
    m.columns = [int(c) for c in m.columns]
    return m


# ---------------------------------------------------------------------------
# 2. Korrespondenzpartner
# ---------------------------------------------------------------------------

def _split(s) -> list[str]:
    return [x.strip() for x in str(s).split("|")] if isinstance(s, str) and s.strip() else []


def add_partner(letters: pd.DataFrame, schoenberg_pattern: str) -> pd.DataFrame:
    pat = re.compile(schoenberg_pattern, re.I)
    rows = []
    for _, r in letters.iterrows():
        s = list(zip(_split(r["sender_types"]), _split(r["sender_names"]), _split(r["sender_keys"])))
        rc = list(zip(_split(r["receiver_types"]), _split(r["receiver_names"]), _split(r["receiver_keys"])))
        s_as = any(pat.search(n) for _, n, _ in s)
        r_as = any(pat.search(n) for _, n, _ in rc)
        if s_as and not r_as:
            direction, others = "von Schönberg", rc
        elif r_as and not s_as:
            direction, others = "an Schönberg", s
        else:
            direction, others = "Dritte", s + rc
        others = [o for o in others if not pat.search(o[1])]
        pers = [o for o in others if o[0] == "persName"]
        orgs = [o for o in others if o[0] == "orgName"]
        main = pers[0] if pers else (orgs[0] if orgs else ("", "unbekannt", ""))
        rows.append({
            "direction": direction,
            "partner": main[1],
            "partner_key": main[2] or main[1],
            "partner_org": orgs[0][1] if orgs else "",
        })
    return pd.concat([letters.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_period(letters: pd.DataFrame, periods=DEFAULT_PERIODS) -> pd.DataFrame:
    def lab(y):
        try:
            y = int(y)
        except (TypeError, ValueError):
            return "undatiert"
        for a, b, name in periods:
            if a <= y <= b:
                return name
        return "andere"
    out = letters.copy()
    out["period"] = out["year"].apply(lab)
    return out


# ---------------------------------------------------------------------------
# 3. Profile
# ---------------------------------------------------------------------------

def profile(lt: pd.DataFrame, letters: pd.DataFrame, by: str, min_letters: int = 1) -> pd.DataFrame:
    """Mittlere Topic-Anteile pro Gruppe; Spalte 'n_letters' = Zahl der Briefe."""
    j = lt.join(letters.set_index("letter_id")[[by]], how="inner")
    prof = j.groupby(by).mean(numeric_only=True)
    prof.insert(0, "n_letters", j.groupby(by).size())
    return prof[prof["n_letters"] >= min_letters].sort_values("n_letters", ascending=False)


def topic_diversity(prof: pd.DataFrame) -> pd.Series:
    """Normierte Shannon-Entropie der Topic-Verteilung (0 = ein Thema, 1 = alle gleich)."""
    p = prof.drop(columns="n_letters").clip(lower=1e-12)
    p = p.div(p.sum(axis=1), axis=0)
    h = -(p * np.log(p)).sum(axis=1)
    return (h / np.log(p.shape[1])).rename("diversity")


# ---------------------------------------------------------------------------
# 4. Abbildungen
# ---------------------------------------------------------------------------

def _short(label: str, n: int = 38) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def plot_heatmap(prof: pd.DataFrame, labels: dict, path: str | Path, title: str, max_rows: int = 15):
    import matplotlib.pyplot as plt

    p = prof.head(max_rows)
    data = p.drop(columns="n_letters")
    cols = [_short(labels.get(c, f"T{c}")) for c in data.columns]
    rows = [f"{_short(str(i), 28)} (n={n})" for i, n in zip(p.index, p["n_letters"])]
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(cols) + 4), max(3, 0.4 * len(rows) + 2)))
    im = ax.imshow(data.values, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    ax.set_title(title, fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, label="mittlerer Anteil der Sätze", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_topics_over_time(lt: pd.DataFrame, letters: pd.DataFrame, labels: dict, path: str | Path,
                          top_n: int = 6, window: int = 3):
    import matplotlib.pyplot as plt

    j = lt.join(letters.set_index("letter_id")[["year"]], how="inner")
    j = j[pd.to_numeric(j["year"], errors="coerce").notna()]
    j["year"] = j["year"].astype(int)
    by_year = j.groupby("year").mean()
    full = by_year.reindex(range(by_year.index.min(), by_year.index.max() + 1))
    smooth = full.rolling(window, center=True, min_periods=1).mean()
    top = lt.mean().sort_values(ascending=False).head(top_n).index
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for t in top:
        ax.plot(smooth.index, smooth[t], label=_short(labels.get(t, f"T{t}"), 45), linewidth=1.8)
    counts = j.groupby("year").size().reindex(full.index).fillna(0)
    ax2 = ax.twinx()
    ax2.bar(counts.index, counts.values, color="grey", alpha=0.15, width=0.8)
    ax2.set_ylabel("Briefe pro Jahr", color="grey")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel(f"mittlerer Anteil ({window}-Jahres-Mittel)")
    ax.set_title("Häufigste Topics im Zeitverlauf", loc="left", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.08, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def partner_topic_graph(prof: pd.DataFrame, labels: dict, min_share: float = 0.12):
    """Bipartites Netz: Partner:innen – Topics; Kante, wenn mittlerer Anteil >= min_share."""
    import networkx as nx

    G = nx.Graph()
    for partner, row in prof.iterrows():
        G.add_node(f"P:{partner}", label=str(partner), kind="partner", n_letters=int(row["n_letters"]))
        for t, share in row.drop("n_letters").items():
            if share >= min_share:
                tid = f"T:{t}"
                if tid not in G:
                    G.add_node(tid, label=labels.get(t, f"T{t}"), kind="topic")
                G.add_edge(f"P:{partner}", tid, weight=float(share))
    return G


def plot_partner_topic_graph(G, path: str | Path, title: str):
    import matplotlib.pyplot as plt
    import networkx as nx

    if G.number_of_edges() == 0:
        print("Netzwerk leer – min_share senken.")
        return
    pos = nx.spring_layout(G, seed=42, k=1.2 / np.sqrt(G.number_of_nodes()), weight="weight")
    fig, ax = plt.subplots(figsize=(11, 8))
    widths = [1 + 8 * d["weight"] for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.35, edge_color="#5b7a99")
    part = [n for n, d in G.nodes(data=True) if d["kind"] == "partner"]
    top = [n for n, d in G.nodes(data=True) if d["kind"] == "topic"]
    nx.draw_networkx_nodes(G, pos, nodelist=part, ax=ax, node_color="#d8893b",
                           node_size=[60 + 6 * G.nodes[n]["n_letters"] for n in part])
    nx.draw_networkx_nodes(G, pos, nodelist=top, ax=ax, node_color="#2f5d8a", node_shape="s", node_size=160)
    nx.draw_networkx_labels(G, pos, {n: _short(G.nodes[n]["label"], 30) for n in G}, ax=ax, font_size=7)
    ax.set_title(title, loc="left", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Export für Gephi
# ---------------------------------------------------------------------------

def correspondence_graph(lt: pd.DataFrame, letters: pd.DataFrame, labels: dict):
    """Gerichtetes Korrespondenznetz (Absender -> Empfänger, Personen bevorzugt) mit Topic-Anteilen je Kante."""
    import networkx as nx

    def main_actor(types, names, keys):
        a = list(zip(_split(types), _split(names), _split(keys)))
        pers = [x for x in a if x[0] == "persName"]
        x = pers[0] if pers else (a[0] if a else ("", "unbekannt", ""))
        return (x[2] or x[1]), x[1]

    G = nx.DiGraph()
    j = letters.set_index("letter_id").join(lt, how="inner")
    tcols = [c for c in lt.columns]
    for lid, r in j.iterrows():
        sk, sn = main_actor(r["sender_types"], r["sender_names"], r["sender_keys"])
        rk, rn = main_actor(r["receiver_types"], r["receiver_names"], r["receiver_keys"])
        G.add_node(sk, label=sn)
        G.add_node(rk, label=rn)
        if G.has_edge(sk, rk):
            e = G[sk][rk]
            e["n_letters"] += 1
            for t in tcols:
                e[f"t{t}"] += float(r[t])
        else:
            G.add_edge(sk, rk, n_letters=1, **{f"t{t}": float(r[t]) for t in tcols})
    for _, _, e in G.edges(data=True):
        shares = {t: e[f"t{t}"] / e["n_letters"] for t in tcols}
        for t in tcols:
            e[f"t{t}"] = round(shares[t], 4)
        dom = max(shares, key=shares.get)
        e["dominant_topic"] = labels.get(dom, f"T{dom}")
        e["weight"] = e["n_letters"]
    return G


def write_gexf(G, path: str | Path):
    import networkx as nx
    nx.write_gexf(G, str(path))
