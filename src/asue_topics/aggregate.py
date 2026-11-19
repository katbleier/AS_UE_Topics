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

COUNT_COLS = ["n_letters", "n_erschlossen"]

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
    """Mittlere Topic-Anteile pro Gruppe.

    n_letters     Briefe mit Text, aus denen das Profil berechnet ist
    n_erschlossen erschlossene Briefe derselben Gruppe (zählen als Kontakt, haben aber kein Topic-Profil)
    """
    j = lt.join(letters.set_index("letter_id")[[by]], how="inner")
    prof = j.groupby(by).mean(numeric_only=True)
    prof.insert(0, "n_letters", j.groupby(by).size())
    if "status" in letters:
        ersch = letters[letters["status"] == "erschlossen"].groupby(by).size()
        prof.insert(1, "n_erschlossen", ersch.reindex(prof.index).fillna(0).astype(int))
    else:
        prof.insert(1, "n_erschlossen", 0)
    return prof[prof["n_letters"] >= min_letters].sort_values("n_letters", ascending=False)


def topic_columns(prof: pd.DataFrame) -> pd.DataFrame:
    return prof.drop(columns=[c for c in COUNT_COLS if c in prof])


def topic_diversity(prof: pd.DataFrame) -> pd.Series:
    """Normierte Shannon-Entropie der Topic-Verteilung (0 = ein Thema, 1 = alle gleich)."""
    p = topic_columns(prof).clip(lower=1e-12)
    p = p.div(p.sum(axis=1), axis=0)
    h = -(p * np.log(p)).sum(axis=1)
    return (h / np.log(p.shape[1])).rename("diversity")


# ---------------------------------------------------------------------------
# 4. Abbildungen
# ---------------------------------------------------------------------------

def _short(label: str, n: int = 38) -> str:
    return label if len(label) <= n else label[: n - 1] + "…"


def plot_heatmap(prof: pd.DataFrame, labels: dict, path: str | Path, title: str, max_rows: int = 15,
                 order: list | None = None, separator_after=None, mode: str = "share",
                 annotate: bool = True, subtitle: str | None = None):
    """Heatmap Gruppen (Zeilen) × Topics (Spalten).

    order            Spaltenreihenfolge (Topic-Schlüssel); fehlende Topics werden hinten angehängt
    separator_after  Topic-Schlüssel, nach dem eine senkrechte Trennlinie gezeichnet wird
                     (z. B. zwischen Themen und Sprechhandlungen)
    mode             "share"     mittlerer Anteil der Sätze
                     "deviation" Abweichung vom Durchschnitt aller Briefe (Prozentpunkte):
                                 zeigt, was für eine Beziehung TYPISCH ist
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    p = prof.head(max_rows)
    data = topic_columns(p)
    if order:
        cols = [c for c in order if c in data.columns] + [c for c in data.columns if c not in order]
        data = data[cols]
    if mode == "deviation":
        allp = topic_columns(prof)
        baseline = allp.mul(prof["n_letters"], axis=0).sum() / prof["n_letters"].sum()
        values = (data - baseline[data.columns]) * 100
        vmax = max(np.abs(values.values).max(), 1e-6)
        cmap, norm = "RdBu", TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        cbar_label = "Abweichung vom Durchschnitt (Prozentpunkte)"
        fmt = lambda v: f"{v:+.0f}"
    else:
        values = data * 100
        cmap, norm = "Blues", None
        cbar_label = "Anteil der Sätze (%)"
        fmt = lambda v: f"{v:.0f}"

    col_labels = [_short(labels.get(c, f"T{c}"), 32) for c in values.columns]
    row_labels = [f"{_short(str(i), 28)} (n={n}" + (f", +{e} erschl.)" if e else ")")
                  for i, n, e in zip(p.index, p["n_letters"], p.get("n_erschlossen", [0] * len(p)))]
    fig, ax = plt.subplots(figsize=(max(9, 0.62 * len(col_labels) + 4.5), max(3.5, 0.48 * len(row_labels) + 2.6)))
    im = ax.imshow(values.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=50, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=9.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if annotate:
        lim = np.abs(values.values).max()
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values.values[i, j]
                strong = abs(v) > 0.55 * lim
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.5,
                        color="white" if strong else "#333333")
    if separator_after is not None and separator_after in list(values.columns):
        x = list(values.columns).index(separator_after) + 0.5
        ax.axvline(x, color="#222222", linewidth=1.6)
    ax.set_title(title, fontsize=12, loc="left", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555555")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_topics_over_time(lt: pd.DataFrame, letters: pd.DataFrame, labels: dict, path: str | Path,
                          top_n: int = 6, window: int = 3, topics: list | None = None,
                          markers: dict | None = None, title: str = "Topics im Zeitverlauf",
                          year_range: tuple | None = None):
    """Mittlerer Topic-Anteil pro Jahr (gleitender Mittelwert), dazu Briefe pro Jahr als graue Balken.

    topics      Auswahl von Topic-Schlüsseln (empfohlen: 2–4); sonst die top_n häufigsten
    markers     {Jahr: "Beschriftung"} für senkrechte Markierungen, z. B. {1933: "Emigration"}
    year_range  (von, bis) – z. B. (1902, 1938), um dünn belegte Randjahre auszublenden
    """
    import matplotlib.pyplot as plt

    j = lt.join(letters.set_index("letter_id")[["year"]], how="inner")
    j = j[pd.to_numeric(j["year"], errors="coerce").notna()]
    j["year"] = j["year"].astype(int)
    if year_range:
        j = j[(j["year"] >= year_range[0]) & (j["year"] <= year_range[1])]
    by_year = j.drop(columns="year").groupby(j["year"]).mean()
    full = by_year.reindex(range(by_year.index.min(), by_year.index.max() + 1))
    smooth = full.rolling(window, center=True, min_periods=1).mean() * 100
    sel = topics or list(lt.mean().sort_values(ascending=False).head(top_n).index)
    counts = j.groupby("year").size().reindex(full.index).fillna(0)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax2 = ax.twinx()
    ax2.bar(counts.index, counts.values, color="#999999", alpha=0.18, width=0.85)
    ax2.set_ylabel("Briefe pro Jahr", color="#888888", fontsize=9)
    ax2.tick_params(colors="#888888", labelsize=8)
    colors = ["#1f5f8b", "#d1782a", "#3b8f4f", "#8b3a62", "#6b6b6b", "#b5a33a"]
    for k, t in enumerate(sel):
        ax.plot(smooth.index, smooth[t], label=labels.get(t, f"T{t}"), linewidth=2.2,
                color=colors[k % len(colors)])
    for yr, txt in (markers or {}).items():
        ax.axvline(yr, color="#444444", linewidth=1, linestyle="--")
        ax.text(yr + 0.3, ax.get_ylim()[1] * 0.97, txt, fontsize=8.5, va="top", color="#444444")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylabel(f"Anteil der Sätze in % ({window}-Jahres-Mittel)", fontsize=9)
    ax.set_title(title, loc="left", fontsize=12)
    for sp in ("top",):
        ax.spines[sp].set_visible(False)
        ax2.spines[sp].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def partner_topic_graph(prof: pd.DataFrame, labels: dict, min_share: float = 0.12):
    """Bipartites Netz: Partner:innen – Topics; Kante, wenn mittlerer Anteil >= min_share."""
    import networkx as nx

    G = nx.Graph()
    for partner, row in prof.iterrows():
        G.add_node(f"P:{partner}", label=str(partner), kind="partner", n_letters=int(row["n_letters"]))
        for t, share in row.drop([c for c in COUNT_COLS if c in row.index]).items():
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
    """Gerichtetes Korrespondenznetz (Absender -> Empfänger, Personen bevorzugt).

    Kantenattribute:
        n_letters        alle Briefe (überliefert + erschlossen + ohne Text)
        n_ueberliefert   Briefe mit Text
        n_erschlossen    erschlossene Briefe
        t<k>             mittlerer Anteil von Topic k – nur aus überlieferten Briefen
        dominant_topic   Topic mit dem höchsten Anteil
    """
    import networkx as nx

    def main_actor(types, names, keys):
        a = list(zip(_split(types), _split(names), _split(keys)))
        pers = [x for x in a if x[0] == "persName"]
        x = pers[0] if pers else (a[0] if a else ("", "unbekannt", ""))
        return (x[2] or x[1]), x[1]

    tcols = list(lt.columns)
    G = nx.DiGraph()
    j = letters.set_index("letter_id").join(lt, how="left")
    for _, r in j.iterrows():
        sk, sn = main_actor(r["sender_types"], r["sender_names"], r["sender_keys"])
        rk, rn = main_actor(r["receiver_types"], r["receiver_names"], r["receiver_keys"])
        G.add_node(sk, label=sn)
        G.add_node(rk, label=rn)
        if not G.has_edge(sk, rk):
            G.add_edge(sk, rk, n_letters=0, n_ueberliefert=0, n_erschlossen=0, _n_topics=0,
                       **{f"t{t}": 0.0 for t in tcols})
        e = G[sk][rk]
        e["n_letters"] += 1
        status = r.get("status", "überliefert")
        if status == "erschlossen":
            e["n_erschlossen"] += 1
        elif status == "überliefert":
            e["n_ueberliefert"] += 1
        has_topics = tcols and pd.notna(r[tcols[0]])
        if has_topics:
            e["_n_topics"] += 1
            for t in tcols:
                e[f"t{t}"] += float(r[t])
    for _, _, e in G.edges(data=True):
        n = e.pop("_n_topics")
        shares = {t: (e[f"t{t}"] / n if n else 0.0) for t in tcols}
        for t in tcols:
            e[f"t{t}"] = round(shares[t], 4)
        dom = max(shares, key=shares.get) if n else None
        e["dominant_topic"] = labels.get(dom, f"T{dom}") if dom is not None else "(kein Text)"
        e["weight"] = e["n_letters"]
    return G


def write_gexf(G, path: str | Path):
    import networkx as nx
    nx.write_gexf(G, str(path))


def plot_partner_profiles(prof: pd.DataFrame, labels: dict, path: str | Path, partners: list | None = None,
                          top_k: int = 5, categories: dict | None = None, ncols: int = 3,
                          title: str = "Worüber schreibt Schönberg mit wem?"):
    """Einfache Alternative zur Heatmap: je Korrespondenzpartner:in ein kleines Balkendiagramm
    mit den top_k häufigsten Topic-Gruppen (Anteil der Sätze in %).

    partners    Auswahl (Index von prof); sonst die ersten 6 Zeilen
    categories  {Topic-Schlüssel: "Thema" | "Sprechhandlung"} – färbt die Balken entsprechend ein
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    sel = partners or list(prof.index[:6])
    data = topic_columns(prof).loc[sel] * 100
    colors = {"Thema": "#2f5d8a", "Sprechhandlung": "#d1782a"}
    nrows = int(np.ceil(len(sel) / ncols))
    xmax = data.max().max() * 1.18
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.5 * nrows + 0.9), squeeze=False)
    for ax, partner in zip(axes.flat, sel):
        row = data.loc[partner].sort_values(ascending=False).head(top_k)[::-1]
        cols = [colors.get((categories or {}).get(t), "#2f5d8a") for t in row.index]
        ax.barh(range(len(row)), row.values, color=cols, height=0.65)
        ax.set_yticks(range(len(row)), [_short(labels.get(t, f"T{t}"), 30) for t in row.index], fontsize=9)
        for i, v in enumerate(row.values):
            ax.text(v + xmax * 0.015, i, f"{v:.0f} %", va="center", fontsize=8.5, color="#333333")
        n = int(prof.loc[partner, "n_letters"])
        e = int(prof.loc[partner, "n_erschlossen"]) if "n_erschlossen" in prof else 0
        ax.set_title(f"{partner}\n{n} Briefe" + (f", {e} erschlossen" if e else ""), fontsize=10, loc="left")
        ax.set_xlim(0, xmax)
        ax.set_xticks([])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
    for ax in list(axes.flat)[len(sel):]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    if categories:
        fig.legend(handles=[Patch(color=c, label=k) for k, c in colors.items()], loc="upper right",
                   frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Richtung der Kommunikation
# ---------------------------------------------------------------------------

def direction_profile(lt: pd.DataFrame, letters: pd.DataFrame) -> pd.DataFrame:
    """Mittlerer Anteil jeder Gruppe in Briefen von bzw. an Schönberg (nur Briefe mit Text)."""
    j = lt.join(letters.set_index("letter_id")[["direction"]], how="inner")
    j = j[j["direction"].isin(["von Schönberg", "an Schönberg"])]
    prof = j.groupby("direction").mean(numeric_only=True).T
    prof.attrs["n"] = j["direction"].value_counts().to_dict()
    return prof[["von Schönberg", "an Schönberg"]]


def plot_direction(lt: pd.DataFrame, letters: pd.DataFrame, labels: dict, path: str | Path,
                   order: list | None = None, separator_after=None,
                   title: str = "Wer schreibt worüber? Briefe von und an Schönberg"):
    """Schmetterlingsdiagramm: links Anteil in Briefen VON Schönberg, rechts in Briefen AN Schönberg."""
    import matplotlib.pyplot as plt

    prof = direction_profile(lt, letters) * 100
    n = direction_profile(lt, letters).attrs["n"]
    keys = [k for k in (order or prof.index) if k in prof.index]
    keys += [k for k in prof.index if k not in keys]
    prof = prof.loc[keys]
    y = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(keys) + 2))
    left, right = prof["von Schönberg"].values, prof["an Schönberg"].values
    ax.barh(y, -left, color="#2f5d8a", height=0.7, label=f"von Schönberg ({n.get('von Schönberg', 0)} Briefe)")
    ax.barh(y, right, color="#d1782a", height=0.7, label=f"an Schönberg ({n.get('an Schönberg', 0)} Briefe)")
    xmax = max(left.max(), right.max()) * 1.25
    for yi, l, r in zip(y, left, right):
        ax.text(-l - xmax * 0.015, yi, f"{l:.0f} %", ha="right", va="center", fontsize=8.5, color="#333333")
        ax.text(r + xmax * 0.015, yi, f"{r:.0f} %", ha="left", va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(y, [_short(labels.get(k, str(k)), 32) for k in keys], fontsize=9.5)
    ax.set_xlim(-xmax, xmax)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    if separator_after is not None and separator_after in keys:
        ys = y[keys.index(separator_after)] - 0.5
        ax.axhline(ys, color="#888888", linewidth=0.8, linestyle="--")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=12, pad=30)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Erwähnte Personen je Gruppe
# ---------------------------------------------------------------------------

def load_person_register(path: str | Path) -> pd.DataFrame:
    """Personenregister der Edition (TEI listPerson) -> key, name, occupation."""
    from lxml import etree

    ns = "{http://www.tei-c.org/ns/1.0}"
    rows = []
    for p in etree.parse(str(path)).getroot().iter(ns + "person"):
        key = p.get("{http://www.w3.org/XML/1998/namespace}id")
        pn = p.find(ns + "persName[@type='reg']")
        if pn is None:
            pn = p.find(ns + "persName")
        sur = pn.findtext(ns + "surname") if pn is not None else None
        fore = pn.findtext(ns + "forename") if pn is not None else None
        name = ", ".join(x.strip() for x in (sur, fore) if x and x.strip()) or \
            (" ".join(pn.itertext()).strip() if pn is not None else key)
        occ = p.find(ns + "occupation")
        rows.append({"key": key, "name": " ".join(name.split()),
                     "occupation": " ".join("".join(occ.itertext()).split()) if occ is not None else ""})
    return pd.DataFrame(rows)


def mentions_by_group(lt: pd.DataFrame, letters: pd.DataFrame, groups: list, register: pd.DataFrame,
                      min_share: float = 0.3, exclude_keys: set | None = None) -> pd.DataFrame:
    """Personen, die in Briefen erwähnt werden, in denen eine Gruppe mindestens min_share der Sätze ausmacht.

    Ausgeschlossen: Absender:innen und Empfänger:innen des jeweiligen Briefs sowie exclude_keys
    (z. B. Schönberg selbst) – es bleiben die „Dritten".
    """
    exclude_keys = set(exclude_keys or ())
    L = letters.set_index("letter_id")
    rows = []
    for g in groups:
        ids = lt.index[lt[g] >= min_share]
        for lid in ids:
            if lid not in L.index:
                continue
            r = L.loc[lid]
            corr = set(_split(r.get("sender_keys"))) | set(_split(r.get("receiver_keys")))
            for k in set(_split(r.get("mentioned_persons"))):
                if k not in corr and k not in exclude_keys:
                    rows.append({"gruppe": g, "key": k, "letter_id": lid})
    df = pd.DataFrame(rows, columns=["gruppe", "key", "letter_id"])
    out = df.groupby(["gruppe", "key"]).letter_id.nunique().rename("n_briefe").reset_index()
    out = out.merge(register, on="key", how="left")
    out["name"] = out["name"].fillna(out["key"])
    return out.sort_values(["gruppe", "n_briefe"], ascending=[True, False]).reset_index(drop=True)


def plot_mentions(df: pd.DataFrame, path: str | Path, groups: list | None = None, top_n: int = 10,
                  title: str = "Wer wird erwähnt? Dritte in den Briefen", show_occupation: bool = True):
    """Je Gruppe ein Balkendiagramm der am häufigsten erwähnten Personen (Zahl der Briefe)."""
    import matplotlib.pyplot as plt

    groups = groups or list(df["gruppe"].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 0.38 * top_n + 1.8), squeeze=False)
    xmax = df[df.gruppe.isin(groups)].groupby("gruppe").n_briefe.max().max() * 1.2
    for ax, g in zip(axes.flat, groups):
        d = df[df.gruppe == g].head(top_n)[::-1]

        def lab(r):
            occ = str(r.get("occupation") or "")
            occ = occ.split(",")[0].split(";")[0].strip()
            return _short(r["name"], 26) + (f" ({_short(occ, 18)})" if show_occupation and occ else "")
        ax.barh(range(len(d)), d["n_briefe"], color="#3b8f4f", height=0.65)
        ax.set_yticks(range(len(d)), [lab(r) for _, r in d.iterrows()], fontsize=9)
        for i, v in enumerate(d["n_briefe"]):
            ax.text(v + xmax * 0.015, i, str(v), va="center", fontsize=8.5, color="#333333")
        ax.set_title(g, loc="left", fontsize=10.5)
        ax.set_xlim(0, xmax)
        ax.set_xticks([])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.text(0.01, 0.005, "Zahl der Briefe, in denen die Person erwähnt wird (ohne Absender:in/Empfänger:in)",
             fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def mentions_graph(df: pd.DataFrame, min_letters: int = 2):
    """Bipartites Netz Gruppe – erwähnte Person (Kantengewicht = Zahl der Briefe), für Gephi."""
    import networkx as nx

    G = nx.Graph()
    for _, r in df[df.n_briefe >= min_letters].iterrows():
        G.add_node(f"G:{r.gruppe}", label=r.gruppe, kind="gruppe")
        G.add_node(r.key, label=r["name"], kind="person", occupation=str(r.get("occupation") or ""))
        G.add_edge(f"G:{r.gruppe}", r.key, weight=int(r.n_briefe))
    return G
