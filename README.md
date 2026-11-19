# AS_UE_Topics

Topic Modeling der Korrespondenz Arnold Schönberg – Universal-Edition / Verlag Dreililien
(Digitale Edition: https://www.schoenberg-ue.at).

**Version 2 (2026):** von der TEI-Edition über Topics zu inhaltlich qualifizierten Korrespondenzbeziehungen –
*wer mit wem* (correspDesc) und *worüber* (Topic Modeling).
Version 1 (Projekt „Machine Learning für digitale Editionen", Universität Graz, WS 2024/25) liegt unverändert in [`v1_2025/`](v1_2025/).

## Daten

Die TEI-Daten der Edition (CC BY-SA 4.0) liegen auf GitHub und im mdwRepository:
- <https://github.com/Schoenberg-UE/Schoenberg-Data>
- <https://datasets.mdw.ac.at/datasets/dataset/cfb9cae7-2729-40a0-958d-be13a539af28>

Das Datenrepo **neben** dieses Repo klonen – das Notebook erwartet `../Schoenberg-Data/data`:

```powershell
cd C:\Users\<name>\Documents\github
git clone https://github.com/Schoenberg-UE/Schoenberg-Data.git
```

Gelesen werden alle Dokumente mit `<correspDesc>` aus `data/letters` und `data/documents`; Register werden übersprungen.
Datenstand v2.5 (Mai 2026): 1.794 Dokumente (1.643 Briefe, 151 Dokumente), davon 368 erschlossene Stücke ohne Text,
15.723 Sätze. Die Editionsdaten werden hier nicht erneut veröffentlicht.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
jupyter lab AS_UE_Topics_v2.ipynb
```

Beim ersten Lauf lädt `sentence-transformers` das Sprachmodell herunter (ca. 1 GB) und berechnet die Embeddings;
diese werden in `output/cache/` zwischengespeichert.

## Ablauf

| Schritt | Modul | Ergebnis |
|---|---|---|
| 1. Extraktion | `src/asue_topics/extract.py` | `output/letters.csv` (Metadaten), `output/units.csv` (Sätze im Originaltext) |
| 2. Topic Modeling | `src/asue_topics/model.py` | `output/<verfahren>_unit_topics.csv`, `_topic_info.csv`, `_topic_labels.csv` |
| 3. Aggregation | `src/asue_topics/aggregate.py` | Heatmaps, Zeitverlauf, Partner–Topic-Netz (PNG), GEXF für Gephi |

Die Extraktion lässt sich auch ohne Notebook aufrufen:

```bash
PYTHONPATH=src python -m asue_topics.extract data/tei output sentence
```

## Editorische und methodische Entscheidungen

**Textgrundlage** (`ExtractConfig`, abgestimmt auf ediarum.asue): nur `<body>`; ausgelassen werden
Editionskommentare (`<note>` mit `@resp`/`@sameAs`), handschriftliche Vermerke (`<note @hand>`, optional einschließbar),
Briefkopf und Formeln (`<opener>`, `<closer>`, `<salute>`, `<signed>`, `<dateline>`, `<address>`, `<stamp>`),
`<del>`, `<metamark>`, `<figure>`, Notenbeispiele (`<notatedMusic>`/MEI).
Bei `<seg type="comment">` bleibt `<orig>` (Brieftext) erhalten; bei `<choice>` wird `<corr>`/`<reg>`/`<expan>` übernommen;
`<supplied>` und `<add>` bleiben; `<lb break="no"/>` verbindet getrennte Wörter.
Zu jeder Einheit wird zusätzlich `text_noent` erzeugt: derselbe Text ohne ausgezeichnete Personen-, Orts- und Organisationsnamen.
Die Liste der ausgelassenen Elemente sollte mit den Kodierrichtlinien der Edition abgeglichen werden
(`extract.element_inventory()` zählt alle Elemente im `<body>`).

**Einheit:** Satz (regelbasierte Segmentierung mit Abkürzungsliste); Sätze unter 4 Wörtern werden nicht modelliert.
Alternativ Absatz oder ganzer Brief.

**BERTopic** (`ModelConfig`): Embeddings aus den unveränderten Sätzen (`paraphrase-multilingual-mpnet-base-v2`);
die Topic-Wörter werden aus `text_noent` gebildet, damit Topics nicht aus Namen bestehen;
Stoppwörter ([stopwords-iso](https://github.com/stopwords-iso/stopwords-de), MIT) und projektspezifische Briefformeln
(`resources/stopwords-domain.txt`) wirken nur auf die Darstellung der Topics. UMAP mit festem Seed.

**Vergleichsverfahren:** TF-IDF + NMF (scikit-learn), um die Abhängigkeit der Topics vom Verfahren sichtbar zu machen.

**Aggregation:** Topic-Anteile pro Brief = Anteil der Sätze je Topic (ohne Outlier).
Profile je Korrespondenzpartner:in (bevorzugt Person, sonst Körperschaft aus `correspDesc`) und Zeitraum;
Themendiversität als normierte Shannon-Entropie.

## Änderungen gegenüber v1

| v1 (2025) | v2 (2026) |
|---|---|
| unveröffentlichte XML (Edition v1.1), 1.393 Briefe | Schoenberg-Data v2.5, 1.794 Dokumente (1.426 mit Text) |
| Worttrennungen als Leerzeichen („durchfüh rung") | `lb break="no"` verbindet Wörter |
| Kleinschreibung und Stoppwortentfernung vor dem Embedding | Originalsätze ins Embedding, Stoppwörter nur für die Darstellung |
| Fragmente („1 u.", „ii.") als eigene Dokumente | Mindestlänge 4 Wörter |
| 231 Topics, 44 % Outlier (6.793 von 15.328 Sätzen) | siehe Notebook |
| keine Metadaten | Absender:in, Empfänger:in, Datum, Bezugsbriefe, Erwähnungen |
| Gestrichenes (`del`) und bei `choice` beide Lesarten (`sic` + `corr`) im Text | an Kodierung der Edition angepasst |
| Namen bestimmen Topics („berlin, paris …") | Topic-Wörter ohne ausgezeichnete Namen |
| kein Seed, Modell als pickle | Seed, Embedding-Cache, safetensors |

## Tests

`tests/sample_tei/` enthält synthetische TEI-Dateien (kein Editionstext) zur Prüfung der Extraktion:

```bash
PYTHONPATH=src python -m asue_topics.extract tests/sample_tei output_test
```

## Lizenz und Zitation

Code: Lizenz noch festzulegen (Vorschlag: MIT). Editionsdaten: CC BY-SA 4.0, zitiert nach: Arnold Schönberg: Briefwechsel mit den Verlagen Universal-Edition
und Dreililien. Hrsg. von Katharina Bleier und Therese Muxeneder, mdw / Arnold Schönberg Center, https://www.schoenberg-ue.at.
