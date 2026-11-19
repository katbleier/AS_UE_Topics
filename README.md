# AS_UE_Topics

Topic Modeling der Korrespondenz Arnold Schönberg – Universal-Edition / Verlag Dreililien
(Digitale Edition: https://www.schoenberg-ue.at).

Von der TEI-Edition über Topics zu inhaltlich qualifizierten Korrespondenzbeziehungen –
*wer mit wem* (correspDesc) und *worüber* (Topic Modeling). Beispiel für den Workshop
„Netzwerken in Wien um 1900“ (November 2026).

- **`AS_UE_Topics_Workshop.ipynb`** – der Ablauf mit den verwendeten Einstellungen, Benennungen, Gruppen und Abbildungen
- `AS_UE_Topics_v2.ipynb` – Arbeitsnotebook mit allen Optionen (u. a. NMF als Vergleichsverfahren)

Eine frühere Fassung (Universität Graz, WS 2024/25) liegt unverändert in [`v1_2025/`](v1_2025/).

## Daten

Die TEI-Daten der Edition (CC BY-SA 4.0) liegen auf GitHub und im mdwRepository:
- <https://github.com/Schoenberg-UE/Schoenberg-Data>
- <https://datasets.mdw.ac.at/datasets/dataset/cfb9cae7-2729-40a0-958d-be13a539af28>

Das Datenrepo **neben** dieses Repo klonen – das Notebook erwartet `../Schoenberg-Data/data`:

```powershell
cd C:\Users\<name>\Documents\github
git clone https://github.com/Schoenberg-UE/Schoenberg-Data.git
```

Gelesen werden standardmäßig nur Briefe (`telota:doctype="letter"`); Dokumente wie Verträge und Abrechnungen
haben einen eigenen Stil und sind ausgeschlossen (zuschaltbar). Register werden übersprungen.
Datenstand v2.5 (Mai 2026): 1.643 Briefe – 1.414 überliefert, 226 erschlossen, 3 ohne Text; 15.605 Sätze. Die Editionsdaten werden hier nicht erneut veröffentlicht.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m spacy download de_core_news_md
jupyter lab AS_UE_Topics_Workshop.ipynb
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

**Normalisierung der Topic-Wörter** (`normalize.py`, nur für `text_repr`, nicht für die Embeddings):
Schreibvarianten werden datengetrieben vereinheitlicht (ss/ß, ue/ae/oe → ü/ä/ö, th → t, c → k/z), aber nur,
wenn die Variante im Korpus selbst häufiger vorkommt; alle Ersetzungen stehen in `output/normalization_map.csv`
(rund 270 Wortformen, z. B. daß → dass, ueber → über, Theil → Teil, Correctur → Korrektur).
Danach Lemmatisierung mit spaCy (`de_core_news_md`) und einheitlich ß → ss.

**Erschlossene Briefe** (Titel „[erschlossen]") haben keinen Text und kein Topic-Profil, zählen aber als Kontakt:
`status` in `letters.csv`, `n_erschlossen` in den Profilen, eigene Kantenattribute im Korrespondenznetz.

**Vergleichsverfahren:** TF-IDF + NMF (scikit-learn), um die Abhängigkeit der Topics vom Verfahren sichtbar zu machen.

**Aggregation:** Topic-Anteile pro Brief = Anteil der Sätze je Topic (ohne Outlier).
Profile je Korrespondenzpartner:in (bevorzugt Person, sonst Körperschaft aus `correspDesc`) und Zeitraum;
Themendiversität als normierte Shannon-Entropie.


## Tests

`tests/sample_tei/` enthält synthetische TEI-Dateien (kein Editionstext) zur Prüfung der Extraktion:

```bash
PYTHONPATH=src python -m asue_topics.extract tests/sample_tei output_test
```

## Lizenz und Zitation

Code: Lizenz noch festzulegen (Vorschlag: MIT). Editionsdaten: CC BY-SA 4.0, zitiert nach: Arnold Schönberg: Briefwechsel mit den Verlagen Universal-Edition
und Dreililien. Hrsg. von Katharina Bleier und Therese Muxeneder, mdw / Arnold Schönberg Center, https://www.schoenberg-ue.at.
