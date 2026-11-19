"""AS_UE_Topics v2 – Topic Modeling für die Korrespondenz Arnold Schönberg – Universal-Edition / Dreililien.

Module:
    extract    TEI-XML -> letters.csv (Metadaten) + units.csv (Textabschnitte)
    model      Topic Modeling (BERTopic, alternativ NMF als leichte Vergleichsbasis)
    aggregate  Topic-Profile pro Brief, Korrespondenzpartner und Zeitraum; Abbildungen; Netzwerk-Export
"""
__version__ = "2.0.0"
