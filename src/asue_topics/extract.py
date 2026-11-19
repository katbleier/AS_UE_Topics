"""TEI-XML -> Tabellen für Topic Modeling und Netzwerkanalyse.

Ergebnis:
    letters.csv  eine Zeile pro Dokument: ID, Datum, Absender:innen, Empfänger:innen (Name + Key),
                 im Brieftext erwähnte Personen/Organisationen/Werke, Wortzahl
    units.csv    eine Zeile pro Textabschnitt (Satz oder Absatz) mit Brief-ID, im ORIGINALTEXT
                 (keine Kleinschreibung, keine Stoppwortentfernung – das passiert erst im Modell)

Editorische Entscheidungen (alle in ExtractConfig einstellbar):
    - nur <text>/<body>, ohne <note> (Stellenkommentar), <opener>, <closer> (Briefformeln)
    - <choice>: <expan>/<corr>/<reg> statt <abbr>/<sic>/<orig>
    - <app>: <lem> statt <rdg>
    - <del> wird ausgelassen, <add> aufgenommen
    - <lb break="no"/> / <pb break="no"/> verbinden getrennte Wörter (v1-Fehler „durchfüh rung")

Aufruf:
    python -m asue_topics.extract data/tei output
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}


def _local(tag) -> str:
    """Lokaler Elementname ohne Namespace ('' für Kommentare/PIs)."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


@dataclass
class ExtractConfig:
    # Elemente, deren gesamter Inhalt NICHT in den Text übernommen wird
    skip_elements: set = field(default_factory=lambda: {
        "note",                      # Stellenkommentar der Edition
        "opener", "closer",          # Anrede, Datumszeile, Gruß, Unterschrift
        "del", "surplus",            # Gestrichenes, Überflüssiges
        "abbr", "sic", "orig",       # in <choice>: es bleibt expan / corr / reg
        "rdg",                       # in <app>: es bleibt lem
        "fw", "figure", "metamark", "index", "gap",
    })
    # Elemente, die als eigener Block gelten (Absatzgrenze)
    block_elements: set = field(default_factory=lambda: {
        "p", "ab", "div", "head", "item", "list", "postscript", "lg", "l",
        "table", "row", "cell", "quote", "floatingText", "dateline", "salute", "signed",
    })
    # Trennzeichen, die vor <lb break="no"/> entfernt werden (Bindestrich bewusst NICHT,
    # damit Komposita wie „Liszt-Partituren" erhalten bleiben)
    strip_before_nobreak: str = "¬="
    unit: str = "sentence"           # "sentence" | "paragraph" | "letter"
    schoenberg_name_pattern: str = r"sch[öo]e?nberg,?\s*arnold|arnold\s+sch[öo]e?nberg"


# ---------------------------------------------------------------------------
# Text aus dem body
# ---------------------------------------------------------------------------

PARA_BREAK = " "  # interner Absatzmarker


def _collect(el, cfg: ExtractConfig, out: list[str]) -> None:
    name = _local(el.tag)
    if name in cfg.skip_elements:
        return  # Inhalt überspringen, tail wird vom Elternaufruf behandelt
    if name in ("lb", "pb", "cb"):
        if el.get("break") == "no":
            # getrennte Wörter zusammenziehen
            joined = "".join(out).rstrip(" \n\t" + cfg.strip_before_nobreak)
            out.clear()
            out.append(joined)
        else:
            out.append(" ")
        return
    is_block = name in cfg.block_elements
    if is_block:
        out.append(PARA_BREAK)
    if el.text:
        out.append(el.text)
    for child in el:
        _collect(child, cfg, out)
        if child.tail:
            tail = child.tail
            # nach lb break="no" führenden Whitespace des Folgetexts entfernen
            if _local(child.tag) in ("lb", "pb", "cb") and child.get("break") == "no":
                tail = tail.lstrip()
            out.append(tail)
    if is_block:
        out.append(PARA_BREAK)


def body_paragraphs(body, cfg: ExtractConfig) -> list[str]:
    parts: list[str] = []
    _collect(body, cfg, parts)
    raw = "".join(parts)
    paras = []
    for chunk in raw.split(PARA_BREAK):
        txt = re.sub(r"\s+", " ", chunk).strip()
        if txt:
            paras.append(txt)
    return paras


# ---------------------------------------------------------------------------
# Satzsegmentierung (regelbasiert, ohne nltk/punkt-Download)
# ---------------------------------------------------------------------------

ABBREV = {
    "z", "b", "u", "a", "d", "h", "s", "v", "vgl", "bzw", "usw", "etc", "ca", "nr", "no", "op",
    "dr", "prof", "hr", "hrn", "fr", "frl", "st", "str", "bd", "bde", "jan", "feb", "febr", "jän",
    "aug", "sept", "sep", "okt", "oct", "nov", "dez", "dec", "mk", "kr", "fl", "m", "s", "ff",
    "evtl", "eventl", "resp", "ev", "inkl", "excl", "exkl", "ue", "u.e", "k", "hrsg", "tel",
    "sr", "ihr", "exz", "wohlgeb", "hochwohlgeb", "dir", "kapellm", "o", "w", "abs", "geb",
}
_SENT_END = re.compile(r"([.!?…]+[\"“”»«')\]]*)(\s+)")


def split_sentences(text: str) -> list[str]:
    sents, start = [], 0
    for m in _SENT_END.finditer(text):
        end = m.end(1)
        before = text[start:end]
        last_tok = re.findall(r"[\wÄÖÜäöüß.]+[.!?…]+$", before)
        tok = last_tok[0].rstrip(".!?…").lower() if last_tok else ""
        nxt = text[m.end():m.end() + 1]
        # kein Satzende: Abkürzung, Zahl (Datum/Ordinalzahl), einzelner Buchstabe, Kleinbuchstabe danach
        if m.group(1).startswith(".") and (
            tok in ABBREV or tok.split(".")[-1] in ABBREV or tok.isdigit()
            or len(tok) == 1 or (nxt and nxt.islower())
        ):
            continue
        s = text[start:end].strip()
        if s:
            sents.append(s)
        start = m.end()
    rest = text[start:].strip()
    if rest:
        sents.append(rest)
    return sents


# ---------------------------------------------------------------------------
# Metadaten
# ---------------------------------------------------------------------------

def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _key(el) -> str:
    k = el.get("key") or el.get("ref") or ""
    return k.lstrip("#").strip()


def _actors(action) -> list[tuple[str, str, str]]:
    """(Typ, Name, Key) aller Akteure einer correspAction."""
    res = []
    if action is None:
        return res
    for el in action:
        n = _local(el.tag)
        if n in ("persName", "orgName", "name"):
            res.append((n, _norm("".join(el.itertext())), _key(el)))
    return res


def _date(action) -> str:
    if action is None:
        return ""
    d = action.find("tei:date", NS)
    if d is None:
        return ""
    for att in ("when", "from", "notBefore", "notAfter", "to"):
        if d.get(att):
            return d.get(att)
    return _norm("".join(d.itertext()))


def _mentions(body, cfg: ExtractConfig) -> dict[str, list[str]]:
    out = {"person": [], "org": [], "place": [], "work": []}
    for el in body.iter():
        n = _local(el.tag)
        # Erwähnungen in Kommentaren/Formeln nicht zählen
        if any(_local(a.tag) in cfg.skip_elements for a in el.iterancestors()):
            continue
        k = _key(el)
        if not k:
            continue
        if n == "persName":
            out["person"].append(k)
        elif n == "orgName":
            out["org"].append(k)
        elif n == "placeName":
            out["place"].append(k)
        elif n == "title" or (n == "rs" and el.get("type") in ("work", "composition", "writing", "bibl")):
            out["work"].append(k)
        elif n == "rs" and el.get("type") in out:
            out[el.get("type")].append(k)
    return {t: sorted(set(v)) for t, v in out.items()}


def parse_file(path: Path, cfg: ExtractConfig):
    tree = etree.parse(str(path))
    root = tree.getroot()
    header = root.find("tei:teiHeader", NS)
    body = root.find(".//tei:text/tei:body", NS)

    idno = header.find(".//tei:publicationStmt/tei:idno", NS) if header is not None else None
    if idno is None and header is not None:
        idno = header.find(".//tei:idno", NS)
    letter_id = _norm(idno.text) if idno is not None and idno.text else path.stem

    sent = root.find(".//tei:correspDesc/tei:correspAction[@type='sent']", NS)
    recv = root.find(".//tei:correspDesc/tei:correspAction[@type='received']", NS)
    s_act, r_act = _actors(sent), _actors(recv)
    date = _date(sent)
    year = re.search(r"\d{4}", date)

    paras = body_paragraphs(body, cfg) if body is not None else []
    text_words = sum(len(p.split()) for p in paras)
    ments = _mentions(body, cfg) if body is not None else {k: [] for k in ("person", "org", "place", "work")}

    meta = {
        "letter_id": letter_id,
        "file": path.name,
        "doc_type": root.get("type") or "",
        "date": date,
        "year": int(year.group()) if year else "",
        "sender_names": " | ".join(a[1] for a in s_act),
        "sender_keys": " | ".join(a[2] for a in s_act),
        "sender_types": " | ".join(a[0] for a in s_act),
        "receiver_names": " | ".join(a[1] for a in r_act),
        "receiver_keys": " | ".join(a[2] for a in r_act),
        "receiver_types": " | ".join(a[0] for a in r_act),
        "place_sent": _norm("".join(sent.find("tei:placeName", NS).itertext()))
        if sent is not None and sent.find("tei:placeName", NS) is not None else "",
        "n_words": text_words,
        "mentioned_persons": " | ".join(ments["person"]),
        "mentioned_orgs": " | ".join(ments["org"]),
        "mentioned_places": " | ".join(ments["place"]),
        "mentioned_works": " | ".join(ments["work"]),
    }

    if cfg.unit == "letter":
        texts = [" ".join(paras)] if paras else []
    elif cfg.unit == "paragraph":
        texts = paras
    else:
        texts = [s for p in paras for s in split_sentences(p)]
    units = [{"letter_id": letter_id, "unit_no": i + 1, "text": t, "n_words": len(t.split())}
             for i, t in enumerate(texts)]
    return meta, units


def element_inventory(tei_dir: Path) -> Counter:
    """Zählt alle Elemente innerhalb von <body> – zur Prüfung der skip_elements gegen die Kodierrichtlinien."""
    c = Counter()
    for f in sorted(tei_dir.glob("**/*.xml")):
        try:
            body = etree.parse(str(f)).getroot().find(".//tei:text/tei:body", NS)
        except etree.XMLSyntaxError:
            continue
        if body is not None:
            c.update(_local(e.tag) for e in body.iter() if _local(e.tag))
    return c


def run(tei_dir: str | Path, out_dir: str | Path, cfg: ExtractConfig | None = None):
    cfg = cfg or ExtractConfig()
    tei_dir, out_dir = Path(tei_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    letters, units, errors = [], [], []
    for f in sorted(tei_dir.glob("**/*.xml")):
        try:
            root_tag = etree.parse(str(f)).getroot().tag
        except etree.XMLSyntaxError as e:
            errors.append((f.name, str(e)))
            continue
        if _local(root_tag) != "TEI":  # Registerdateien u. ä. überspringen
            continue
        try:
            if etree.parse(str(f)).getroot().find(".//tei:correspDesc", NS) is None:
                continue  # keine Korrespondenz (z. B. Register, Bibliographie)
            m, u = parse_file(f, cfg)
        except Exception as e:  # noqa: BLE001
            errors.append((f.name, repr(e)))
            continue
        letters.append(m)
        units.extend(u)

    _write(out_dir / "letters.csv", letters)
    _write(out_dir / "units.csv", units)
    if errors:
        _write(out_dir / "extract_errors.csv", [{"file": a, "error": b} for a, b in errors])
    print(f"{len(letters)} Dokumente, {len(units)} Einheiten ({cfg.unit}), {len(errors)} Fehler -> {out_dir}")
    return letters, units


def _write(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Aufruf: python -m asue_topics.extract <TEI-Ordner> <Ausgabe-Ordner> [sentence|paragraph|letter]")
    c = ExtractConfig()
    if len(sys.argv) > 3:
        c.unit = sys.argv[3]
    run(sys.argv[1], sys.argv[2], c)
