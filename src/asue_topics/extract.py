"""TEI-XML (Schoenberg-UE-Edition) -> Tabellen für Topic Modeling und Netzwerkanalyse.

Ergebnis:
    letters.csv  eine Zeile pro Dokument: Signatur (ASCC), xml:id, Dokumenttyp, Datum, Absender:innen,
                 Empfänger:innen (Name + Register-Key), Bezugsbriefe (correspContext),
                 im Text erwähnte Personen/Organisationen/Orte/Werke, Wortzahl
    units.csv    eine Zeile pro Textabschnitt (Satz, Absatz oder Brief) mit
                    text         Originaltext (für die Embeddings)
                    text_noent   derselbe Text ohne ausgezeichnete Personen-, Orts- und Organisationsnamen
                                 (für die Topic-Darstellung, damit Topics nicht aus Namen bestehen)

Editorische Entscheidungen (in ExtractConfig einstellbar), abgestimmt auf die Kodierung der Edition
(ediarum.asue, Datenpaket Schoenberg-Data):
    - Text nur aus <text>/<body>
    - ausgelassen: Briefformeln und Kopf (<opener>, <closer>, <salute>, <signed>, <dateline>, <address>, <stamp>),
      Gestrichenes (<del>), <metamark>, <figure>, Notenbeispiele (<notatedMusic>, MEI)
    - <note>: Editionskommentare (@resp, @sameAs, ohne @hand) immer ausgelassen; handschriftliche Vermerke
      (@hand) nur, wenn ihre Hand in include_note_hands steht (Standard: keine)
    - <seg type="comment"><orig>…</orig><note/></seg>: <orig> ist Brieftext und bleibt erhalten
    - <choice>: <corr> statt <sic>, <reg> statt <orig>, <expan> statt <abbr>
    - <supplied> (editorisch ergänzt) bleibt erhalten, <add> ebenso
    - <lb break="no"/> verbindet getrennte Wörter (<g ref="#typoHyphen"/> trägt keinen Text)

Aufruf:
    python -m asue_topics.extract <TEI-Ordner> <Ausgabe-Ordner> [sentence|paragraph|letter]
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
TELOTA_NS = "http://www.telota.de"
NS = {"tei": TEI_NS}
_RECOVER = etree.XMLParser(recover=True)


def _local(tag) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _is_tei(tag) -> bool:
    return isinstance(tag, str) and tag.startswith("{" + TEI_NS + "}")


def parse_xml(path: Path):
    """Parsen; bei Fehlern (z. B. doppelte xml:id) tolerant erneut."""
    try:
        return etree.parse(str(path)), None
    except etree.XMLSyntaxError as e:
        return etree.parse(str(path), _RECOVER), str(e)


@dataclass
class ExtractConfig:
    skip_elements: set = field(default_factory=lambda: {
        "opener", "closer", "salute", "signed", "dateline", "address", "stamp",   # Briefformeln, Kopf
        "del", "surplus", "metamark", "fw", "figure", "figDesc", "index", "gap", "anchor",
        "notatedMusic", "rdg",
    })
    include_note_hands: set = field(default_factory=set)   # z. B. {"#author"} für Vermerke des Schreibers
    choice_preference: tuple = ("corr", "reg", "expan")
    block_elements: set = field(default_factory=lambda: {
        "p", "ab", "div", "head", "item", "list", "postscript", "lg", "l",
        "table", "row", "cell", "quote", "floatingText",
    })
    entity_elements: set = field(default_factory=lambda: {"persName", "placeName", "orgName"})
    entity_rs_types: set = field(default_factory=lambda: {"person", "place", "org"})
    strip_before_nobreak: str = "¬="
    unit: str = "sentence"            # "sentence" | "paragraph" | "letter"
    doc_types: set = field(default_factory=lambda: {"letter", "document"})
    require_correspdesc: bool = True
    schoenberg_name_pattern: str = r"sch[öo]e?nberg,?\s*arnold|arnold\s+sch[öo]e?nberg"


# ---------------------------------------------------------------------------
# Text aus dem body
# ---------------------------------------------------------------------------

PARA = " "    # Absatzmarker
E_ON = ""    # Beginn einer Entität (Name)
E_OFF = ""   # Ende einer Entität


def _skip(el, cfg: ExtractConfig) -> bool:
    if not _is_tei(el.tag):
        return True                       # MEI (Notenbeispiele) u. ä.
    name = _local(el.tag)
    if name in cfg.skip_elements:
        return True
    if name == "note":
        hand = el.get("hand")
        return not (hand and hand in cfg.include_note_hands)
    return False


def _collect(el, cfg: ExtractConfig, out: list[str]) -> None:
    if _skip(el, cfg):
        return
    name = _local(el.tag)
    if name in ("lb", "pb", "cb"):
        if el.get("break") == "no":
            joined = "".join(out).rstrip(" \n\t" + cfg.strip_before_nobreak)
            out.clear()
            out.append(joined)
        else:
            out.append(" ")
        return
    if name == "choice":
        kids = [c for c in el if _is_tei(c.tag)]
        for pref in cfg.choice_preference:
            chosen = [c for c in kids if _local(c.tag) == pref]
            if chosen:
                _collect(chosen[0], cfg, out)
                return
        for c in kids:
            _collect(c, cfg, out)
        return

    is_block = name in cfg.block_elements
    is_ent = name in cfg.entity_elements or (name == "rs" and el.get("type") in cfg.entity_rs_types)
    if is_block:
        out.append(PARA)
    if is_ent:
        out.append(E_ON)
    if el.text:
        out.append(el.text)
    for child in el:
        _collect(child, cfg, out)
        if child.tail:
            tail = child.tail
            if _local(child.tag) in ("lb", "pb", "cb") and child.get("break") == "no":
                tail = tail.lstrip()
            out.append(tail)
    if is_ent:
        out.append(E_OFF)
    if is_block:
        out.append(PARA)


def body_paragraphs(body, cfg: ExtractConfig) -> list[str]:
    """Absätze mit Entitätsmarkern (E_ON/E_OFF)."""
    parts: list[str] = []
    _collect(body, cfg, parts)
    paras = []
    for chunk in "".join(parts).split(PARA):
        txt = re.sub(r"\s+", " ", chunk).strip()
        if re.sub(f"[{E_ON}{E_OFF}\\s]", "", txt):
            paras.append(txt)
    return paras


def clean(text: str) -> str:
    t = text.replace(E_ON, "").replace(E_OFF, "")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?)\]])", r"\1", t)      # „Krenek ." -> „Krenek."
    t = re.sub(r"([(\[])\s+", r"\1", t)
    t = re.sub(r"\(\s*\)|\[\s*\]", "", t)           # leere Klammern nach Namensentfernung
    return re.sub(r"\s+", " ", t).strip()


def without_entities(text: str) -> str:
    t = text
    pat = re.compile(f"{E_ON}[^{E_ON}{E_OFF}]*{E_OFF}")
    while pat.search(t):          # verschachtelte Namen von innen nach außen
        t = pat.sub(" ", t)
    return clean(t)


# ---------------------------------------------------------------------------
# Satzsegmentierung (regelbasiert, ohne nltk/punkt-Download)
# ---------------------------------------------------------------------------

ABBREV = {
    "z", "b", "u", "a", "d", "h", "s", "v", "vgl", "bzw", "usw", "etc", "ca", "nr", "no", "op",
    "dr", "prof", "hr", "hrn", "fr", "frl", "st", "str", "bd", "bde", "jan", "feb", "febr", "jän",
    "aug", "sept", "sep", "okt", "oct", "nov", "dez", "dec", "mk", "kr", "fl", "m", "ff",
    "evtl", "eventl", "resp", "ev", "inkl", "excl", "exkl", "ue", "k", "hrsg", "tel",
    "sr", "exz", "wohlgeb", "hochwohlgeb", "dir", "kapellm", "o", "w", "abs", "geb", "ds", "mts",
    "lfd", "j", "pp", "ggf", "sog", "allg", "gen", "mr", "mrs", "pt", "ord", "kgl", "kk",
}
_SENT_END = re.compile(r"([.!?…]+[\"“”»«')\]" + E_OFF + r"]*)(\s+)")


def split_sentences(text: str) -> list[str]:
    sents, start = [], 0
    for m in _SENT_END.finditer(text):
        end = m.end(1)
        before = clean(text[start:end])
        last_tok = re.findall(r"[\wÄÖÜäöüß.]+[.!?…]+[\"“”»«')\]]*$", before)
        tok = last_tok[0].rstrip(".!?…\"“”»«')]").lower() if last_tok else ""
        nxt = clean(text[m.end():m.end() + 3])[:1]
        if m.group(1).startswith(".") and (
            tok in ABBREV or tok.split(".")[-1] in ABBREV or tok.isdigit()
            or len(tok) == 1 or (nxt and nxt.islower())
        ):
            continue
        s = text[start:end].strip()
        if clean(s):
            sents.append(s)
        start = m.end()
    rest = text[start:].strip()
    if clean(rest):
        sents.append(rest)
    return sents


# ---------------------------------------------------------------------------
# Metadaten
# ---------------------------------------------------------------------------

def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _key(el) -> str:
    return (el.get("key") or el.get("ref") or "").lstrip("#").strip()


def _actors(action) -> list[tuple[str, str, str]]:
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
        if not _is_tei(el.tag):
            continue
        if any(_skip(a, cfg) for a in el.iterancestors() if _is_tei(a.tag)):
            continue
        n, k = _local(el.tag), _key(el)
        if not k:
            continue
        if n == "persName":
            out["person"].append(k)
        elif n == "orgName":
            out["org"].append(k)
        elif n == "placeName":
            out["place"].append(k)
        elif n == "rs" and el.get("type") in out:
            out[el.get("type")].append(k)
        elif n == "rs" and el.get("type") in ("work", "werk", "composition", "writing"):
            out["work"].append(k)
    # Werke stehen in dieser Edition v. a. in Kommentaren (<bibl sameAs="werke.…">) – separat erfasst
    for b in body.iter("{%s}bibl" % TEI_NS):
        s = b.get("sameAs") or ""
        if s.startswith("werke."):
            out["work"].append(s)
    return {t: sorted(set(v)) for t, v in out.items()}


def parse_file(path: Path, cfg: ExtractConfig, tree=None):
    tree = tree if tree is not None else parse_xml(path)[0]
    root = tree.getroot()
    sig = root.find(".//tei:idno[@type='signature']", NS)
    xml_id = root.get("{http://www.w3.org/XML/1998/namespace}id") or ""
    letter_id = _norm(sig.text) if sig is not None and sig.text else (xml_id or path.stem)
    url = root.find(".//tei:publicationStmt/tei:idno[@type='URLWeb']", NS)

    sent = root.find(".//tei:correspDesc/tei:correspAction[@type='sent']", NS)
    recv = root.find(".//tei:correspDesc/tei:correspAction[@type='received']", NS)
    s_act, r_act = _actors(sent), _actors(recv)
    date = _date(sent)
    year = re.search(r"\d{4}", date)
    ctx = [f"{r.get('type')}:{r.get('target')}" for r in root.findall(".//tei:correspContext/tei:ref", NS)]

    body = root.find(".//tei:text/tei:body", NS)
    paras = body_paragraphs(body, cfg) if body is not None else []
    ments = _mentions(body, cfg) if body is not None else {k: [] for k in ("person", "org", "place", "work")}
    place = sent.find("tei:placeName", NS) if sent is not None else None

    meta = {
        "letter_id": letter_id,
        "xml_id": xml_id,
        "url": _norm(url.text) if url is not None else "",
        "file": str(path.name),
        "doc_type": root.get("{%s}doctype" % TELOTA_NS) or root.get("type") or "",
        "date": date,
        "year": int(year.group()) if year else "",
        "sender_names": " | ".join(a[1] for a in s_act),
        "sender_keys": " | ".join(a[2] for a in s_act),
        "sender_types": " | ".join(a[0] for a in s_act),
        "receiver_names": " | ".join(a[1] for a in r_act),
        "receiver_keys": " | ".join(a[2] for a in r_act),
        "receiver_types": " | ".join(a[0] for a in r_act),
        "place_sent": _norm("".join(place.itertext())) if place is not None else "",
        "corresp_context": " | ".join(ctx),
        "n_words": sum(len(clean(p).split()) for p in paras),
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
    units = []
    for i, t in enumerate(texts):
        c = clean(t)
        units.append({"letter_id": letter_id, "unit_no": i + 1, "text": c,
                      "text_noent": without_entities(t), "n_words": len(c.split())})
    return meta, units


def element_inventory(tei_dir: str | Path) -> Counter:
    """Alle Elemente innerhalb von <body> zählen – zum Abgleich mit den Kodierrichtlinien."""
    c = Counter()
    for f in sorted(Path(tei_dir).glob("**/*.xml")):
        body = parse_xml(f)[0].getroot().find(".//tei:text/tei:body", NS)
        if body is not None:
            c.update(_local(e.tag) for e in body.iter() if _local(e.tag))
    return c


def run(tei_dir: str | Path, out_dir: str | Path, cfg: ExtractConfig | None = None):
    cfg = cfg or ExtractConfig()
    tei_dir, out_dir = Path(tei_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    letters, units, issues = [], [], []
    seen = set()
    for f in sorted(tei_dir.glob("**/*.xml")):
        tree, err = parse_xml(f)
        root = tree.getroot()
        if root is None or _local(root.tag) != "TEI":
            continue
        if err:
            issues.append({"file": str(f.relative_to(tei_dir)), "issue": "XML-Fehler (tolerant gelesen): " + err})
        if cfg.require_correspdesc and root.find(".//tei:correspDesc", NS) is None:
            continue
        dtype = root.get("{%s}doctype" % TELOTA_NS) or root.get("type") or ""
        if cfg.doc_types and dtype and dtype not in cfg.doc_types:
            continue
        try:
            m, u = parse_file(f, cfg, tree)
        except Exception as e:  # noqa: BLE001
            issues.append({"file": str(f.relative_to(tei_dir)), "issue": repr(e)})
            continue
        if m["letter_id"] in seen:
            issues.append({"file": str(f.relative_to(tei_dir)), "issue": f"doppelte ID {m['letter_id']}"})
            m["letter_id"] = f"{m['letter_id']} ({f.stem})"
            for x in u:
                x["letter_id"] = m["letter_id"]
        seen.add(m["letter_id"])
        letters.append(m)
        units.extend(u)

    _write(out_dir / "letters.csv", letters)
    _write(out_dir / "units.csv", units)
    _write(out_dir / "extract_issues.csv", issues or [{"file": "", "issue": ""}])
    print(f"{len(letters)} Dokumente, {len(units)} Einheiten ({cfg.unit}), {len(issues)} Hinweise -> {out_dir}")
    return letters, units


def _write(path: Path, rows: list[dict]):
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
