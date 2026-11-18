"""
BAREC pair extraction — shared utilities.

Imported by extract_pairs.py.

Provides:
  - File paths, schema constants
  - Linguistic skip-list constants (kāna sisters, impersonal verbs, quasi-quant heads)
  - Generic helpers: nfc, is_garbage_lemma, is_coord_head, parse_sync_file,
                     load_manual_excludes, build_pair_row, write_summary
  - Subject-verb pair classifier (used by both VS and SV)
"""

import csv
import unicodedata
from pathlib import Path
from collections import Counter

# ============================================================
# Paths
# ============================================================
# Corpus and output locations are passed as command-line arguments to
# extract_pairs.py; no paths are hard-coded in this module.


# ============================================================
# Linguistic constants
# ============================================================
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# Kāna and its sisters — copular/aspectual; the post-verbal NP is the topic,
# not an agreement-driving subject.
KAANA_SISTERS = {nfc(s) for s in (
    "كَان", "صَار", "أَصْبَح", "أَمْسَى", "أَضْحَى",
    "ظَلّ", "بَات", "لَيْس",
    "زَال", "دَام", "بَرِح", "ٱِنْفَكّ", "فَتِئ",
)}

# Impersonal / passive-thetic verbs — DP after V is not a real agreement subject.
IMPERSONAL_VERBS = {nfc(s) for s in (
    "أَمْكَن", "وَجَب", "يَنْبَغِي", "اِنْبَغَى", "جاز",
    "تَمّ", "حَدَث", "عَدّ", "ٱِعْتَبَر",
)}

# Quasi-prepositions and quantifier heads — actual semantic subject is the
# noun they govern (مضاف إليه), not the quantifier itself.
QUASI_QUANT_HEADS = {nfc(s) for s in (
    "نَحْو", "مِثْل", "نَفْس", "غَيْر",                            # quasi-preps
    "أَيّ", "كُلّ", "كِلَا", "كِلْتَا",                            # universal quantifiers
    "ذُو", "ذَات",                                                  # 'possessor of'
    "عَدَد", "أَحَد", "إِحْدَى", "مُعْظَم", "جَمِيع",              # quantifier heads
    "بَعْض", "كَثِير", "قَلِيل", "غَالِبِيَّة",
)}

MA_FORMS = {nfc(s) for s in ("ما", "مَا")}
COORD_PRC2_VALUES = {"wa_conj", "fa_conj"}

# ---- Subj-Pred clause heads (strict) ----
# Only base كان; sisters (لَيْسَ, أَصْبَح, زَال…) are excluded.
KAANA_LEX = nfc("كَان")
# Only أَنَّ and إِنَّ; لِأَنَّ/بِأَنَّ are encoded with proclitics in BAREC,
# so keying on lex catches them automatically. Excludes كَأَنَّ, لَٰكِنَّ, لَا.
INNA_LEX_SET = {nfc(s) for s in ("أَنَّ", "إِنَّ")}
INNA_HEAD_POS = {"conj_sub", "verb_pseudo"}
# PRD pos values accepted. Excludes:
#   noun       — nouns don't agree (lexically marked, no gen/num agreement)
#   noun_quant — quantifier-headed PRDs distort agreement validation
#   noun_num   — numeral-PRDs have their own agreement profile (tamyiz etc.)
#   adj_comp / adj_num — agreement-defective (same reason as noun-adj)
SP_PRD_POS = {"noun_prop", "adj"}

# Forms of زال (a kaana sister) get mistagged as noun_prop by MADA when in
# PRD position. The lex field stores the full multi-word analysis (e.g.,
# "مازالت") so we match on lemma instead.
SP_PRD_BLOCKED_LEMMA = {nfc("زال")}

# Quasi-prepositions tagged as nouns by MADA but functioning as PPs when
# they appear in PRD position (e.g., الكتاب لدى الرجل = "the book is with
# the man"). xpos-based filter doesn't catch these because BAREC's tagger
# assigns pos=noun. Filtered explicitly by lex.
QUASI_PREP_PRD_LEX = {nfc(s) for s in (
    "لَدَى", "عِنْد", "بَيْن", "تَحْت", "فَوْق",
    "وَرَاء", "أَمَام", "خَلْف", "حَوْل", "مَع",
    "نَحْو", "غَيْر",
)}


# ============================================================
# Haal extraction constants
# ============================================================
# Surface form stoplist (diacritics + alef variants stripped).
import re as _re
_DIACRITICS_RE = _re.compile(r"[ً-ْٰـ]")
_ALEF_TRANS = str.maketrans({
    "ٱ": "ا", "أ": "ا",
    "إ": "ا", "آ": "ا",
})

def strip_dia(s: str) -> str:
    s = nfc(s)
    s = _DIACRITICS_RE.sub("", s)
    return s.translate(_ALEF_TRANS)

# ------------------------------------------------------------
# Curated exclusion lexicons — loaded from scripts/extraction/stoplists/.
# These are open-class, corpus-curated RESOURCES, not grammar constants:
# no grammatical feature separates a manner adverb (رسمياً 'officially')
# from a true circumstantial haal, so a lexical list is the only possible
# mechanism. Extend or replace the files for other corpora. Matching is
# applied after strip_dia, so diacritization in the files is irrelevant.
# ------------------------------------------------------------
_STOPLIST_DIR = Path(__file__).resolve().parent / "stoplists"


def _load_stoplist(filename: str) -> frozenset:
    """One entry per line; '#' comments and blank lines skipped;
    first tab-separated field taken; strip_dia applied."""
    entries = []
    with (_STOPLIST_DIR / filename).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line.split("	")[0].strip())
    return frozenset(strip_dia(nfc(e)) for e in entries)


# Dictionary forms (lemmas) of words whose indefinite-accusative use with
# tanwin fath functions as a manner/domain adverbial (e.g. رسمياً
# 'officially') rather than a circumstantial haal. The candidate token's
# dictionary form is looked up in this lexicon.
HAAL_ADVERBIAL_STOPLIST = _load_stoplist("haal_adverbials.tsv")

# Event verb lemma stoplist. Diacritics stripped.
# ظنّ-class double-accusative 'consider' verbs (اعتبر، عَدّ، أَعَدّ): their
# second accusative is a predicate, not a circumstantial haal.
HAAL_VERB_LEMMA_STOPLIST = frozenset(strip_dia(s) for s in (
    "اعتبر", "عَدّ", "أَعَدّ",
))

# Active verbs that, in the haal construction, take an OBJ that the haal
# describes. Largely the closed classes of traditional grammar — أفعال
# القلوب / ظنّ وأخواتها (رأى، ظن، حسب، خال، زعم، علم، وجد) and أفعال
# التحويل (جعل، ترك، سمى) — plus modern perception/reporting verbs
# (نقل، لاحظ). When the
# event verb's lemma is one of these, the OBJ must be extractable from the
# dependency tree; if no OBJ child is present, the triple is skipped with
# reason 'transitive_obj_missing'. (اعتبر, عَدّ, أَعَدّ are already filtered
# above via HAAL_VERB_LEMMA_STOPLIST and never reach this check.)
HAAL_OBJ_REQUIRING_LEMMAS = frozenset(strip_dia(s) for s in (
    "رأى",   # see
    "وجد",   # find
    "جعل",   # make/render
    "ظن",    # think
    "حسب",   # consider
    "خال",   # imagine
    "زعم",   # claim
    "ترك",   # leave (in a state)
    "سمى",   # name
    "نقل",   # transmit/report
    "علم",   # know
    "لاحظ",  # notice
))

# Controller pos values (excludes noun_quant — quantifier heads distort).
HAAL_CONTROLLER_POS = {
    "noun", "noun_prop", "noun_num",
    "pron", "pron_rel", "pron_dem",
}

# ctrl_role labels used by the haal extractor (and downstream validation).
HAAL_DP_CTRL_ROLES = {"sbj", "grammatical_sbj"}
HAAL_PRODROP_CTRL_ROLES = {"verb_prodrop", "verb_prodrop_grammatical_sbj"}


# ============================================================
# SYNC file column map (29 columns — vox added at index 7)
# ============================================================
SYNC_COL = {
    "id": 0, "form": 1, "lemma": 2, "upos": 3, "xpos": 4,
    "per": 5, "asp": 6, "vox": 7, "mod": 8,
    "gen": 9, "form_gen": 10, "num": 11, "form_num": 12,
    "cas": 13, "rat": 14,
    "pos": 15, "lex": 16, "source": 17,
    "prc0": 18, "prc1": 19, "prc2": 20, "prc3": 21, "enc0": 22,
    "pattern": 23, "root": 24,
    "head": 25, "deprel": 26, "deps": 27, "misc": 28,
}


# ============================================================
# Output schema (pair format) — 38 columns (added target_vox/head_vox)
# ============================================================
_TOK_FIELDS = (
    "id", "form", "lemma", "upos", "xpos", "deprel", "pos",
    "per", "asp", "vox", "mod",
    "gen", "form_gen", "num", "form_num", "cas", "rat",
)

OUTPUT_COLUMNS_PAIR = (
    ["document", "domain", "sent_id"]
    + [f"target_{f}" for f in _TOK_FIELDS]
    + [f"head_{f}"   for f in _TOK_FIELDS]
    + ["skip_reason"]
)

# 15 placeholder columns for downstream rule validation (filled by
# scripts/validation/validate_pairs.py, NOT by extraction).
PRED_COLUMNS = [
    "pred_gen_form", "pred_num_form", "match_form",
    "pred_gen_func", "pred_num_func", "match_func",
    "pred_gen_rule", "pred_num_rule", "match_rule",
    "pred_gen_form_rule", "pred_num_form_rule", "match_form_rule",
    "pred_gen_extended_rule", "pred_num_extended_rule", "match_extended_rule",
]

# Noun-Adj output schema — base + PRED_COLUMNS + skip_reason.
OUTPUT_COLUMNS_PAIR_NA = (
    ["document", "domain", "sent_id"]
    + [f"target_{f}" for f in _TOK_FIELDS]
    + [f"head_{f}"   for f in _TOK_FIELDS]
    + PRED_COLUMNS
    + ["skip_reason"]
)

# Demonstrative output schema — same as Noun-Adj.
OUTPUT_COLUMNS_PAIR_DEM = (
    ["document", "domain", "sent_id"]
    + [f"target_{f}" for f in _TOK_FIELDS]
    + [f"head_{f}"   for f in _TOK_FIELDS]
    + PRED_COLUMNS
    + ["skip_reason"]
)

# Demonstrative valid lemma allowlist (standard MSA demonstratives).
# Diacritics stripped before comparison. Any pron_dem token whose lemma is
# NOT in this set is dropped with reason 'non_standard_dem' (catches MADA
# mistags like ها, هات, etc.).
DEM_VALID_LEMMAS = frozenset(strip_dia(s) for s in (
    "هذا", "ذلك", "هذه", "تلك", "هؤلاء",
    "ذاك", "ذلكم", "ذينك",
    "هاتان", "هاتين", "هذان", "هذين",
    "أولاء", "أولئك", "أولائك",
))

# ============================================================
# Number-Noun extraction constants
# ============================================================

# Decades (عقود) 20-90 — stripped of diacritics for comparison.
DECADE_LEMMAS_STRIPPED = frozenset(strip_dia(s) for s in (
    "عشرون", "ثلاثون", "أربعون", "خمسون",
    "ستون", "سبعون", "ثمانون", "تسعون",
))

# عشر / عشرة — the ten-part of compound 11-19, OR standalone 10.
ASHAR_LEMMA_STRIPPED = strip_dia("عشر")

# Units 1-9: stripped lemma → numeric value (also covers common variant lemmas).
UNIT_LEMMA_TO_VAL: dict[str, int] = {
    strip_dia(s): v for s, v in {
        "واحد": 1, "أحد": 1, "إحدى": 1,
        "ٱثن": 2,  "اثن": 2,                   # variant MADA lemmas for 2
        "ثلاث": 3,
        "أربع": 4,
        "خمس": 5,
        "ست": 6,   "سِت": 6,
        "سبع": 7,
        "ثماني": 8, "ثمان": 8, "ثمانية": 8,
        "تسع": 9,
    }.items()
}

# Form-based fallback for units (some MADA lemma fields are corrupted).
UNIT_FORM_TO_VAL: dict[str, int] = {
    strip_dia(s): v for s, v in {
        "واحد": 1, "الواحد": 1, "واحدة": 1, "الواحدة": 1,
        "أحد": 1, "إحدى": 1,
        "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2,
        "اثنا": 2, "اثنتا": 2,
        "ثلاثة": 3, "ثلاث": 3,
        "أربعة": 4, "أربع": 4,
        "خمسة": 5, "خمس": 5,
        "ستة": 6,  "ست": 6,
        "سبعة": 7, "سبع": 7,
        "ثمانية": 8, "ثماني": 8, "ثمان": 8,
        "تسعة": 9, "تسع": 9,
    }.items()
}


def cardinal_value(tok: dict) -> int | None:
    """Return numeric value 1-10 of a noun_num token, or None."""
    v = UNIT_LEMMA_TO_VAL.get(strip_dia(tok.get("lemma", "")))
    if v is not None:
        return v
    v = UNIT_FORM_TO_VAL.get(strip_dia(tok.get("form", "")))
    if v is not None:
        return v
    if strip_dia(tok.get("lemma", "")) == ASHAR_LEMMA_STRIPPED:
        return 10
    if strip_dia(tok.get("form", "")) in (strip_dia("عشر"), strip_dia("عشرة")):
        return 10
    return None


# Fields carried on the second number token (عشر/عشرة for 11-19; decade for 21-99)
_SECOND_FIELDS = ("form", "lemma", "pos", "gen", "form_gen", "num", "form_num")

# Ordinal head pos values
ORDINAL_HEAD_POS = frozenset({"noun", "noun_prop", "noun_quant"})

# Ordinal compounds: decades 20-90 (used as the second part of ordinal compounds)
ORDINAL_UQUD_LEMMAS_STRIPPED = DECADE_LEMMAS_STRIPPED
ORDINAL_ASHAR_LEMMA_STRIPPED  = ASHAR_LEMMA_STRIPPED

# ---- Output schemas ----

# Simple schema: 1_2 and 3_10 — no second_* columns.
OUTPUT_COLUMNS_NUM_SIMPLE = (
    ["document", "domain", "sent_id"]
    + [f"target_{f}" for f in _TOK_FIELDS]
    + [f"head_{f}"   for f in _TOK_FIELDS]
    + PRED_COLUMNS
    + ["skip_reason"]
)

# Compound schema: 11_12, 13_19, 20_90_compounds, ordinal.
# Column order mirrors surface order: target → second → head (matches the
# Arabic linear order: units عشر counted_noun / unit و-decade counted_noun).
OUTPUT_COLUMNS_NUM_COMPOUND = (
    ["document", "domain", "sent_id"]
    + [f"target_{f}" for f in _TOK_FIELDS]
    + [f"second_{f}" for f in _SECOND_FIELDS]
    + [f"head_{f}"   for f in _TOK_FIELDS]
    + ["subtype_detail"]
    + PRED_COLUMNS
    + ["skip_reason"]
)


# ---- Structural helpers ----

def num_find_children(tok: dict, tokens: dict):
    """Return children of tok grouped by role."""
    tid = tok["id"]
    kids = [c for c in tokens.values() if c.get("head") == tid]
    return {
        "ashar":    [k for k in kids if k["pos"] == "noun_num"
                     and k["deprel"] == "---"],
        "tmz":      [k for k in kids if k["deprel"] == "TMZ"
                     and k["pos"] not in ("noun_num", "digit", "adj_num")],
        "idf":      [k for k in kids if k["deprel"] == "IDF"
                     and k["pos"] not in ("noun_num", "digit", "adj_num")],
        "obj_num":  [k for k in kids if k["deprel"] == "OBJ"
                     and k["pos"] == "noun_num"],          # units of 21-99
        "idf_num":  [k for k in kids if k["deprel"] == "IDF"
                     and k["pos"] in ("noun_num", "digit")],  # millions structure
        "all":      kids,
    }


def num_classify_skip(N: dict, NUM: dict, doc: str, sent_id: str,
                      tokens: dict, manual_excludes: set):
    """Quality-gate on the counted noun N and the number NUM.
    Returns (action, reason) — same drop/flag pattern as dem/rel."""
    if is_coord_head(N, tokens):
        return "drop", "coordination"
    if N.get("deprel") == "PRD":
        return "drop", "head_prd"
    h_lex = nfc(N.get("lex", ""))
    if h_lex in QUASI_QUANT_HEADS:
        return "drop", "quasi_quant_head"
    key = (doc, sent_id, N["id"], NUM["id"])
    if key in manual_excludes:
        return "flag", "manual_exclude"
    if is_garbage_lemma(N["lemma"]) or is_garbage_lemma(NUM["lemma"]):
        return "flag", "garbage_lemma"
    if N["rat"] not in ("i", "r"):
        return "flag", "bad_rat"
    return "keep", ""


def num_build_simple_row(document, domain, sent_id, target, head,
                         skip_reason):
    """Build a row for OUTPUT_COLUMNS_NUM_SIMPLE."""
    row = build_pair_row(document, domain, sent_id, target, head, skip_reason)
    for col in PRED_COLUMNS:
        row[col] = ""
    return row


def num_build_compound_row(document, domain, sent_id, target, head,
                            second, subtype_detail, skip_reason):
    """Build a row for OUTPUT_COLUMNS_NUM_COMPOUND.

    Column order: target_* → second_* → head_* (matches Arabic surface order).
    """
    row = {"document": document, "domain": domain, "sent_id": sent_id,
           "subtype_detail": subtype_detail, "skip_reason": skip_reason}
    for f in _TOK_FIELDS:
        row[f"target_{f}"] = target.get(f, "")
    for f in _SECOND_FIELDS:
        row[f"second_{f}"] = second.get(f, "") if second is not None else ""
    for f in _TOK_FIELDS:
        row[f"head_{f}"] = head.get(f, "")
    for col in PRED_COLUMNS:
        row[col] = ""
    return row

# Relative pronouns — invariable forms that don't show agreement.
# Caught with skip_reason='invariable_rel'.
REL_INVARIABLE_FORMS = frozenset(strip_dia(s) for s in (
    "ما", "من", "مهما", "أي", "أيا", "أية", "ماذا",
))

# Canonical agreement-bearing relative pronouns (form allowlist).
# Anything outside this set (and not in REL_INVARIABLE_FORMS) is dropped
# with reason 'non_standard_rel' (catches MADA mistags / rare variants).
REL_VALID_FORMS = frozenset(strip_dia(s) for s in (
    "الذي", "التي",
    "اللذان", "اللذين",
    "اللتان", "اللتين",
    "الذين",
    "اللاتي", "اللائي", "اللواتي",
))

# Relative output schema — same as Noun-Adj.
OUTPUT_COLUMNS_PAIR_REL = (
    ["document", "domain", "sent_id"]
    + [f"target_{f}" for f in _TOK_FIELDS]
    + [f"head_{f}"   for f in _TOK_FIELDS]
    + PRED_COLUMNS
    + ["skip_reason"]
)

# Subj-Pred output schema — adds clause_type + clause_head_lemma metadata.
OUTPUT_COLUMNS_PAIR_SP = (
    ["document", "domain", "sent_id"]
    + [f"target_{f}" for f in _TOK_FIELDS]
    + [f"head_{f}"   for f in _TOK_FIELDS]
    + ["clause_type", "clause_head_lemma"]
    + PRED_COLUMNS
    + ["skip_reason"]
)

# Haal output schema.
# Used for BOTH haal_triples.tsv (only keepers, skip_reason empty) and
# haal_skipped.tsv (all excluded cases, skip_reason populated).
# PRED columns left empty for validation to fill.
OUTPUT_COLUMNS_HAAL = (
    ["document", "domain", "sent_id"]
    + ["haal_id", "haal_form", "haal_lemma",
       "haal_gen", "haal_form_gen", "haal_num", "haal_form_num"]
    + ["ctrl_id", "ctrl_pos", "ctrl_role", "ctrl_form", "ctrl_lemma",
       "ctrl_gen", "ctrl_form_gen", "ctrl_num", "ctrl_form_num", "ctrl_rat"]
    + ["verb_id", "verb_xpos", "verb_form", "verb_lemma",
       "verb_gen", "verb_form_gen", "verb_num", "verb_form_num"]
    + PRED_COLUMNS
    + ["skip_reason"]
)


# ============================================================
# Helpers
# ============================================================
def is_garbage_lemma(lemma: str) -> bool:
    """Latin char in Arabic lemma = MADA glitch."""
    return any(c.isascii() and c.isalpha() for c in lemma)


def is_coord_head(N: dict, tokens: dict) -> bool:
    """N is a coord head if any noun-like child has prc2 in coord set."""
    nid = N["id"]
    for child in tokens.values():
        if child["head"] == nid:
            if child["pos"].startswith("noun") and child.get("prc2", "0") in COORD_PRC2_VALUES:
                return True
    return False


def parse_sync_file(path: Path):
    """Yield (sent_id, tokens_dict) per sentence."""
    cur_sent_id = None
    cur_tokens: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_tokens:
                    yield cur_sent_id, cur_tokens
                    cur_sent_id = None
                    cur_tokens = {}
                continue
            if line.startswith("# sent_id ="):
                cur_sent_id = line.split("=", 1)[1].strip()
                continue
            if line.startswith("#") or line.startswith("ID\t"):
                continue
            cols = line.split("\t")
            if len(cols) < 29:
                continue
            tok = {k: cols[idx] for k, idx in SYNC_COL.items()}
            cur_tokens[tok["id"]] = tok
    if cur_tokens:
        yield cur_sent_id, cur_tokens


def load_manual_excludes(path: Path) -> set:
    """Load manual exclude tuples (document, sent_id, head_id, target_id)."""
    s: set = set()
    if not path.exists():
        return s
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                s.add((parts[0], parts[1], parts[2], parts[3]))
    return s


def build_pair_row(document: str, domain: str, sent_id: str,
                   target: dict, head: dict, skip_reason: str = "") -> dict:
    """Build an output row dict matching OUTPUT_COLUMNS_PAIR."""
    row = {
        "document": document,
        "domain": domain,
        "sent_id": sent_id,
        "skip_reason": skip_reason,
    }
    for prefix, tok in (("target", target), ("head", head)):
        for f in _TOK_FIELDS:
            row[f"{prefix}_{f}"] = tok[f]
    return row


def write_summary(
    construction: str,
    out_path: Path,
    files_count: int,
    candidates: int,
    pairs_written: int,
    drop_counts: Counter,
    flag_counts: Counter,
    domain_clean: Counter,
) -> str:
    """Format and write extraction summary. Return summary text."""
    total_dropped = sum(drop_counts.values())
    total_flagged = sum(flag_counts.values())
    clean = pairs_written - total_flagged

    lines = [
        "=" * 64,
        f"BAREC {construction} extraction summary",
        "=" * 64,
        f"  SYNC files processed             : {files_count}",
        f"  Candidate pairs                  : {candidates}",
        f"  Dropped entirely                 : {total_dropped}",
        f"  Pairs written to file            : {pairs_written}",
        f"    of which flagged (skip_reason) : {total_flagged}",
        f"    of which clean (no skip_reason): {clean}",
        "",
        "Drop reasons (NOT in output file):",
    ]
    for r, n in drop_counts.most_common():
        lines.append(f"  {r:30s} {n}")
    lines.append("")
    lines.append("Flag reasons (in file with skip_reason):")
    for r, n in flag_counts.most_common():
        lines.append(f"  {r:30s} {n}")
    lines.append("")
    lines.append("Clean pair counts per domain:")
    for d, n in sorted(domain_clean.items(), key=lambda x: -x[1]):
        lines.append(f"  {d:25s} {n}")

    text = "\n".join(lines)
    out_path.write_text(text + "\n", encoding="utf-8")
    return text


# ============================================================
# Subject-Verb pair classifier (shared by VS and SV)
# ============================================================
def classify_subject_verb_pair(
    N: dict, V: dict, doc: str, sent_id: str,
    tokens: dict, manual_excludes: set,
):
    """Skip taxonomy for VS/SV pairs (head=N noun, target=V verb).

    Returns (action, reason) where action ∈ {"drop", "flag", "keep"}.
    """
    # ---- Drop rules ----
    if is_coord_head(N, tokens):
        return "drop", "coordination"
    v_lex = nfc(V.get("lex", ""))
    if v_lex in KAANA_SISTERS:
        return "drop", "kaana_sister"
    if V.get("per") in ("1", "2"):
        return "drop", "first_second_person_verb"
    if v_lex in IMPERSONAL_VERBS:
        return "drop", "impersonal_verb"
    if nfc(V["form"]) in MA_FORMS:
        return "drop", "ma_verb"
    h_lex = nfc(N.get("lex", ""))
    if h_lex in QUASI_QUANT_HEADS:
        return "drop", "quasi_quant_head"

    # ---- Flag rules ----
    key = (doc, sent_id, N["id"], V["id"])
    if key in manual_excludes:
        return "flag", "manual_exclude"
    if is_garbage_lemma(N["lemma"]) or is_garbage_lemma(V["lemma"]):
        return "flag", "garbage_lemma"
    if N["rat"] not in ("i", "r"):
        return "flag", "bad_rat"

    return "keep", ""


# ============================================================
# Demonstrative pair classifier
# ============================================================
def classify_dem_pair(
    N: dict, D: dict, doc: str, sent_id: str,
    tokens: dict, manual_excludes: set,
):
    """Skip taxonomy for Dem-Noun pairs (head=N noun, target=D demonstrative).

    Structural rules plus a demonstrative-lemma allowlist.
    Note: dual_head_singular_dem uses LEXICAL num (not form_num) because
    BAREC's MADA gives form_num='s' for ALL demonstratives regardless of
    actual number.

    Drops:
      coordination, head_prd, dual_head_singular_dem, quasi_quant_head,
      non_standard_dem (dem lemma not in DEM_VALID_LEMMAS)
    Flags:
      manual_exclude, garbage_lemma, bad_rat
    """
    # ---- Drop rules ----
    if is_coord_head(N, tokens):
        return "drop", "coordination"
    if N.get("deprel") == "PRD":
        return "drop", "head_prd"
    # Dual head + singular dem (using lexical num — MADA's form_num is unreliable
    # for demonstratives, always reported as 's').
    if N.get("num") == "d" and D.get("num") == "s":
        return "drop", "dual_head_singular_dem"
    h_lex = nfc(N.get("lex", ""))
    if h_lex in QUASI_QUANT_HEADS:
        return "drop", "quasi_quant_head"
    # Allowlist on demonstrative lemma — catches MADA mistags (ها, هات, etc.)
    if strip_dia(D.get("lemma", "")) not in DEM_VALID_LEMMAS:
        return "drop", "non_standard_dem"

    # ---- Flag rules ----
    key = (doc, sent_id, N["id"], D["id"])
    if key in manual_excludes:
        return "flag", "manual_exclude"
    if is_garbage_lemma(N["lemma"]) or is_garbage_lemma(D["lemma"]):
        return "flag", "garbage_lemma"
    if N["rat"] not in ("i", "r"):
        return "flag", "bad_rat"

    return "keep", ""


# ============================================================
# Relative pronoun pair classifier
# ============================================================
def classify_rel_pair(
    N: dict, R: dict, doc: str, sent_id: str,
    tokens: dict, manual_excludes: set,
):
    """Skip taxonomy for Rel-Noun pairs (head=N noun, target=R relative pronoun).

    Mirrors the dem classifier. Note: dual_head_singular_rel uses LEXICAL num
    because BAREC's MADA gives form_num='s' for ALL relative pronouns.

    Drops:
      coordination, head_prd, dual_head_singular_rel, quasi_quant_head,
      invariable_rel (form in {ما, من, مهما, أي, أيا, أية, ماذا}),
      non_standard_rel (form outside REL_VALID_FORMS allowlist)
    Flags:
      manual_exclude, garbage_lemma, bad_rat
    """
    # ---- Drop rules ----
    if is_coord_head(N, tokens):
        return "drop", "coordination"
    if N.get("deprel") == "PRD":
        return "drop", "head_prd"
    # Dual head + singular rel — use LEXICAL num (form_num unreliable for rels)
    if N.get("num") == "d" and R.get("num") == "s":
        return "drop", "dual_head_singular_rel"
    h_lex = nfc(N.get("lex", ""))
    if h_lex in QUASI_QUANT_HEADS:
        return "drop", "quasi_quant_head"
    r_form_stripped = strip_dia(R.get("form", ""))
    if r_form_stripped in REL_INVARIABLE_FORMS:
        return "drop", "invariable_rel"
    if r_form_stripped not in REL_VALID_FORMS:
        return "drop", "non_standard_rel"

    # ---- Flag rules ----
    key = (doc, sent_id, N["id"], R["id"])
    if key in manual_excludes:
        return "flag", "manual_exclude"
    if is_garbage_lemma(N["lemma"]) or is_garbage_lemma(R["lemma"]):
        return "flag", "garbage_lemma"
    if N["rat"] not in ("i", "r"):
        return "flag", "bad_rat"

    return "keep", ""


# ============================================================
# Subj-Pred helpers
# ============================================================
def sp_clause_type(H: dict) -> str | None:
    """Return 'kaana' | 'inna' | None for the predication head."""
    h_lex = nfc(H.get("lex", ""))
    if H.get("pos") == "verb" and h_lex == KAANA_LEX:
        return "kaana"
    if H.get("pos") in INNA_HEAD_POS and h_lex in INNA_LEX_SET:
        return "inna"
    return None


def classify_subj_pred_pair(
    S: dict, P: dict, doc: str, sent_id: str,
    tokens: dict, manual_excludes: set,
):
    """Skip taxonomy for Subj-Pred pairs (head=S subject noun, target=P predicate).

    Drops:
      coordination          — SBJ-side coord head
      prd_coord_non_first   — non-first conjunct in PRD-side coordination
      quasi_quant_head      — SBJ lex is a quantifier/quasi-prep head
      dual_head_singular_target — SBJ dual + adjectival PRD singular
                                  (only applied to adj-PRD; noun-PRD agreement
                                  is laxer)
    Flags:
      manual_exclude, garbage_lemma, bad_rat
    """
    # ---- Drop rules ----
    if is_coord_head(S, tokens):
        return "drop", "coordination"
    if P.get("prc2", "0") in COORD_PRC2_VALUES:
        return "drop", "prd_coord_non_first"
    s_lex = nfc(S.get("lex", ""))
    if s_lex in QUASI_QUANT_HEADS:
        return "drop", "quasi_quant_head"
    if nfc(P.get("lex", "")) in QUASI_PREP_PRD_LEX:
        return "drop", "quasi_prep_prd"
    if nfc(P.get("lemma", "")) in SP_PRD_BLOCKED_LEMMA:
        return "drop", "kaana_sister_as_prd"
    if (P.get("pos") == "adj"
            and S.get("num") == "d" and P.get("num") == "s"):
        return "drop", "dual_head_singular_target"

    # ---- Flag rules ----
    key = (doc, sent_id, S["id"], P["id"])
    if key in manual_excludes:
        return "flag", "manual_exclude"
    if is_garbage_lemma(S["lemma"]) or is_garbage_lemma(P["lemma"]):
        return "flag", "garbage_lemma"
    if S["rat"] not in ("i", "r"):
        return "flag", "bad_rat"

    return "keep", ""


# ============================================================
# Noun-Adjective pair classifier
# ============================================================
def classify_noun_adj_pair(
    N: dict, A: dict, doc: str, sent_id: str,
    tokens: dict, manual_excludes: set,
):
    """Skip taxonomy for Noun-Adj pairs (head=N noun, target=A adjective).

    Subset of the SV/VS taxonomy: verb-side rules (kaana sisters, impersonal,
    person filter, ma_verb) are dropped; noun-side rules (coord head,
    quasi-quant head, garbage_lemma, bad_rat, manual_exclude) are kept.

    Adj-side coordination (e.g., الجديد والمفيد): non-first conjuncts are
    dropped entirely — only the first conjunct adj is extracted as the pair.

    Returns (action, reason) where action ∈ {"drop", "flag", "keep"}.
    """
    # ---- Drop rules ----
    if is_coord_head(N, tokens):
        return "drop", "coordination"
    if A.get("prc2", "0") in COORD_PRC2_VALUES:
        return "drop", "adj_coord_non_first"
    if N.get("deprel") == "PRD":
        # Head noun is itself a predicate — adj here is a modifier inside a
        # predicate NP, not the canonical attributive case we want to validate.
        return "drop", "head_prd"
    if N.get("num") == "d" and A.get("num") == "s":
        # Dual head with singular adj — almost always a tagging artifact
        # where a coordinated second adj was dropped or mis-attached.
        return "drop", "dual_head_singular_adj"
    h_lex = nfc(N.get("lex", ""))
    if h_lex in QUASI_QUANT_HEADS:
        return "drop", "quasi_quant_head"

    # ---- Flag rules ----
    key = (doc, sent_id, N["id"], A["id"])
    if key in manual_excludes:
        return "flag", "manual_exclude"
    if is_garbage_lemma(N["lemma"]) or is_garbage_lemma(A["lemma"]):
        return "flag", "garbage_lemma"
    if N["rat"] not in ("i", "r"):
        return "flag", "bad_rat"

    return "keep", ""


# ============================================================
# Haal extraction helpers
# ============================================================
def haal_is_definite(A: dict) -> bool:
    """BAREC encodes definiteness in prc0 (Al_det) rather than xpos."""
    return A.get("prc0", "0") == "Al_det"


def haal_is_passive(V: dict) -> bool:
    """BAREC stores voice in the vox column (a=active, p=passive)."""
    return V.get("vox", "") == "p"


def haal_event_verb_under_kaana(V: dict, tokens: dict) -> bool:
    """True iff event verb is PRD child of a kaana-family auxiliary —
    those triples belong to subj_pred, not haal."""
    if V.get("deprel") != "PRD":
        return False
    vhid = V.get("head", "0")
    if vhid not in tokens:
        return False
    par = tokens[vhid]
    return par.get("pos") == "verb" and nfc(par.get("lex", "")) in KAANA_SISTERS


def _find_role_child(V: dict, tokens: dict, deprel: str) -> dict | None:
    """Find first child of V with the given deprel and controller-eligible pos."""
    vid = V["id"]
    for child in tokens.values():
        if child.get("head") != vid:
            continue
        if child.get("deprel") != deprel:
            continue
        cpos = child.get("pos", "")
        if cpos == "noun_quant":
            continue
        if cpos in HAAL_CONTROLLER_POS:
            return child
    return None


def haal_find_sbj_child(V: dict, tokens: dict) -> dict | None:
    """SBJ child of V (excludes noun_quant)."""
    return _find_role_child(V, tokens, "SBJ")


def haal_find_obj_child(V: dict, tokens: dict) -> dict | None:
    """OBJ child of V (excludes noun_quant). Used for 1st/2nd person active
    verbs where the haal typically modifies the object (رأيتُ الولدَ ضاحكاً)."""
    return _find_role_child(V, tokens, "OBJ")


def haal_make_prodrop(V: dict, sync_with_haal: dict | None = None) -> dict:
    """Synthetic prodrop controller: copies the verb's own form_gen /
    form_num / gen / num (the verb's morphology reflects the dropped
    pronominal subject). If `sync_with_haal` is given, the haal's features
    are copied instead (not used by the current extraction flow).
    """
    if sync_with_haal is not None:
        H = sync_with_haal
        return {
            "id": "", "form": "Ø", "lemma": "",
            "pos": "pro",
            "gen": H.get("gen", ""),
            "form_gen": H.get("form_gen", ""),
            "num": H.get("num", ""),
            "form_num": H.get("form_num", ""),
            "rat": V.get("rat", ""),
        }
    return {
        "id": "", "form": "Ø", "lemma": "",
        "pos": "pro",
        "gen": V.get("gen", ""),
        "form_gen": V.get("form_gen", ""),
        "num": V.get("num", ""),
        "form_num": V.get("form_num", ""),
        "rat": V.get("rat", ""),
    }


def haal_agrees(H: dict, C: dict) -> bool:
    """Strict agreement: both form_gen AND form_num must match haal's.

    Additionally, if both haal and ctrl have informative underlying gen/num
    (not na/u), those must match too. This catches MADA-inconsistent verb
    analyses where form_gen=m surfaces over a lexical gen=f (e.g., أتت
    tagged 3rd-fem on gen but masc on form_gen) — these are tagger errors
    that would otherwise sneak through a form_gen-only check.
    """
    if C.get("form_gen") in ("na", "u") or C.get("form_num") in ("na", "u"):
        return False
    if H["form_gen"] != C["form_gen"] or H["form_num"] != C["form_num"]:
        return False
    # Underlying-feature check: only enforced when both sides are informative
    h_gen, c_gen = H.get("gen", ""), C.get("gen", "")
    if h_gen not in ("", "na", "u") and c_gen not in ("", "na", "u"):
        if h_gen != c_gen:
            return False
    h_num, c_num = H.get("num", ""), C.get("num", "")
    if h_num not in ("", "na", "u") and c_num not in ("", "na", "u"):
        if h_num != c_num:
            return False
    return True


def haal_pick_controller(H: dict, V: dict,
                          sbj: dict | None, obj: dict | None = None,
                          enforce_agreement: bool = True):
    """Returns (ctrl_dict, ctrl_role, skip_reason).

    Controller policy:

      OBJ-preference is restricted to perception/cognition verbs (lemma in
      HAAL_OBJ_REQUIRING_LEMMAS). For other transitive verbs (e.g., رفض)
      the haal typically describes the SBJ, so we use SBJ-first even if an
      OBJ child exists.

      Prodrop controllers take their features from the verb's own
      morphology (the verb reflects the dropped pronominal subject);
      prodrops with uninformative features are skipped.

      Passive transitive verbs (lemma in HAAL_OBJ_REQUIRING_LEMMAS) with no
      SBJ child also skip as 'transitive_obj_missing' — the semantic OBJ
      should be the grammatical SBJ but isn't there.

    Flow:
      passive verb:
        SBJ exists + agrees     → use SBJ (role=grammatical_sbj)
        SBJ exists + disagrees  → skip 'ctrl_disagrees'
        no SBJ, lemma in OBJ_REQUIRING → skip 'transitive_obj_missing'
        no SBJ                  → prodrop:
            form_gen/num in {u,na} → skip 'uninformative_prodrop_feats'
            disagrees              → skip 'ctrl_disagrees'
            else                    → keep
      active verb, lemma in OBJ_REQUIRING (perception/cognition):
        OBJ exists + agrees       → use OBJ (role=obj)
        OBJ exists + disagrees + SBJ agrees → use SBJ
        OBJ exists, neither agrees → skip 'ctrl_disagrees' (OBJ)
        no OBJ                     → skip 'transitive_obj_missing'
      active verb, lemma NOT in OBJ_REQUIRING:
        SBJ exists + agrees       → use SBJ
        SBJ exists + disagrees    → skip 'ctrl_disagrees'
        no SBJ                    → prodrop (as above)
    """
    passive = haal_is_passive(V)
    v_lemma_stripped = strip_dia(V.get("lemma", ""))
    obj_required    = v_lemma_stripped in HAAL_OBJ_REQUIRING_LEMMAS

    sbj_role     = "grammatical_sbj"               if passive else "sbj"
    prodrop_role = "verb_prodrop_grammatical_sbj"  if passive else "verb_prodrop"

    def _prodrop_with_checks():
        """Build prodrop ctrl from verb; apply strict uninformative + agreement.
        Used uniformly for all persons."""
        C = haal_make_prodrop(V)
        # Strict uninformative: any of gen/form_gen/num/form_num in {u,na}
        if (C.get("gen") in ("u", "na")
                or C.get("form_gen") in ("u", "na")
                or C.get("num") in ("u", "na")
                or C.get("form_num") in ("u", "na")):
            return C, prodrop_role, "uninformative_prodrop_feats"
        if enforce_agreement and not haal_agrees(H, C):
            return C, prodrop_role, "ctrl_disagrees"
        return C, prodrop_role, ""

    # ---- Passive verb ----
    if passive:
        if sbj is not None:
            if not enforce_agreement or haal_agrees(H, sbj):
                return sbj, sbj_role, ""
            return sbj, sbj_role, "ctrl_disagrees"
        if obj_required:
            return haal_make_prodrop(V), prodrop_role, "transitive_obj_missing"
        return _prodrop_with_checks()

    # ---- Active perception/cognition verb: OBJ-first ----
    if obj_required:
        if obj is not None:
            if not enforce_agreement:
                return obj, "obj", ""
            if haal_agrees(H, obj):
                return obj, "obj", ""
            if sbj is not None and haal_agrees(H, sbj):
                return sbj, "sbj", ""
            return obj, "obj", "ctrl_disagrees"
        return haal_make_prodrop(V), prodrop_role, "transitive_obj_missing"

    # ---- Active non-perception verb: SBJ-first ----
    if sbj is not None:
        if not enforce_agreement or haal_agrees(H, sbj):
            return sbj, "sbj", ""
        return sbj, "sbj", "ctrl_disagrees"

    # No SBJ — prodrop with uniform check (no per-based exceptions)
    return _prodrop_with_checks()


def classify_haal_pre(
    H: dict, V: dict, doc: str, sent_id: str,
    tokens: dict, manual_excludes: set,
):
    """Pre-controller checks on haal H + event verb V.

    Returns (action, reason) where action ∈ {"skip", "keep"}.
    Every excluded case carries a skip_reason and goes to haal_skipped.tsv —
    no silent drops.
    """
    if haal_is_definite(H):
        return "skip", "definite_haal"
    try:
        if int(H["id"]) <= int(V["id"]):
            return "skip", "haal_before_verb"
    except ValueError:
        return "skip", "haal_before_verb"
    if H.get("form_gen") in ("na", "u") or H.get("form_num") in ("na", "u"):
        return "skip", "na_form_feats"
    if H.get("num") == "d":
        # Dual haal — excluded (agreement profile differs).
        return "skip", "dual_haal"
    lemma_stripped = strip_dia(H.get("lemma", ""))
    if lemma_stripped in HAAL_ADVERBIAL_STOPLIST:
        return "skip", "adverbial_haal"
    if haal_event_verb_under_kaana(V, tokens):
        return "skip", "verb_prd_of_kaana"
    if strip_dia(V.get("lemma", "")) in HAAL_VERB_LEMMA_STOPLIST:
        return "skip", "verb_is_itabar"
    # Event verb itself is a kaana-family copula (صار، ليس، أصبح، ظل،…).
    # The post-verbal "haal-like" element is actually PRD of the copula, not
    # a haal of an action verb. Belongs in subj_pred, not haal.
    if nfc(V.get("lex", "")) in KAANA_SISTERS:
        return "skip", "verb_is_kaana_sister"
    # MADA sometimes mistags definite proper nouns as verbs (e.g., الارقم).
    # Real verbs never carry the Al- proclitic. Catch via either prc0=Al_det
    # or surface form starting with ال/ٱل (MADA doesn't always split out the
    # proclitic on mistagged words).
    v_form = nfc(V.get("form", ""))
    if V.get("prc0", "0") == "Al_det" \
            or v_form.startswith("ال") or v_form.startswith("ٱل") \
            or v_form.startswith("الْ"):
        return "skip", "verb_definite_form"
    if is_garbage_lemma(H["lemma"]) or is_garbage_lemma(V["lemma"]):
        return "skip", "garbage_lemma"
    # Manual excludes route to plural_haal_wrong_ctrl.
    key = (doc, sent_id, V["id"], H["id"])
    if key in manual_excludes:
        return "skip", "plural_haal_wrong_ctrl"
    return "keep", ""


def classify_haal_post(C: dict, ctrl_role: str):
    """Post-controller checks. Returns (action, reason).

    unknown_ctrl_rat — controller is a DP (sbj/grammatical_sbj) and its
                       rat is 'u' (unknown). Verb-prodrop controllers are
                       exempt (their 'rat' comes from the verb and is not
                       meaningful for haal agreement).
    """
    if ctrl_role in HAAL_DP_CTRL_ROLES and C.get("rat") == "u":
        return "skip", "unknown_ctrl_rat"
    return "keep", ""
