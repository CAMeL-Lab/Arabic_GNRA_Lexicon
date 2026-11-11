"""
Agreement rule validation for extracted controller-target pairs.

One script, all nine constructions:

  --construction na    Noun-Adjective            <- noun_adj_pairs.tsv
  --construction sv    Subject-Verb              <- sv_pairs.tsv
  --construction vs    Verb-Subject              <- vs_pairs.tsv
  --construction dem   Demonstrative-Noun        <- dem_pairs.tsv
  --construction rel   Noun-Relative pronoun     <- rel_pairs.tsv
  --construction sp    Subject-Predicate         <- subj_pred_pairs.tsv
  --construction haal  Haal                      <- haal_triples.tsv
  --construction num   Numeral-Noun (all bands)  <- num_noun_{band}_pairs.tsv
  --construction ord   Ordinal-Noun              <- ordinal_noun_pairs.tsv
  --construction all   everything above

The validator is a generic matching engine driven entirely by the rule
inventory (``rules/rule_inventory.tsv``). For every pair and every rule set,
the rules whose input conditions (controller form features, functional
features, AGG value, numeral band) match the controller contribute their
``output`` to the licensed set; the pair is *matched* if the target's
observed FUNCTIONAL gender-number is in the licensed set. 

Grading policy: all constructions are graded against the target's
functional features, uniformly across the five rule sets.

Extended system extras (Extended only):
  * AGG-conditioned rules (``agg_value`` G/D) fire when the controller lemma
    carries that AGG value in the lexicon given via --agg-lexicon
    (TSV with a lemma column [lemma/lemma_ar] and an AGG column
    [agg/aggregation/AGR], values A/C/G/D). Without --agg-lexicon these rules
    never fire.
  * Dual-gender (common-gender) nouns: a lexical lookup against
    --dual-gender (default: rules/dual_gender_nouns.tsv). If the controller
    is in the list, rule matching is repeated with each licensed functional
    gender and the licensed outputs are unioned.

Outputs (in --out):
  <pairs-file-stem>_validated.tsv  input rows + pred_*/match_* columns per
                                   rule set (matched / mismatched / na)
  accuracy_summary.tsv             construction x band x rule set accuracies
                                   (accuracy = matched / (matched+mismatched);
                                   'na' rows are out of scope, e.g. the
                                   gender-less 'uqud decades)
"""

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = REPO_ROOT / "rules" / "rule_inventory.tsv"
DEFAULT_DUAL = REPO_ROOT / "rules" / "dual_gender_nouns.tsv"

RULE_SETS = [
    ("form_match", "Form Match"),
    ("func_match", "Func Match"),
    ("form_rules", "Form Rules"),
    ("func_rules", "Func Rules"),
    ("extended", "Extended"),
]
EXTENDED_KEY = "extended"

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
_DIA_RE = re.compile(r"[ً-ْٰـ]")  # harakat, dagger alef, tatweel


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def strip_dia(s: str) -> str:
    return _DIA_RE.sub("", nfc(s))


def is_na(v: str) -> bool:
    return (v or "").strip().lower() in ("", "na", "-", "none", "u")


# ---------------------------------------------------------------------------
# Rule inventory
# ---------------------------------------------------------------------------
_UNIT_RE = re.compile(r"Unit(?:Gen)?\s*=\s*([MF])", re.I)
_TEN_RE = re.compile(r"Ten(?:Gen)?\s*=\s*([MF])", re.I)


def parse_output(out: str):
    """Parse the inventory ``output`` column into a comparable tuple.

    'MS'                 -> ('m', 's')       gender-number target
    'MP (syncretic)'     -> ('m', 'p')
    'F'                  -> ('f',)           gender-only (numerals/ordinals)
    'Unit=F ∧ Ten=M'     -> ('f', 'm')       compound numeral (unit, ten)
    'Unit=F'             -> ('f',)           21-99 compound (unit only)
    'NA'                 -> 'NA'             gender-less ('uqud decades)
    """
    s = (out or "").strip()
    if s.upper() == "NA":
        return "NA"
    if "Unit" in s or "unit" in s:
        u = _UNIT_RE.search(s)
        t = _TEN_RE.search(s)
        if u and t:
            return (u.group(1).lower(), t.group(1).lower())
        if u:
            return (u.group(1).lower(),)
    tok = s.split()[0].strip()
    if len(tok) == 2 and tok[0] in "MFmf" and tok[1] in "SDPsdp":
        return (tok[0].lower(), tok[1].lower())
    if len(tok) == 1 and tok in "MFmf":
        return (tok.lower(),)
    raise ValueError(f"Unparseable rule output: {out!r}")


class Rule:
    __slots__ = ("name", "conds", "agg", "num_range", "output")

    def __init__(self, row):
        self.name = row["rule_name"]
        self.conds = {
            k: row[k].strip()
            for k in ("form_gen", "form_num", "form_rat",
                      "func_gen", "func_num", "func_rat")
        }
        self.agg = row["agg_value"].strip()
        self.num_range = row["num_range"].strip()
        self.output = parse_output(row["output"])

    def matches(self, ctrl, band):
        if self.num_range and self.num_range != band:
            return False
        for feat, cond in self.conds.items():
            if cond == "*":
                continue
            val = ctrl.get(feat, "")
            if is_na(val) or cond.lower() != val.lower():
                return False
        if self.agg != "*" and self.agg not in ctrl.get("agg", ()):
            return False
        return True


def load_rules(path: Path):
    """rules[(construction, rule_set_label)] = [Rule, ...]"""
    rules = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rules[(row["construction"], row["rule_set"])].append(Rule(row))
    return rules


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
def load_agg_lexicon(path: Path):
    """lemma/form (dia-stripped) -> set of AGG values (A/C/G/D)."""
    lex = defaultdict(set)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        key_cols = [c for c in cols
                    if c.lower() in ("lemma", "lemma_ar", "form", "noun_form",
                                     "inflected_form_ar")]
        agg_cols = [c for c in cols
                    if c.lower() in ("agg", "agg_value", "aggregation", "agr")]
        if not key_cols or not agg_cols:
            sys.exit(f"ERROR: --agg-lexicon {path} needs a lemma column "
                     f"(lemma/lemma_ar) and an AGG column (agg/aggregation/AGR); "
                     f"found columns: {cols}")
        for row in reader:
            vals = {v.upper() for v in re.split(r"[^A-Za-z]+", row[agg_cols[0]] or "")
                    if v.upper() in ("A", "C", "G", "D")}
            if not vals:
                continue
            for kc in key_cols:
                k = strip_dia(row.get(kc, ""))
                if k:
                    lex[k] |= vals
    return lex


def load_dual_gender(path: Path):
    """lemma/form (dia-stripped) -> set of licensed functional genders."""
    dual = defaultdict(set)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            g = (row.get("functional_gen") or "").strip().lower()
            if g not in ("m", "f"):
                continue
            for kc in ("lemma_ar", "noun_form", "inflected_form_ar"):
                k = strip_dia(row.get(kc, ""))
                if k:
                    dual[k].add(g)
    return dual


def lex_lookup(lex, lemma, form):
    for key in (strip_dia(lemma), strip_dia(form)):
        if key and key in lex:
            return lex[key]
    return set()


# ---------------------------------------------------------------------------
# Licensed-set computation and grading
# ---------------------------------------------------------------------------
def licensed_outputs(rules, ctrl, band, rule_set_key, dual_genders):

    gender_variants = [ctrl["func_gen"]]
    if rule_set_key == EXTENDED_KEY:
        for g in sorted(dual_genders):
            if g not in gender_variants:
                gender_variants.append(g)
    out = []
    for g in gender_variants:
        c = dict(ctrl)
        c["func_gen"] = g
        for r in rules:
            if r.matches(c, band) and r.output not in out:
                out.append(r.output)
    return out


def gold_tuple(row, kind, band):
    """Observed target agreement, from FUNCTIONAL features."""
    if band == "20-90":  # 'uqud decades: gender-less, out of scope
        return "NA"
    tg = row.get("target_gen", row.get("haal_gen", ""))
    if is_na(tg):
        return None
    if kind == "gn":
        tn = row.get("target_num", row.get("haal_num", ""))
        if is_na(tn):
            return None
        return (tg.lower(), tn.lower())
    # gender-only constructions (numerals / ordinals)
    if band in ("11-12", "13-19"):
        sg = row.get("second_gen", "")
        if is_na(sg):
            return None
        return (tg.lower(), sg.lower())
    return (tg.lower(),)


def evaluate(gold, outputs):
    if gold == "NA" or "NA" in outputs:
        return "na"
    if gold is None or not outputs:
        return "na"
    return "matched" if gold in outputs else "mismatched"


def fmt_output(o):
    if o == "NA":
        return "NA"
    return "".join(v.upper() for v in o) if len(o) == 2 and o[1] in "sdp" \
        else "+".join(v.upper() for v in o)


def fmt_pred(outputs):
    return "|".join(fmt_output(o) for o in outputs) if outputs else ""


# ---------------------------------------------------------------------------
# Numeral / ordinal band classification
# ---------------------------------------------------------------------------
ORD_UNIT_LEMMA_TO_VAL = {strip_dia(k): v for k, v in {
    "أول": 1, "اول": 1, "أولي": 1, "اولي": 1, "حادي": 1,
    "ثاني": 2, "ثانية": 2, "ثانيه": 2,
    "ثالث": 3, "ثالثه": 3, "رابع": 4, "رابعه": 4,
    "خامس": 5, "خامسه": 5, "سادس": 6, "سادسه": 6,
    "سابع": 7, "سابعه": 7, "ثامن": 8, "ثامنه": 8,
    "تاسع": 9, "تاسعه": 9, "عاشر": 10, "عاشره": 10,
}.items()}

DECADE_LEMMAS = {strip_dia(k) for k in
                 ("عشرون", "ثلاثون", "أربعون", "خمسون",
                  "ستون", "سبعون", "ثمانون", "تسعون")}

ASHAR_LEMMAS = {strip_dia(k) for k in ("عشر", "عشرة")}


def ordinal_band(row):
    sub = row.get("subtype_detail", "") or row.get("subtype", "")
    tlem = strip_dia(row.get("target_lemma", ""))
    slem = strip_dia(row.get("second_lemma", ""))
    if sub == "ashar" or (not sub and slem in ASHAR_LEMMAS):
        return "11-12" if ORD_UNIT_LEMMA_TO_VAL.get(tlem, 0) <= 2 else "13-19"
    if sub == "uqud" or (not sub and slem in DECADE_LEMMAS):
        return "21-99"
    if tlem in DECADE_LEMMAS:
        return "20-90"
    return "1-10"


NUM_FILE_BANDS = [
    ("num_noun_1_2_pairs.tsv", "1-2"),
    ("num_noun_3_10_pairs.tsv", "3-10"),
    ("num_noun_11_12_pairs.tsv", "11-12"),
    ("num_noun_13_19_pairs.tsv", "13-19"),
    ("num_noun_20_90_compounds_pairs.tsv", "mixed"),
]


def cardinal_band(row, file_band):
    if file_band == "1-2":
        # band 1 vs 2 is decided by the rules' own conditions on the noun
        # (S vs D); both bands are open for this file.
        return "1-2"
    if file_band == "mixed":
        return ("21-99" if row.get("subtype_detail", "") == "compound_21_99"
                else "20-90")
    return file_band


# ---------------------------------------------------------------------------
# Per-construction drivers
# ---------------------------------------------------------------------------
def ctrl_features(row, prefix, agg_lex, dual_lex):
    lemma = row.get(f"{prefix}_lemma", "")
    form = row.get(f"{prefix}_form", "")
    rat = row.get(f"{prefix}_rat", "")
    return {
        "form_gen": row.get(f"{prefix}_form_gen", ""),
        "form_num": row.get(f"{prefix}_form_num", ""),
        "form_rat": rat,  # rationality is lexical, not form-inflected
        "func_gen": row.get(f"{prefix}_gen", ""),
        "func_num": row.get(f"{prefix}_num", ""),
        "func_rat": rat,
        "agg": lex_lookup(agg_lex, lemma, form),
    }, lex_lookup(dual_lex, lemma, form)


def validate_file(pairs_path, out_dir, rules_by_set, kind, agg_lex, dual_lex,
                  ctrl_prefix="head", band_fn=None, rules_key_fn=None,
                  summary=None, label=None):
    """Generic driver. kind: 'gn' or 'gender'.

    rules_by_set: {rule_set_key: [Rule,...]} OR callable(row) -> that dict
    band_fn(row) -> band string ('' for non-numeral constructions)
    """
    with pairs_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    extra_cols = []
    for key, _ in RULE_SETS:
        extra_cols += [f"pred_{key}", f"match_{key}"]
    out_fields = fieldnames + [c for c in extra_cols if c not in fieldnames]

    counts = defaultdict(Counter)  # (label_for_row, set_key) -> Counter
    for row in rows:
        if (row.get("skip_reason") or "").strip():
            for key, _ in RULE_SETS:
                row[f"pred_{key}"] = ""
                row[f"match_{key}"] = "na"
            continue
        band = band_fn(row) if band_fn else ""
        row_label = label(row) if callable(label) else (label or "")
        ruleset = rules_key_fn(row) if rules_key_fn else rules_by_set
        ctrl, duals = ctrl_features(
            row, ctrl_prefix(row) if callable(ctrl_prefix) else ctrl_prefix,
            agg_lex, dual_lex)
        gold = gold_tuple(row, kind, band)
        for key, _ in RULE_SETS:
            bands = ("1", "2") if band == "1-2" else (band,)
            outs = []
            for b in bands:
                for o in licensed_outputs(ruleset[key], ctrl, b, key, duals):
                    if o not in outs:
                        outs.append(o)
            res = evaluate(gold, outs)
            row[f"pred_{key}"] = fmt_pred(outs)
            row[f"match_{key}"] = res
            counts[(row_label, key)][res] += 1

    out_path = out_dir / (pairs_path.stem + "_validated.tsv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    for (row_label, key), c in sorted(counts.items()):
        m, mm, na = c["matched"], c["mismatched"], c["na"]
        acc = m / (m + mm) if (m + mm) else 0.0
        summary.append({
            "construction": row_label.split("::")[0],
            "band": row_label.split("::")[1] if "::" in row_label else "",
            "rule_set": dict(RULE_SETS)[key],
            "pairs": m + mm + na, "matched": m, "mismatched": mm, "na": na,
            "accuracy": f"{acc:.4f}",
        })
    print(f"  {pairs_path.name}: {len(rows):,} rows -> {out_path.name}")


def sets_for(rules, construction):
    d = {}
    for key, sheet_label in RULE_SETS:
        d[key] = rules.get((construction, sheet_label), [])
        if not d[key]:
            print(f"  WARNING: no rules for ({construction}, {sheet_label})")
    return d


def run_construction(flag, args, rules, agg_lex, dual_lex, summary):
    pairs = Path(args.pairs)
    out_dir = Path(args.out)

    simple = {
        "na": ("noun_adj", "noun_adj_pairs.tsv"),
        "sv": ("sv", "sv_pairs.tsv"),
        "vs": ("vs", "vs_pairs.tsv"),
        "dem": ("dem", "dem_pairs.tsv"),
        "rel": ("rel", "rel_pairs.tsv"),
        "sp": ("subj_pred", "subj_pred_pairs.tsv"),
    }
    if flag in simple:
        ckey, fname = simple[flag]
        p = pairs / fname
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")
        validate_file(p, out_dir, sets_for(rules, ckey), "gn",
                      agg_lex, dual_lex, label=ckey, summary=summary)

    elif flag == "haal":
        p = pairs / "haal_triples.tsv"
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")
        dp_sets = sets_for(rules, "haal_dp")
        vb_sets = sets_for(rules, "haal_verb")

        def is_verb_ctrl(row):
            return row.get("ctrl_role", "").startswith("verb_prodrop")

        validate_file(
            p, out_dir, None, "gn", agg_lex, dual_lex,
            ctrl_prefix=lambda r: "verb" if is_verb_ctrl(r) else "ctrl",
            rules_key_fn=lambda r: vb_sets if is_verb_ctrl(r) else dp_sets,
            label=lambda r: "haal_verb" if is_verb_ctrl(r) else "haal_dp",
            summary=summary)

    elif flag == "num":
        csets = sets_for(rules, "num_noun")
        for fname, file_band in NUM_FILE_BANDS:
            p = pairs / fname
            if not p.exists():
                print(f"  WARNING: {p} not found — skipping")
                continue
            validate_file(
                p, out_dir, csets, "gender", agg_lex, dual_lex,
                band_fn=lambda r, fb=file_band: cardinal_band(r, fb),
                label=lambda r, fb=file_band: f"num_noun::{cardinal_band(r, fb)}",
                summary=summary)

    elif flag == "ord":
        p = pairs / "ordinal_noun_pairs.tsv"
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")
        validate_file(
            p, out_dir, sets_for(rules, "ordinal"), "gender",
            agg_lex, dual_lex, band_fn=ordinal_band,
            label=lambda r: f"ordinal::{ordinal_band(r)}",
            summary=summary)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Validate agreement rules on extracted pairs "
                    "(driven by the rule inventory TSV).")
    ap.add_argument("--construction", required=True,
                    choices=["na", "sv", "vs", "dem", "rel", "sp", "haal",
                             "num", "ord", "all"])
    ap.add_argument("--pairs", required=True,
                    help="Directory holding the extracted pair TSVs")
    ap.add_argument("--rules", type=Path, default=DEFAULT_RULES,
                    help=f"Rule inventory TSV (default: {DEFAULT_RULES})")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--agg-lexicon", type=Path, default=None,
                    help="AGG lexicon TSV (lemma + agg/AGR columns, values "
                         "A/C/G/D). Without it, AGG-conditioned Extended "
                         "rules never fire.")
    ap.add_argument("--dual-gender", type=Path, default=DEFAULT_DUAL,
                    help=f"Dual-gender noun list (default: {DEFAULT_DUAL})")
    args = ap.parse_args()

    rules = load_rules(args.rules)
    print(f"Loaded {sum(len(v) for v in rules.values())} rules "
          f"from {args.rules}")

    agg_lex = {}
    if args.agg_lexicon:
        agg_lex = load_agg_lexicon(args.agg_lexicon)
        print(f"AGG lexicon: {len(agg_lex):,} keys")
    else:
        print("NOTE: no --agg-lexicon given; AGG-conditioned Extended rules "
              "will not fire.")

    dual_lex = {}
    if args.dual_gender and Path(args.dual_gender).exists():
        dual_lex = load_dual_gender(Path(args.dual_gender))
        print(f"Dual-gender list: {len(dual_lex):,} keys")

    Path(args.out).mkdir(parents=True, exist_ok=True)

    flags = (["na", "sv", "vs", "dem", "rel", "sp", "haal", "num", "ord"]
             if args.construction == "all" else [args.construction])
    summary = []
    for flag in flags:
        print(f"== {flag}")
        run_construction(flag, args, rules, agg_lex, dual_lex, summary)

    sum_path = Path(args.out) / "accuracy_summary.tsv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["construction", "band", "rule_set", "pairs",
                           "matched", "mismatched", "na", "accuracy"],
            delimiter="\t")
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary -> {sum_path}")


if __name__ == "__main__":
    main()
