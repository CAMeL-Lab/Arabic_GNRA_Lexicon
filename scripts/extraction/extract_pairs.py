"""
BAREC-10M — Extract controller-target agreement pairs, all constructions.

One entry point for the nine constructions evaluated in the paper:

  --construction na    Noun-Adjective            -> noun_adj_pairs.tsv
  --construction sv    Subject-Verb (S before V) -> sv_pairs.tsv
  --construction vs    Verb-Subject (V before S) -> vs_pairs.tsv
  --construction dem   Demonstrative-Noun        -> dem_pairs.tsv (+ dem_skipped.tsv)
  --construction rel   Noun-Relative pronoun     -> rel_pairs.tsv (+ rel_skipped.tsv)
  --construction sp    Subject-Predicate         -> subj_pred_pairs.tsv
  --construction haal  Haal (circumstantial acc) -> haal_triples.tsv (+ haal_skipped.tsv)
  --construction num   Numeral-Noun AND Ordinals -> num_noun_{band}_pairs.tsv,
                                                    ordinal_noun_pairs.tsv (+ *_skipped.tsv)
  --construction all   run everything

PURE STRUCTURAL EXTRACTION: no agreement predictions here — those are produced
later by scripts/validation/validate_pairs.py, which fills the pred_*/match_* columns.

Two extraction-time filters apply to Noun-Adjective (such pairs are never
emitted):
  1. not_attributive_definiteness — head noun is definite by article
     (prc0=Al_det) while the adjective is not: attributive adjectives agree in
     definiteness, so such pairs are predicates/adverbs/haal, not Noun-Adj.
     Definiteness is read from the analyzer feature prc0, not the surface Al.
  2. idafa_wrong_head — the head is the genitive N2 of a construct
     (deprel=IDF) and the adjective agrees (deflection-aware, functional
     features) with the governing noun N1 but not with N2: the parse attached
     the adjective to the wrong noun.

Skip taxonomies live in barec_pair_utils (classify_* functions). Manual
exclude files are optional; extraction runs with none by default.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from barec_pair_utils import (
    # shared
    PRED_COLUMNS, nfc, strip_dia,
    parse_sync_file, load_manual_excludes, build_pair_row, write_summary,
    # noun-adjective
    OUTPUT_COLUMNS_PAIR_NA, classify_noun_adj_pair,
    # subject-verb / verb-subject
    OUTPUT_COLUMNS_PAIR, classify_subject_verb_pair,
    # demonstratives
    OUTPUT_COLUMNS_PAIR_DEM, classify_dem_pair,
    # relatives
    OUTPUT_COLUMNS_PAIR_REL, classify_rel_pair,
    # subject-predicate
    OUTPUT_COLUMNS_PAIR_SP, SP_PRD_POS, sp_clause_type, classify_subj_pred_pair,
    # haal
    OUTPUT_COLUMNS_HAAL, classify_haal_pre, classify_haal_post,
    haal_find_sbj_child, haal_find_obj_child, haal_pick_controller,
    # numerals + ordinals
    OUTPUT_COLUMNS_NUM_SIMPLE, OUTPUT_COLUMNS_NUM_COMPOUND,
    DECADE_LEMMAS_STRIPPED,
    ORDINAL_HEAD_POS, ORDINAL_UQUD_LEMMAS_STRIPPED, ORDINAL_ASHAR_LEMMA_STRIPPED,
    cardinal_value, num_find_children,
    num_classify_skip, num_build_simple_row, num_build_compound_row,
)


# ============================================================
# Shared helpers
# ============================================================
def _sync_files(sync_root: Path) -> list[Path]:
    files = sorted(sync_root.rglob("*.conllx"))
    print(f"Reading SYNC files from: {sync_root}")
    print(f"SYNC files: {len(files)}\n")
    return files


def _doc_domain(sync_path: Path, sync_root: Path) -> tuple[str, str]:
    document = str(sync_path.relative_to(sync_root).with_suffix(""))
    return document, document.split("/", 1)[0]


def _excludes(exclusions_dir: Path | None, name: str) -> set:
    if exclusions_dir is None:
        return set()
    return load_manual_excludes(exclusions_dir / name)


def _writer(handle, columns):
    w = csv.DictWriter(handle, fieldnames=columns, delimiter="\t",
                       quoting=csv.QUOTE_NONE, escapechar="\\")
    w.writeheader()
    return w


# ============================================================
# Noun-Adjective
# ============================================================
def _na_adj_agrees(ag, an, ng, nn, nr):
    """Deflection-aware adjective agreement over FUNCTIONAL features
    (used by the folded idafa wrong-head filter)."""
    if nn == "s":
        return an == "s" and ag == ng
    if nn == "d":
        return ag == ng and an in ("d", "s")
    if nn == "p":
        return (ag == ng and an == "p") if nr == "r" else (ag == "f" and an == "s")
    return ag == ng and an == nn


def run_na(sync_root: Path, pairs_dir: Path, exclusions_dir: Path | None) -> None:
    out_pairs = pairs_dir / "noun_adj_pairs.tsv"
    out_summary = pairs_dir / "noun_adj_extraction_summary.txt"
    manual_excludes = _excludes(exclusions_dir, "noun_adj_manual_excludes.tsv")

    files = _sync_files(sync_root)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    candidates = pairs_written = 0
    drop_counts, flag_counts, domain_clean = Counter(), Counter(), Counter()

    with out_pairs.open("w", encoding="utf-8", newline="") as fout:
        writer = _writer(fout, OUTPUT_COLUMNS_PAIR_NA)
        for fi, sync_path in enumerate(files, 1):
            if fi % 1000 == 0:
                print(f"  {fi}/{len(files)} files processed...")
            document, domain = _doc_domain(sync_path, sync_root)

            for sent_id, tokens in parse_sync_file(sync_path):
                if not tokens:
                    continue
                for tid, A in tokens.items():
                    if A["deprel"] != "MOD":
                        continue
                    if A["pos"] != "adj":
                        continue
                    hid = A["head"]
                    if hid not in tokens:
                        continue
                    N = tokens[hid]
                    if not N["pos"].startswith("noun"):
                        continue

                    candidates += 1

                    # Folded filter 1: definite-by-article head + non-article
                    # adjective cannot be attributive.
                    if N["prc0"] == "Al_det" and A["prc0"] != "Al_det":
                        drop_counts["not_attributive_definiteness"] += 1
                        continue

                    # Folded filter 2: idafa wrong-head — adjective agrees with
                    # the governing N1 but not with the extracted head N2.
                    if N["deprel"] == "IDF":
                        n1 = tokens.get(N.get("head"))
                        if (n1 and n1.get("pos", "").startswith("noun")
                                and _na_adj_agrees(A["gen"], A["num"],
                                                   n1["gen"], n1["num"], n1["rat"])
                                and not _na_adj_agrees(A["gen"], A["num"],
                                                       N["gen"], N["num"], N["rat"])):
                            drop_counts["idafa_wrong_head"] += 1
                            continue

                    action, reason = classify_noun_adj_pair(
                        N, A, document, sent_id, tokens, manual_excludes)
                    if action == "drop":
                        drop_counts[reason] += 1
                        continue

                    skip_reason = reason if action == "flag" else ""
                    if action == "flag":
                        flag_counts[reason] += 1
                    else:
                        domain_clean[domain] += 1

                    row = build_pair_row(document, domain, sent_id, A, N, skip_reason)
                    for col in PRED_COLUMNS:
                        row[col] = ""
                    writer.writerow(row)
                    pairs_written += 1

    text = write_summary("Noun-Adj", out_summary, len(files),
                         candidates, pairs_written, drop_counts, flag_counts,
                         domain_clean)
    print("\n" + text)
    print(f"\nWrote: {out_pairs}\nWrote: {out_summary}")


# ============================================================
# Subject-Verb / Verb-Subject (shared skeleton; order flag differs)
# ============================================================
def _run_subject_verb(sync_root: Path, pairs_dir: Path,
                      exclusions_dir: Path | None, order: str) -> None:
    label = "SV" if order == "sv" else "VS"
    out_pairs = pairs_dir / f"{order}_pairs.tsv"
    out_summary = pairs_dir / f"{order}_extraction_summary.txt"
    manual_excludes = _excludes(exclusions_dir, f"{order}_manual_excludes.tsv")

    files = _sync_files(sync_root)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    candidates = pairs_written = 0
    drop_counts, flag_counts, domain_clean = Counter(), Counter(), Counter()

    with out_pairs.open("w", encoding="utf-8", newline="") as fout:
        writer = _writer(fout, OUTPUT_COLUMNS_PAIR)
        for fi, sync_path in enumerate(files, 1):
            if fi % 1000 == 0:
                print(f"  {fi}/{len(files)} files processed...")
            document, domain = _doc_domain(sync_path, sync_root)

            for sent_id, tokens in parse_sync_file(sync_path):
                if not tokens:
                    continue
                for tid, N in tokens.items():
                    if N["deprel"] != "SBJ":
                        continue
                    if not N["pos"].startswith("noun"):
                        continue
                    hid = N["head"]
                    if hid not in tokens:
                        continue
                    V = tokens[hid]
                    if V["pos"] != "verb":
                        continue
                    try:
                        if order == "sv" and int(V["id"]) <= int(N["id"]):
                            continue   # VS order, not SV
                        if order == "vs" and int(V["id"]) >= int(N["id"]):
                            continue   # SV order, not VS
                    except ValueError:
                        continue
                    vhid = V.get("head", "0")
                    if vhid in tokens and tokens[vhid]["pos"] == "pron_rel":
                        continue

                    candidates += 1
                    action, reason = classify_subject_verb_pair(
                        N, V, document, sent_id, tokens, manual_excludes)
                    if action == "drop":
                        drop_counts[reason] += 1
                        continue

                    skip_reason = reason if action == "flag" else ""
                    if action == "flag":
                        flag_counts[reason] += 1
                    else:
                        domain_clean[domain] += 1

                    writer.writerow(
                        build_pair_row(document, domain, sent_id, V, N, skip_reason))
                    pairs_written += 1

    text = write_summary(label, out_summary, len(files),
                         candidates, pairs_written, drop_counts, flag_counts,
                         domain_clean)
    print("\n" + text)
    print(f"\nWrote: {out_pairs}\nWrote: {out_summary}")


def run_sv(sync_root, pairs_dir, exclusions_dir):
    _run_subject_verb(sync_root, pairs_dir, exclusions_dir, "sv")


def run_vs(sync_root, pairs_dir, exclusions_dir):
    _run_subject_verb(sync_root, pairs_dir, exclusions_dir, "vs")


# ============================================================
# Demonstratives / Relatives (shared kept+skipped two-file skeleton)
# ============================================================
def _run_dem_rel(sync_root: Path, pairs_dir: Path,
                 exclusions_dir: Path | None, which: str) -> None:
    if which == "dem":
        label, columns, classify = "Dem", OUTPUT_COLUMNS_PAIR_DEM, classify_dem_pair
        target_pos = "pron_dem"
        out_pairs = pairs_dir / "dem_pairs.tsv"
        out_skipped = pairs_dir / "dem_skipped.tsv"
        out_summary = pairs_dir / "dem_extraction_summary.txt"
        manual_excludes = _excludes(exclusions_dir, "dem_manual_excludes.tsv")
    else:
        label, columns, classify = "Rel", OUTPUT_COLUMNS_PAIR_REL, classify_rel_pair
        target_pos = "pron_rel"
        out_pairs = pairs_dir / "rel_pairs.tsv"
        out_skipped = pairs_dir / "rel_skipped.tsv"
        out_summary = pairs_dir / "rel_extraction_summary.txt"
        manual_excludes = _excludes(exclusions_dir, "rel_manual_excludes.tsv")

    files = _sync_files(sync_root)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    candidates = pairs_kept = pairs_skipped = 0
    skip_counts, domain_kept = Counter(), Counter()

    f_keep = out_pairs.open("w", encoding="utf-8", newline="")
    f_skip = out_skipped.open("w", encoding="utf-8", newline="")
    try:
        w_keep = _writer(f_keep, columns)
        w_skip = _writer(f_skip, columns)
        for fi, sync_path in enumerate(files, 1):
            if fi % 1000 == 0:
                print(f"  {fi}/{len(files)} files processed...")
            document, domain = _doc_domain(sync_path, sync_root)

            for sent_id, tokens in parse_sync_file(sync_path):
                if not tokens:
                    continue
                for tid, T in tokens.items():
                    if T["pos"] != target_pos:
                        continue
                    if T["deprel"] != "MOD":
                        continue
                    hid = T["head"]
                    if hid not in tokens:
                        continue
                    N = tokens[hid]
                    if not N["pos"].startswith("noun"):
                        continue
                    if N["pos"] == "noun_num":
                        continue

                    candidates += 1
                    action, reason = classify(
                        N, T, document, sent_id, tokens, manual_excludes)
                    row = build_pair_row(document, domain, sent_id, T, N,
                                         reason if action != "keep" else "")
                    for col in PRED_COLUMNS:
                        row[col] = ""
                    if action == "keep":
                        w_keep.writerow(row)
                        pairs_kept += 1
                        domain_kept[domain] += 1
                    else:
                        # Both 'drop' and 'flag' go to the skipped file.
                        w_skip.writerow(row)
                        pairs_skipped += 1
                        skip_counts[reason] += 1
    finally:
        f_keep.close()
        f_skip.close()

    text = write_summary(label, out_summary, len(files),
                         candidates, pairs_kept, Counter(), Counter(), domain_kept)
    extra = ["", f"  Skipped (written to {out_skipped.name}) : {pairs_skipped}",
             "", f"Skip reasons (in {out_skipped.name}):"]
    for r, n in skip_counts.most_common():
        extra.append(f"  {r:30s} {n}")
    out_summary.write_text(text + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    print("\n" + text)
    print("\n".join(extra))
    print(f"\nWrote: {out_pairs}\nWrote: {out_skipped}\nWrote: {out_summary}")


def run_dem(sync_root, pairs_dir, exclusions_dir):
    _run_dem_rel(sync_root, pairs_dir, exclusions_dir, "dem")


def run_rel(sync_root, pairs_dir, exclusions_dir):
    _run_dem_rel(sync_root, pairs_dir, exclusions_dir, "rel")


# ============================================================
# Subject-Predicate
# ============================================================
def run_sp(sync_root: Path, pairs_dir: Path, exclusions_dir: Path | None) -> None:
    out_pairs = pairs_dir / "subj_pred_pairs.tsv"
    out_summary = pairs_dir / "subj_pred_extraction_summary.txt"
    manual_excludes = _excludes(exclusions_dir, "subj_pred_manual_excludes.tsv")

    files = _sync_files(sync_root)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    candidates = pairs_written = 0
    drop_counts, flag_counts = Counter(), Counter()
    domain_clean, clause_type_clean = Counter(), Counter()

    with out_pairs.open("w", encoding="utf-8", newline="") as fout:
        writer = _writer(fout, OUTPUT_COLUMNS_PAIR_SP)
        for fi, sync_path in enumerate(files, 1):
            if fi % 1000 == 0:
                print(f"  {fi}/{len(files)} files processed...")
            document, domain = _doc_domain(sync_path, sync_root)

            for sent_id, tokens in parse_sync_file(sync_path):
                if not tokens:
                    continue

                children_by_head: dict[str, list[dict]] = {}
                for tok in tokens.values():
                    children_by_head.setdefault(tok["head"], []).append(tok)

                for hid, kids in children_by_head.items():
                    if hid not in tokens:
                        continue
                    H = tokens[hid]
                    ctype = sp_clause_type(H)
                    if ctype is None:
                        continue

                    sbjs = [k for k in kids
                            if k["deprel"] == "SBJ" and k["pos"].startswith("noun")]
                    prds = [k for k in kids
                            if k["deprel"] == "PRD" and k["pos"] in SP_PRD_POS
                            and "PREP" not in k.get("xpos", "")]
                    if not sbjs or not prds:
                        continue

                    h_lemma = nfc(H.get("lemma", ""))
                    for S in sbjs:
                        for P in prds:
                            candidates += 1
                            action, reason = classify_subj_pred_pair(
                                S, P, document, sent_id, tokens, manual_excludes)
                            if action == "drop":
                                drop_counts[reason] += 1
                                continue

                            skip_reason = reason if action == "flag" else ""
                            if action == "flag":
                                flag_counts[reason] += 1
                            else:
                                domain_clean[domain] += 1
                                clause_type_clean[ctype] += 1

                            row = build_pair_row(
                                document, domain, sent_id, P, S, skip_reason)
                            row["clause_type"] = ctype
                            row["clause_head_lemma"] = h_lemma
                            for col in PRED_COLUMNS:
                                row[col] = ""
                            writer.writerow(row)
                            pairs_written += 1

    text = write_summary("Subj-Pred", out_summary, len(files),
                         candidates, pairs_written, drop_counts, flag_counts,
                         domain_clean)
    extra = ["", "Clean pair counts per clause_type:"]
    for ct, n in sorted(clause_type_clean.items(), key=lambda x: -x[1]):
        extra.append(f"  {ct:25s} {n}")
    out_summary.write_text(text + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    print("\n" + text)
    print("\n".join(extra))
    print(f"\nWrote: {out_pairs}\nWrote: {out_summary}")


# ============================================================
# Haal
# ============================================================
def _build_haal_row(document, domain, sent_id, H, V, C, role, skip_reason):
    row = {
        "document": document, "domain": domain, "sent_id": sent_id,
        "haal_id": H["id"], "haal_form": H["form"], "haal_lemma": H["lemma"],
        "haal_gen": H["gen"], "haal_form_gen": H["form_gen"],
        "haal_num": H["num"], "haal_form_num": H["form_num"],
        "ctrl_id": C.get("id", ""), "ctrl_pos": C.get("pos", ""),
        "ctrl_role": role,
        "ctrl_form": C.get("form", ""), "ctrl_lemma": C.get("lemma", ""),
        "ctrl_gen": C.get("gen", ""), "ctrl_form_gen": C.get("form_gen", ""),
        "ctrl_num": C.get("num", ""), "ctrl_form_num": C.get("form_num", ""),
        "ctrl_rat": C.get("rat", ""),
        "verb_id": V["id"], "verb_xpos": V.get("xpos", ""),
        "verb_form": V["form"], "verb_lemma": V["lemma"],
        "verb_gen": V["gen"], "verb_form_gen": V["form_gen"],
        "verb_num": V["num"], "verb_form_num": V["form_num"],
        "skip_reason": skip_reason,
    }
    for c in PRED_COLUMNS:
        row[c] = ""
    return row


def _empty_ctrl():
    return {k: "" for k in
            ("id", "form", "lemma", "pos",
             "gen", "form_gen", "num", "form_num", "rat")}


def run_haal(sync_root: Path, pairs_dir: Path, exclusions_dir: Path | None) -> None:
    out_triples = pairs_dir / "haal_triples.tsv"
    out_skipped = pairs_dir / "haal_skipped.tsv"
    out_summary = pairs_dir / "haal_extraction_summary.txt"
    manual_excludes = _excludes(exclusions_dir, "haal_manual_excludes.tsv")

    files = _sync_files(sync_root)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    candidates = triples_written = skipped_written = 0
    skip_counts, domain_kept, ctrl_role_kept = Counter(), Counter(), Counter()

    f_trip = out_triples.open("w", encoding="utf-8", newline="")
    f_skip = out_skipped.open("w", encoding="utf-8", newline="")
    try:
        w_trip = _writer(f_trip, OUTPUT_COLUMNS_HAAL)
        w_skip = _writer(f_skip, OUTPUT_COLUMNS_HAAL)
        for fi, sync_path in enumerate(files, 1):
            if fi % 1000 == 0:
                print(f"  {fi}/{len(files)} files processed...")
            document, domain = _doc_domain(sync_path, sync_root)

            for sent_id, tokens in parse_sync_file(sync_path):
                if not tokens:
                    continue
                for hid, H in tokens.items():
                    if H["pos"] != "adj":
                        continue
                    if H["deprel"] != "MOD":
                        continue
                    if H["cas"] != "a":
                        continue
                    vid = H["head"]
                    if vid not in tokens:
                        continue
                    V = tokens[vid]
                    if V["pos"] != "verb":
                        continue

                    candidates += 1

                    action, reason = classify_haal_pre(
                        H, V, document, sent_id, tokens, manual_excludes)
                    if action == "skip":
                        skip_counts[reason] += 1
                        w_skip.writerow(_build_haal_row(
                            document, domain, sent_id, H, V,
                            _empty_ctrl(), "", reason))
                        skipped_written += 1
                        continue

                    sbj = haal_find_sbj_child(V, tokens)
                    obj = haal_find_obj_child(V, tokens)
                    # Extraction runs WITHOUT the agreement filter — form
                    # disagreements are kept for the validator to judge
                    # (parallel to noun-adjective).
                    C, role, ctrl_skip = haal_pick_controller(
                        H, V, sbj, obj, enforce_agreement=False)
                    if ctrl_skip:
                        skip_counts[ctrl_skip] += 1
                        w_skip.writerow(_build_haal_row(
                            document, domain, sent_id, H, V, C, role, ctrl_skip))
                        skipped_written += 1
                        continue

                    action, reason = classify_haal_post(C, role)
                    if action == "skip":
                        skip_counts[reason] += 1
                        w_skip.writerow(_build_haal_row(
                            document, domain, sent_id, H, V, C, role, reason))
                        skipped_written += 1
                        continue

                    w_trip.writerow(_build_haal_row(
                        document, domain, sent_id, H, V, C, role, ""))
                    triples_written += 1
                    domain_kept[domain] += 1
                    ctrl_role_kept[role] += 1
    finally:
        f_trip.close()
        f_skip.close()

    text = write_summary("Haal", out_summary, len(files),
                         candidates, triples_written, Counter(), Counter(),
                         domain_kept)
    extra = ["", f"  Skipped (written to {out_skipped.name}) : {skipped_written}",
             "", "Skip reasons (in haal_skipped.tsv):"]
    for r, n in skip_counts.most_common():
        extra.append(f"  {r:30s} {n}")
    extra.append("")
    extra.append("ctrl_role distribution (haal_triples.tsv):")
    for r, n in ctrl_role_kept.most_common():
        extra.append(f"  {r:32s} {n}")
    out_summary.write_text(text + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    print("\n" + text)
    print("\n".join(extra))
    print(f"\nWrote: {out_triples}\nWrote: {out_skipped}\nWrote: {out_summary}")


# ============================================================
# Numerals + Ordinals
# ============================================================
# أحد / إحدى function as indefinite pronouns ("someone"), not the cardinal
# "one", when they appear as standalone noun_num tokens. Drop from 1_2.
NON_CARDINAL_LEMMAS_1_2 = frozenset(strip_dia(s) for s in ("أحد", "إحدى"))

_SUBTYPES_SIMPLE = ["1_2", "3_10"]
_SUBTYPES_COMPOUND = ["11_12", "13_19", "20_90_compounds"]


def _num_open_writers(pairs_dir: Path) -> dict[str, dict]:
    out = {}
    for stem in _SUBTYPES_SIMPLE:
        pp = pairs_dir / f"num_noun_{stem}_pairs.tsv"
        sp = pairs_dir / f"num_noun_{stem}_skipped.tsv"
        fp = pp.open("w", encoding="utf-8", newline="")
        fs = sp.open("w", encoding="utf-8", newline="")
        out[stem] = dict(fp=fp, fs=fs,
                         wp=_writer(fp, OUTPUT_COLUMNS_NUM_SIMPLE),
                         ws=_writer(fs, OUTPUT_COLUMNS_NUM_SIMPLE),
                         kept=0, skipped=0, skip_counts=Counter(),
                         domain_kept=Counter(), pair_path=pp, skip_path=sp)
    for stem in _SUBTYPES_COMPOUND + ["ordinal"]:
        pp = pairs_dir / (f"num_noun_{stem}_pairs.tsv"
                          if stem != "ordinal" else "ordinal_noun_pairs.tsv")
        sp = pairs_dir / (f"num_noun_{stem}_skipped.tsv"
                          if stem != "ordinal" else "ordinal_noun_skipped.tsv")
        fp = pp.open("w", encoding="utf-8", newline="")
        fs = sp.open("w", encoding="utf-8", newline="")
        out[stem] = dict(fp=fp, fs=fs,
                         wp=_writer(fp, OUTPUT_COLUMNS_NUM_COMPOUND),
                         ws=_writer(fs, OUTPUT_COLUMNS_NUM_COMPOUND),
                         kept=0, skipped=0, skip_counts=Counter(),
                         domain_kept=Counter(), pair_path=pp, skip_path=sp)
    return out


def _num_write_row(info: dict, row: dict, action: str, reason: str,
                   domain: str) -> None:
    if action == "keep":
        info["wp"].writerow(row)
        info["kept"] += 1
        info["domain_kept"][domain] += 1
    else:
        info["ws"].writerow(row)
        info["skipped"] += 1
        info["skip_counts"][reason] += 1


def _find_ordinal_second(tok: dict, tokens: dict) -> tuple[dict | None, str]:
    """Find the second part of a compound ordinal: ('ashar', 'uqud', or '')."""
    tid_int = int(tok["id"])
    tid = tok["id"]
    for c in tokens.values():
        if c.get("head") != tid or c["pos"] != "adj_num":
            continue
        lem = strip_dia(c.get("lemma", ""))
        if lem == ORDINAL_ASHAR_LEMMA_STRIPPED:
            return c, "ashar"
        if lem in ORDINAL_UQUD_LEMMAS_STRIPPED:
            return c, "uqud"
    for off in (1, 2, 3, 4):
        cid = str(tid_int + off)
        if cid in tokens and tokens[cid]["pos"] == "adj_num":
            lem = strip_dia(tokens[cid].get("lemma", ""))
            if lem in ORDINAL_UQUD_LEMMAS_STRIPPED:
                return tokens[cid], "uqud"
            if lem == ORDINAL_ASHAR_LEMMA_STRIPPED:
                return tokens[cid], "ashar"
    return None, ""


def _is_ordinal_second_half(tok: dict, tokens: dict) -> bool:
    """True if this adj_num token is the second half of a preceding compound."""
    lem = strip_dia(tok.get("lemma", ""))
    if lem not in ORDINAL_UQUD_LEMMAS_STRIPPED and lem != ORDINAL_ASHAR_LEMMA_STRIPPED:
        return False
    try:
        tid_int = int(tok["id"])
    except ValueError:
        return False
    for off in (1, 2, 3, 4):
        pid = str(tid_int - off)
        if (pid in tokens
                and tokens[pid]["pos"] == "adj_num"
                and tokens[pid]["deprel"] == "MOD"):
            return True
    return False


def run_num(sync_root: Path, pairs_dir: Path, exclusions_dir: Path | None) -> None:
    manual_excludes = _excludes(exclusions_dir, "num_manual_excludes.tsv")
    files = _sync_files(sync_root)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    info = _num_open_writers(pairs_dir)
    candidates = Counter()

    try:
        for fi, sync_path in enumerate(files, 1):
            if fi % 1000 == 0:
                print(f"  {fi}/{len(files)} files processed...")
            document, domain = _doc_domain(sync_path, sync_root)

            for sent_id, tokens in parse_sync_file(sync_path):
                if not tokens:
                    continue
                for tid, T in tokens.items():
                    pos = T["pos"]

                    # ---- Cardinals: noun_num --------------------------------
                    if pos == "noun_num":
                        hid = T["head"]
                        if hid in tokens and tokens[hid]["pos"] in ("noun_num", "digit"):
                            continue

                        ch = num_find_children(T, tokens)
                        if ch["idf_num"]:
                            continue   # millions/billions

                        is_decade = strip_dia(T.get("lemma", "")) in DECADE_LEMMAS_STRIPPED

                        if is_decade:
                            counted = ch["tmz"][0] if ch["tmz"] else None
                            if not counted:
                                continue
                            units_child = ch["obj_num"][0] if ch["obj_num"] else None
                            if units_child:
                                u_val = cardinal_value(units_child)
                                u_lem = strip_dia(units_child.get("lemma", ""))
                                if (u_val is None
                                        or not (1 <= u_val <= 9)
                                        or u_lem in DECADE_LEMMAS_STRIPPED):
                                    units_child = None
                            if units_child:
                                subtype, target, second = "20_90_compounds", units_child, T
                                subtype_detail = "compound_21_99"
                            else:
                                subtype, target, second = "20_90_compounds", T, None
                                subtype_detail = "standalone_decade"

                            candidates[subtype] += 1
                            action, reason = num_classify_skip(
                                counted, target, document, sent_id,
                                tokens, manual_excludes)
                            row = num_build_compound_row(
                                document, domain, sent_id, target, counted,
                                second, subtype_detail,
                                reason if action != "keep" else "")
                            _num_write_row(info[subtype], row, action, reason, domain)

                        else:
                            val = cardinal_value(T)
                            if val is None:
                                continue

                            if val == 10:
                                counted = (ch["tmz"][0] if ch["tmz"]
                                           else ch["idf"][0] if ch["idf"] else None)
                                if not counted:
                                    continue
                                candidates["3_10"] += 1
                                action, reason = num_classify_skip(
                                    counted, T, document, sent_id, tokens,
                                    manual_excludes)
                                row = num_build_simple_row(
                                    document, domain, sent_id, T, counted,
                                    reason if action != "keep" else "")
                                _num_write_row(info["3_10"], row, action, reason, domain)

                            elif ch["ashar"]:
                                ashar = ch["ashar"][0]
                                counted = ch["tmz"][0] if ch["tmz"] else None
                                if not counted:
                                    continue
                                if val in (1, 2):
                                    subtype, subtype_detail = "11_12", "compound_11_12"
                                else:
                                    subtype, subtype_detail = "13_19", "compound_13_19"
                                candidates[subtype] += 1
                                action, reason = num_classify_skip(
                                    counted, T, document, sent_id, tokens,
                                    manual_excludes)
                                row = num_build_compound_row(
                                    document, domain, sent_id, T, counted,
                                    ashar, subtype_detail,
                                    reason if action != "keep" else "")
                                _num_write_row(info[subtype], row, action, reason, domain)

                            else:
                                counted = (ch["idf"][0] if ch["idf"]
                                           else ch["tmz"][0] if ch["tmz"] else None)
                                if not counted:
                                    continue
                                if val in (1, 2):
                                    candidates["1_2"] += 1
                                    if strip_dia(T.get("lemma", "")) in NON_CARDINAL_LEMMAS_1_2:
                                        action, reason = "drop", "non_cardinal_pronoun"
                                    else:
                                        action, reason = num_classify_skip(
                                            counted, T, document, sent_id, tokens,
                                            manual_excludes)
                                    row = num_build_simple_row(
                                        document, domain, sent_id, T, counted,
                                        reason if action != "keep" else "")
                                    _num_write_row(info["1_2"], row, action, reason, domain)
                                elif 3 <= val <= 9:
                                    candidates["3_10"] += 1
                                    action, reason = num_classify_skip(
                                        counted, T, document, sent_id, tokens,
                                        manual_excludes)
                                    row = num_build_simple_row(
                                        document, domain, sent_id, T, counted,
                                        reason if action != "keep" else "")
                                    _num_write_row(info["3_10"], row, action, reason, domain)

                    # ---- Ordinals: adj_num MOD after noun -------------------
                    elif pos == "adj_num" and T["deprel"] == "MOD":
                        hid = T["head"]
                        if hid not in tokens:
                            continue
                        H = tokens[hid]
                        if H["pos"] not in ORDINAL_HEAD_POS:
                            continue
                        try:
                            if int(tid) <= int(hid):
                                continue   # must be after head
                        except ValueError:
                            continue
                        if _is_ordinal_second_half(T, tokens):
                            continue

                        second, sec_type = _find_ordinal_second(T, tokens)
                        subtype_detail = sec_type if sec_type else "standalone"

                        candidates["ordinal"] += 1
                        action, reason = num_classify_skip(
                            H, T, document, sent_id, tokens, manual_excludes)
                        row = num_build_compound_row(
                            document, domain, sent_id, T, H,
                            second, subtype_detail,
                            reason if action != "keep" else "")
                        _num_write_row(info["ordinal"], row, action, reason, domain)

    finally:
        for v in info.values():
            v["fp"].close()
            v["fs"].close()

    print("\n" + "=" * 64)
    print("BAREC Number-Noun extraction summary")
    print("=" * 64)
    for stem, label in [
        ("1_2",             "Cardinals 1-2"),
        ("3_10",            "Cardinals 3-10"),
        ("11_12",           "Cardinals 11-12"),
        ("13_19",           "Cardinals 13-19"),
        ("20_90_compounds", "Cardinals 20-90 + compounds"),
        ("ordinal",         "Ordinals"),
    ]:
        v = info[stem]
        print(f"\n  {label}")
        print(f"    candidates={candidates[stem]}  kept={v['kept']}  skipped={v['skipped']}")
        print(f"    pairs:   {v['pair_path'].name}")
        print(f"    skipped: {v['skip_path'].name}")
        if v["skip_counts"]:
            for r, n in v["skip_counts"].most_common():
                print(f"      {r:25s} {n}")


# ============================================================
# CLI
# ============================================================
CONSTRUCTIONS = {
    "na": run_na,
    "sv": run_sv,
    "vs": run_vs,
    "dem": run_dem,
    "rel": run_rel,
    "sp": run_sp,
    "haal": run_haal,
    "num": run_num,   # numerals AND ordinals (they share structural detection)
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract controller-target agreement pairs from the "
                    "synchronized BAREC corpus (SYNC).")
    ap.add_argument("--construction", required=True,
                    choices=list(CONSTRUCTIONS) + ["all"],
                    help="Which construction to extract "
                         "(num covers numerals and ordinals); 'all' runs everything")
    ap.add_argument("--sync", required=True, type=Path,
                    help="Root of the synchronized corpus (Step 5 --out)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory for the pair TSVs")
    ap.add_argument("--exclusions", type=Path, default=None,
                    help="Optional directory of *_manual_excludes.tsv files "
                         "(default: none)")
    args = ap.parse_args()

    todo = list(CONSTRUCTIONS) if args.construction == "all" else [args.construction]
    for name in todo:
        print("\n" + "#" * 64)
        print(f"# Extraction: {name}")
        print("#" * 64)
        CONSTRUCTIONS[name](args.sync, args.out, args.exclusions)


if __name__ == "__main__":
    main()
