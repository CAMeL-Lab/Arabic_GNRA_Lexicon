"""
BAREC-10M Step 3: Token-count alignment with junk filter and skip tracking.

For each matched document pair (from Step 1):
- Extracts conllx basewords (clitic rows dropped) and JSON word tokens.
- Applies a symmetric junk filter (tokens with no Arabic letter, no basic
  Latin letter, and no digit are dropped on both sides).
- Records per-sentence skip indices (for Step 5 sync) and a form-skip mask
  (kept tokens excluded from Step 4 form comparison: punctuation, digits,
  backoff/spvar analyses).
- Writes a per-sentence alignment CSV, a mismatch report, a per-document
  summary, and the bad-sentences list consumed by Step 5.

Use --dry-run to process one file per domain (fast setup check); outputs are
then suffixed `_dryrun` so they don't overwrite full-run files.
"""

import argparse
import csv
import difflib
import json
import unicodedata
from pathlib import Path


# ============================================================
# Junk filter
# ============================================================
def is_junk(tok: str) -> bool:
    """
    A token is JUNK if it contains no Arabic letter, no basic Latin letter, and no digit.

    Implementation uses Unicode categories so that Arabic PUNCTUATION (`،`, `؛`, `؟`,
    Tatweel etc. — category P*) is correctly classified as junk even though it lives in
    the Arabic block. Greek letters (`μ`, `α`, `θ`) and math italics (`𝜃`) are also junk
    because they're not in the Arabic block and not basic ASCII Latin.

    Keeps:
      - Arabic letters (Arabic block 0600-06FF or 0750-077F with Unicode category L*)
      - Basic ASCII Latin letters (a-z, A-Z)
      - Digits (Unicode category Nd → covers ASCII 0-9 and Arabic-Indic ٠-٩)

    Drops:
      - Pure-symbol tokens: •, °, →, ∞, ✓, ❐, ⠀
      - Greek/math letters: μ, α, θ, σ, λ, 𝜃
      - Quran ornate parens: ﴾, ﴿
      - Arabic punctuation: ،, ؛, ؟, ۔
      - ASCII punctuation: ., ,, :, (, ), ", ', etc.
      - Empty / whitespace-only tokens
    """
    if not tok or tok.isspace():
        return True
    for ch in tok:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            c = ord(ch)
            # Arabic block letters (Lo for Arabic letters)
            if 0x0600 <= c <= 0x06FF or 0x0750 <= c <= 0x077F:
                return False
            # Basic ASCII Latin letters only (a-z, A-Z)
            if 0x0041 <= c <= 0x005A or 0x0061 <= c <= 0x007A:
                return False
            # Other letters (Greek, math italics, Cyrillic, etc.) → fall through, considered junk
            continue
        if cat == "Nd":  # decimal digit (ASCII or Arabic-Indic)
            return False
    return True


# ============================================================
# Parsers
# ============================================================
def parse_conllx(path: Path) -> list[list[tuple[str, str]]]:
    """Return list of sentences. Each sentence: list of (form, token_type) tuples."""
    sents: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("# text ="):
                cur = []
            elif line.startswith("#"):
                continue
            elif line == "":
                if cur is not None:
                    sents.append(cur)
                    cur = None
            elif cur is not None:
                parts = line.split("\t")
                if len(parts) >= 6:
                    form = parts[1]
                    feat = parts[5]
                    ttype = next(
                        (v.split("=", 1)[1] for v in feat.split("|") if v.startswith("token_type=")),
                        "?",
                    )
                    cur.append((form, ttype))
    if cur is not None:
        sents.append(cur)
    return sents


def parse_json_doc(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Per-sentence alignment
# ============================================================
def align_sentence(
    c_tokens: list[tuple[str, str]],
    j_words: list[str],
    j_pos: list[str],
    j_source: list[str],
) -> dict:
    """
    Build per-sentence alignment record:
      - extract conllx basewords (filter clitics)
      - apply junk filter symmetrically
      - record indices of stripped tokens (for sync)
      - record form-skip mask for kept JSON tokens
    """
    basewords = [form for (form, t) in c_tokens if t == "baseword"]

    c_junk_mask = [is_junk(b) for b in basewords]
    j_junk_mask = [is_junk(w) for w in j_words]

    filt_c = [b for b, junk in zip(basewords, c_junk_mask) if not junk]
    filt_j = [w for w, junk in zip(j_words, j_junk_mask) if not junk]

    c_skip_idx = [i for i, junk in enumerate(c_junk_mask) if junk]
    j_skip_idx = [i for i, junk in enumerate(j_junk_mask) if junk]

    # Form skip indices: positions in the FILTERED json list where the token should be
    # skipped in form comparison (pos=punc / source in {punc, digit, backoff, spvar}).
    # Stored as indices (sparse) instead of full mask (dense) for compactness.
    form_skip_idx: list[int] = []
    kept_pos = 0
    for w, p, s, junk in zip(j_words, j_pos, j_source, j_junk_mask):
        if junk:
            continue
        if (p == "punc") or (s in ("punc", "digit", "backoff", "spvar")):
            form_skip_idx.append(kept_pos)
        kept_pos += 1

    return {
        "basewords": basewords,
        "filt_c": filt_c,
        "filt_j": filt_j,
        "c_skip_idx": c_skip_idx,
        "j_skip_idx": j_skip_idx,
        "form_skip_idx": form_skip_idx,
        "counts_match": len(filt_c) == len(filt_j),
        "diff": len(filt_j) - len(filt_c),
    }


def find_extras(c_tokens: list[str], j_tokens: list[str]) -> tuple[list[str], list[str]]:
    """
    Sequence-aligned diff. Reports tokens genuinely inserted or deleted.

    Important: when difflib emits an unequal-length 'replace' opcode (which happens when
    nearby tokens differ by surface form due to proclitic attachment), the excess tokens
    from the longer side are still genuinely extra and must be reported. Without this,
    extras can come back blank even though counts differ.
    """
    matcher = difflib.SequenceMatcher(None, c_tokens, j_tokens, autojunk=False)
    extra_in_json: list[str] = []
    extra_in_conllx: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            extra_in_json.extend(j_tokens[j1:j2])
        elif tag == "delete":
            extra_in_conllx.extend(c_tokens[i1:i2])
        elif tag == "replace":
            c_len = i2 - i1
            j_len = j2 - j1
            if j_len > c_len:
                # Excess tokens at the tail of the j-side region → genuine extras
                extra_in_json.extend(j_tokens[j1 + c_len : j2])
            elif c_len > j_len:
                extra_in_conllx.extend(c_tokens[i1 + j_len : i2])
    return extra_in_json, extra_in_conllx


# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 3: token-count alignment with junk filter.")
    ap.add_argument("--workdir", type=Path, default=Path("work"),
                    help="Directory holding Step 1 output and receiving Step 3 outputs (default: ./work)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Process one file per domain only; outputs suffixed _dryrun")
    args = ap.parse_args()
    DRY_RUN = args.dry_run
    step1_csv = args.workdir / "barec_step1_file_alignment.csv"

    suffix = "_dryrun" if DRY_RUN else ""
    out_doc_summary  = args.workdir / f"barec_step3_doc_summary{suffix}.csv"
    out_sent_align   = args.workdir / f"barec_step3_sent_alignment{suffix}.csv"
    out_sent_mismatch = args.workdir / f"barec_step3_sent_mismatches{suffix}.csv"
    out_bad_sents    = args.workdir / f"barec_bad_sentences{suffix}.csv"

    with step1_csv.open(encoding="utf-8") as f:
        all_pairs = [r for r in csv.DictReader(f) if r["status"] == "matched"]

    if DRY_RUN:
        seen = set()
        pairs = []
        for p in all_pairs:
            dom = p["document"].split("/")[0]
            if dom not in seen:
                pairs.append(p)
                seen.add(dom)
        print(f"DRY RUN: {len(pairs)} files (1 per domain).")
    else:
        pairs = all_pairs
        print(f"FULL RUN: {len(pairs)} files.")

    total_sents = 0
    sents_matched = 0
    sents_mismatched = 0
    sents_recovered = 0  # was mismatch v1, now matched after junk filter

    doc_rows: list[dict] = []
    sent_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    bad_rows: list[dict] = []

    for idx, pair in enumerate(pairs, 1):
        if not DRY_RUN and idx % 1000 == 0:
            print(f"  {idx}/{len(pairs)}...")

        doc = pair["document"]
        try:
            c_sents = parse_conllx(Path(pair["syntax_path"]))
            j_data  = parse_json_doc(Path(pair["morph_path"]))
        except Exception as e:
            print(f"  ERROR parsing {doc}: {e}")
            continue

        n = min(len(c_sents), len(j_data["raw_sents"]))
        doc_matched = doc_mismatched = doc_recovered = 0

        for i in range(n):
            total_sents += 1

            r = align_sentence(
                c_sents[i],
                j_data["word"][i],
                j_data["pos"][i],
                j_data["source"][i],
            )

            raw_c = len(r["basewords"])
            raw_j = len(j_data["word"][i])
            raw_match = raw_c == raw_j
            filt_c = len(r["filt_c"])
            filt_j = len(r["filt_j"])

            if r["counts_match"]:
                sents_matched += 1
                doc_matched += 1
                if not raw_match:
                    sents_recovered += 1
                    doc_recovered += 1
                status = "matched"
                skip = "False"
            else:
                sents_mismatched += 1
                doc_mismatched += 1
                status = "mismatch"
                skip = "True"

                extra_j, extra_c = find_extras(r["filt_c"], r["filt_j"])
                mismatch_rows.append({
                    "document": doc,
                    "sent_idx": i,
                    "sent_text": j_data["raw_sents"][i],
                    "conllx_count_raw": raw_c,
                    "conllx_count_filtered": filt_c,
                    "json_count_raw": raw_j,
                    "json_count_filtered": filt_j,
                    "diff": r["diff"],
                    "conllx_filtered_tokens": " | ".join(r["filt_c"]),
                    "json_filtered_tokens":   " | ".join(r["filt_j"]),
                    "extra_in_json":   " | ".join(extra_j),
                    "extra_in_conllx": " | ".join(extra_c),
                })
                bad_rows.append({
                    "document": doc,
                    "sent_idx": i,
                    "reason": "count_mismatch_after_junk_filter",
                })

            sent_rows.append({
                "document": doc,
                "sent_idx": i,
                "conllx_count_raw": raw_c,
                "conllx_count_filtered": filt_c,
                "json_count_raw": raw_j,
                "json_count_filtered": filt_j,
                "json_skip_indices":   json.dumps(r["j_skip_idx"], separators=(",", ":")),
                "conllx_skip_indices": json.dumps(r["c_skip_idx"], separators=(",", ":")),
                "form_skip_indices":   json.dumps(r["form_skip_idx"], separators=(",", ":")),
                "status": status,
                "skip": skip,
            })

        doc_rows.append({
            "document": doc,
            "total_sents": n,
            "sents_matched": doc_matched,
            "sents_mismatched": doc_mismatched,
            "sents_recovered_by_junk_filter": doc_recovered,
        })

    # ---------- Write outputs ----------
    with out_doc_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "document", "total_sents", "sents_matched",
            "sents_mismatched", "sents_recovered_by_junk_filter",
        ])
        w.writeheader()
        w.writerows(doc_rows)

    sent_rows.sort(key=lambda r: (r["document"], r["sent_idx"]))
    with out_sent_align.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "document", "sent_idx",
            "conllx_count_raw", "conllx_count_filtered",
            "json_count_raw",   "json_count_filtered",
            "json_skip_indices", "conllx_skip_indices", "form_skip_indices",
            "status", "skip",
        ])
        w.writeheader()
        w.writerows(sent_rows)

    mismatch_rows.sort(key=lambda r: (r["document"], r["sent_idx"]))
    with out_sent_mismatch.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "document", "sent_idx", "sent_text",
            "conllx_count_raw", "conllx_count_filtered",
            "json_count_raw",   "json_count_filtered",
            "diff",
            "conllx_filtered_tokens", "json_filtered_tokens",
            "extra_in_json", "extra_in_conllx",
        ])
        w.writeheader()
        w.writerows(mismatch_rows)

    bad_rows.sort(key=lambda r: (r["document"], r["sent_idx"]))
    with out_bad_sents.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["document", "sent_idx", "reason"])
        w.writeheader()
        w.writerows(bad_rows)

    print()
    print("=" * 60)
    print(f"STEP 3 v2 SUMMARY ({'DRY RUN' if DRY_RUN else 'FULL RUN'})")
    print("=" * 60)
    print(f"  Documents processed             : {len(pairs):>8}")
    print(f"  Total sentences                 : {total_sents:>8}")
    print(f"  Sentences matched (after junk)  : {sents_matched:>8}")
    print(f"  Sentences still mismatched      : {sents_mismatched:>8}")
    print(f"  Recovered by junk filter        : {sents_recovered:>8}")
    if total_sents:
        print(f"  Mismatch rate                   : {100*sents_mismatched/total_sents:>7.2f}%")
    print()
    print("Outputs:")
    print(f"  {out_doc_summary}")
    print(f"  {out_sent_align}")
    print(f"  {out_sent_mismatch}")
    print(f"  {out_bad_sents}")


if __name__ == "__main__":
    main()
