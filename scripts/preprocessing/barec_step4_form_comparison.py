"""
BAREC-10M Step 4: Token form comparison via atbtok normalization.

For each MATCHED sentence (status="matched" in Step 3 alignment CSV):
  - Get conllx basewords; drop junk indices (from Step 3 conllx_skip_indices)
  - Get JSON atbtok sequence; drop junk indices (from Step 3 json_skip_indices)
  - Skip kept positions where Step 3 marked form_skip_indices (pos=punc/source=digit/backoff/spvar)
  - For each remaining position k:
      conllx_form_norm = normalize_form(filt_basewords[k])
      json_atbtok_norm = normalize_atbtok(filt_atbtoks[k])
      if they differ → record token-level form mismatch

Outputs:
  barec_step4_form_mismatches.csv  — one row per token-level form mismatch
  barec_step4_doc_summary.csv      — per-document tokens_compared / tokens_mismatched / rate

Step 5 sync will read form_mismatches and exclude those token positions from feature merge.
"""

import argparse
import csv
import json
import re
from pathlib import Path


# ============================================================
# Normalization
# ============================================================
# Arabic diacritics: U+064B (fathatan) through U+0652 (sukun) — covers
# tanwin/fatha/damma/kasra/shadda/sukun. Plus U+0670 (superscript alef).
_DIACRITICS_RE = re.compile(r"[ً-ْٰ]")

# Alef variants → bare alef (U+0627)
# أ (0623), إ (0625), آ (0622), ٱ (0671 alef wasla), ٲ (0672), ٳ (0673)
_ALEF_VARIANTS_RE = re.compile(r"[آأإٱٲٳ]")

# Ya / alef-maksura / ya-with-hamza → bare ya (U+064A)
# ى (0649) alef maksura, ئ (0626) ya with hamza
_YA_VARIANTS_RE = re.compile(r"[ىئ]")

# Ta marbuta → ha (U+0647) — common BAREC vs conllx spelling variation
_TA_MARBUTA_RE = re.compile(r"[ة]")

# BAREC's atbtok separator format is INCONSISTENT — both `+_` and `_+` can mark
# either a proclitic or an enclitic boundary depending on the file. We disambiguate
# by splitting on any of the four separator patterns (`+_`, `_+`, `+ـ`, `ـ+`)
# and using the prc1/prc2/prc3 + enc0 field counts to pick the stem segment.
# (prc0=Al_det does NOT add an atbtok separator; the Al stays merged with the noun.)
_SEP_RE = re.compile(r"\+[_ـ]|[_ـ]\+")


def _final_norm(text: str) -> str:
    text = _DIACRITICS_RE.sub("", text)
    text = _ALEF_VARIANTS_RE.sub("ا", text)
    text = _YA_VARIANTS_RE.sub("ي", text)
    text = _TA_MARBUTA_RE.sub("ه", text)
    return text


def normalize_atbtok(atbtok: str, prc0: str, prc1: str, prc2: str, prc3: str, enc0: str) -> str:
    """Strip clitics from atbtok using prc/enc field info, then normalize."""
    if not atbtok:
        return ""
    segments = _SEP_RE.split(atbtok)
    if len(segments) == 1:
        return _final_norm(segments[0])

    def is_clitic(v: str) -> bool:
        return v not in ("0", "na", "")

    # Number of proclitic separators in atbtok.
    # prc0=Al_det does NOT add a separator (Al stays merged); prc0=lA_neg does add one.
    n_pre = sum(1 for v in (prc1, prc2, prc3) if is_clitic(v))
    if prc0 == "lA_neg":
        n_pre += 1
    n_post = 1 if is_clitic(enc0) else 0

    # Stem index = number of preceding proclitics.
    if len(segments) == 1 + n_pre + n_post and n_pre < len(segments):
        stem = segments[n_pre]
    else:
        # Fallback: pick the longest segment.
        stem = max(segments, key=len)
    return _final_norm(stem)


def normalize_form(form: str) -> str:
    """Normalize conllx form: same pipeline as JSON stem (idempotent)."""
    if not form:
        return ""
    return _final_norm(form)


# ============================================================
# Parsers (same as Step 3)
# ============================================================
def parse_conllx_basewords(path: Path) -> list[list[str]]:
    """Return list of sentences. Each sentence: list of baseword forms (in order)."""
    sents: list[list[str]] = []
    cur: list[str] | None = None
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
                    feat = parts[5]
                    ttype = next(
                        (v.split("=", 1)[1] for v in feat.split("|") if v.startswith("token_type=")),
                        "?",
                    )
                    if ttype == "baseword":
                        cur.append(parts[1])
    if cur is not None:
        sents.append(cur)
    return sents


def parse_json_doc(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 4: token form comparison via atbtok normalization.")
    ap.add_argument("--workdir", type=Path, default=Path("work"),
                    help="Directory holding Step 1/3 outputs and receiving Step 4 outputs (default: ./work)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Process one file per domain only; reads/writes _dryrun-suffixed files")
    args = ap.parse_args()
    DRY_RUN = args.dry_run
    suffix = "_dryrun" if DRY_RUN else ""
    step1_csv = args.workdir / "barec_step1_file_alignment.csv"
    step3_csv = args.workdir / f"barec_step3_sent_alignment{suffix}.csv"
    out_mismatches = args.workdir / f"barec_step4_form_mismatches{suffix}.csv"
    out_doc_summary = args.workdir / f"barec_step4_doc_summary{suffix}.csv"

    # Load step1 path lookup
    with step1_csv.open(encoding="utf-8") as f:
        path_lookup = {r["document"]: r for r in csv.DictReader(f) if r["status"] == "matched"}

    # Load step3 alignment — only matched sentences
    sent_align: dict[tuple[str, int], dict] = {}
    with step3_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] == "matched":
                sent_align[(r["document"], int(r["sent_idx"]))] = r

    if DRY_RUN:
        seen = set()
        docs = []
        for doc in sorted(path_lookup.keys()):
            dom = doc.split("/")[0]
            if dom not in seen:
                docs.append(doc)
                seen.add(dom)
        print(f"DRY RUN: {len(docs)} documents (1 per domain)")
    else:
        docs = sorted(path_lookup.keys())
        print(f"FULL RUN: {len(docs)} documents")

    total_tokens = 0
    total_compared = 0
    total_form_skipped = 0
    total_form_mismatches = 0

    mismatch_rows: list[dict] = []
    doc_summary_rows: list[dict] = []

    for idx, doc in enumerate(docs, 1):
        if not DRY_RUN and idx % 1000 == 0:
            print(f"  {idx}/{len(docs)}...")

        p = path_lookup[doc]
        try:
            c_sents = parse_conllx_basewords(Path(p["syntax_path"]))
            j_data  = parse_json_doc(Path(p["morph_path"]))
        except Exception as e:
            print(f"  ERROR parsing {doc}: {e}")
            continue

        n = min(len(c_sents), len(j_data["raw_sents"]))
        doc_compared = 0
        doc_mismatched = 0

        for i in range(n):
            key = (doc, i)
            if key not in sent_align:
                continue  # sentence had count_mismatch — skip entirely

            row = sent_align[key]
            j_skip   = set(json.loads(row["json_skip_indices"]))
            c_skip   = set(json.loads(row["conllx_skip_indices"]))
            form_skip = set(json.loads(row["form_skip_indices"]))

            basewords = c_sents[i]
            j_words   = j_data["word"][i]
            j_atbtoks = j_data["atbtok"][i]
            j_pos     = j_data["pos"][i]
            j_source  = j_data["source"][i]
            j_prc0 = j_data["prc0"][i]
            j_prc1 = j_data["prc1"][i]
            j_prc2 = j_data["prc2"][i]
            j_prc3 = j_data["prc3"][i]
            j_enc0 = j_data["enc0"][i]

            def filt(seq):
                return [v for k, v in enumerate(seq) if k not in j_skip]

            filt_basewords = [b for k, b in enumerate(basewords) if k not in c_skip]
            filt_jwords    = filt(j_words)
            filt_jatbtoks  = filt(j_atbtoks)
            filt_jpos      = filt(j_pos)
            filt_jsource   = filt(j_source)
            filt_jprc0     = filt(j_prc0)
            filt_jprc1     = filt(j_prc1)
            filt_jprc2     = filt(j_prc2)
            filt_jprc3     = filt(j_prc3)
            filt_jenc0     = filt(j_enc0)

            if len(filt_basewords) != len(filt_jwords):
                # Defensive — Step 3 already gated this; skip if drift
                continue

            for kept_idx in range(len(filt_basewords)):
                total_tokens += 1
                if kept_idx in form_skip:
                    total_form_skipped += 1
                    continue

                b  = filt_basewords[kept_idx]
                w  = filt_jwords[kept_idx]
                a  = filt_jatbtoks[kept_idx]
                ps = filt_jpos[kept_idx]
                sr = filt_jsource[kept_idx]

                c_norm = normalize_form(b)
                j_norm = normalize_atbtok(
                    a,
                    filt_jprc0[kept_idx], filt_jprc1[kept_idx],
                    filt_jprc2[kept_idx], filt_jprc3[kept_idx],
                    filt_jenc0[kept_idx],
                )
                total_compared += 1
                doc_compared += 1

                if c_norm != j_norm:
                    total_form_mismatches += 1
                    doc_mismatched += 1
                    mismatch_rows.append({
                        "document":         doc,
                        "sent_idx":         i,
                        "kept_idx":         kept_idx,
                        "conllx_form":      b,
                        "conllx_form_norm": c_norm,
                        "json_word":        w,
                        "json_atbtok":      a,
                        "json_atbtok_norm": j_norm,
                        "json_pos":         ps,
                        "json_source":      sr,
                    })

        doc_summary_rows.append({
            "document": doc,
            "tokens_compared":   doc_compared,
            "tokens_mismatched": doc_mismatched,
            "mismatch_rate": (
                f"{doc_mismatched / doc_compared:.6f}"
                if doc_compared else "0.000000"
            ),
        })

    # ---------- Write outputs ----------
    mismatch_rows.sort(key=lambda r: (r["document"], r["sent_idx"], r["kept_idx"]))
    with out_mismatches.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "document", "sent_idx", "kept_idx",
            "conllx_form", "conllx_form_norm",
            "json_word", "json_atbtok", "json_atbtok_norm",
            "json_pos", "json_source",
        ])
        w.writeheader()
        w.writerows(mismatch_rows)

    with out_doc_summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "document", "tokens_compared", "tokens_mismatched", "mismatch_rate",
        ])
        w.writeheader()
        w.writerows(doc_summary_rows)

    print()
    print("=" * 60)
    print(f"STEP 4 SUMMARY ({'DRY RUN' if DRY_RUN else 'FULL RUN'})")
    print("=" * 60)
    print(f"  Documents processed                  : {len(docs):>10}")
    print(f"  Tokens seen (excl. skipped sentences): {total_tokens:>10}")
    print(f"  Tokens skipped (punc/digit/backoff/spvar): {total_form_skipped:>6}")
    print(f"  Tokens compared                      : {total_compared:>10}")
    print(f"  Form mismatches                      : {total_form_mismatches:>10}")
    if total_compared:
        print(f"  Mismatch rate                        : {100*total_form_mismatches/total_compared:>9.4f}%")
    print()
    print("Outputs:")
    print(f"  {out_mismatches}")
    print(f"  {out_doc_summary}")


if __name__ == "__main__":
    main()
