"""
BAREC-10M Step 5: Sync features from JSON morphology into conllx baseword rows.

Per-document, mirrors the input directory structure under the --out root.

For each (matched) sentence:
  - Walk conllx rows in original order
  - Drop clitic rows (token_type != baseword)
  - Drop basewords whose position is in conllx_skip_indices (junk)
  - Drop tokens whose (doc, sent_idx, kept_idx) is in step4 form_mismatches
  - Emit remaining baseword rows with merged JSON features

Skip whole sentence if (doc, sent_idx) is in barec_bad_sentences.csv.

Output columns (29 total):
  ID FORM LEMMA UPOS XPOS
  per asp vox mod
  gen form_gen num form_num cas rat
  pos lex source
  prc0 prc1 prc2 prc3 enc0
  pattern root
  HEAD DEPREL DEPS MISC

The output directory is only replaced when --overwrite is given.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import argparse

OUTPUT_COLUMNS = [
    "ID", "FORM", "LEMMA", "UPOS", "XPOS",
    "per", "asp", "vox", "mod",
    "gen", "form_gen", "num", "form_num", "cas", "rat",
    "pos", "lex", "source",
    "prc0", "prc1", "prc2", "prc3", "enc0",
    "pattern", "root",
    "HEAD", "DEPREL", "DEPS", "MISC",
]


# ============================================================
# Parsing
# ============================================================
def parse_feat2(feat: str) -> dict:
    if feat == "_" or not feat:
        return {}
    return dict(kv.split("=", 1) for kv in feat.split("|") if "=" in kv)


def parse_conllx_full(path: Path) -> list[dict]:
    """Return list of sentences with full row info."""
    sents: list[dict] = []
    cur: dict | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("# text ="):
                cur = {"text": line[9:].strip(), "rows": []}
            elif line.startswith("#"):
                continue
            elif line == "":
                if cur is not None:
                    sents.append(cur)
                    cur = None
            elif cur is not None:
                parts = line.split("\t")
                if len(parts) >= 10:
                    feats = parse_feat2(parts[5])
                    cur["rows"].append({
                        "id":     parts[0],
                        "form":   parts[1],
                        "lemma":  parts[2],
                        "upos":   parts[3],
                        "xpos":   parts[4],
                        "feats":  feats,
                        "head":   parts[6],
                        "deprel": parts[7],
                        "deps":   parts[8],
                        "misc":   parts[9],
                        "token_type": feats.get("token_type", "?"),
                    })
    if cur is not None:
        sents.append(cur)
    return sents


def parse_json_doc(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Per-sentence sync
# ============================================================
def _safe_int(s: str) -> int:
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def sync_sentence(
    c_rows: list[dict],
    j_data: dict,
    sent_idx: int,
    c_skip: set[int],
    j_skip: set[int],
    form_skip: set[int],
    form_mismatch_keys: set[tuple],
    doc: str,
) -> tuple[list[list[str]], dict]:
    """
    Walk conllx rows, classify kept vs dropped, then:
      - Renumber kept rows sequentially (1, 2, 3, ...)
      - Remap HEAD refs through dropped rows: walk up the original tree to the nearest
        kept ancestor (or root) so the output tree stays valid.

    Returns (output_rows, stats).
    """
    j_words = j_data["word"][sent_idx]
    kept_to_jorig = [j for j in range(len(j_words)) if j not in j_skip]
    sorted_c_skip = sorted(c_skip)

    stats = {
        "emitted": 0,
        "skipped_form_mismatch": 0,
        "skipped_junk": 0,
        "form_skip_emitted": 0,
        "clitic_rows_dropped": 0,
    }

    # ---- Pass 1: classify each conllx row, collect kept metadata ----
    kept_orig_ids: set[int] = set()
    kept_meta: list[tuple[dict, int]] = []  # (row, j_orig)

    baseword_pos = 0
    for row in c_rows:
        if row["token_type"] != "baseword":
            stats["clitic_rows_dropped"] += 1
            continue

        cur_idx = baseword_pos
        baseword_pos += 1

        if cur_idx in c_skip:
            stats["skipped_junk"] += 1
            continue

        kept_idx = cur_idx - sum(1 for s in sorted_c_skip if s < cur_idx)

        if (doc, sent_idx, kept_idx) in form_mismatch_keys:
            stats["skipped_form_mismatch"] += 1
            continue

        if kept_idx >= len(kept_to_jorig):
            stats["skipped_junk"] += 1
            continue

        if kept_idx in form_skip:
            stats["form_skip_emitted"] += 1

        j_orig = kept_to_jorig[kept_idx]
        kept_orig_ids.add(_safe_int(row["id"]))
        kept_meta.append((row, j_orig))
        stats["emitted"] += 1

    # ---- Pass 2: build HEAD replacement map ----
    id_to_head = {_safe_int(r["id"]): _safe_int(r["head"]) for r in c_rows}

    def kept_ancestor(orig_id: int) -> int:
        cur = orig_id
        seen: set[int] = set()
        while True:
            if cur == 0:
                return 0
            if cur in kept_orig_ids:
                return cur
            if cur in seen:
                return 0
            seen.add(cur)
            cur = id_to_head.get(cur, 0)

    # ---- Pass 3: renumber & remap ----
    orig_to_new = {_safe_int(row["id"]): new_id for new_id, (row, _) in enumerate(kept_meta, 1)}

    output: list[list[str]] = []
    for new_id, (row, j_orig) in enumerate(kept_meta, 1):
        old_head = _safe_int(row["head"])
        new_head = orig_to_new.get(kept_ancestor(old_head), 0) if old_head != 0 else 0

        out_row = [
            str(new_id),
            row["form"],
            row["lemma"],
            row["upos"],
            row["xpos"],
            row["feats"].get("per", "na"),
            row["feats"].get("asp", "na"),
            row["feats"].get("vox", "na"),
            row["feats"].get("mod", "na"),
            j_data["gen"][sent_idx][j_orig],
            j_data["form_gen"][sent_idx][j_orig],
            j_data["num"][sent_idx][j_orig],
            j_data["form_num"][sent_idx][j_orig],
            row["feats"].get("cas", "na"),
            j_data["rat"][sent_idx][j_orig],
            j_data["pos"][sent_idx][j_orig],
            j_data["lex"][sent_idx][j_orig],
            j_data["source"][sent_idx][j_orig],
            j_data["prc0"][sent_idx][j_orig],
            j_data["prc1"][sent_idx][j_orig],
            j_data["prc2"][sent_idx][j_orig],
            j_data["prc3"][sent_idx][j_orig],
            j_data["enc0"][sent_idx][j_orig],
            j_data["pattern"][sent_idx][j_orig],
            j_data["root"][sent_idx][j_orig],
            str(new_head),
            row["deprel"],
            row["deps"],
            row["misc"],
        ]
        output.append(out_row)

    return output, stats


# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 5: write the synchronized corpus (SYNC).")
    ap.add_argument("--workdir", type=Path, default=Path("work"),
                    help="Directory holding Step 1/3/4 outputs (default: ./work)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output root for the synchronized corpus (e.g. BAREC_10M_SYNC)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Process one file per domain only; output root suffixed _dryrun")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace the output root if it already exists (deletes it first)")
    args = ap.parse_args()
    DRY_RUN = args.dry_run
    suffix = "_dryrun" if DRY_RUN else ""

    step1_csv          = args.workdir / "barec_step1_file_alignment.csv"
    step3_align_csv    = args.workdir / f"barec_step3_sent_alignment{suffix}.csv"
    step4_mismatch_csv = args.workdir / f"barec_step4_form_mismatches{suffix}.csv"
    bad_sents_csv      = args.workdir / f"barec_bad_sentences{suffix}.csv"

    out_root = args.out.with_name(args.out.name + "_dryrun") if DRY_RUN else args.out
    summary_path = args.workdir / f"barec_step5_sync_summary{suffix}.csv"

    print(f"Output root: {out_root}")

    # ---- Load lookups ----
    with step1_csv.open(encoding="utf-8") as f:
        path_lookup = {r["document"]: r for r in csv.DictReader(f) if r["status"] == "matched"}

    with bad_sents_csv.open(encoding="utf-8") as f:
        bad_sents = {(r["document"], int(r["sent_idx"])) for r in csv.DictReader(f)}
    print(f"Loaded {len(bad_sents)} bad sentences (skip whole sentence)")

    with step4_mismatch_csv.open(encoding="utf-8") as f:
        form_mismatches = {
            (r["document"], int(r["sent_idx"]), int(r["kept_idx"]))
            for r in csv.DictReader(f)
        }
    print(f"Loaded {len(form_mismatches)} form-mismatched tokens (skip individual tokens)")

    sent_align: dict[tuple[str, int], dict] = {}
    with step3_align_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] == "matched":
                sent_align[(r["document"], int(r["sent_idx"]))] = r
    print(f"Loaded {len(sent_align)} matched sentence alignments")

    # ---- Pick docs ----
    if DRY_RUN:
        seen = set()
        docs = []
        for doc in sorted(path_lookup.keys()):
            dom = doc.split("/")[0]
            if dom not in seen:
                docs.append(doc)
                seen.add(dom)
        print(f"\nDRY RUN: {len(docs)} documents (1 per domain)")
    else:
        docs = sorted(path_lookup.keys())
        print(f"\nFULL RUN: {len(docs)} documents")

    # ---- Stats ----
    totals = {
        "sents_emitted": 0,
        "sents_skipped_bad": 0,
        "tokens_emitted": 0,
        "tokens_skipped_form_mismatch": 0,
        "tokens_skipped_junk": 0,
        "tokens_form_skip_emitted": 0,
        "clitic_rows_dropped": 0,
    }
    summary_rows: list[dict] = []

    # Replace output dir only when explicitly requested
    if out_root.exists():
        if not args.overwrite:
            sys.exit(
                f"Output root already exists: {out_root}\n"
                "Re-run with --overwrite to replace it (this deletes the directory first)."
            )
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    for idx, doc in enumerate(docs, 1):
        if not DRY_RUN and idx % 1000 == 0:
            print(f"  {idx}/{len(docs)}...")

        p = path_lookup[doc]
        try:
            c_sents = parse_conllx_full(Path(p["syntax_path"]))
            j_data  = parse_json_doc(Path(p["morph_path"]))
        except Exception as e:
            print(f"  ERROR parsing {doc}: {e}", file=sys.stderr)
            continue

        out_path = out_root / f"{doc}.conllx"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        doc_stats = {
            "doc": doc,
            "sents_emitted": 0,
            "sents_skipped_bad": 0,
            "tokens_emitted": 0,
            "tokens_skipped_form_mismatch": 0,
            "tokens_skipped_junk": 0,
            "tokens_form_skip_emitted": 0,
        }

        domain = doc.split("/")[0]
        with out_path.open("w", encoding="utf-8") as out:
            out.write(f"# columns = {' '.join(OUTPUT_COLUMNS)}\n")
            out.write(f"# doc = {doc}\n")
            out.write(f"# domain = {domain}\n")
            out.write("\n")

            n = min(len(c_sents), len(j_data["raw_sents"]))
            for i in range(n):
                key = (doc, i)
                if key in bad_sents or key not in sent_align:
                    doc_stats["sents_skipped_bad"] += 1
                    totals["sents_skipped_bad"] += 1
                    continue

                row = sent_align[key]
                j_skip    = set(json.loads(row["json_skip_indices"]))
                c_skip    = set(json.loads(row["conllx_skip_indices"]))
                form_skip = set(json.loads(row["form_skip_indices"]))

                output_rows, sstats = sync_sentence(
                    c_sents[i]["rows"], j_data, i,
                    c_skip, j_skip, form_skip,
                    form_mismatches, doc,
                )

                if output_rows:
                    out.write(f"# sent_id = {doc}/sent_{i}\n")
                    out.write(f"# text = {j_data['raw_sents'][i]}\n")
                    for orow in output_rows:
                        out.write("\t".join(orow) + "\n")
                    out.write("\n")
                    doc_stats["sents_emitted"] += 1
                    totals["sents_emitted"] += 1

                doc_stats["tokens_emitted"]               += sstats["emitted"]
                doc_stats["tokens_skipped_form_mismatch"] += sstats["skipped_form_mismatch"]
                doc_stats["tokens_skipped_junk"]          += sstats["skipped_junk"]
                doc_stats["tokens_form_skip_emitted"]     += sstats["form_skip_emitted"]
                totals["tokens_emitted"]                   += sstats["emitted"]
                totals["tokens_skipped_form_mismatch"]     += sstats["skipped_form_mismatch"]
                totals["tokens_skipped_junk"]              += sstats["skipped_junk"]
                totals["tokens_form_skip_emitted"]         += sstats["form_skip_emitted"]
                totals["clitic_rows_dropped"]              += sstats["clitic_rows_dropped"]

        summary_rows.append(doc_stats)

    # ---- Write summary ----
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "doc", "sents_emitted", "sents_skipped_bad",
            "tokens_emitted",
            "tokens_skipped_form_mismatch", "tokens_skipped_junk",
            "tokens_form_skip_emitted",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    # ---- Print summary ----
    print()
    print("=" * 64)
    print(f"STEP 5 SYNC SUMMARY ({'DRY RUN' if DRY_RUN else 'FULL RUN'})")
    print("=" * 64)
    print(f"  Documents processed             : {len(docs):>10}")
    print(f"  Sentences emitted               : {totals['sents_emitted']:>10}")
    print(f"  Sentences skipped (bad)         : {totals['sents_skipped_bad']:>10}")
    print(f"  Tokens emitted                  : {totals['tokens_emitted']:>10}")
    print(f"    of which form-skip (punc etc.): {totals['tokens_form_skip_emitted']:>10}")
    print(f"  Tokens dropped (form mismatch)  : {totals['tokens_skipped_form_mismatch']:>10}")
    print(f"  Tokens dropped (junk filter)    : {totals['tokens_skipped_junk']:>10}")
    print(f"  Clitic conllx rows dropped      : {totals['clitic_rows_dropped']:>10}")
    print()
    print(f"Output dir : {out_root}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
