"""
BAREC-10M Step 2: Sentence-level alignment check.
For each matched document pair, compares sentence counts between
the .conllx syntax file and the .json morphology file.

Reads Step 1 CSV to get matched pairs. Outputs a new CSV report.
"""

import argparse
import csv
import json
from pathlib import Path


def count_conllx_sentences(path: Path) -> int:
    """Count sentences by counting '# text =' header lines."""
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("# text ="):
                count += 1
    return count


def count_json_sentences(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return len(json.load(f)["raw_sents"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 2: sentence-level alignment check.")
    ap.add_argument("--workdir", type=Path, default=Path("work"),
                    help="Directory holding Step 1 output and receiving Step 2 output (default: ./work)")
    args = ap.parse_args()
    step1_csv = args.workdir / "barec_step1_file_alignment.csv"
    csv_out   = args.workdir / "barec_step2_sentence_alignment.csv"

    # Load Step 1 CSV — only process matched pairs
    with step1_csv.open(encoding="utf-8") as f:
        pairs = [r for r in csv.DictReader(f) if r["status"] == "matched"]

    total = len(pairs)
    print(f"Processing {total} matched document pairs...")

    rows = []
    mismatch_count = 0

    for i, pair in enumerate(pairs, 1):
        if i % 1000 == 0:
            print(f"  {i}/{total}...")

        doc          = pair["document"]
        syntax_path  = Path(pair["syntax_path"])
        morph_path   = Path(pair["morph_path"])

        try:
            conllx_sents = count_conllx_sentences(syntax_path)
        except Exception as e:
            conllx_sents = -1
            print(f"  ERROR reading conllx {doc}: {e}")

        try:
            json_sents = count_json_sentences(morph_path)
        except Exception as e:
            json_sents = -1
            print(f"  ERROR reading json {doc}: {e}")

        match  = conllx_sents == json_sents
        diff   = json_sents - conllx_sents  # positive = json has more
        status = "matched" if match else "mismatch"

        if not match:
            mismatch_count += 1

        rows.append({
            "document":     doc,
            "conllx_sents": conllx_sents,
            "json_sents":   json_sents,
            "diff":         diff,
            "status":       status,
        })

    # Sort: mismatches first, then matched
    rows.sort(key=lambda r: (r["status"] == "matched", r["document"]))

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["document", "conllx_sents", "json_sents", "diff", "status"]
        )
        writer.writeheader()
        writer.writerows(rows)

    matched_count = total - mismatch_count
    print("\n" + "=" * 60)
    print("SENTENCE-LEVEL ALIGNMENT SUMMARY")
    print("=" * 60)
    print(f"  Total documents checked  : {total:>6}")
    print(f"  Sentence counts match    : {matched_count:>6}")
    print(f"  Sentence count mismatch  : {mismatch_count:>6}")
    print(f"\nCSV written to: {csv_out}")

    if mismatch_count == 0:
        print("All sentence counts match. Ready for Step 3 (token-level alignment).")
    else:
        print(f"Mismatches found in {mismatch_count} documents. Review CSV before proceeding.")


if __name__ == "__main__":
    main()
