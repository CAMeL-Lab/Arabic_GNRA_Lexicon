"""
BAREC-10M Step 1: File-level alignment check.
Compares .conllx syntax files against .json morphology files by document name.
Writes a CSV report. Does not open any file content — filenames only.
"""

import argparse
import csv
from pathlib import Path


def collect(root: Path, suffix: str) -> dict[str, Path]:
    return {
        str(p.relative_to(root).with_suffix("")): p
        for p in sorted(root.rglob(f"*{suffix}"))
        if not p.name.startswith(".")
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 1: file-level alignment of treebank vs. morphology.")
    ap.add_argument("--treebank", required=True, type=Path,
                    help="Root directory of the BAREC treebank (.conllx files)")
    ap.add_argument("--morphology", required=True, type=Path,
                    help="Root directory of the BAREC morphology (.json files)")
    ap.add_argument("--workdir", type=Path, default=Path("work"),
                    help="Directory for pipeline artifact CSVs (default: ./work)")
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    csv_out = args.workdir / "barec_step1_file_alignment.csv"

    syntax = collect(args.treebank, ".conllx")
    morph  = collect(args.morphology,  ".json")

    all_keys = sorted(set(syntax) | set(morph))

    rows = []
    for key in all_keys:
        in_s = key in syntax
        in_m = key in morph
        status = "matched" if (in_s and in_m) else ("syntax_only" if in_s else "morph_only")
        rows.append({
            "document":    key,
            "syntax_path": str(syntax[key]) if in_s else "",
            "morph_path":  str(morph[key])  if in_m else "",
            "status":      status,
        })

    counts = {s: sum(1 for r in rows if r["status"] == s)
              for s in ("matched", "syntax_only", "morph_only")}

    print("=" * 60)
    print("FILE-LEVEL ALIGNMENT SUMMARY")
    print("=" * 60)
    print(f"  Syntax files   (.conllx) : {len(syntax):>6}")
    print(f"  Morphology files (.json) : {len(morph):>6}")
    print(f"  Matched documents        : {counts['matched']:>6}")
    print(f"  Only in syntax           : {counts['syntax_only']:>6}")
    print(f"  Only in morphology       : {counts['morph_only']:>6}")

    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["document", "syntax_path", "morph_path", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV written to: {csv_out}")

    if counts["syntax_only"] == 0 and counts["morph_only"] == 0:
        print("All files matched. Ready for Step 2 (sentence-level alignment).")
    else:
        print("Mismatches found. Investigate before proceeding to Step 2.")


if __name__ == "__main__":
    main()
