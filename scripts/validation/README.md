# Validation: evaluate agreement rules

[`validate_pairs.py`](validate_pairs.py) reads the TSVs produced by
[extraction](../extraction/README.md) and evaluates five rule configurations:
Form Match, Func Match, Form Rules, Func Rules, and Extended. The machine-readable
[rule inventory](../../rules/README.md) and dual-gender noun list are loaded
from `rules/` by default.

Run from the repository root after extraction:

```bash
python3 scripts/validation/validate_pairs.py --construction all --pairs pairs/ --out reports/ --agg-lexicon lexicon/AGG_lexicon.tsv
```

`--construction` accepts `na`, `sv`, `vs`, `sp`, `dem`, `rel`, `haal`, `num`,
`ord`, or `all`. Extraction's `num` flag creates both the cardinal files and
`ordinal_noun_pairs.tsv`; validation evaluates them with `num` and `ord`,
respectively. The `all` flag evaluates both.

The optional `--agg-lexicon` argument reads the bundled
[AGG lexicon](../../lexicon/README.md). If omitted, the AGG-conditioned
Extended rules never fire. `--rules` and `--dual-gender` can override their
respective default TSV files.

For each input pair file, the script writes a `*_validated.tsv` under
`--out`. The `pred_*` fields contain licensed agreement outputs (for example,
`MS` or `MS|MP`); the `match_*` fields contain `matched`, `mismatched`, or
`na`. Matching compares the observed target's **functional** features with
the licensed outputs. Skipped rows, missing features, and out-of-scope cases
can be marked `na`.

`accuracy_summary.tsv` reports matched, mismatched, and `na` counts by
construction, numeral band where relevant, and rule configuration. Accuracy
is `matched / (matched + mismatched)`; `na` rows are excluded from that ratio.
