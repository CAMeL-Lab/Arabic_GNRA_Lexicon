# Extraction: controller–target pairs

[`extract_pairs.py`](extract_pairs.py) reads the synchronized `.conllx` corpus
written by [preprocessing](../preprocessing/README.md). It identifies syntactic
controller–target configurations, attaches their analyzed features, and writes
TSV files under `--out`. [`barec_pair_utils.py`](barec_pair_utils.py) provides
the shared parser, output schemas, and skip classifiers.

Run from the repository root:

```bash
python3 scripts/extraction/extract_pairs.py --construction all --sync BAREC_10M_SYNC/ --out pairs/
```

| Extraction flag | Construction | Main output |
|---|---|---|
| `na` | Noun–adjective | `noun_adj_pairs.tsv` |
| `sv` | Subject–verb | `sv_pairs.tsv` |
| `vs` | Verb–subject | `vs_pairs.tsv` |
| `sp` | Subject–predicate | `subj_pred_pairs.tsv` |
| `dem` | Demonstrative–noun | `dem_pairs.tsv` |
| `rel` | Noun–relative pronoun | `rel_pairs.tsv` |
| `haal` | Circumstantial ḥāl | `haal_triples.tsv` |
| `num` | Numeral–noun **and ordinal–noun** | Five `num_noun_*_pairs.tsv` files and `ordinal_noun_pairs.tsv` |

`all` runs every extraction flag. There is no separate `ord` extraction flag:
`num` produces the ordinal file because the two share structural detection.
The validator accepts `ord` separately.

Some constructions write a matching `*_skipped.tsv` with excluded candidates;
others retain flagged rows with a `skip_reason` in the pair file. Extraction
also writes summaries of candidate and skip counts. Noun–adjective extraction
excludes definiteness-mismatch predicates and likely wrong-head iḍāfa
attachments before output.

The `haal` extractor uses the curated
[adverbial stoplist](stoplists/README.md) to exclude manner/domain adverbials.
The optional `--exclusions DIR` flag reads construction-specific
`*_manual_excludes.tsv` files from `DIR` when present; each data row has four
tab-separated fields: document, sentence ID, controller ID, target ID.
Without this flag, no manual exclusions are applied.

Extraction does not score agreement. Pass its pair files to
[validation](../validation/README.md) for predictions and match decisions.
