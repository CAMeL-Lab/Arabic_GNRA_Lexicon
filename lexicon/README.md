# Aggregation lexicon

[`AGG_lexicon.tsv`](AGG_lexicon.tsv) is the lexical resource accompanying the
paper's aggregation (AGG) feature. Its 952 rows describe Arabic noun lemmas and
forms; a lemma may appear in more than one row. Use this TSV with the
validator.

AGG is a property of the lexeme. Gender, number, and rationality give the
default agreement pattern; AGG indicates whether an additional agreement
reading is available.

| `AGR` | Name | Additional agreement reading |
|---|---|---|
| `A` | Atomic | None beyond the GNR default |
| `C` | Collective | None beyond the GNR default |
| `G` | Groupable | A rational plural can be construed as a unit, licensing feminine singular |
| `D` | Divisible | A unit's members can be accessed, licensing masculine plural |

The TSV is UTF-8 and tab-separated. `Lemma` and `Form` identify the noun;
`AGR` holds one of the four labels above. Other columns provide a gloss,
functional gender and number, rationality, default GNR agreement, and extra
agreement. See the [paper](https://aclanthology.org/2026.arabicnlp-main.7.pdf)
for the annotation criteria and linguistic interpretation.

To use the lexicon, pass `--agg-lexicon lexicon/AGG_lexicon.tsv` to
[`scripts/validation/validate_pairs.py`](../scripts/validation/validate_pairs.py).
The validator accepts the `AGR` header and matches controller lemmas or forms
after removing diacritics. It uses `G` and `D` labels in the AGG-conditioned
Extended rules; `A` and `C` do not add an agreement option.
