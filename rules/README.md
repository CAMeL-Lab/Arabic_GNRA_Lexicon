# Rule inventory

[`rule_inventory.tsv`](rule_inventory.tsv) is the machine-readable agreement
rule inventory used by [`scripts/validation/validate_pairs.py`](../scripts/validation/validate_pairs.py).
It has one row per rule and covers five configurations for each evaluated
construction. [`dual_gender_nouns.tsv`](dual_gender_nouns.tsv) is a separate
list of common-gender nouns used by the Extended configuration. The validator
loads both files by default.

The [AGG lexicon](../lexicon/README.md) is a separate resource: its noun labels
are matched against `agg_value` conditions in this inventory when
`--agg-lexicon` is supplied to the validator.

## Columns

| Column | Meaning |
|---|---|
| `construction` | `noun_adj`, `sv`, `vs`, `subj_pred`, `dem`, `rel`, `haal_dp` (nominal controller), `haal_verb` (pro-drop verb controller), `num_noun`, `ordinal` |
| `rule_set` | `Form Match`, `Func Match`, `Form Rules`, `Func Rules`, `Extended` — the five evaluated configurations |
| `rule_name` | Unique rule identifier within its construction + rule set |
| `form_gen` / `form_num` / `form_rat` | Conditions on the controller's form features F(N); `*` = unconstrained |
| `func_gen` / `func_num` / `func_rat` | Conditions on the controller's functional features G(N); `*` = unconstrained |
| `agg_value` | Aggregation (AGG) condition: `G` (Groupable), `D` (Divisible), or `*` (none) |
| `num_range` | Numeral/ordinal band (`1`, `2`, `3-10`, `11-12`, `13-19`, `21-99`, `20-90`, `1-10`); empty for non-numeral constructions |
| `output` | Licensed target agreement (e.g. `MS`, `FS`, `MP`; numerals: gender only, compounds as `Unit=… ∧ Ten=…`; `NA` for the gender-less ʿuqūd decades) |
| `member_access` | `LICENSED` (Divisible member access), `BLOCKED` (Groupable unit reading), `*` otherwise — a semantic readout, not an input condition |
| `formula` | Human-readable statement of the rule. The formula column stays in the TSV purely as human documentation; the validator never reads it — matching is done entirely on the input/output columns. |

## Semantics

- Within a rule set, **every rule whose conditions match the controller
  contributes its output to the licensed set**. In the baseline sets each
  controller matches exactly one rule; in `Extended` the
  AGG-conditioned rules (`*_EXT_G`, `*_EXT_D`) and construction-specific
  extension rules license additional options on top of the functional default.
- Dual-gender (common-gender) nouns are handled lexically by the validator
  (membership lookup against a dual-gender noun list), not by rules in this
  inventory.
