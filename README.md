# Arabic GNRA Lexicon and Agreement Rules

This repository accompanies [*Beyond Gender, Number, and Rationality: Modeling
Extended Arabic Morpho-Syntactic Agreement with Aggregation*](https://aclanthology.org/2026.arabicnlp-main.7.pdf)
by Bailasan Zaina, Salam Khalifa, and Nizar Habash (ArabicNLP 2026).

The main resources are an Arabic aggregation (AGG) lexicon and a machine-readable
inventory of agreement rules. Together they model which target agreement
patterns a noun licenses beyond the default predicted by functional gender,
number, and rationality (GNR).

## AGG lexicon

The [AGG lexicon](lexicon/README.md) assigns noun lexemes one of four
aggregation values. Its [TSV file](lexicon/AGG_lexicon.tsv) includes lemmas,
forms, functional features, rationality, and default and additional agreement
patterns.

| AGG value | Interpretation | Additional agreement option |
|---|---|---|
| `A` — Atomic | Individuated entity | None beyond GNR |
| `C` — Collective | Non-individuated aggregate | None beyond GNR |
| `G` — Groupable | Rational plural construed as a unit | Feminine singular |
| `D` — Divisible | Unit accessed through its members | Masculine plural |

AGG is assigned at the lexeme level; its additional option may appear only
with particular forms. The [lexicon README](lexicon/README.md) describes the
columns and labels in detail.

## Agreement rules

The [rule inventory](rules/rule_inventory.tsv) specifies the controller
features and target agreement patterns for five configurations: Form Match,
Func Match, Form Rules, Func Rules, and Extended (GNRA). Extended retains the
functional GNR defaults and adds options conditioned on AGG and other lexical
properties. The separate [dual-gender noun list](rules/dual_gender_nouns.tsv)
supports nouns that can license either functional gender.

See the [rules README](rules/README.md) for the inventory's columns, rule
semantics, and relationship to the lexicon.

## Repository contents

```text
├── lexicon/
│   ├── README.md
│   └── AGG_lexicon.tsv          # AGG lexical entries for rule evaluation
├── rules/
│   ├── README.md
│   ├── rule_inventory.tsv      # five rule configurations across constructions
│   └── dual_gender_nouns.tsv   # common-gender noun resource
├── scripts/
│   ├── README.md
│   ├── preprocessing/          # align and synchronize BAREC analyses
│   ├── extraction/             # extract agreement pairs
│   └── validation/             # evaluate pairs against the rules
├── README.md
└── LICENSE
```

The [scripts README](scripts/README.md) covers data requirements, commands,
and generated outputs to reproduce the results presented in the paper. The [preprocessing](scripts/preprocessing/README.md),
[extraction](scripts/extraction/README.md), and
[validation](scripts/validation/README.md) READMEs document each stage. BAREC
and its upstream analyses are not distributed in this repository.

## Citation

```bibtex
@inproceedings{zaina-etal-2026-beyond,
  title     = {Beyond Gender, Number, and Rationality: Modeling Extended Arabic Morpho-Syntactic Agreement with Aggregation},
  author    = {Zaina, Bailasan and Khalifa, Salam and Habash, Nizar},
  booktitle = {Proceedings of the Fourth Arabic Natural Language Processing Conference (ArabicNLP 2026)},
  year      = {2026},
  address   = {Budapest, Hungary}
}
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). BAREC and the
upstream CAMeL morphological and parsing resources remain under their own
licenses; this repository does not grant rights to the underlying corpus text.

## Contact

Bailasan Zaina (btz3@georgetown.edu)
Nizar Habash (nizar.habash@nyu.edu)
