# BAREC pipeline scripts

These scripts reproduce the BAREC part of the paper's agreement-pair analysis.
They require Python 3.10 or newer and two prepared views of BAREC: dependency
parses (`.conllx`) and morphological analyses (`.json`). The corpus and
upstream analyses are not included in this repository.

Run the stages from the repository root in this order:

1. [Preprocessing](preprocessing/README.md) aligns the two views and writes a
   synchronized corpus of `.conllx` files.
2. [Extraction](extraction/README.md) finds controller–target pairs in that
   synchronized corpus and writes TSV files.
3. [Validation](validation/README.md) applies the [rule inventory](../rules/README.md)
   to those pairs and writes per-pair decisions and accuracy summaries.

Each stage README contains its commands, input requirements, output files, and
optional flags. Run them from the repository root in the order above. Obtain
BAREC from its official release and follow the applicable data license.
