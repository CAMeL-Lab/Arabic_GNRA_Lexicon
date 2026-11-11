# Preprocessing: synchronize BAREC analyses

The five scripts in this directory align a CATiB-style dependency treebank
(`.conllx`) with a morphological analysis (`.json`) for the same BAREC
documents. They do not run a parser or morphological analyzer. Each input root
must contain corresponding files at the same relative paths, apart from the
file extension. Use Python 3.10 or newer; the scripts use the standard library
and require no additional packages. Run the commands below from the repository
root.

| Stage | Script | Main output in `work/` |
|---|---|---|
| 1 | `barec_step1_file_alignment.py` | `barec_step1_file_alignment.csv`, with matched document paths |
| 2 | `barec_step2_sentence_alignment.py` | `barec_step2_sentence_alignment.csv`, a sentence-count report |
| 3 | `barec_step3_token_alignment.py` | Sentence alignment, mismatch, document-summary, and bad-sentence CSVs |
| 4 | `barec_step4_form_comparison.py` | Form-mismatch and document-summary CSVs |
| 5 | `barec_step5_sync.py` | Synchronized `.conllx` files in the chosen `--out` directory and a summary CSV |

```bash
python3 scripts/preprocessing/barec_step1_file_alignment.py --treebank "/path/to/BAREC/treebank" --morphology "/path/to/BAREC/morphology" --workdir work/
python3 scripts/preprocessing/barec_step2_sentence_alignment.py --workdir work/
python3 scripts/preprocessing/barec_step3_token_alignment.py --workdir work/
python3 scripts/preprocessing/barec_step4_form_comparison.py --workdir work/
python3 scripts/preprocessing/barec_step5_sync.py --workdir work/ --out BAREC_10M_SYNC/
```

Stage 1 records the source paths used by later stages. Stage 2 reports sentence
count differences; stages 3–5 read stage 1's file alignment, while stages 4–5
also consume stage 3's token alignment. Stage 3 drops junk tokens symmetrically
and marks sentences whose token counts still differ. Stage 4 compares
normalized token forms. Stage 5 skips marked sentences and form-mismatched
tokens, merges the remaining morphological features, and remaps dependency
heads after dropping tokens. Its output is the input to
[pair extraction](../extraction/README.md).

The JSON input is expected to contain parallel sentence/token arrays,
including `raw_sents`, `word`, `atbtok`, `pos`, `source`, `gen`, `form_gen`,
`num`, `form_num`, `rat`, `lex`, `prc0`–`prc3`, `enc0`, `pattern`, and `root`.
The `.conllx` input must use `# text =` sentence headers and identify basewords
in the `token_type` feature.

Stages 3–5 accept `--dry-run`, processing one document per domain and using
`_dryrun` output names. Pass that flag to all three stages for a consistent
trial run. Stage 5 refuses to replace an existing output directory unless
`--overwrite` is supplied; that flag deletes the existing output directory.
