# Extraction stoplists

[`haal_adverbials.tsv`](haal_adverbials.tsv) lists dictionary forms whose
indefinite-accusative use functions as a manner or domain adverbial rather than
a circumstantial ḥāl. The ḥāl extractor uses it to exclude those candidates.

The file has one lemma per line. Lines beginning with `#` and empty lines are
ignored; the extractor reads the first tab-separated field and normalizes
diacritics for matching. The list was curated for the paper's dataset and may
need review when used with another corpus.
