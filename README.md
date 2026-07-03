# Temporal-Operator Reanalysis of Sperm Whale Codas

This package contains a self-contained LaTeX manuscript and reproducibility materials for:

**From Coda Categories to Temporal Operators: An Unsupervised Reanalysis of Sperm Whale Coda Structure in Annotated Dominica Data and DSWP Audio Recordings**

## Data sources

1. Annotated coda/dialogue artifacts associated with `pratyushasharma/sw-combinatoriality`:
   - `sperm-whale-dialogues.csv`
   - `rhythms.p`
   - `mean_codas.p`
   - `ornaments.p`
   - `tempos-dict.p`

2. Dominica Sperm Whale Project audio recordings from the Hugging Face dataset `orrp/DSWP`.

The two corpora are treated as independent. No individual DSWP audio recording is assumed to map to an individual row of the annotated dialogue table.

## Main source files

- `main.tex`: LaTeX manuscript.
- `main.pdf`: compiled manuscript.
- `figures/`: publication figures in PDF and PNG formats.
- `tables/`: derived tables used in the manuscript.
- `data/`: tabular/artifact inputs used by the reproduction scripts.
- `scripts/reproduce_statistics.py`: recomputes the annotated-dialogue statistics, audio-density summaries, motif compression, and AUC results from included data tables.
- `scripts/make_figures.py`: regenerates figures from the derived tables.
- `scripts/extract_audio_recording_features.py`: transparent click-detection script for regenerating DSWP audio feature tables from a local DSWP audio snapshot.

## Reproduce the manuscript from included tables

```bash
python scripts/reproduce_statistics.py
python scripts/make_figures.py
latexmk -pdf main.tex
```

The recomputation script writes outputs to `tables_recomputed/` so that the original tables used by the manuscript remain unchanged.

## Reproduce DSWP audio features from raw recordings

Download the `orrp/DSWP` audio dataset locally, then run:

```bash
python scripts/extract_audio_recording_features.py /path/to/DSWP --out data/dswp_audio_recording_features.csv
python scripts/reproduce_statistics.py
python scripts/make_figures.py
latexmk -pdf main.tex
```

Exact audio-derived counts may vary across dataset snapshots, audio decoders, and peak-detection thresholds. Annotated-dialogue values should be exactly reproducible from the included dialogue table and pickle artifacts.
