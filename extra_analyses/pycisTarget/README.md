# pycisTarget motif enrichment analysis

To compare the results from CREsted to a regular (non-deeplearning) motif enrichment tools we performed motif enrichment analysis using [pycisTarget](https://github.com/aertslab/pycistarget).

We performed this analysis on the following datasets:
- PBMC

This procedure has the following steps:

1. Download motif collection(`download_motif_collection.sh`).

2. For each dataset a cisTarget database is generated: `PBMC/create_cistarget_database.sh`

3. Regions to perform motif enrichment on are defined: `PBMC/regions_for_motif_enrichment.sh`

4. Finally, motif enrichment is performed: `PBMC/run_cistarget.sh`

