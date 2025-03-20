# Figure 4: DeepZebrafish can be used to design cell type-specific enhancers in the whole zebrafish over development.

For this figure we analysed data a zebrafish developmental atlas [Sun K. 2024 et al.](https://www.nature.com/articles/s41556-024-01449-0)

We downloaded following files
1. [fragments file](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243256/suppl/GSE243256%5FZEPA.All.sample.bed.gz) containing scATAC-seq fragments for all cells.
2. [cell metadata file](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243256/suppl/GSE243256%5FZEPA%5Fmetadata.xlsx) containing cell metadata (cell type annotation etc.).

As well as the [Zebrafish genome](https://hgdownload.cse.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.fa.gz) and [chromosome sizes](https://hgdownload.cse.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.chrom.sizes).

First, we processed the data using Snapatac2 to generate cell-type by consensus peak insertion site matrix.

See: **1_preprocessing_snapatac.md** for details.

Next we trained a CREsted model and fine-tuned the model on specific peaks

See: **2_train.py** and **3_finetune.py** for details.

