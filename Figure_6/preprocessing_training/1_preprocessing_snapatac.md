
# Data

Data used for this study is publicly available, see [Sun K. 2024 et al.](https://www.nature.com/articles/s41556-024-01449-0) for more info.
In particular two files were used:
1. [fragments file](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243256/suppl/GSE243256%5FZEPA.All.sample.bed.gz) containing scATAC-seq fragments for all cells.
2. [cell metadata file](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/GSE243256/suppl/GSE243256%5FZEPA%5Fmetadata.xlsx) containing cell metadata (cell type annotation etc.).

# Import data

```python

import snapatac2 as snap
import os
import pandas as pd

out_dir = "sun_et_al_zebrafish_emb_scatac/snapatac/outs"
os.makedirs(out_dir, exist_ok = True)

chromsizes = pd.read_table(
    "https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.chrom.sizes",
    header = None
).set_index(0).to_dict()[1]

data = snap.pp.import_data(
    "sun_et_al_zebrafish_emb_scatac/data/fragments_and_metadata/GSE243256_ZEPA.All.sample.fixed.bed.gz",
    chrom_sizes = chromsizes,
    file = os.path.join(out_dir, "danRer_dev.h5ad"),
    sorted_by_barcode = False
)


data.close()

```
# Add cell metadata
```python

import snapatac2 as snap
import os
import polars as pl

out_dir = "sun_et_al_zebrafish_emb_scatac/snapatac/outs"

metadata = pl.read_csv(
    "sun_et_al_zebrafish_emb_scatac/data/fragments_and_metadata/metadata.tsv",
    separator = "\t") \
    .drop("")

data = snap.read(
    os.path.join(out_dir, "danRer_dev.h5ad"),
    backed = "r")

cb_index = {
    cb: idx
    for idx, cb in enumerate(data.obs_names)
}

data.close()

metadata = metadata.with_columns(
    (
        pl.col("cell_barcodes") \
            .replace(cb_index)
    ) \
    .alias("index") \
    .cast(pl.UInt64)
)

assert len(set(metadata.select(pl.col("cell_barcodes")).to_numpy()[:, 0]) & set(cb_index.keys())) \
    == len(cb_index.keys()), "Some cell barcodes don't have metadata!"

metadata = metadata.sort("index")

obs_to_save = [
    'cell_barcodes',
    'FRIP',
    'Main_Celltypes',
    'DoubletEnrichment',
    'DoubletScore',
    'nFrags',
    'technology',
    'TSSEnrichment',
    'sample',
    'cell_type',
    'stage',
    'experimental_batch',
]

data = snap.read(
    os.path.join(out_dir, "danRer_dev.h5ad"),
    backed = "r+")

result = metadata.select(pl.col(obs_to_save)).to_dict()

for k, v in result.items():
    print(k)
    # fill nulls
    if isinstance(v.dtype, pl.String):
        data.obs[k] = v.fill_null(pl.lit(""))
    else:
        data.obs[k] = v.fill_null(pl.lit(0))

data.close()

```

# Call peaks per cell type

```python

import snapatac2 as snap
import os

out_dir = "sun_et_al_zebrafish_emb_scatac/snapatac/outs"

data = snap.read(
    os.path.join(out_dir, "danRer_dev.h5ad"),
    backed = "r")

peaks = snap.tl.macs3(
    data,
    groupby = "cell_type",
    inplace = False,
    tempdir = "sun_et_al_zebrafish_emb_scatac/snapatac/outs/peak_tmp/",
    n_jobs = 36)

data.close()

for cell_type in peaks.keys():
    print(cell_type)
    peaks[cell_type].write_parquet(
        os.path.join(
            "sun_et_al_zebrafish_emb_scatac/snapatac/outs/peaks_per_cell_type",
            f"{cell_type}_peaks.parquet"
        )
    )

```


# Create consensus peak set and generate insertion peak matrix per cell type

```python

import polars as pl
import os
import snapatac2 as snap

peaks = {}
for f in os.listdir(
    "sun_et_al_zebrafish_emb_scatac/snapatac/outs/peaks_per_cell_type"):
    print(f)
    peaks[f.replace(".parquet", "")] = pl.read_parquet(
        os.path.join(
            "sun_et_al_zebrafish_emb_scatac/snapatac/outs/peaks_per_cell_type",
            f
        )
    )

import pandas as pd
chromsizes = pd.read_table(
    "https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.chrom.sizes",
    header = None
).set_index(0).to_dict()[1]

consensus_peaks = snap.tl.merge_peaks(
    peaks,
    chromsizes
)

consensus_peaks.write_parquet(
    "sun_et_al_zebrafish_emb_scatac/snapatac/outs/consensus_peaks.parquet"
)

consensus_peaks = pl.read_parquet(
    "sun_et_al_zebrafish_emb_scatac/snapatac/outs/consensus_peaks.parquet"
)

out_dir = "sun_et_al_zebrafish_emb_scatac/snapatac/outs"

data = snap.read(
    os.path.join(out_dir, "danRer_dev.h5ad"),
    backed = "r")

snap.pp.make_peak_matrix(
    data,
    use_rep = consensus_peaks["Peaks"],
    file = os.path.join(out_dir, "danRer_dev_insertion_consensus_peak_matrix.h5ad"),
    counting_strategy = "insertion"
)

data.close()

data = snap.read(
    os.path.join(out_dir, "danRer_dev_insertion_consensus_peak_matrix.h5ad"),
    backed = "r")

from snapatac2.tools._misc import aggregate_X

count_per_ct = aggregate_X(
    data, "cell_type",
    normalize = "RPM",
    file = os.path.join(out_dir, "danRer_dev_insertion_consensus_peak_matrix_per_celltype.h5ad")
)

count_per_ct.close()

```
