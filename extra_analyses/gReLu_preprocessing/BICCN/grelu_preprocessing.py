from grelu.data import preprocess                                 # type: ignore
import grelu.io.bed                                               # type: ignore
import os
from tqdm import tqdm                                             # type: ignore
import pandas as pd                                               # type: ignore
import grelu.data.dataset                                         # type: ignore
import anndata                                                    # type: ignore

GENOME="mm10"
SEQ_LEN=2114
LABEL_LEN=1000
VAL_CHROMS=["chr8", "chr10"]
TEST_CHROMS=["chr9", "chr18"]

for frag in tqdm(os.listdir("pseudobulk"), total = 19):
    if not frag.endswith(".tsv.gz"):
        continue
    bw_file = preprocess.make_insertion_bigwig(
        frag_file=os.path.join("pseudobulk", frag),
        plus_shift=0,
        minus_shift=1,
        genome=GENOME,
        chroms="autosomes"
    )

bw_files = [
    x for x in os.listdir()
    if x.endswith(".fragments.tsv.bw")
]

peaks = grelu.io.bed.read_bed("crested_data/Figure_2/consensus_regions.bed")


peaks = grelu.data.preprocess.filter_chromosomes(peaks, 'autosomes')

peaks = preprocess.filter_blacklist(
    peaks,
    genome=GENOME,
    window=50 # Remove peaks if they are within 50 bp of a blacklist region
)


negatives = preprocess.get_gc_matched_intervals(
    peaks,
    binwidth=0.02, # resolution of measuring GC content
    genome=GENOME,
    chroms=list(set(peaks["chrom"])),
    seed=0,
)

regions = pd.concat([peaks, negatives])

train, val, test = grelu.data.preprocess.split(
    regions, val_chroms=VAL_CHROMS, test_chroms=TEST_CHROMS)

train_ds = grelu.data.dataset.BigWigSeqDataset(
    intervals = train,
    bw_files=bw_files,
    label_len=LABEL_LEN,
    label_aggfunc="sum",
    rc=False, # reverse complement
    max_seq_shift=0, # Shift the sequence
    max_pair_shift=0, # Shift both sequence and label
    augment_mode="serial",
    seed=0,
    genome=GENOME,
)

data = {}

for (_, label), (_, (chrom, start, end)) in tqdm(
    zip(
        train_ds, train[["chrom", "start", "end"]].iterrows(),
    ),
    total=len(train_ds)
):
    data[f"{chrom}:{start}-{end}"] = label.numpy().squeeze()

df = pd.DataFrame(
    data,
).T

df.columns = [
    x.replace(".fragments.tsv", "")
    for x in train_ds.tasks.index
]

train.index = df.index

adata = anndata.AnnData(
    df,
    obs = train[
        ["chrom", "start", "end"]
    ].rename({"chrom": "chr"}, axis = 1)
)

adata.write_h5ad(
    "adata_biccn_grelu.h5ad"
)