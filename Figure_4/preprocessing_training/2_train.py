import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) == 0:
    print("NO GPUS :( exiting")
    import sys

    sys.exit(1)

import os
import anndata
import crested

data_dir = "sun_et_al_zebrafish_emb_scatac/snapatac/outs"

print("Loading data ...")

adata = anndata.read_h5ad(
    os.path.join(
        data_dir, "danRer_dev_insertion_consensus_peak_matrix_per_celltype.h5ad"
    )
)

print(f"Done loading object: {adata}")

print("Splitting train, test, val ...")

adata.var["chr"] = [x.split(":")[0] for x in adata.var_names]
adata.var["start"] = [int(x.split(":")[1].split("-")[0]) for x in adata.var_names]
adata.var["end"] = [int(x.split(":")[1].split("-")[1]) for x in adata.var_names]

crested.pp.train_val_test_split(
    adata,
    strategy="chr",
    val_chroms=["chr8", "chr10"],
    test_chroms=["chr9", "chr18"],
    random_state=123,
)

print(adata.var["split"].value_counts())
print(adata.var)

print("Extending regions ...")

crested.pp.change_regions_width(
    adata,
    2114,
    chromsizes_file="sun_et_al_zebrafish_emb_scatac/crested/danRer11.chrom.sizes",
)

print("scaling data")

crested.pp.normalize_peaks(adata)

datamodule = crested.tl.data.AnnDataModule(
    adata,
    genome_file="sun_et_al_zebrafish_emb_scatac/crested/danRer11.fa.gz",
    chromsizes_file="sun_et_al_zebrafish_emb_scatac/crested/danRer11.chrom.sizes",
    batch_size=128,
    in_memory=False,
    max_stochastic_shift=3,  # optional augmentation
    always_reverse_complement=True,  # default True
)

from crested.tl.zoo import chrombpnet

print(f"Creating model with {len(adata.obs_names)} classes!")
model_architecture = chrombpnet(
    seq_len=2114,
    num_classes=len(adata.obs_names),
    first_conv_filters=1024,
    num_filters=1024,
)

# Load the default configuration for training a topic classication model
from crested.tl import default_configs, TaskConfig

import keras

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
print("max_weight = 100")
loss = crested.tl.losses._cosinemse_log.CosineMSELogLoss(max_weight=100)
metrics = [
    keras.metrics.MeanAbsoluteError(),
    keras.metrics.MeanSquaredError(),
    keras.metrics.CosineSimilarity(axis=1),
    crested.tl.metrics.PearsonCorrelation(),
    crested.tl.metrics.ConcordanceCorrelationCoefficient(),
    crested.tl.metrics.PearsonCorrelationLog(),
    crested.tl.metrics.ZeroPenaltyMetric(),
]

config = TaskConfig(optimizer, loss, metrics)
print("Starting training ... ")

try:
    trainer = crested.tl.Crested(
        data=datamodule,
        model=model_architecture,
        config=config,
        project_name="danRer_development_rescaled",
        logger="wandb",
    )
    trainer.fit(
        epochs=200,
        early_stopping_patience=6,
        mixed_precision=False,
        learning_rate_reduce=True,
        learning_rate_reduce_patience=3,
    )
except Exception as e:
    import wandb

    wandb.finish(1)
    raise e
