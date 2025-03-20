import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) == 0:
    print("NO GPUS :(")
    sys.exit(0)
else:
    print("Yay! GPUS!")


import os
import anndata
import crested
import keras

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

print("FILTERING")

crested.pp.filter_regions_on_specificity(adata, gini_std_threshold=1.0)
adata


datamodule = crested.tl.data.AnnDataModule(
    adata,
    genome_file="sun_et_al_zebrafish_emb_scatac/crested/danRer11.fa.gz",
    chromsizes_file="sun_et_al_zebrafish_emb_scatac/crested/danRer11.chrom.sizes",
    batch_size=128,
    in_memory=False,
    max_stochastic_shift=3,  # optional augmentation
    always_reverse_complement=True,  # default True
)


print("Loading model ...")
model_architecture = keras.models.load_model(
    "sun_et_al_zebrafish_emb_scatac/crested/danRer_development_rescaled/2024-08-30_15:37/checkpoints/53.keras",
    compile=False,
)

import keras

optimizer = keras.optimizers.Adam(learning_rate=2e-6)
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

from crested.tl import default_configs, TaskConfig

config = TaskConfig(optimizer, loss, metrics)


try:
    trainer = crested.tl.Crested(
        data=datamodule,
        model=model_architecture,
        config=config,
        project_name="danRer_development_rescaled_finetune",
        logger="wandb",
    )
    trainer.fit(
        epochs=60,
        early_stopping_patience=6,
        mixed_precision=False,
        learning_rate_reduce=True,
        learning_rate_reduce_patience=3,
    )
except Exception as e:
    import wandb

    wandb.finish(1)
    raise e
