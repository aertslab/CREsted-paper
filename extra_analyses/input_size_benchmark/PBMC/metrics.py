import crested
import os
import anndata
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

def evaluate(d: str) -> dict:
    run_name = os.path.basename(d).replace("_same_shape", "")
    adata = anndata.read_h5ad(
        os.path.join(d, f"adata_{run_name}.h5ad")
    )
    crested.pp.filter_regions_on_specificity(adata, gini_std_threshold=1.0)
    datamodule = crested.tl.data.AnnDataModule(
        adata=adata,
        batch_size=256
    )
    evaluator = crested.tl.Crested(data=datamodule)

    ft_model_name = sorted(
        os.listdir(
            os.path.join(d, "PBMC_INPUT_SIZE", f"{run_name}_finetune", "checkpoints")
        ),
        key = lambda x: int(x.split(".")[0])
    )[-1]

    evaluator.load_model(
        os.path.join(d, "PBMC_INPUT_SIZE", f"{run_name}_finetune",  "checkpoints", ft_model_name),
        compile=True,
    )

    return evaluator.test(return_metrics=True)


GENOME_FA="../../../../../../../resources/hg38/hg38.fa"
GENOME_CHROM_SIZES="../../../../../../../resources/hg38/hg38.chrom.sizes"

crested.register_genome(
    crested.Genome(
        fasta=GENOME_FA,
        chrom_sizes=GENOME_CHROM_SIZES
    )
)

model_dirs = [x for x in os.listdir() if x.startswith("input_")]

model_to_eval = {
    model_dir: evaluate(model_dir)
    for model_dir in tqdm(model_dirs)
}

models_same_shape = [x for x in model_dirs if "same_shape" in x]
models_diff_shape = [x for x in model_dirs if "same_shape" not in x]

models_same_shape = sorted(
    models_same_shape, key = lambda x: int(x.rsplit("_")[-2])
)
models_diff_shape = sorted(
    models_diff_shape, key = lambda x: int(x.rsplit("_")[-2])
)

metrics = pd.DataFrame(model_to_eval)

fig, axs = plt.subplots(nrows = 2,  ncols=4, figsize = (16, 8))
for ax, metric in zip(axs.ravel(), metrics.index):
    ax.scatter(
        [int(x.rsplit("_")[-2]) for x in models_diff_shape],
        metrics.loc[metric, models_diff_shape],
        color = "black"
    )
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_ylabel(metric)
    ax.set_xlabel("input size")
fig.tight_layout()
fig.savefig("figures/metrics_diff_shape.png")

fig, axs = plt.subplots(nrows = 2,  ncols=4, figsize = (16, 8))
for ax, metric in zip(axs.ravel(), metrics.index):
    ax.scatter(
        [int(x.rsplit("_")[-2]) for x in models_same_shape],
        metrics.loc[metric, models_same_shape],
        color = "black"
    )
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_ylabel(metric)
    ax.set_xlabel("input size")
fig.tight_layout()
fig.savefig("figures/metrics_same_shape.png")

metrics.to_csv(
    "metrics.tsv",
    sep = "\t",
    header = True,
    index = True
)