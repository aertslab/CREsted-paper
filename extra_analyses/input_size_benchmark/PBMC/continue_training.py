#!/data/groups/vib.ai/stein.aerts/sdewin/software/miniforge3/envs/crested_1.5.0/bin/python

import crested
import anndata
import keras
import argparse
import os

def train(adata: anndata.AnnData, run_id: str):
    
    model_f = sorted(os.listdir(f"PBMC_INPUT_SIZE/{run_id}/checkpoints"), key = lambda x: int(x.split(".")[0]))[-1]
 
    model = keras.saving.load_model(
        f"PBMC_INPUT_SIZE/{run_id}/checkpoints/{model_f}"
    )

    datamodule = crested.tl.data.AnnDataModule(
        adata=adata,
        batch_size=128,
        max_stochastic_shift=3,
        always_reverse_complement=True
    )
 
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss = crested.tl.losses.CosineMSELogLoss(max_weight=100)
    metrics = [
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.MeanSquaredError(),
                keras.metrics.CosineSimilarity(axis=1),
                crested.tl.metrics.PearsonCorrelation(),
                crested.tl.metrics.ConcordanceCorrelationCoefficient(),
                crested.tl.metrics.PearsonCorrelationLog(),
                crested.tl.metrics.ZeroPenaltyMetric(),
    ]

    config = crested.tl.TaskConfig(optimizer=optimizer, loss=loss, metrics=metrics)

    trainer = crested.tl.Crested(
        data = datamodule,
        model = model,
        config = config,
        project_name = "PBMC_INPUT_SIZE",
        run_name = run_id,
        logger = 'wandb',
    )

    trainer.fit(epochs=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model_directory", required = True, type = str)

    args = parser.parse_args()
    model_dir = args.model_directory

    os.chdir(model_dir)

    run_id = model_dir.replace("same_shape_", "")

    adata = anndata.read_h5ad(f"adata_{run_id}.h5ad")

    GENOME_FA="../../../../../../../../resources/hg38/hg38.fa"
    GENOME_CHROM_SIZES="../../../../../../../../resources/hg38/hg38.chrom.sizes"

    crested.register_genome(
        crested.Genome(
            fasta=GENOME_FA,
            chrom_sizes=GENOME_CHROM_SIZES
        )
    )


    train(adata, run_id)


if __name__ == "__main__":
    main()
