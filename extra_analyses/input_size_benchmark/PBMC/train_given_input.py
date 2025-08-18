#!/data/groups/vib.ai/stein.aerts/sdewin/software/miniforge3/envs/crested_1.5.0/bin/python

import crested
import anndata
import keras
from datetime import datetime
import argparse
import os

now = datetime.now()
timestamp = str(now.year)
timestamp += str(now.month)
timestamp += str(now.day)
timestamp += str(now.hour)
timestamp += str(now.minute)
timestamp += str(now.second)

BIGWIGS_FOLDER="../crested_data/Figure_3/pbmc/bw/"
REGION_FILE="../crested_data/Figure_3/pbmc/consensus_regions.bed"

def calc_output_width_after_conv(input_width: int, kernel_size: int, dilation_factor: int) -> int:
    k_eff = dilation_factor * (kernel_size - 1) + 1
    return input_width - k_eff + 1

def calc_max_number_of_dil_layers(input_width: int, kernel_size) -> int:
    i = 0
    while input_width > 0:
        input_width = calc_output_width_after_conv(input_width, kernel_size, 2**(i + 1))
        i += 1
    return i - 1

def preprocess(input_size: int) -> anndata.AnnData:
    adata = crested.import_bigwigs(
        bigwigs_folder=BIGWIGS_FOLDER,
        regions_file=REGION_FILE,
        target_region_width=1000,
        target="mean",
    )
    crested.pp.train_val_test_split(adata, strategy="chr", val_chroms=["chr8", "chr10"], test_chroms=["chr9", "chr18"])
    print(f"CHANGING REGIONS WIDTH TO: {input_size}")
    crested.pp.change_regions_width(adata=adata, width=input_size)
    crested.pp.normalize_peaks(adata, top_k_percent = 0.03)
    return adata


def train(adata: anndata.AnnData, input_size: int):
    n_dil_layers = calc_max_number_of_dil_layers(input_size, kernel_size=3) - 1
    print(f"Setting up model training for input size: {input_size} using {n_dil_layers} dilation layers")
    datamodule = crested.tl.data.AnnDataModule(
        adata=adata,
        batch_size=128,
        max_stochastic_shift=3,
        always_reverse_complement=True
    )

    model_architecture = crested.tl.zoo.dilated_cnn(
        seq_len=input_size,
        num_classes=7,
        n_dil_layers=n_dil_layers
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
        model = model_architecture,
        config = config,
        project_name = "PBMC_INPUT_SIZE",
        run_name = f"input_{input_size}_{timestamp}",
        logger = 'wandb',
    )

    trainer.fit(epochs=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_width", required = True, type = int)

    args = parser.parse_args()
    input_size = args.input_width

    # change directory because import_bigwigs
    # creates a temp file, so otherwise not possible to train
    # multiple models in parallel
    os.makedirs(f"input_{input_size}_{timestamp}")
    os.chdir(f"input_{input_size}_{timestamp}")

    GENOME_FA="../../../../../../../../resources/hg38/hg38.fa"
    GENOME_CHROM_SIZES="../../../../../../../../resources/hg38/hg38.chrom.sizes"


    crested.register_genome(
        crested.Genome(
            fasta=GENOME_FA,
            chrom_sizes=GENOME_CHROM_SIZES
        )
    )

    adata = preprocess(input_size)
    adata.write_h5ad(f"adata_input_{input_size}_{timestamp}.h5ad")

    train(adata, input_size)


if __name__ == "__main__":
    main()