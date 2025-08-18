#!/data/groups/vib.ai/stein.aerts/sdewin/software/miniforge3/envs/crested_1.5.0/bin/python

import crested
import anndata
import keras
import argparse
import os

def get_contribution(adata: anndata.AnnData, run_id: str):
    model_f = sorted(os.listdir(f"PBMC_INPUT_SIZE/{run_id}/checkpoints"), key = lambda x: int(x.split(".")[0]))[-1]

    print(f"PBMC_INPUT_SIZE/{run_id}/checkpoints/{model_f}")

    model = keras.saving.load_model(
        f"PBMC_INPUT_SIZE/{run_id}/checkpoints/{model_f}"
    )

    predictions = crested.tl.predict(adata, model)

    adata.layers["pbmc_model"] = predictions.T

    adata_combined = adata.copy()  # Copy the peak heights
    adata_combined.X = (
        adata_combined.X + adata_combined.layers["pbmc_model"]
    ) / 2  # Take the average with the predictions

    adata_filtered = adata_combined.copy()
    top_k = 2000
    crested.pp.sort_and_filter_regions_on_specificity(
        adata_filtered, top_k=top_k, method="proportion"
    )

    crested.tl.contribution_scores_specific(
        input=adata_filtered,
        target_idx=None,  # We calculate for all classes
        model=model,
        output_dir="modisco_results_ft_2000",
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--model_directory", required = True, type = str)

    args = parser.parse_args()
    model_dir = args.model_directory

    os.chdir(model_dir)

    run_id = model_dir.replace("same_shape_", "") + "_finetune"

    adata = anndata.read_h5ad(f"adata_{run_id.replace('_finetune', '')}.h5ad")

    # Filter on specificity
    crested.pp.filter_regions_on_specificity(adata, gini_std_threshold=1.0)

    GENOME_FA="../../../../../../../../resources/hg38/hg38.fa"
    GENOME_CHROM_SIZES="../../../../../../../../resources/hg38/hg38.chrom.sizes"

    crested.register_genome(
        crested.Genome(
            fasta=GENOME_FA,
            chrom_sizes=GENOME_CHROM_SIZES
        )
    )


    get_contribution(adata, run_id)


if __name__ == "__main__":
    main()

