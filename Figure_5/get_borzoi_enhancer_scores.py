import os

import anndata as ad
import keras
import numpy as np
import pandas as pd
from tqdm import tqdm

import crested

# Set paths
output_file = "data/mouse_biccin_baseborzoi_preds.h5ad"

adata_path = "data/mouse_biccn_data_specific.h5ad"
targets_file = "data/targets_bor_mouse.txt"
model_file, output_names = crested.get_model('Borzoi_mouse_rep0')
resources_dir = ? # Directory with your downloaded mm10 genome
genome_file = os.path.join(resources_dir, "mm10.fa")
chromsizes_file = os.path.join(resources_dir, "mm10.chrom.sizes")
# Set parameters
untransform = False
n_bins = 32 # (1000/32, rounded up to make even)
batch_size = 6
big_comparison_tracks = np.array([i for i, output_name in enumerate(output_names) if output_name.startswith('ATAC')])


# Load in model
borzoi = keras.models.load_model(model_file)
targets = pd.read_csv(targets_file, sep = '\t', index_col = 0)
borzoi_seq_len = borzoi.input_shape[-2]
borzoi_output_len = borzoi.output_shape[-2]

# Load in genome
genome = crested.Genome(genome_file, chromsizes_file)
crested.register_genome(genome)

# Load in adata
adata = ad.read_h5ad(adata_path)
if 'split' not in adata.var.columns:
    crested.pp.train_val_test_split(adata_ct, strategy="chr", val_chroms=["chr8", "chr10"], test_chroms=["chr9", "chr18"])
if (adata.var['end'].iloc[0]-adata.var['start'].iloc[0]) != 2048:
    crested.pp.change_regions_width(adata_ct, 2048)

# Calc values for use during calculations
bin_range = (borzoi_output_len-n_bins)//2, (borzoi_output_len+n_bins)//2
borzoi_preds = np.zeros((len(big_comparison_tracks), adata.n_vars, n_bins))

batch_sizes_list = [batch_size]*(adata.n_vars//batch_size)+[adata.n_vars % batch_size]

# Get factors for transformation (if used)
scaling_factors = targets['scale'][big_comparison_tracks].values
clipsoft_factors = targets['clip_soft'][big_comparison_tracks].values[None, None, :]
sumstat_sqrt_factors = np.array([0.5 if sum_stat.endswith('sqrt') else 1 for sum_stat in targets['sum_stat'][big_comparison_tracks]])[None, None, :]
sumstat_mean_factors = np.array([32 if sum_stat.startswith('mean') else 1 for sum_stat in targets['sum_stat'][big_comparison_tracks]])[None, None, :]


print(f"{len(batch_sizes_list)} batches")
chrs, seq_starts, seq_ends, bin_starts, bin_ends = [], [], [], [], []
for batch_idx, batch in enumerate(tqdm(batch_sizes_list)):
    batch_sequences = []
    for inbatch_idx in range(batch):
        region_idx = batch_idx*batch_size + inbatch_idx
        region = adata.var_names[region_idx]
        chr, start_end = region.split(':')
        start, end = map(int, start_end.split('-'))
        center = (start+end)//2

        # Calculate positions
        seq_start, seq_end = center-borzoi_seq_len//2, center+borzoi_seq_len//2
        bin_start, bin_end = center-(borzoi_output_len//2-bin_range[0])*32, center-(borzoi_output_len//2-bin_range[1])*32
        chrs.append(chr)
        seq_starts.append(seq_start)
        seq_ends.append(seq_end)
        bin_starts.append(bin_start)
        bin_ends.append(bin_end)

        # Get sequence, dealing with out-of-chrom overhangs
        if seq_start < 0:
            start_padding = -seq_start
            seq_start = 0
        else:
            start_padding = 0
        if seq_end > genome.chrom_sizes[chr]:
            end_padding = seq_end - genome.chrom_sizes[chr]
            seq_end = genome.chrom_sizes[chr]
        else:
            end_padding = 0

        seq = genome.fetch(chr, seq_start, seq_end)
        if start_padding > 0:
            seq = "N"*start_padding+seq
        if end_padding > 0:
            seq = seq+"N"*end_padding
        seq_1hot = crested.utils.one_hot_encode_sequence(seq)
        batch_sequences.append(seq_1hot)
    batch_sequences = np.concatenate(batch_sequences, axis=0)

    # Get Borzoi preds
    preds = borzoi(batch_sequences)
    preds_select = preds.numpy()[:, :, big_comparison_tracks][:, bin_range[0]:bin_range[1]]

    # Undo Borzoi transforms - untransform_old
    if untransform:
        preds_select = preds_select / scaling_factors
        preds_select = np.where(
            preds_select > clipsoft_factors,
            (preds_select - clipsoft_factors) ** 2 + clipsoft_factors,
            preds_select,
        )
        preds_select = preds_select ** (1.0 / sumstat_sqrt_factors)

    # Save to big matrix
    borzoi_preds[:, (batch_idx*batch_size):(batch_idx*batch_size+batch), :] = preds_select.transpose(2, 0, 1)

# Save non-summed preds as temp npz, just in case creating anndata goes wrong somehow
np.savez("data/temp_"+output_file.replace(".h5ad", "_wide.npz"), borzoi_preds)

# Create and save anndata
borzoi_ad = ad.AnnData(
    borzoi_preds.mean(axis=-1),
    layers = {'full_preds': borzoi_preds},
    obs = pd.DataFrame(index = np.array(output_names)[big_comparison_tracks]),
    var = pd.DataFrame({'chr': chrs, 'borzoi_start': seq_starts, 'borzoi_end': seq_ends, 'bin_start': bin_starts, 'bin_end': bin_ends}, index = adata.var_names)
)

borzoi_ad.write_h5ad(output_file, compression = 'gzip', compression_opts=9)

