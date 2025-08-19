import tfmindi as tm
import os
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt

motif_collection_dir = tm.fetch_motif_collection()
motif_annotations_file = tm.fetch_motif_annotations()

with open("sampled_motifs.txt") as f:
    motif_names = [line.strip() for line in f.readlines()]

# load them as dictionary of PPM matrices
motif_collection = tm.load_motif_collection(motif_collection_dir, motif_names=motif_names)
motif_annotations = tm.load_motif_annotations(motif_annotations_file)

# load motif to dna-binding domain (DBD) mapping
motif_to_db = tm.load_motif_to_dbd(motif_annotations)

def get_seqlets(d: str):
    d = os.path.join(d, "modisco_results_ft_2000")

    cell_types = list(set([x.rsplit("_", 1)[0] for x in os.listdir(d)]))

    ohs = []
    contribs = []

    for cell_type in tqdm(cell_types, leave=False):
        ohs.append(np.load(os.path.join(d, f"{cell_type}_oh.npz"))["arr_0"])
        contribs.append(np.load(os.path.join(d, f"{cell_type}_contrib.npz"))["arr_0"])

    oh = np.concatenate(ohs)
    contrib = np.concatenate(contribs)

    return (tm.pp.extract_seqlets(contrib=contrib, oh=oh, threshold=0.01, additional_flanks=3), (oh, contrib))

model_dirs = [x for x in os.listdir() if x.startswith("input_")]


model_dirs = [x for x in os.listdir() if x.startswith("input_")]

model_dir_to_seqlets = {
    model_dir: get_seqlets(model_dir)
    for model_dir in tqdm(model_dirs)
}

N_TO_SAMPLE=5_000

seqlet_df = []
seqlet_matrics = []
oh = []
contrib = []

random.seed(123)

max_size = max([int(x.rsplit("_")[-2]) for x in model_dirs])

prev_max_idx = 0

for model_dir in tqdm(model_dirs):
    (_seqlet_df, _seqlet_matrics), (oh_, contrib_) = model_dir_to_seqlets[model_dir]
    _seqlet_df["model"] = model_dir
    _seqlet_df["example_idx"] = _seqlet_df["example_idx"] + prev_max_idx
    idx_to_keep = random.sample(list(np.arange(len(_seqlet_matrics))), min(N_TO_SAMPLE, len(_seqlet_matrics)))
    seqlet_df.append(_seqlet_df.iloc[idx_to_keep])
    seqlet_matrics.extend([_seqlet_matrics[x] for x in idx_to_keep])
    pad_width = ((0, 0),
                 (0, 0),
                 (0, max_size - oh_.shape[2])
    )
    oh.append(np.pad(oh_, pad_width, mode="constant", constant_values=0))
    contrib.append(np.pad(contrib_, pad_width, mode="constant", constant_values=0))
    prev_max_idx += oh_.shape[0]

seqlet_df = pd.concat(seqlet_df)
oh = np.concatenate(oh)
contrib = np.concatenate(contrib)

sim_matrix = tm.pp.calculate_motif_similarity(
    seqlet_matrics, motif_collection, chunk_size=5000
)

adata = tm.pp.create_seqlet_adata(
    sim_matrix,  # mandatory
    seqlet_df,  # mandatory
    seqlet_matrices=seqlet_matrics,
    oh_sequences=oh,
    contrib_scores=contrib,
    motif_collection=motif_collection,
    motif_annotations=motif_annotations,
    motif_to_dbd=motif_to_db,
)
adata

tm.tl.cluster_seqlets(adata, resolution=3.0)

tm.pl.tsne(
    adata,
    color_by="cluster_dbd",
    width=12,
    height=10,
    s=2,
    alpha=0.2,
    save_path="figures/seqlets_tsne_cluster_dbd.png",
)

tm.io.save_h5ad(
    adata,
    "tfmindi_adata.h5ad"
)

color_dict = {
        "input_132_20258602526": "#60a85f",
        "input_same_shape_132_20258516744": "#026200",
        "input_264_202585234212": "#a863c3",
        "input_same_shape_264_20258516744": "#460161",
        "input_528_20258521175": "#b1963f",
        "input_same_shape_528_20258516744": "#896F19",
        "input_1057_202585185910": "#648ace",
        "input_same_shape_1057_20258516745": "#001842",
        "input_2114_20258591653": "#cb613f",
        "input_same_shape_2114_20258516745": "#651900",
        "input_4228_20258591653": "#c9557e",
        "input_same_shape_4228_20258516745": "#ff4083"
}

sc.pl.tsne(
    adata,
    color = "model",
    palette = color_dict,
    save = "by_model.png",
    size = 10
)

fig, axs = plt.subplots(nrows = 3, ncols = 4, figsize = (16, 12))
for ax, model_dir in zip(axs.ravel(), color_dict.keys()):
    sc.pl.tsne(
        adata[adata.obs["model"] == model_dir],
        color = "model",
        palette = color_dict,
        ax = ax,
        size = 10,
        legend_loc = None
    )
    ax.set_title(model_dir)

fig.tight_layout()
fig.savefig("figures/tsne_by_model_split.png")

model_per_leiden = pd.crosstab(
    adata.obs["leiden"], adata.obs["model"]
)

fig = sns.clustermap(
    (model_per_leiden.T / model_per_leiden.sum(1)).T * 100, 
    vmin = 0, vmax = 70,
    cmap = "rainbow"
    )
fig.savefig("figures/fraction_seqlets_per_cluster_per_model.png")

model_per_leiden = pd.crosstab(
    adata.obs["cluster_dbd"], adata.obs["model"]
)

fig = sns.clustermap(
    (model_per_leiden.T / model_per_leiden.sum(1)).T * 100, 
    vmin = 0, vmax = 70,
    cmap = "rainbow"
    )
fig.savefig("figures/fraction_seqlets_per_dbd_per_model.png")