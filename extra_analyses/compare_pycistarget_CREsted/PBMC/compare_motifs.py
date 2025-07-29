import h5py
import numpy as np
import os
from tqdm import tqdm
import modiscolite
from memelite import tomtom
import scanpy as sc
import pandas as pd
import seaborn as sns

sc.settings.figdir = "plots"

def trim_by_ic(ic, min_v):
    if len(np.where(np.diff((ic > min_v) * 1))[0]) == 0:
        return 0, 0
    start_index = min(np.where(np.diff((ic > min_v) * 1))[0])
    end_index = max(np.where(np.diff((ic > min_v) * 1))[0])
    return start_index, end_index + 1

def load_motif_from_modisco(filename, ic_thr, avg_ic_thr):
    with h5py.File(filename) as f:
        for pos_neg in ["pos_patterns", "neg_patterns"]:
            if pos_neg not in f.keys():
                continue
            for pattern in f[pos_neg].keys():
                ppm = f[pos_neg][pattern]["sequence"][:]
                ic = modiscolite.util.compute_per_position_ic(
                    ppm=ppm, background=[0.27, 0.23, 0.23, 0.27], pseudocount=1e-3
                )
                start, stop = trim_by_ic(ic, ic_thr)
                if stop - start <= 1:
                    continue
                if ic[start:stop].mean() < avg_ic_thr:
                    continue
                yield (
                    filename.split("/")[-1].rsplit(".", 1)[0]
                    + "_"
                    + pos_neg.split("_")[0]
                    + "_"
                    + pattern,
                    pos_neg == "pos_patterns",
                    ppm[start:stop],
                    ic[start:stop],
                )

def load_motif(n, d):
    motifs = []
    names = []
    with open(os.path.join(d, f"{n}.cb")) as f:
        tmp = None
        for l in f:
            l = l.strip()
            if l.startswith(">"):
                names.append(l.replace(">", ""))
                if tmp is not None:
                    tmp = np.array(tmp)
                    tmp = (tmp.T / tmp.sum(1)).T
                    motifs.append(tmp)
                tmp = []
            else:
                tmp.append([float(x) for x in l.split()])
        tmp = np.array(tmp)
        tmp = (tmp.T / tmp.sum(1)).T
        motifs.append(tmp)
    return motifs, names

# Load enriched motifs as ppm

MOTIF_DIR = "../../pycisTarget/v10nr_clust_public/singletons/"

enriched_motifs: list[str] = []
with open("enriched_motifs.txt") as f:
    for line in f:
        enriched_motifs.append(line.strip())

enriched_motifs_ppm: list[np.ndarray] = []
enriched_motifs_sub_names: list[str] = []
enriched_motifs_names: list[str] = []

for motif in tqdm(enriched_motifs):
    _motifs, _m_sub_names = load_motif(
        motif,
        MOTIF_DIR
    )
    enriched_motifs_ppm.extend(_motifs)
    enriched_motifs_sub_names.extend(_m_sub_names)
    enriched_motifs_names.extend(np.repeat(motif, len(_motifs)))

# Load modisco patterns

modisco_results: list[str] = [
    f for f in os.listdir("crested_data/Figure_3/pbmc/modisco")
    if f.endswith("_modisco_results.h5")
]

modisco_motifs_ppm: list[np.ndarray] = []
modisco_motifs_names: list[str] = []
modisco_motifs_ic: list[np.ndarray] = []
modisco_motifs_is_pos: list[bool] = []
for f_modisco_res in tqdm(modisco_results):
    for name, is_pos, ppm, ic in load_motif_from_modisco(
        filename=os.path.join(
            "crested_data/Figure_3/pbmc/modisco", f_modisco_res
        ),
        ic_thr=0.2,
        avg_ic_thr=0.5,
    ):
        modisco_motifs_ppm.append(ppm)
        modisco_motifs_names.append(name)
        modisco_motifs_ic.append(ic)
        modisco_motifs_is_pos.append(is_pos)


all_motifs = [
    ppm.T for ppm in [
        *enriched_motifs_ppm,
        *modisco_motifs_ppm
    ]   
]

motif_names = [*enriched_motifs_sub_names, *modisco_motifs_names]

motif_technique = [
    *np.repeat("pycisTarget", len(enriched_motifs_ppm)),
    *np.repeat("deepPBMC", len(modisco_motifs_ppm))
]

motif_metadata = pd.DataFrame(
    index = motif_names,
    data = {"technique": motif_technique}
)

pvals, scores, offsets, overlaps, strands = tomtom(
    all_motifs, all_motifs
)

evals = pvals * len(all_motifs)
adata_motifs_dl_ctx = sc.AnnData(
    evals, 
    obs=motif_metadata
)

sc.tl.pca(adata_motifs_dl_ctx)
sc.pp.neighbors(adata_motifs_dl_ctx)
sc.tl.tsne(adata_motifs_dl_ctx)

sc.pl.tsne(
    adata_motifs_dl_ctx,
    color = "technique",
    palette = {"pycisTarget": "#0D3B66", "deepPBMC": "#F95738"},
    save = "_technique.png"
)

sc.tl.leiden(adata_motifs_dl_ctx, resolution=1)
sc.pl.tsne(
    adata_motifs_dl_ctx,
    color=["leiden"],
    save="_leiden.png",
    legend_loc="on data",
)

corr_matrix = adata_motifs_dl_ctx.to_df().corr()

color_dict = {"pycisTarget": "#0D3B66", "deepPBMC": "#F95738"}
fig = sns.clustermap(
    corr_matrix.to_numpy(),
    row_colors=[color_dict[t] for t in adata_motifs_dl_ctx.obs["technique"]],
    col_colors=[color_dict[t] for t in adata_motifs_dl_ctx.obs["technique"]],
    robust = True, 
    cmap = "bwr",
    xticklabels=False, yticklabels=False
)
fig.savefig("plots/heatmap_corr_matrix.png")

