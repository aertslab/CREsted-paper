import h5py
import numpy as np
import os
from tqdm import tqdm
import modiscolite
from memelite import tomtom
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logomaker
import tfmindi as tm

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

def load_motif_chromvar(p):
    nuc_index = list("ACGT")
    motifs = []
    names = []
    current_motif = 1
    with open(p) as f:
        motif = [[], [], [], []]
        for line in f:
            if line.startswith("A:"):
                motif = np.array(motif)
                if np.array(motif).shape[1] > 0 and motif.min() >= 0 :
                    motif = (motif / motif.sum(0))
                    motifs.append(motif.T)
                    names.append(f"chromvar_{current_motif}")
                current_motif += 1
                motif = [[], [], [], []]
            if not (
                line.startswith("A:") or \
                line.startswith("C:") or \
                line.startswith("G:") or \
                line.startswith("T:")
            ):
                continue
            line = line.replace("-inf", "0.0")
            nuc = line.split()[0].replace(":", "")
            motif[nuc_index.index(nuc)] = [
                float(x) for x in line.split()[1:]
            ]
    return motifs, names

def ic(ppm, bg=np.array([0.27, 0.23, 0.23, 0.27]), eps=1e-3) -> np.ndarray:
    return (
        ppm * np.log(ppm + eps) / np.log(2) - bg * np.log(bg) / np.log(2)
    ).sum(1)

# Load enriched motifs as ppm

MOTIF_DIR = "../../pycisTarget/v10nr_clust_public/singletons/"

motif_annotations_file = tm.fetch_motif_annotations()
motif_annotations = tm.load_motif_annotations(motif_annotations_file)

motif_to_db = tm.load_motif_to_dbd(motif_annotations)

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

# Load chromvar motifs

chromvar_motifs, chromvar_names = load_motif_chromvar(
    "/data/groups/vib.ai/stein.aerts/dabaffy/crested_revision/enriched_motifs.csv"
)

all_motifs = [
    ppm.T for ppm in [
        *enriched_motifs_ppm,
        *modisco_motifs_ppm,
        *chromvar_motifs
    ]   
]

motif_names = [
    *enriched_motifs_sub_names, *modisco_motifs_names, *chromvar_names
]

motif_technique = [
    *np.repeat("pycisTarget", len(enriched_motifs_ppm)),
    *np.repeat("deepPBMC", len(modisco_motifs_ppm)),
    *np.repeat("chromVAR", len(chromvar_motifs))
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

adata_motifs_dl_ctx.obs["dbd"] = [
    motif_to_db.get(x, np.nan) for x in adata_motifs_dl_ctx.obs.index
]

sc.pl.tsne(
    adata_motifs_dl_ctx,
    color = "technique",
    #palette = {"pycisTarget": "#0D3B66", "deepPBMC": "#F95738"},
    save = "_technique.png"
)

sc.tl.leiden(adata_motifs_dl_ctx, resolution=1)

cluster_to_dbd = (
    adata_motifs_dl_ctx.obs[["leiden", "dbd"]]
        .groupby("leiden", observed=True)["dbd"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan)
        .to_dict()
)

adata_motifs_dl_ctx.obs["cluster_dbd"] = [
    cluster_to_dbd[cl] for cl in adata_motifs_dl_ctx.obs["leiden"]
]

sc.pl.tsne(
    adata_motifs_dl_ctx,
    color=["leiden"],
    save="_leiden.png",
    legend_loc="on data",
)

for cluster in adata_motifs_dl_ctx.obs["leiden"].unique():
    print(cluster)
    fig, axs = plt.subplots(figsize = (12, 2), ncols = 3)
    for technique, ax in zip(
            adata_motifs_dl_ctx.obs["technique"].unique(),
            axs.ravel()):
        motifs_of_cluster_and_tech = np.logical_and(
            adata_motifs_dl_ctx.obs["leiden"] == cluster,
            adata_motifs_dl_ctx.obs["technique"] == technique
        )
        if sum(motifs_of_cluster_and_tech) == 0:
            continue
        print(technique)
        motifs = [
            all_motifs[x] for x in np.where(motifs_of_cluster_and_tech)[0]
        ]
        ics = [
            np.nan_to_num(ic(m.T)) for m in motifs
        ]
        motif_to_show = 0#np.argmax([x.sum() for x in ics])
        _ = logomaker.Logo(
            pd.DataFrame(
                motifs[motif_to_show].T * ics[motif_to_show][:, None],
                columns = list("ACGT")
            ),
            ax = ax
        )
    fig.savefig(f"plots/logos_cluster_{cluster}.pdf")
    fig.savefig(f"plots/logos_cluster_{cluster}.png")


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

leiden_color_dict = {
    cluster: color
    for cluster, color in zip(
        adata_motifs_dl_ctx.obs["leiden"].unique(),
        adata_motifs_dl_ctx.uns["leiden_colors"]
    )   
}

technique_marker = {
    "chromVAR": "^",
    "deepPBMC": "o",
    "pycisTarget": "."
}

technique_to_zorder = {
    "chromVAR": 2,
    "deepPBMC": 3,
    "pycisTarget": 1
}

fig, ax = plt.subplots(figsize = (4,4))
for technique, marker in technique_marker.items():
    v_adata_motifs_dl_ctx = adata_motifs_dl_ctx[
        adata_motifs_dl_ctx.obs["technique"] == technique
    ]
    ax.scatter(
        v_adata_motifs_dl_ctx.obsm["X_tsne"][:, 0],
        v_adata_motifs_dl_ctx.obsm["X_tsne"][:, 1],
        c = [
            leiden_color_dict[cl]
            for cl in v_adata_motifs_dl_ctx.obs["leiden"]
        ],
        marker=marker,
        label = technique,
        edgecolors="black",
        zorder=technique_to_zorder[technique]
    )

ax.legend()
ax.set_axis_off()
fig.tight_layout()
fig.savefig("plots/tSNE_leiden_and_technique.png", dpi = 300)
fig.savefig("plots/tSNE_leiden_and_technique.pdf")


sc.pl.tsne(
    adata_motifs_dl_ctx,
    color = "cluster_dbd",
    save = "_cluster_dbd.png"
)

dbd_color_dict = {
    cluster: color
    for cluster, color in zip(
        adata_motifs_dl_ctx.obs["cluster_dbd"].unique().dropna(),
        adata_motifs_dl_ctx.uns["cluster_dbd_colors"]
    )   
}

dbd_color_dict[np.nan] = "gray"


fig, ax = plt.subplots(figsize = (4,4))
for technique, marker in technique_marker.items():
    v_adata_motifs_dl_ctx = adata_motifs_dl_ctx[
        adata_motifs_dl_ctx.obs["technique"] == technique
    ]
    ax.scatter(
        v_adata_motifs_dl_ctx.obsm["X_tsne"][:, 0],
        v_adata_motifs_dl_ctx.obsm["X_tsne"][:, 1],
        c = [
            dbd_color_dict[cl]
            for cl in v_adata_motifs_dl_ctx.obs["cluster_dbd"]
        ],
        marker=marker,
        label = technique,
        edgecolors="black",
        zorder=technique_to_zorder[technique]
    )

ax.legend()
ax.set_axis_off()
fig.tight_layout()
fig.savefig("plots/tSNE_dbd_and_technique.png", dpi = 300)
fig.savefig("plots/tSNE_dbd_and_technique.pdf")

fig, ax = plt.subplots()
for dbd, color in dbd_color_dict.items():
    ax.scatter([], [], label = dbd, color = color)
fig.legend()
fig.tight_layout()
fig.savefig("plots/dbd_legend.pdf")

for dbd in ["HMG/Sox", "Homeodomain"]:
    print(dbd)
    for technique in adata_motifs_dl_ctx.obs["technique"].unique():
        print(technique)
        motifs_of_dbd_and_tech = np.logical_and(
                adata_motifs_dl_ctx.obs["cluster_dbd"] == dbd,
                adata_motifs_dl_ctx.obs["technique"] == technique
            )
        motifs = [
            all_motifs[x] for x in np.where(motifs_of_dbd_and_tech)[0]
        ]
        ics = [
            np.nan_to_num(ic(m.T)) for m in motifs
        ]
        for i, (motif_, ic_) in enumerate(zip(motifs, ics)):
            print(f"{i + 1}/{len(motifs)}", end = "\r")
            fig, ax = plt.subplots(figsize = (4, 2))
            _ = logomaker.Logo(
                pd.DataFrame(
                    motif_.T * ic_[:, None],
                    columns = list("ACGT")
                ),
                ax = ax
            )
            fig.savefig(f"plots/logo_{dbd.replace('/', '_')}_{technique}_{i}.png")
            fig.savefig(f"plots/logo_{dbd.replace('/', '_')}_{technique}_{i}.pdf")

for dbd in ["bZIP"]:
    print(dbd)
    technique = "chromVAR"
    print(technique)
    motifs_of_dbd_and_tech = np.logical_and(
            adata_motifs_dl_ctx.obs["cluster_dbd"] == dbd,
            adata_motifs_dl_ctx.obs["technique"] == technique
        )
    motifs = [
        all_motifs[x] for x in np.where(motifs_of_dbd_and_tech)[0]
    ]
    ics = [
        np.nan_to_num(ic(m.T)) for m in motifs
    ]
    for i, (motif_, ic_) in enumerate(zip(motifs, ics)):
        print(f"{i + 1}/{len(motifs)}", end = "\r")
        fig, ax = plt.subplots(figsize = (4, 2))
        _ = logomaker.Logo(
            pd.DataFrame(
                motif_.T * ic_[:, None],
                columns = list("ACGT")
            ),
            ax = ax
        )
        fig.savefig(f"plots/logo_{dbd.replace('/', '_')}_{technique}_{i}.png")
        fig.savefig(f"plots/logo_{dbd.replace('/', '_')}_{technique}_{i}.pdf")

