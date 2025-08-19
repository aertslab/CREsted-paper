import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow.keras.backend as K
import os
import keras
from tangermeme.seqlet import recursive_seqlets
from tqdm import tqdm
import tfmindi as tm
import logomaker

def calc_trainable_params(p: str) -> int:
    model = keras.models.load_model(
        p, compile=False
    )
    return np.sum([K.count_params(w) for w in model.trainable_weights])

matplotlib.rcParams["pdf.fonttype"] = 42

color_dict = {
    "method": {"padding": "#3F88C5", "shrink param": "#AF125A"},
    "model_name": {
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
}

model_name_to_method = {
    "input_132_20258602526":                "shrink param",
    "input_264_202585234212":               "shrink param",
    "input_528_20258521175":                "shrink param",
    "input_1057_202585185910":              "shrink param",
    "input_2114_20258591653":               "shrink param",
    "input_4228_20258591653":               "shrink param",
    "input_same_shape_132_20258516744":     "padding",
    "input_same_shape_264_20258516744":     "padding",
    "input_same_shape_528_20258516744":     "padding",
    "input_same_shape_1057_20258516745":    "padding",
    "input_same_shape_2114_20258516745":    "padding",
    "input_same_shape_4228_20258516745":    "padding"
}

model_name_to_input_size = {
    "input_132_20258602526":                132,
    "input_264_202585234212":               264,
    "input_528_20258521175":                528,
    "input_1057_202585185910":              1057,
    "input_2114_20258591653":               2114,
    "input_4228_20258591653":               4228,
    "input_same_shape_132_20258516744":     132,
    "input_same_shape_264_20258516744":     264,
    "input_same_shape_528_20258516744":     528,
    "input_same_shape_1057_20258516745":    1057,
    "input_same_shape_2114_20258516745":    2114,
    "input_same_shape_4228_20258516745":    4228
}

model_to_n_params: dict[str, int] = {}

for model_name in os.listdir():
    if not model_name.startswith("input_"):
        continue
    print(model_name)
    model_to_n_params[model_name] = calc_trainable_params(
        os.path.join(
            model_name,
            "PBMC_INPUT_SIZE",
            model_name.replace("_same_shape", ""),
            "checkpoints",
            "01.keras"
        )
    )

metrics = pd.read_table("metrics.tsv", index_col = 0)

shrink_models = [
    model for model, method in model_name_to_method.items()
    if method == "shrink param"
]

padding_models = [
    model for model, method in model_name_to_method.items()
    if method == "padding"
]

shrink_models = sorted(
    shrink_models, 
    key = lambda x: int(x.rsplit("_")[-2])
)

padding_models = sorted(
    padding_models, 
    key = lambda x: int(x.rsplit("_")[-2])
)


################################################################################
#                               PLOT METRICS
################################################################################

input_size_shrink = [
    model_name_to_input_size[model] for model in shrink_models
]
input_size_padding = [
    model_name_to_input_size[model] for model in padding_models
]
n_params_shrink = [
    model_to_n_params[model] for model in shrink_models
]
n_params_padding = [
     model_to_n_params[model] for model in padding_models
]
fig, ax = plt.subplots(figsize = (4, 4))
_ = ax.scatter(
    input_size_shrink,
    n_params_shrink,
    color = color_dict["method"]["shrink param"],
    label = "shrink param",
    edgecolors="black",
    s = 50
)
_ = ax.scatter(
    input_size_padding,
    n_params_padding,
    color = color_dict["method"]["padding"],
    label = "padding",
    edgecolors="black",
    marker="^",
    s = 50
)
_ = ax.set_xlabel("Input size")
_ = ax.set_ylabel("# trainable param")
ax.grid(True)
ax.set_axisbelow(True)
ax.legend(loc = "lower right")
ax.axvline(2114, color = "black", ls = "dashed")
fig.tight_layout()
fig.savefig("figures/for_fig/input_size_v_param.png")
fig.savefig("figures/for_fig/input_size_v_param.pdf")

fig, axs = plt.subplots(nrows = 2,  ncols=4, figsize = (16, 8))
for ax, metric in zip(axs.ravel(), metrics.index):
    ax.scatter(
        [model_name_to_input_size[model] for model in shrink_models],
        metrics.loc[metric, shrink_models],
        color = color_dict["method"]["shrink param"],
        label = "shrink param",
        edgecolors="black",
        s=100
    )
    ax.scatter(
        [model_name_to_input_size[model] for model in padding_models],
        metrics.loc[metric, padding_models],
        color = color_dict["method"]["padding"],
        label = "padding",
        edgecolors="black",
        marker = "^",
        s = 100
    )
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_ylabel(metric)
    ax.set_xlabel("input size")
    ax.axvline(2114, color = "black", ls = "dashed")
fig.tight_layout()
fig.savefig("figures/for_fig/metric.png")
fig.savefig("figures/for_fig/metric.pdf")

################################################################################
#                              PLOT SEQLET LOC
################################################################################

def get_seqlets(d: str) -> pd.DataFrame:
    d = os.path.join(d, "modisco_results_ft_2000")

    cell_types = list(set([x.rsplit("_", 1)[0] for x in os.listdir(d)]))

    ohs = []
    contribs = []

    for cell_type in tqdm(cell_types, leave=False):
        ohs.append(
            np.load(os.path.join(d, f"{cell_type}_oh.npz"))["arr_0"])
        contribs.append(
            np.load(os.path.join(d, f"{cell_type}_contrib.npz"))["arr_0"])

    oh = np.concatenate(ohs)
    contrib = np.concatenate(contribs)

    seqlets = recursive_seqlets(
        (oh * contrib).sum(1)
    )

    return seqlets

model_dirs = [x for x in os.listdir() if x.startswith("input_")]

model_dir_to_seqlets = {
    model_dir: get_seqlets(model_dir)
    for model_dir in tqdm(model_dirs)
}

n_bp_per_bin = 10

fig, axs = plt.subplots(
    ncols=len(padding_models), figsize = (len(padding_models) * 4, 4),
    sharex=True)
for ax, model_dir in zip(axs, padding_models):
    input_size = int(model_dir.rsplit("_")[-2])
    _ = ax.hist(
        model_dir_to_seqlets[model_dir]["start"] - input_size // 2, 
        bins = input_size // n_bp_per_bin,
        color = color_dict["method"]["padding"]
    )
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_title(input_size)
    ax.set_xticks(
        np.arange(-2000, 2500, 250),
        [x if x % 500 else None for x in np.arange(-2000, 2500, 250)]
    )
    mn = np.mean(model_dir_to_seqlets[model_dir]["start"] - input_size // 2)
    st = np.std(model_dir_to_seqlets[model_dir]["start"] - input_size // 2)
    ax.text(
        0.05, 0.9,
        f"u={np.round(mn, 2)} sd={np.round(st, 2)}",
        transform = ax.transAxes
    )
fig.tight_layout()
fig.savefig("figures/for_fig/seqlet_location_same_shape.png")
fig.savefig("figures/for_fig/seqlet_location_same_shape.pdf")

fig, axs = plt.subplots(
    ncols=len(shrink_models), figsize = (len(shrink_models) * 4, 4),
    sharex=True)
for ax, model_dir in zip(axs, shrink_models):
    input_size = int(model_dir.rsplit("_")[-2])
    _ = ax.hist(
        model_dir_to_seqlets[model_dir]["start"] - input_size // 2,
        bins = input_size // n_bp_per_bin,
        color = color_dict["method"]["shrink param"]
    )
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_title(input_size)
    ax.set_xticks(
        np.arange(-2000, 2500, 250),
        [x if x % 500 else None for x in np.arange(-2000, 2500, 250)]
    )
    mn = np.mean(model_dir_to_seqlets[model_dir]["start"] - input_size // 2)
    st = np.std(model_dir_to_seqlets[model_dir]["start"] - input_size // 2)
    ax.text(
        0.05, 0.9,
        f"u={np.round(mn, 2)} sd={np.round(st, 2)}",
        transform = ax.transAxes
    )
fig.tight_layout()
fig.savefig("figures/for_fig/seqlet_location_diff_shape.png")
fig.savefig("figures/for_fig/seqlet_location_diff_shape.pdf")

################################################################################
#                              TF-MINDI RESULTS
################################################################################

adata = tm.io.load_h5ad("tfmindi_adata.h5ad")

dbd_to_color = {
    dbd: plt.cm.tab20(i)
    for i, dbd in enumerate(
        adata.obs["cluster_dbd"].value_counts().sort_values()[::-1].index)
}

fig, ax = plt.subplots(figsize = (8, 8), frameon=False)
_ = ax.scatter(
    adata.obsm["X_tsne"][:, 0],
    adata.obsm["X_tsne"][:, 1],
    c = [dbd_to_color[dbd] for dbd in adata.obs["cluster_dbd"]],
    s = 5
)
ax.set_axis_off()
fig.tight_layout()
fig.savefig("figures/for_fig/tsne_dbd.png")

fig, ax = plt.subplots()
for dbd, color in dbd_to_color.items():
    ax.scatter([], [], color = color, label = dbd)
ax.legend()
fig.tight_layout()
fig.savefig("figures/for_fig/tsne_legend.pdf")

for model in adata.obs["model"].unique():
    print(model)
    fig, ax = plt.subplots(figsize = (4,4), frameon=False)
    ax.scatter(
        adata.obsm["X_tsne"][adata.obs["model"] == model, 0],
        adata.obsm["X_tsne"][adata.obs["model"] == model, 1],
        color = color_dict["model_name"][model],
        s = 5
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_axis_off()
    fig.tight_layout(pad=0.05)
    fig.savefig(f"figures/for_fig/tsne_{model}.png", bbox_inches='tight')

################################################################################
#                              DEEP EXPLAINERS
################################################################################

selected_regions_onehot = np.load("selected_regions_one_hot.npz")
selected_regions_scores = np.load("selected_regions_scores.npz")

target_size = 250

for model_name in shrink_models:
    print(model_name)
    de = selected_regions_scores[model_name][-1] \
        * selected_regions_onehot[model_name][-1]
    if de.shape[0] < target_size:
        diff = target_size - de.shape[0]
        de = np.pad(
            de, 
            (
                (diff // 2, int(diff / 2 + 0.5)), 
                (0, 0)
            )
        )
    else:
        de = de[
            de.shape[0] // 2 - target_size // 2:
            de.shape[0] // 2 + int(target_size / 2 + 0.5)
        ]
    print(de.shape)
    fig, ax = plt.subplots(figsize = (10, 2))
    _ = logomaker.Logo(
        pd.DataFrame(
            de, columns = list("ACGT")
        ),
        ax = ax
    )
    fig.tight_layout()
    fig.savefig(f"figures/for_fig/{model_name}_de_enhanceosome.png")
    fig.savefig(f"figures/for_fig/{model_name}_de_enhanceosome.pdf")

for model_name in shrink_models:
    print(model_name)
    de = selected_regions_scores[model_name][0] \
        * selected_regions_onehot[model_name][0]
    if de.shape[0] < target_size:
        diff = target_size - de.shape[0]
        de = np.pad(
            de, 
            (
                (diff // 2, int(diff / 2 + 0.5)), 
                (0, 0)
            )
        )
    else:
        de = de[
            de.shape[0] // 2 - target_size // 2:
            de.shape[0] // 2 + int(target_size / 2 + 0.5)
        ]
    print(de.shape)
    fig, ax = plt.subplots(figsize = (10, 2))
    _ = logomaker.Logo(
        pd.DataFrame(
            de, columns = list("ACGT")
        ),
        ax = ax
    )
    fig.tight_layout()
    fig.savefig(f"figures/for_fig/{model_name}_de_cd79.png")
    fig.savefig(f"figures/for_fig/{model_name}_de_cd79.pdf")

