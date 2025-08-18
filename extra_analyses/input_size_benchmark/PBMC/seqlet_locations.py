from tangermeme.seqlet import recursive_seqlets
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def get_seqlets(d: str) -> pd.DataFrame:
    d = os.path.join(d, "modisco_results_ft_2000")

    cell_types = list(set([x.rsplit("_", 1)[0] for x in os.listdir(d)]))

    ohs = []
    contribs = []

    for cell_type in tqdm(cell_types, leave=False):
        ohs.append(np.load(os.path.join(d, f"{cell_type}_oh.npz"))["arr_0"])
        contribs.append(np.load(os.path.join(d, f"{cell_type}_contrib.npz"))["arr_0"])

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


models_same_shape = [x for x in model_dirs if "same_shape" in x]
models_diff_shape = [x for x in model_dirs if "same_shape" not in x]

models_same_shape = sorted(
    models_same_shape, key = lambda x: int(x.rsplit("_")[-2])
)
models_diff_shape = sorted(
    models_diff_shape, key = lambda x: int(x.rsplit("_")[-2])
)

n_bp_per_bin = 10

fig, axs = plt.subplots(
    ncols=len(models_same_shape), figsize = (len(models_same_shape) * 4, 4),
    sharex=True)
for ax, model_dir in zip(axs, models_same_shape):
    input_size = int(model_dir.rsplit("_")[-2])
    _ = ax.hist(
        model_dir_to_seqlets[model_dir]["start"] - input_size // 2, bins = input_size // n_bp_per_bin,
        color = "gray"
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
fig.savefig("figures/seqlet_location_same_shape.png")

fig, axs = plt.subplots(
    ncols=len(models_diff_shape), figsize = (len(models_diff_shape) * 4, 4),
    sharex=True)
for ax, model_dir in zip(axs, models_diff_shape):
    input_size = int(model_dir.rsplit("_")[-2])
    _ = ax.hist(
        model_dir_to_seqlets[model_dir]["start"] - input_size // 2, bins = input_size // n_bp_per_bin,
        color = "gray"
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
fig.savefig("figures/seqlet_location_diff_shape.png")


