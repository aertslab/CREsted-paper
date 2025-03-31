"""Helper custom functions for the cistopic visualization notebook."""
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects

import random
import sklearn
import numpy as np
import pandas as pd
from adjustText import adjust_text
from pycisTopic.utils import subset_list


def plot_metadata_custom(
    cistopic_obj_projections: dict,
    cell_data: pd.DataFrame,
    reduction_name: str,
    variables: list[str],
    target: str = "cell",
    remove_nan: bool = True,
    show_label: bool = True,
    show_legend: bool = False,
    cmap: matplotlib.cm = cm.viridis,
    dot_size: int = 10,
    text_size: int = 10,
    alpha: float = 1.0,
    seed: int = 555,
    color_dictionary: dict[str, dict[str, str]] | None=None,
    figsize: tuple[float, float]=(6.4, 4.8),
    num_columns: int = 1,
    selected_features: list[str] | None = None,
    save: str | None = None,
):
    """
    Plot categorical and continuous metadata into dimensionality reduction.

    Parameters
    ----------
    cistopic_obj_projections: dict
            A dictionary with cisTopic object dimensionality reductions.
    cell_data: pd.DataFrame
            A dataframe containing sample and cell type/state annotations.
    reduction_name: str
            Name of the dimensionality reduction to use
    variables: list[str]
            List of variables to plot. They should be included in cell_data.
    target: str
            Whether cells ('cell') or regions ('region') should be used. Default: 'cell'
    remove_nan: bool
            Whether to remove data points for which the variable value is 'nan'. Default: True
    show_label: bool
            For categorical variables, whether to show the label in the plot. Default: True
    show_legend: bool
            For categorical variables, whether to show the legend next to the plot. Default: False
    cmap: str or 'matplotlib.cm'
            For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    dot_size: int
            Dot size in the plot. Default: 10
    text_size: int
            For categorical variables and if show_label is True, size of the labels in the plot. Default: 10
    alpha: float
            Transparency value for the dots in the plot. Default: 1
    seed: int
            Random seed used to select random colors. Default: 555
    color_dictionary: dict, optional
            A dictionary containing an entry per variable, whose values are dictionaries with variable levels as keys and corresponding colors as values.
            Default: None
    figsize: tuple[float, float], optional
            Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
            default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    num_columns: int
            For multiplot figures, indicates the number of columns (the number of rows will be automatically determined based on the number of plots). Default: 1
    selected_features: list,[str] optional
            A list with selected features (cells or regions) to plot. This is recommended when working with regions (e.g. selecting
            regions in binarized topics), as working with all regions can be time consuming. Default: None (use all features)
    save: str, optional
            Path to save plot. Default: None.

    """
    embedding = cistopic_obj_projections[target][reduction_name]
    data_mat = cell_data

    if selected_features is not None:
        data_mat = data_mat.loc[selected_features]
        embedding = embedding.loc[selected_features]

    data_mat = data_mat.loc[embedding.index.to_list()]
    pdf = None
    if (save is not None) and (num_columns == 1):
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    if num_columns > 1:
        num_rows = int(np.ceil(len(variables) / num_columns))
        if figsize == (6.4, 4.8):
            figsize = (6.4 * num_columns, 4.8 * num_rows)
        i = 1

    fig = plt.figure(figsize=figsize)

    for var in variables:
        var_data = data_mat.copy().loc[:, var].dropna().to_list()
        if isinstance(var_data[0], str):
            if remove_nan and (data_mat[var].isnull().sum() > 0):
                var_data = data_mat.copy().loc[:, var].dropna().to_list()
                emb_nan = embedding.loc[
                    data_mat.copy().loc[:, var].dropna().index.tolist()
                ]
                label_pd = pd.concat(
                    [emb_nan, data_mat.loc[:, [var]].dropna()], axis=1, sort=False
                )
            else:
                var_data = (
                    data_mat.copy().astype(str).fillna("NA").loc[:, var].to_list()
                )
                label_pd = pd.concat(
                    [embedding, data_mat.astype(str).fillna("NA").loc[:, [var]]],
                    axis=1,
                    sort=False,
                )

            if color_dictionary is None:
                color_dictionary = {}
            categories = set(var_data)

            if var in color_dictionary:
                color_dict = color_dictionary[var]
            else:
                random.seed(seed)
                color = [
                    mcolors.to_rgb("#" + "%06x" % random.randint(0, 0xFFFFFF))
                    for i in range(len(categories))
                ]
                color_dict = dict(zip(categories, color))

            if num_columns > 1:
                plt.subplot(num_rows, num_columns, i)
                i = i + 1

            if remove_nan and (data_mat[var].isnull().sum() > 0):
                plt.scatter(
                    emb_nan.iloc[:, 0],
                    emb_nan.iloc[:, 1],
                    c=data_mat.loc[:, var].dropna().apply(lambda x: color_dict[x]),
                    s=dot_size,
                    alpha=alpha,
                )
                plt.xlabel(emb_nan.columns[0])
                plt.ylabel(emb_nan.columns[1])
            else:
                plt.scatter(
                    embedding.iloc[:, 0],
                    embedding.iloc[:, 1],
                    c=data_mat.astype(str)
                    .fillna("NA")
                    .loc[:, var]
                    .apply(lambda x: color_dict[x]),
                    s=dot_size,
                    alpha=alpha,
                )
                plt.xlabel(embedding.columns[0])
                plt.ylabel(embedding.columns[1])

            if show_label:
                label_pos = label_pd.groupby(var).agg(
                    {label_pd.columns[0]: np.mean, label_pd.columns[1]: np.mean}
                )
                texts = []
                for label in label_pos.index.tolist():
                    texts.append(
                        plt.text(
                            label_pos.loc[label][0],
                            label_pos.loc[label][1],
                            label,
                            horizontalalignment="center",
                            verticalalignment="center",
                            size=text_size,
                            weight="bold",
                            color=color_dict[label],
                            path_effects=[
                                PathEffects.withStroke(linewidth=3, foreground="w")
                            ],
                        )
                    )
                adjust_text(texts)

            plt.title(var)
            patchList = []
            for key in color_dict:
                data_key = mpatches.Patch(color=color_dict[key], label=key)
                patchList.append(data_key)
            if show_legend:
                plt.legend(
                    handles=patchList, bbox_to_anchor=(1.04, 1), loc="upper left"
                )

            if num_columns == 1:
                if save is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                plt.show()
        else:
            var_data = data_mat.copy().loc[:, var].to_list()
            o = np.argsort(var_data)
            if num_columns > 1:
                plt.subplot(num_rows, num_columns, i)
                i = i + 1
            plt.scatter(
                embedding.iloc[o, 0],
                embedding.iloc[o, 1],
                c=subset_list(var_data, o),
                cmap=cmap,
                s=dot_size,
                alpha=alpha,
            )
            plt.xlabel(embedding.columns[0])
            plt.ylabel(embedding.columns[1])
            plt.title(var)
            # setup the colorbar
            normalize = mcolors.Normalize(
                vmin=np.array(var_data).min(), vmax=np.array(var_data).max()
            )
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
            scalarmappaple.set_array(var_data)
            plt.colorbar(scalarmappaple)
            if num_columns == 1:
                if save is not None:
                    pdf.savefig(fig, bbox_inches="tight")
                plt.show()

    if num_columns > 1:
        plt.tight_layout()
        if save is not None:
            fig.savefig(save, bbox_inches="tight")
        plt.show()
    if (save is not None) and (num_columns == 1):
        pdf = pdf.close()


def plot_topic_custom(
    cistopic_obj_projections: dict,
    cell_topic: pd.DataFrame,
    reduction_name: str,
    target: str = "cell",
    cmap: matplotlib.cm = cm.viridis,
    dot_size: int = 10,
    alpha: float = 1.0,
    scale: bool = False,
    selected_topics: list[int] | None = None,
    selected_features: str | None = None,
    harmony: bool = False,
    figsize: tuple[float, float]=(6.4, 4.8),
    num_columns: int = 1,
    save: str | None = None,
):
    """
    Plot topic distributions into dimensionality reduction.

    Parameters
    ----------
    cistopic_obj_projections: dict
            A dictionary with cisTopic object dimensionality reductions.
    cell_topic: pd.DataFrame
            A dataframe with the cell-topic distributions.
    reduction_name: str
            Name of the dimensionality reduction to use
    target: str
            Whether cells ('cell') or regions ('region') should be used. Default: 'cell'
    cmap: str or 'matplotlib.cm'
            For continuous variables, color map to use for the legend color bar. Default: cm.viridis
    dot_size: int
            Dot size in the plot. Default: 10
    alpha: float
            Transparency value for the dots in the plot. Default: 1
    scale: bool
            Whether to scale the cell-topic or topic-regions contributions prior to plotting. Default: False
    selected_topics: list[int], optional
            A list with selected topics to be used for plotting. Default: None (use all topics)
    selected_features: list[str], optional
            A list with selected features (cells or regions) to plot. This is recommended when working with regions (e.g. selecting
            regions in binarized topics), as working with all regions can be time consuming. Default: None (use all features)
    harmony: bool
            If target is 'cell', whether to use harmony processed topic contributions. Default: False
    figsize: tuple[float, float], optional
            Size of the figure. If num_columns is 1, this is the size for each figure; if num_columns is above 1, this is the overall size of the figure (if keeping
            default, it will be the size of each subplot in the figure). Default: (6.4, 4.8)
    num_columns: int
            For multiplot figures, indicates the number of columns (the number of rows will be automatically determined based on the number of plots). Default: 1
    save: str, optional
            Path to save plot. Default: None.

    """
    embedding = cistopic_obj_projections[target][reduction_name]
    data_mat = cell_topic

    if selected_features is not None:
        data_mat = data_mat.loc[selected_features]
        embedding = embedding.loc[selected_features]

    data_mat = data_mat.loc[:, embedding.index.to_list()]

    if selected_topics is not None:
        data_mat = data_mat.loc[["Topic" + str(x) for x in selected_topics]]

    if scale:
        data_mat = pd.DataFrame(
            sklearn.preprocessing.StandardScaler().fit_transform(data_mat),
            index=data_mat.index.to_list(),
            columns=data_mat.columns,
        )
    data_mat = data_mat.T

    if selected_topics is None:
        topic = data_mat.columns.to_list()
    else:
        topic = ["Topic" + str(t) for t in selected_topics]

    if (save is not None) and (num_columns == 1):
        pdf = matplotlib.backends.backend_pdf.PdfPages(save)

    if num_columns > 1:
        num_rows = int(np.ceil(len(topic) / num_columns))
        if figsize == (6.4, 4.8):
            figsize = (6.4 * num_columns, 4.8 * num_rows)
        i = 1

    fig = plt.figure(figsize=figsize)

    for var in topic:
        var_data = data_mat.loc[:, var]
        var_data = var_data.sort_values()
        embedding_plot = embedding.loc[var_data.index.tolist()]
        o = np.argsort(var_data)
        if num_columns > 1:
            plt.subplot(num_rows, num_columns, i)
            i = i + 1
        if not scale:
            plt.scatter(
                embedding_plot.iloc[o, 0],
                embedding_plot.iloc[o, 1],
                c=subset_list(var_data, o),
                cmap=cmap,
                s=dot_size,
                alpha=alpha,
                vmin=0,
                vmax=max(var_data),
            )
            normalize = mcolors.Normalize(vmin=0, vmax=np.array(var_data).max())
        else:
            plt.scatter(
                embedding_plot.iloc[o, 0],
                embedding_plot.iloc[o, 1],
                c=subset_list(var_data, o),
                cmap=cmap,
                s=dot_size,
                alpha=alpha,
            )
            normalize = mcolors.Normalize(
                vmin=np.array(var_data).min(), vmax=np.array(var_data).max()
            )
        plt.xlabel(embedding_plot.columns[0])
        plt.ylabel(embedding_plot.columns[1])
        plt.title(var)
        # setup the colorbar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(var_data)
        plt.colorbar(scalarmappaple)
        if num_columns == 1:
            if save is not None:
                pdf.savefig(fig, bbox_inches="tight")
            plt.show()

    if num_columns > 1:
        plt.tight_layout()
        if save is not None:
            fig.savefig(save, bbox_inches="tight")
        plt.show()

    if (save is not None) and (num_columns == 1):
        pdf.close()
