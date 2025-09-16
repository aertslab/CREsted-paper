import anndata                                                    # type: ignore

PATH_TO_ADATA="../../../../data/biccn/Mouse/Mouse_atac.h5ad"
CELL_TYPE_COL="subclass_Bakken_2022"
SAMPLE_COL="Sample"


adata = anndata.read_h5ad(
    PATH_TO_ADATA
)

cell_type_metadata = adata.obs[[SAMPLE_COL, CELL_TYPE_COL]].reset_index()

cell_type_metadata["index"] = [
    x.split("_")[1] for x in cell_type_metadata["index"]
]

cell_type_metadata = cell_type_metadata.rename(
    {
        "index": "cell_barcode",
        CELL_TYPE_COL: "cell_type",
        SAMPLE_COL: "sample"
    },
    axis=1
)

cell_type_metadata[["sample", "cell_type", "cell_barcode"]].to_csv(
    "cell_type_to_cell_barcode.tsv",
    header=True,
    index=False,
    sep="\t"
)
