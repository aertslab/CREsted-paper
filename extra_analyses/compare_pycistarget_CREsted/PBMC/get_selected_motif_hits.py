# B-cells:
# - EBF1
# - PAX5
# - POU2F2

# CD14 monocytes:
# - CEBPA
# - SPI1

# CD4 + T-cells
# - RUNX1
# - ETS1
# - GATA3
from pycistarget.input_output import read_hdf5
import os

MENR_DIR="../../pycisTarget/PBMC/cistarget_out/"

CELLTYPES = ["B_cell", "CD14_monocyte", "CD4_T_cell"]

OUTDIR = "menr_target_regions"

CELLTYPE_TO_MENR = {
    "B_cell": os.path.join(
        MENR_DIR, "motif_enrichment_cistarget_Merged__B_cell.hdf5"),
    "CD14_monocyte": os.path.join(
        MENR_DIR, "motif_enrichment_cistarget_Merged__CD14_monocyte.hdf5"),
    "CD4_T_cell": os.path.join(
        MENR_DIR, "motif_enrichment_cistarget_Merged__CD4_T_cell.hdf5"
    )
}

CELLTYPE_TO_TF = {
    "B_cell":           ["EBF1", "PAX5", "POU2F2"],
    "CD14_monocyte":    ["CEBPA", "SPI1"],
    "CD4_T_cell":       ["RUNX1", "ETS1", "GATA3"]
}

def write_cistrome_to_bed(cistrome: list[str], outf: str):
    with open(outf, "w") as f:
        for r in cistrome:
            _ = f.write(
                "\t".join(r.replace(":", "-").split("-")) + "\n"
            )


assert all([os.path.exists(f) for f in CELLTYPE_TO_MENR.values()])

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

for cell_type in CELLTYPES:
    print(f"Processing: {cell_type}")
    menr_result = read_hdf5(
        CELLTYPE_TO_MENR[cell_type]
    )[f"Merged__{cell_type}"]
    for tf in CELLTYPE_TO_TF[cell_type]:
        cistromes_tf = [
            key for key in menr_result.cistromes["region_set"].keys()
            if key.split("_")[0] == tf
        ]
        print(f"\tFor {tf} found following cistroms: {', '.join(cistromes_tf)}")
        for cistrome in cistromes_tf:
            write_cistrome_to_bed(
                cistrome=menr_result.cistromes["region_set"][cistrome],
                outf=os.path.join(OUTDIR, f"{cistrome.split('_(')[0]}.bed")
            )
