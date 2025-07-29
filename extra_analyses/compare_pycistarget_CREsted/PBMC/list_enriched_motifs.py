from pycistarget.input_output import read_hdf5
import os

enriched_motifs: set[str] = set()

for f in os.listdir("../../pycisTarget/PBMC/cistarget_out"):
    if not f.endswith(".hdf5"):
        continue
    print(f)
    results = read_hdf5(
        os.path.join(
            "../../pycisTarget/PBMC/cistarget_out", f
        )
    )
    for result in results.values():
        enriched_motifs.update(
            result.motif_hits["region_set"].keys()
        )

with open("enriched_motifs.txt", "w") as f:
    for motif in enriched_motifs:
        _ = f.write(f"{motif}\n")
