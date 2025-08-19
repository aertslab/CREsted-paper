import crested
import keras
import os
import anndata
import dataclasses
import numpy as np

GENOME_FA="../../../../../../../resources/hg38/hg38.fa"
GENOME_CHROM_SIZES="../../../../../../../resources/hg38/hg38.chrom.sizes"

crested.register_genome(
    crested.Genome(
        fasta=GENOME_FA,
        chrom_sizes=GENOME_CHROM_SIZES
    )
)

@dataclasses.dataclass
class Region:
    chrom: str
    start: int
    end: int

    @property
    def middle(self):
        return self.start + (self.end - self.start) // 2

    def resize(self, size:int):
        return Region(
            self.chrom,
            self.middle - size // 2,
            self.middle + size // 2 + (1 if size % 2 else 0)
        )
    
    def __str__(self):
        return f"{self.chrom}:{self.start}-{self.end}"

regions_to_score: list[tuple[Region, str]] = [
    (
        Region("chr19", 41_876_056, 41_878_170), "Bcell"
    ),
    (
        Region("chr14", 22_555_403, 22_557_517), "CD4_Tcell"
    ),
    (
        Region("chr9", 21_076_963, 21_079_077), "Dendritic_cell"
    )
]

def calc_deep_explainer_for_model(
        model_name: str) -> tuple[np.ndarray, np.ndarray]:
    adata = anndata.read_h5ad(
        os.path.join(
            model_name,
            "adata_" + model_name.replace("_same_shape", "") + ".h5ad"
        )
    )

    fine_tuned_models = os.listdir(
        os.path.join(
            model_name,
            "PBMC_INPUT_SIZE",
            model_name.replace("_same_shape", "") + "_finetune",
            "checkpoints"
        )
    )

    model = keras.models.load_model(
        os.path.join(
            model_name,
            "PBMC_INPUT_SIZE",
            model_name.replace("_same_shape", "") + "_finetune",
            "checkpoints",
            sorted(fine_tuned_models)[-1]
        ),
        compile=False
    )

    all_scores = []
    all_one_hot = []
    for region, cell_type in regions_to_score:
        print((region, cell_type))
        class_idx = list(adata.obs_names.get_indexer([cell_type]))
        scores, one_hot_encoded_sequences = crested.tl.contribution_scores(
            str(region.resize(model.input[0].shape[1])),
            target_idx=class_idx,
            model=model,
        )
        all_scores.append(scores)
        all_one_hot.append(one_hot_encoded_sequences)
    
    return (
        np.concatenate(all_scores).squeeze(), 
        np.concatenate(all_one_hot).squeeze()
    )

model_to_scores: dict[str, np.ndarray] = {}
model_to_onehot: dict[str, np.ndarray] = {}

for model_name in os.listdir():
    if not model_name.startswith("input_"):
        continue
    print(model_name)

    scores, onehot = calc_deep_explainer_for_model(model_name)

    model_to_scores[model_name] = scores
    model_to_onehot[model_name] = onehot

np.savez(
    "selected_regions_one_hot.npz",
    **model_to_onehot
)

np.savez(
    "selected_regions_scores.npz",
    **model_to_scores
)