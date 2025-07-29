ml tqdm

# extract modisco results
tqdm --bytes --total $filesize --desc "Extracting modisco results" < ../../crested_data.tar.gz \
    | pigz -dc \
    | tar xvf - crested_data/Figure_3/pbmc/modisco/Bcell_modisco_results.h5 \
        crested_data/Figure_3/pbmc/modisco/CD14_monocyte_modisco_results.h5 \
        crested_data/Figure_3/pbmc/modisco/CD16_monocyte_modisco_results.h5 \
        crested_data/Figure_3/pbmc/modisco/CD4_Tcell_modisco_results.h5 \
        crested_data/Figure_3/pbmc/modisco/Cytotoxic_T_cell_modisco_results.h5 \
        crested_data/Figure_3/pbmc/modisco/Dendritic_cell_modisco_results.h5 \
        crested_data/Figure_3/pbmc/modisco/Natural_killer_cell_modisco_results.h5
