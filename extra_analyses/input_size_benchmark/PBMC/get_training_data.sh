ml tqdm

filesize=$(stat -c %s ../../crested_data.tar.gz)
tqdm --bytes --total $filesize --desc "Extracting PBMC training data" < ../../crested_data.tar.gz \
    | pigz -dc \
    | tar xvf - crested_data/Figure_3/pbmc/consensus_regions.bed \
            crested_data/Figure_3/pbmc/bw/Bcell.bw \
            crested_data/Figure_3/pbmc/bw/CD4_Tcell.bw \
            crested_data/Figure_3/pbmc/bw/Cytotoxic_T_cell.bw \
            crested_data/Figure_3/pbmc/bw/Dendritic_cell.bw \
            crested_data/Figure_3/pbmc/bw/CD14_monocyte.bw \
            crested_data/Figure_3/pbmc/bw/Natural_killer_cell.bw \
            crested_data/Figure_3/pbmc/bw/CD16_monocyte.bw
