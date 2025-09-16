ml tqdm

filesize=$(stat -c %s ../../crested_data.tar.gz)
tqdm --bytes --total $filesize --desc "Extracting PBMC training data" < ../../crested_data.tar.gz \
    | pigz -dc \
    | tar xvf - crested_data/Figure_2/consensus_regions.bed
