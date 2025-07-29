# get file size to generate progressbar using tqdm (optional)
filesize=$(stat -c %s ../../crested_data.tar.gz)

# extract PBMC modisco regions
tqdm --bytes --total $filesize --desc "Extracting modisco regions" < ../../crested_data.tar.gz \
    | pigz -dc \
    | tar xvf - crested_data/Figure_3/pbmc/modisco_regions.csv

celltypes=$(tail --lines=+2 crested_data/Figure_3/pbmc/modisco_regions.csv | cut -f 6 -d "," | sort -u); echo ${celltypes}

mkdir modisco_peaks_per_cell_type

for celltype in ${celltypes}
do
    echo ${celltype}
    awk -F',' -v ct="$celltype" '$6 == ct' crested_data/Figure_3/pbmc/modisco_regions.csv \
        | cut -f 2,3,4 -d "," \
        | tr "," "\t" \
        | sort -k 1,1 -k2,2n \
        > modisco_peaks_per_cell_type/${celltype}.bed
done