module load parallel

mamba create --name pycisTarget python=3.11 -y
mamba activate pycisTarget

pip install git+https://github.com/aertslab/pycistarget
pip install attrs
pip install tables

n_celltypes=$(ls modisco_peaks_per_cell_type/*.bed | wc -l)

mkdir cistarget_out

ls modisco_peaks_per_cell_type/*.bed \
    | parallel --jobs ${n_celltypes} --bar '
    pycistarget cistarget \
        --cistarget_db_fname hg38_CTX_PBMC.regions_vs_motifs.rankings.feather \
        --bed_fname {} \
        --output_folder cistarget_out \
        --species homo_sapiens \
        --write_html
    '