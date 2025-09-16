# we will use https://github.com/aertslab/scatac_fragment_tools
# to create pseudobulk (per cell type) fragment files

echo -e "sample\tpath_to_fragment_file
2C_rep1\t../../../../data/biccn/Mouse/ATAC_Fragments/2C_rep1_atac_fragments.tsv.gz
2C_rep2\t../../../../data/biccn/Mouse/ATAC_Fragments/2C_rep2_atac_fragments.tsv.gz
3C_rep1\t../../../../data/biccn/Mouse/ATAC_Fragments/3C_rep1_atac_fragments.tsv.gz
3C_rep2\t../../../../data/biccn/Mouse/ATAC_Fragments/3C_rep2_atac_fragments.tsv.gz
4B_rep1\t../../../../data/biccn/Mouse/ATAC_Fragments/4B_rep1_atac_fragments.tsv.gz
4B_rep2\t../../../../data/biccn/Mouse/ATAC_Fragments/4B_rep2_atac_fragments.tsv.gz
5D_rep1\t../../../../data/biccn/Mouse/ATAC_Fragments/5D_rep1_atac_fragments.tsv.gz
5D_rep2\t../../../../data/biccn/Mouse/ATAC_Fragments/5D_rep2_atac_fragments.tsv.gz" \
    > sample_to_fragment.tsv

mkdir pseudobulk

scatac_fragment_tools split \
    -f sample_to_fragment.tsv \
    -b cell_type_to_cell_barcode.tsv \
    -c  ../../../../../../../resources/mm10/mm10.chrom.sizes \
    -o pseudobulk
