#!/usr/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=gp_64C_128T_512GB
#SBATCH --time=48:00:00
#SBATCH --mem=400gb
#SBATCH --cpus-per-task=40

module load BEDTools
module load tqdm

HOME="../../../../../../.."
PYTHON=${HOME}/software/miniforge3/envs/create_cistarget_databases/bin/python

cd ${HOME}/PhD/papers/CREsted_2025/CREsted-paper/extra_analyses/pycisTarget/PBMC/

# to create this conda environment see: https://github.com/aertslab/create_cisTarget_databases
source  ${HOME}/software/miniforge3/etc/profile.d/conda.sh
conda activate create_cistarget_databases

# get file size to generate progressbar using tqdm (optional)
filesize=$(stat -c %s ../../crested_data.tar.gz)

# extract PBMC consensus regions
cat ../../crested_data.tar.gz \
    | tqdm --bytes --total $filesize --desc "Extracting consensus peaks" \
    | pigz -dc \
    | tar xvf - crested_data/Figure_3/pbmc/consensus_regions.bed


REGION_BED="crested_data/Figure_3/pbmc/consensus_regions.bed"
GENOME_FASTA="${HOME}/resources/hg38/hg38.fa"
CHROMSIZES="${HOME}/resources/hg38/hg38.chrom.sizes"
DATABASE_PREFIX="hg38_CTX_PBMC"
BG_PADDING=1000
CBDIR="../v10nr_clust_public/singletons/"
SEED=123

ls ${CBDIR} > motif_list.txt

git clone https://github.com/aertslab/create_cisTarget_databases.git

# generate fasta file containing sequence of each consensus peak and 1_000 bp background padding
echo "Generating fasta"
create_cisTarget_databases/create_fasta_with_padded_bg_from_bed.sh \
    ${GENOME_FASTA} \
    ${CHROMSIZES} \
    ${REGION_BED} \
    ${DATABASE_PREFIX}.fa \
    ${BG_PADDING} \
    yes

# generate cistarget database
echo "Generating database"
${PYTHON} create_cisTarget_databases/create_cistarget_motif_databases.py \
    -f ${DATABASE_PREFIX}.fa \
    -M ${CBDIR} \
    -m motif_list.txt \
    -o ${DATABASE_PREFIX} \
    -t 40 \
    -b ${BG_PADDING} \
    -s ${SEED} \

${PYTHON} create_cisTarget_databases/convert_motifs_or_tracks_vs_regions_or_genes_scores_to_rankings_cistarget_dbs.py \
    -i ${DATABASE_PREFIX}.motifs_vs_regions.scores.feather