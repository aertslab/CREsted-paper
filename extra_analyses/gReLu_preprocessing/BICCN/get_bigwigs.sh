ml tqdm

filesize=$(stat -c %s ../../crested_data.tar.gz)
tqdm --bytes --total $filesize --desc "Extracting PBMC training data" < ../../crested_data.tar.gz \
    | pigz -dc \
    | tar xvf - crested_data/Figure_2/cut_sites_bws/L6CT.bw \
                crested_data/Figure_2/cut_sites_bws/Sncg.bw \
                crested_data/Figure_2/cut_sites_bws/Lamp5.bw \
                crested_data/Figure_2/cut_sites_bws/Pvalb.bw \
                crested_data/Figure_2/cut_sites_bws/Endo.bw \
                crested_data/Figure_2/cut_sites_bws/L5ET.bw \
                crested_data/Figure_2/cut_sites_bws/Micro_PVM.bw \
                crested_data/Figure_2/cut_sites_bws/OPC.bw \
                crested_data/Figure_2/cut_sites_bws/L6IT.bw \
                crested_data/Figure_2/cut_sites_bws/Vip.bw \
                crested_data/Figure_2/cut_sites_bws/Oligo.bw \
                crested_data/Figure_2/cut_sites_bws/SstChodl.bw \
                crested_data/Figure_2/cut_sites_bws/L5_6NP.bw \
                crested_data/Figure_2/cut_sites_bws/Sst.bw \
                crested_data/Figure_2/cut_sites_bws/Astro.bw \
                crested_data/Figure_2/cut_sites_bws/VLMC.bw \
                crested_data/Figure_2/cut_sites_bws/L5IT.bw \
                crested_data/Figure_2/cut_sites_bws/L6b.bw \
                crested_data/Figure_2/cut_sites_bws/L2_3IT.bw
