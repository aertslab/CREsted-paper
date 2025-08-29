# Extra analysis: Fine-tuning 'genomic language models' for chromatin accessibility prediction.

## Description

**compare_glms.ipynb**  
A Jupyter notebook comparing the fine-tuned models, reproducing fig. S??.

**finetune_hyenadna.ipynb**  
A Jupyter notebook showcasing the fine-tuning for the HyenaDNA model.

**finetune_nuctrans.py**  
A script to fine-tune the Nucleotide Transformer model on all regions (first-round finetuning).

**finetune_nuctrans_further.py**  
A script to fine-tune the Nucleotide Transformer further on specific regions (second-round finetuning).

**finetune_nuctrans_pt.ipynb**  
A Jupyter notebook showcasing the results of the fine-tuning for the Nucleotide Transformer model.

**standalone_hyenadna.py**  
HyenaDNA utilities, as downloaded from [their repository](https://github.com/HazyResearch/hyena-dna/blob/main/standalone_hyenadna.py).

**utils_finetuning.py**  
Functions and classes that wrap around CREsted and PyTorch Lightning to provide tokenizer-compatible DataLoaders for AnnDatas and a base LightningModule to add a scalar-predicting head.

**utils_hyenadna.py**  
Utilities specific for HyenaDNA, including a reworked tokenizer, base model loading functions, and HyenaDNA-specific versions of the AnnDataModule and LightningModule.

**utils_nuctrans.py**  
Utilities specific for the Nucleotide Transformer, including Nucleotide Transformer-specific versions of the AnnDataModule and LightningModule.