from pathlib import Path
import os
import anndata as ad
import torch
import crested
import lightning
import transformers
from transformers import AutoModel, AutoTokenizer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from utils_nuctrans import NucTransTokenizerAnnDataModule, NucTransScalarModel

torch.set_float32_matmul_precision('medium')

print(f"PyTorch version: {torch.__version__}")
print(f"CREsted version: {crested.__version__}")
print(f"Transformers version: {transformers.__version__}")

# Prepare data paths
# Download the data for the notebooks from the dedicated Zenodo link of the CREsted paper

data_dir = Path("crested_data/Figure_5/") # CHANGE TO OWN LOCATION OF EXTRACTED crested_data.tar.gz
adata_filtered_file = data_dir / "data/mouse_biccn_data_specific.h5ad"
# Genome paths
resources_dir = Path("mm10_ucsc/fasta/") # CHANGE TO OWN
genome_file = resources_dir / "mm10.fa"
chromsizes_file = resources_dir / "mm10.chrom.sizes"

genome = crested.Genome(genome_file, chromsizes_file)
crested.register_genome(genome)

# Load data
adata = ad.read_h5ad(adata_filtered_file)

# Load model
tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-1000g")
base_model = AutoModel.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-1000g")

loss = crested.tl.losses.CosineMSELogLoss(max_weight=100, multiplier=1)
model = NucTransScalarModel.load_from_checkpoint(
    "nuctrans/lightning_logs/version_3/checkpoints/epoch=5-step=1322982.ckpt",
    base_model = base_model, 
    loss = loss, 
    lr = 1e-5, 
).to('cuda')
print(model)
    
# Create datamodule and trainer
datamodule = NucTransTokenizerAnnDataModule(
    adata = adata,
    genome = genome,
    batch_size=4,  # lower this if you encounter OOM errors
    max_stochastic_shift=3,
    always_reverse_complement=True,
    tokenizer = tokenizer,
    in_memory=True
)
datamodule.setup('fit')
datamodule.setup('test')

trainer = lightning.Trainer(
    default_root_dir = 'nuctrans', 
    logger = True, 
    max_epochs = 20,
    callbacks=[EarlyStopping(monitor = "val_loss", mode = "min"), ModelCheckpoint(monitor = "val_loss", mode = "min")]
)

# Train
trainer.fit(model, datamodule.train_dataloader.data, datamodule.val_dataloader.data)