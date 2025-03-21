# Figure 5: Transfer learning from and benchmarking against a large pre-trained sequence-to-function model.

## Description

**compare_models.ipynb**  
A Jupyter notebook comparing the fine-tuned models, reproducing fig. 5b/c/e and S12/S13/S14/S16.

**finetune_fullmodel.ipynb**  
A Jupyter notebook showcasing the fine-tuning for the full Borzoi model.

**finetune_cnnmodel.ipynb**  
A Jupyter notebook showcasing the fine-tuning for the CNN-only Borzoi model.

**validated_enhancer_scoring.ipynb**  
A Jupyter notebook showcasing performance of Base Borzoi, the double fine-tuned Borzoi model, and DeepBICCN2 on validated mouse enhancers.

**scoring_utils.py**
Helper functions for the validated_enhancer_scoring notebook.

**get_borzoi_enhancer_scores.py**  
A script to get Borzoi predictions for certain regions and classes, creating a (n_classes, n_regions) anndata of prediction counts.