# Figure 4: Comparing MES-like states across cancers in cell lines and biopsies.

## Description

**train_deepccl.ipynb**  
A Jupyter notebook illustrating the training of the base DeepCCL model and its finetuning on cell-type specific regions.

**evaluate_deepccl_chrombpnet.ipynb**  
A Jupyter notebook evaluating and comparing the trained DeepCCL and ChromBPNet models, reproducing Fig. 4b/c/d and S10.

**enhancer_code_deepccl.ipynb**  
A Jupyter notebook analyzing the enhancer code used of the base and finetuned DeepCCL model, reproducing Fig. 4e.

**cistopic_visualization.ipynb**  
A Jupyter notebook showing the tSNEs obtained from pycisTopic of the GBM scATAC dataset, reproducing Fig. 4f and S11a.

**train_evaluate_deepglioma.ipynb**  
A Jupyter notebook illustrating the training and evaluation of the DeepGlioma topic model, reproducing Fig. S11b and part of 4d.

**topic_scoring.ipynb**  
A Jupyter notebook illustrating the scoring of biopsy topics using the DeepCCL model, reproducing Fig. S11c.

**contribution_comparison.ipynb**  
A Jupyter notebook demonstrating the contribution score comparison of the DeepCCL and DeepGlioma models, reproducing Fig. 4g.

**motif_comparison.ipynb**  
A Jupyter notebook comparing the pattern clustering between DeepCCL and DeepGlioma, reproducing Fig. 4h/i.

**chrombpnet_porting.ipynb**  
A Jupyter notebook showing the porting of the ChromBPNet models to CREsted.

**cistopic_utils.py**  
Helper custom functions for the pycisTopic visualization notebook.
