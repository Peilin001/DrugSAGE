# DrugSAGE: 
<img width="1845" height="901" alt="Figure1" src="https://github.com/user-attachments/assets/a63435b0-e008-4e6a-b2f8-38dc0e111e3e" />

DrugSAGE is an aggregation-based method for drug response imputation. It takes gene expression data as input for the task.
 
Step #1. Prepare files for training:
Use 1.CCLE.INTC.V1.R to prepare the input gene expression "KEGG.INTC.CCLE.tsv" for the reference network. The code also generates the gene-pathway annotation file "KEGG.CCLE.gmt" and the cell lines similarity network "KEGG.E11.CCLE.cites". Similarly, use 2.GDSC2.INTC.V1.R for the GDSC data.

Use 3.CCLE_Y.R, 4.GDSC2_Y.R, and 5.GDSC1_Y.R to prepare the drug response data.

Use 6.CCLE.cell_label.R to prepare the most sensitive cell lines and the most insensitive cell lines for each drug.

Step #2. Run the training process:
The code 11.round_CCLE_V2.py, 11.round_GDSC1_V2.py, and 11.round_GDSC2_V2.py apply 10 rounds of 10-fold cross validation model training. The input parameters include drug name, R, and N. For example:
python 11.round_CCLE_V2.py 11 11
This will train models for the drug Lapatinib with R = 11 and N = 11. For each fold in each round, the code generates three files: 
- Lapatinib.CCLE.R11N11.round_i.fold_j.best_model.torch
- Lapatinib.CCLE.R11N11.round_i.fold_j.info.txt
- Lapatinib.CCLE.R11N11.round_i.fold_j.torch

