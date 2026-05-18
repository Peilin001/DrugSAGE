# DrugSAGE: 
<img width="1845" height="901" alt="Figure1" src="https://github.com/user-attachments/assets/a63435b0-e008-4e6a-b2f8-38dc0e111e3e" />

DrugSAGE is an aggregation-based method for drug response imputation. It takes gene expression data as input for the task.
 
Step #1. Prepare files for training:
Use 1.CCLE.INTC.V1.R to prepare the input gene expression "KEGG.INTC.CCLE.tsv" for the reference network. The code also generates the gene-pathway annotation file "KEGG.CCLE.gmt" and the cell lines similarity network "KEGG.E11.CCLE.cites".
