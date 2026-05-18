########### 12.29.2024: DRUG_ID
#> which(x1 == 2)
#    Acetalax Dactinomycin    Docetaxel  Fulvestrant       GSK343  Oxaliplatin 
#          18           84           90          106          126          194 
# Selumetinib  Ulixertinib   Uprosertib 
#         233          260          265 

X = read.table(paste("data/GDSC2/KEGG.INTC.GDSC2.tsv", sep=""), header=T)
cor(t(X)) -> mm

cell.line.anno = read.csv("data/raw/GDSC/download/Cell_Lines_Details.csv", as.is=T)
cell_lines = intersect(rownames(mm), paste("DATA.", cell.line.anno$COSMIC.identifier, sep="") )
match(cell_lines, paste("DATA.", cell.line.anno$COSMIC.identifier, sep="") ) -> idx
cell.line.anno = cell.line.anno[idx, ]

cell_lines = rownames(X)
anno = read.csv("data/raw/GDSC/download/GDSC2_fitted_dose_response_24Jul22.csv", as.is=T)

tapply(anno$DRUG_ID, anno$DRUG_NAME, function(u)length(unique(u))) -> check
double_drugs = names(which(check == 2))

drug_IDs = sort(unique(anno$DRUG_ID))
y_mat = matrix(0, nrow=nrow(X), ncol=length(drug_IDs))

drug_info = cbind(DRUG_ID=drug_IDs, DRUG_NAME=0, N=0, N_POS = 0, N_NEG = 0)

for(k in 1:length(drug_IDs)){
    
    drug_ID = drug_IDs[k]

    anno.1 = anno[which(anno$DRUG_ID==drug_ID),]
    anno.1.match1 = anno.1[which(  anno.1$COSMIC_ID %in% cell.line.anno$COSMIC.identifier ), ]
    
    drug = anno.1[1, "DRUG_NAME"]
    if(drug %in% double_drugs)drug = paste(drug, drug_ID, sep="_")    
    
    ### LN_IC50: higher means sensitive
    ### AUC: lower means sensitive
    x = -anno.1.match1[, "LN_IC50"] 
    names(x) = paste("DATA.",anno.1.match1$COSMIC_ID,sep="")
    
    x1 = x[cell_lines]
    x1[which(is.na(x1))] = -9
    y_mat[,k] = x1
    
    x = x[x!=-9]
    nn = ceiling(length(x) * 0.05)
    
    x = sort(x, decreasing=T)
    pos_cells = names(x)[1:nn]
    
    x = sort(x, decreasing=F)
    neg_cells = names(x)[1:nn]
    
    drug_info[k, ] = c(drug_ID, drug, length(x), length(pos_cells), length(neg_cells))
    
    cell_dat = rbind(cbind(pos_cells, 'positive'), cbind(neg_cells, 'negative'))
    write.table(cell_dat, file=paste("data/GDSC2/cell_label/",drug, ".cell_label.txt", sep=""), row.names=F, col.names=F, quote=F, sep='\t')
    
}

rownames(y_mat) = cell_lines
colnames(y_mat) = drug_IDs
write.table(y_mat, file="data/GDSC2/GDSC2__Y.V2.txt", quote=F, sep="\t", col.names=F)
write.table(cbind(seq(1, length(drug_IDs)), drug_info), file="data/GDSC2/GDSC2__Y.drug_names.V2.txt", quote=F, sep="\t", col.names=F, row.names=F)

