X = read.table(paste("data/CCLE/", ptype, ".INTC.CCLE.tsv", sep=""), header=T)
anno = read.csv("data/raw/CCLE_NP24.2009_Drug_data_2015.02.24.csv", as.is=T)
drugs = sort(unique(anno$Compound))

####
original.ss.PP = rownames(X)
sapply(original.ss.PP, function(x){
        new.u = u = strsplit(x, split="\\.")[[1]][1]
        if(grepl("^X", u)){
            substr(u, 2, nchar(u)) -> new.u
        }
        new.u
}) -> ss.PP
names(ss.PP) = NULL

################################
y_mat = matrix(0, nrow=nrow(X), ncol=length(drugs))
for(k in 1:length(drugs)){
    drug = drugs[k]
    anno.1 = anno[which(anno$Compound==drugs[k] & anno[, "ActArea"] != -9 ),]
    match(ss.PP, anno.1[,1]) -> ii
    y_mat[, k] = anno.1[ii, 'ActArea']
    y_mat[which(is.na(ii)),k] = -9
}
rownames(y_mat) = rownames(X)
colnames(y_mat) = drugs

sapply(rownames(y_mat), function(x){
    strsplit(x, split="\\.")[[1]][1] -> u
    strsplit(u, split="_")[[1]] -> v; v= v[-1]; paste(v, collapse="_")
}) -> tt
names(tt) = NULL

cbind(y_mat, tissue=tt) -> y_tissue
write.table(y_tissue, file="data/CCLE/CCLE__Y.txt", quote=F, sep="\t", col.names=F)

