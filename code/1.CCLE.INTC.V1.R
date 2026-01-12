INT = function(x){
	N = length(x)
	C = 3/8
	qnorm( (rank(x) - C)/(N - 2*C + 1) )
}

##########################################################################################
gene_info = read.delim('data/Homo_sapiens.gene_info')
which(gene_info$type_of_gene == 'protein-coding') -> pc_idx
#> length(idx)
#[1] 20632

hkgenes = read.table('data/HK_genes.txt')
hkgenes_TR = hkgenes[,1]


ccle = read.table("data/raw/CCLE_DepMap_18q3_RNAseq_RPKM_20180718.gct", skip=2, sep="\t", header=T, as.is=T)
ccle.gene.mat = as.matrix(ccle[, c(-1, -2)])
rownames(ccle.gene.mat) = ccle[,2]
print(paste('Raw CCLE matrix: ', dim(ccle.gene.mat), sep="" ))

#### 1. choose Protein-coding genes

genes = intersect(rownames(ccle.gene.mat), gene_info[pc_idx, 3])
ccle.gene.mat = ccle.gene.mat[genes, ]

print(paste('Protein-coding genes: ', nrow(ccle.gene.mat), sep="" ))

#### 2. choose genes with zero in < 5% samples

apply(ccle.gene.mat, 1, function(u)sum(u==0)/length(u)) -> rowZero
which(rowZero > 0.05) -> ii
ccle.gene.mat = ccle.gene.mat[-ii, ]

print(paste('After removing zero: ', nrow(ccle.gene.mat), sep="" ))

#### 3. choose genes not lowly expressed, avg.RPKM > 1 in CCLE

apply(ccle.gene.mat, 1, mean) -> rowMean
which(rowMean > 1) -> expressed
ccle.gene.mat = ccle.gene.mat[expressed, ]

print(paste('After removing avg.RPKM < 1: ', nrow(ccle.gene.mat), sep="" ))

#### 4. choose genes with var > 1 in CCLE

apply(ccle.gene.mat, 1, var) -> rowVar
which(rowVar > 1) -> expressed
ccle.gene.mat = ccle.gene.mat[expressed, ]

print(paste('After removing var < 1: ', nrow(ccle.gene.mat), sep="" ))


apply(ccle.gene.mat, 1, var) -> rowVar
apply(ccle.gene.mat, 1, mean) -> rowMean


####################################################
## choose tissues with >= 20 cell lines
sapply(colnames(ccle.gene.mat), function(x){
	strsplit(x, split="\\.")[[1]][1] -> u
	strsplit(u, split="_")[[1]] -> v; v= v[-1]; paste(v, collapse="_")
}) -> tt

names(tt) = NULL
table(tt) -> t1
names(which(t1 >= 20)) -> tissues.int
which(tt %in% tissues.int) -> ii

ccle.gene.mat = ccle.gene.mat[, ii]
sapply(colnames(ccle.gene.mat), function(x){
	strsplit(x, split="\\.")[[1]][1] -> u
	strsplit(u, split="_")[[1]] -> v; v= v[-1]; paste(v, collapse="_")
}) -> tt
names(tt) = NULL
dim(ccle.gene.mat)

### log2-transform CCLE data
log2.ccle.gene.mat = log2(ccle.gene.mat+1)
apply(log2.ccle.gene.mat, 1, var) -> rowVar
apply(log2.ccle.gene.mat, 1, mean) -> rowMean

####################################################

gmt_file = 'data/c2.all.v2022.1.Hs.symbols.gmt'

tmp = readLines(gmt_file)
path_list = list()
tags = c()

hk_ratio = c()
 
for(k in 1:length(tmp)){
	lines = strsplit(tmp[k], split="\t")[[1]]
	if(grepl("RIBOSOME", lines[1] ) ){
		print(lines[1])
		next
	}
	pathway_genes = lines[c(-1,-2)]
	
	### filter #1
	hkTR_path_genes = intersect(pathway_genes, hkgenes_TR)
	if(length(hkTR_path_genes)/length(pathway_genes) > 0.3)next
	
	pathway_genes = intersect(lines[c(-1,-2)], names(rowMean))
	
	### filter #2
	match(pathway_genes, rownames(ccle.gene.mat)) -> ii
	tmp2 = cor(t(ccle.gene.mat[ii, ]))
	
	which(apply(tmp2,1,function(u)sum(u>0.7))!=1) -> idx
	if(length(idx) > 0){
		tmp_genes = rownames(ccle.gene.mat)[ii[idx]]
		keep_gene = tmp_genes[which.max(rowVar[tmp_genes])]
		tmp_genes = setdiff(tmp_genes, keep_gene)
		pathway_genes = setdiff(pathway_genes, tmp_genes )
	}
	
	### filter #3
	if(length(pathway_genes) < 20 | length(pathway_genes) > 1000)next
	
	path_list[[lines[1]]] = pathway_genes
	tag = strsplit(lines[1], split="_")[[1]][1]
	if(tag == "WP") tag = "WP___"
	if(tag == "PID") tag = "PID___"
	tags = c(tags, tag)
}

### filter #4
ptypes = c("KEGG")

to_remove = c()
for(ptype in ptypes){
	grep(ptype, toupper(tags)) -> ii
	new_path_list = path_list[ii]
	new_genes = unique(unlist(new_path_list))
	
	info = c()
	for(k in 1:(length(new_path_list)-1) ){
		genes_1 = new_path_list[[k]]
		for(k2 in (k+1):length(new_path_list)){
			genes_2 = new_path_list[[k2]]
			genes = intersect(genes_1, genes_2)
			info = rbind(info, c(k,k2,length(genes_1), length(genes_2), length(genes)/length(union(genes_1,genes_2)) ) )
		}
	}
	which(info[,5] > 0.5) -> idx
	ll = c(length(genes_1), length(genes_2))
	to_remove = c(to_remove, names(new_path_list)[info[idx, which.min(ll)  ]])
	
	print(c(ptype, length(new_path_list), length(idx)))
}
to_remove = unique(to_remove)
print(to_remove)
print(paste("before overlap check: ",length(path_list)))
match(to_remove, names(path_list)) -> ii

path_list = path_list[-ii]
tags = tags[-ii]

print(length(path_list))
######################## end of filter #4


##### Part 2
ptypes = c("KEGG")

for(ptype in ptypes){
	grep(ptype, toupper(tags)) -> ii
	new_path_list = path_list[ii]
	new_genes = unique(unlist(new_path_list))
	
	path_gmt = c()
	for(k in 1:length(new_path_list)){
		genes = new_path_list[[k]]
		genes = intersect(genes, names(rowMean) )
		pname = names(new_path_list)[k]
		a = paste(c(pname, ptype, genes), collapse="\t")
		path_gmt = c(path_gmt, a)
	}
	
	write.table(path_gmt, file=paste("data/CCLE/", ptype, ".CCLE.F124.gmt", sep=""), row.names=F, col.names=F, quote=F)
	print(c(ptype, length(path_gmt), length(new_genes)))
}


##############################################################################
#### generate INTC for CCLE
##############################################################################

for(ptype in ptypes){
	grep(ptype, toupper(tags)) -> ii
	new_path_list = path_list[ii]
	new_genes = unique(unlist(new_path_list))
	mad.genes = intersect(new_genes, rownames(ccle.gene.mat))
	
	ccle.train.mat = log2.ccle.gene.mat[mad.genes,]  ### gene by sample
	qx.ccle.gene.mat = apply(ccle.train.mat, 2, INT )
	print(dim(qx.ccle.gene.mat))
	
	write.table(t(qx.ccle.gene.mat), file=paste("data/CCLE/", ptype, ".INTC.CCLE.tsv", sep=""), row.names=T, quote=F, sep="\t")
}

##########################################################
#### generate INTC for TCGA (step #1 for TCGA)
##########################################################

cancers = dir("data/raw/TCGA")
for(ptype in ptypes){
	X = read.table(paste("data/CCLE/", ptype, ".INTC.CCLE.tsv", sep=""), header=T)
	kh.genes = colnames(X)
	kh.genes = gsub('\\.', '-', kh.genes)
	for(cancer in cancers){
		original.TCGA.RPKM = read.delim(paste("/home/peilin/work/23-GNN-Drug/data/raw/TCGA/",cancer,"/HiSeqV2_percentile.gz", sep=""), as.is=T)
			
		match(kh.genes, original.TCGA.RPKM[,1]) -> ii
		shared.TCGA.RPKM = original.TCGA.RPKM[ii, -1]
		rownames(shared.TCGA.RPKM) = kh.genes
			
		apply(shared.TCGA.RPKM, 1, function(u){u[is.na(u)] = 0; u} ) -> TCGA.RPKM.mat
		qx.TCGA.RPKM.mat = apply(TCGA.RPKM.mat, 1, INT )  ### rows are samples
			
		print(c(cancer, nrow(qx.TCGA.RPKM.mat), ncol(qx.TCGA.RPKM.mat) ))
		write.table(t(qx.TCGA.RPKM.mat),        file=paste("data/TCGA/CCLE/",cancer,".INTC.txt", sep=""), row.names=F, col.names=F, quote=F, sep="\t")
		write.table(colnames(qx.TCGA.RPKM.mat), file=paste("data/TCGA/CCLE/",cancer,".INTC.rownames.txt",  sep=""), row.names=F, col.names=F, quote=F, sep="\t")
	}
	print(paste(ptype, ": #genes = ", length(kh.genes), sep=""))
}

##########################################################
#### for different # neighbors, generate edges
##########################################################

for(n_neighbors in c(3:20)){
	
	##########################################################
	#### generate interactions for CCLE
	##########################################################
	
	ptype = "KEGG"
	X = read.table(paste("data/CCLE/", ptype, ".INTC.CCLE.tsv", sep=""), header=T)
	cor(t(X)) -> mm
	cell_lines = rownames(mm)

	expr_edgelist = c()
	for(k in 1:nrow(mm)){
		sort(mm[k, ], decreasing=T) -> x
		x = x[x!=1]
		for(k1 in seq(1,n_neighbors)){
		expr_edgelist = rbind(expr_edgelist, c(cell_lines[k], names(x)[k1], "EXPR", "CO", mm[cell_lines[k], names(x) [k1] ] ) )
		}
	}

	expr_edgelist_str = apply(expr_edgelist, 1, function(u)paste(sort(u[1:2]), collapse="---"))
	match(unique(expr_edgelist_str), expr_edgelist_str) -> idx
	expr_edgelist = expr_edgelist[idx, ]
	write.table(expr_edgelist, file=paste("data/CCLE/", ptype,".E",n_neighbors,".CCLE.cites",sep=""), row.names=F, quote=F, col.names=F, sep="\t")


	##########################################################
	#### generate interactions for TCGA (step #2 for TCGA)
	##########################################################
	CCLE_X1 = read.table(paste("data/CCLE/", ptype, ".INTC.CCLE.tsv", sep=""), header=T)
	for(cancer in cancers){
		cat(paste(cancer, ".", sep=""))
		query_mat  = read.table(paste("data/TCGA/CCLE/",cancer,".INTC.txt", sep=""), header=F)
		query_name = read.table(paste("data/TCGA/CCLE/",cancer,".INTC.rownames.txt", sep=""), header=F)
		rownames(query_mat) = query_name[,1]
		colnames(query_mat) = colnames(CCLE_X1)
		
		rbind(CCLE_X1, query_mat) -> X2
		cor(t(X2)) -> mm

		all_query_edgelist = c()
		for(g in (nrow(CCLE_X1)+1):nrow(mm)){
			tmp = mm[g, 1:nrow(CCLE_X1)]
			tmp = sort(tmp, decreasing=T)
			names(tmp)[1:n_neighbors] -> int_cell_lines
			all_query_edgelist = rbind(all_query_edgelist, cbind(rownames(mm)[g], int_cell_lines, tmp[1:n_neighbors]))
		}
		write.table(all_query_edgelist,  file=paste("data/TCGA/CCLE/",cancer,".E",n_neighbors,".txt",        sep=""), row.names=F, col.names=F, quote=F, sep="\t")
	}
} ### n_neighbors

	
