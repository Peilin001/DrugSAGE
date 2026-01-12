INT = function(x){
	N = length(x)
	C = 3/8
	qnorm( (rank(x) - C)/(N - 2*C + 1) )
}

hkgenes = read.table('data/HK_genes.txt')
hkgenes_TR = hkgenes[,1]

print(c("HK genes from Trends Genet.:", length(hkgenes_TR)))

##############################################################################
gene_info = read.delim('data/Homo_sapiens.gene_info')
which(gene_info$type_of_gene == 'protein-coding') -> pc_idx

tmp = read.delim("data/raw/GDSC/download/Cell_line_RMA_proc_basalExp.txt")
dat = tmp[,-2]

dat = dat[dat[,1] %in% gene_info[pc_idx, 3], ]

cell.line.anno = read.csv("data/raw/GDSC/download/Cell_Lines_Details.csv", as.is=T)
cell_lines = intersect(colnames(dat), paste("DATA.", cell.line.anno$COSMIC.identifier, sep="") )
match(cell_lines, paste("DATA.", cell.line.anno$COSMIC.identifier, sep="") ) -> idx
cell.line.anno = cell.line.anno[idx, ]

dat = dat[, c(1, match(cell_lines, colnames(dat)))]
print(c('# gdsc2_genes (before HK) =', nrow(dat)))

################## Housekeeping genes
#dat = dat[-which(dat[,1] %in% hkgenes),]
############################################
print(c('# gdsc2_genes (after HK) =', nrow(dat)))

gdsc2_genes = dat[,1]
print(c('# gdsc2_genes =', length(gdsc2_genes)))

apply(dat[, -1], 1, mean) -> rowMean
apply(dat[, -1], 1, var) -> rowVar
names(rowMean) = gdsc2_genes
names(rowVar) = gdsc2_genes

#####################################
which(rowVar < 0.1) -> idx
dat = dat[-idx, ]
gdsc2_genes = dat[,1]
print(c('# gdsc2_genes =', length(gdsc2_genes)))

apply(dat[, -1], 1, mean) -> rowMean
apply(dat[, -1], 1, var) -> rowVar
names(rowMean) = gdsc2_genes
names(rowVar) = gdsc2_genes

##############################################################################
#### INT
##############################################################################
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
	match(pathway_genes, dat[,1]) -> ii
	tmp2 = cor(t(dat[ii, -1]))
	which(apply(tmp2,1,function(u)sum(u>0.7))!=1) -> idx
	if(length(idx) > 0){
		#print( c(lines[1], paste(dat[ii[idx],1],collapse=",") ) )
		tmp_genes = dat[ii[idx],1]
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
	to_remove = c(to_remove, names(new_path_list)[info[idx, 2]])
	print(c(ptype, length(new_path_list), length(idx)))
}

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
	
	write.table(path_gmt, file=paste("data/GDSC2/", ptype, ".GDSC2.F124.gmt", sep=""), row.names=F, col.names=F, quote=F)
	print(c(ptype, length(path_gmt), length(new_genes)))
}

##############################################################################
#### generate INTC for GDSC2
##############################################################################

for(ptype in ptypes){
	grep(ptype, toupper(tags)) -> ii
	new_path_list = path_list[ii]
	new_genes = unique(unlist(new_path_list))
	mad.genes = intersect(new_genes, dat[,1])
	
	new_dat = dat[match(mad.genes, dat[,1]), -1]
	rownames(new_dat) = mad.genes
	
	##########################################################################
	# Rank-based inverse normal transformation
	##########################################################################
	qx.ccle.gene.mat = apply(new_dat, 2, INT )
	write.table(t(qx.ccle.gene.mat), file=paste("data/GDSC2/", ptype, ".INTC.GDSC2.tsv", sep=""), row.names=T, quote=F, sep="\t")
}

##########################################################
#### generate INTC for TCGA (step #1 for TCGA)
##########################################################

cancers = dir("data/raw/TCGA")
for(ptype in ptypes){
	X = read.table(paste("data/CCLE/", ptype, ".INTC.GDSC2.tsv", sep=""), header=T)
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
		write.table(t(qx.TCGA.RPKM.mat),        file=paste("data/TCGA/GDSC2/",cancer,".INTC.txt", sep=""), row.names=F, col.names=F, quote=F, sep="\t")
		write.table(colnames(qx.TCGA.RPKM.mat), file=paste("data/TCGA/GDSC2/",cancer,".INTC.rownames.txt",  sep=""), row.names=F, col.names=F, quote=F, sep="\t")
	}
	print(paste(ptype, ": #genes = ", length(kh.genes), sep=""))
}

##########################################################
#### for different # neighbors, generate edges
##########################################################

for(n_neighbors in c(3:20)){

	##########################################################
	#### generate interactions for GDSC2
	##########################################################
	
	ptype = "KEGG"
	X = read.table(paste("data/GDSC2/", ptype, ".INTC.GDSC2.tsv", sep=""), header=T)
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
	write.table(expr_edgelist, file=paste("data/GDSC2/", ptype,".E",n_neighbors,".GDSC2.cites",sep=""), row.names=F, quote=F, col.names=F, sep="\t")


	##########################################################
	#### generate interactions for TCGA (step #2 for TCGA)
	##########################################################
	CCLE_X1 = read.table(paste("/home/peilin/work/23-GNN-Drug/data/Pathways/", ptype, "/", ptype, ".INTC.GDSC2.tsv", sep=""), header=T)

	for(cancer in cancers){
		cat(paste(cancer, ".", sep=""))
		query_mat  = read.table(paste("/home/peilin/work/23-GNN-Drug/data/Pathways/",ptype, "/TCGA/GDSC2/",cancer,".INTC.txt", sep=""), header=F)
		query_name = read.table(paste("/home/peilin/work/23-GNN-Drug/data/Pathways/",ptype, "/TCGA/GDSC2/",cancer,".INTC.rownames.txt", sep=""), header=F)
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
		write.table(all_query_edgelist,  file=paste("data/TCGA/GDSC2/",cancer,".E",n_neighbors,".txt", sep=""), row.names=F, col.names=F, quote=F, sep="\t")
	}
	print(ptype)
} ### n_neighbors

