phenotype_data <- read.delim("/share/terra/Users/gz2294/ld1/Data/geneDx.anno.hg19/GDX_denovos_May_2019.phenotype.tsv") #for NDD
source('~/Pipeline/annotate.var.class.R')
phenotype_data <- annotate.Dmis(phenotype_data, 'REVEL', 0.5)
input.dir <- "/share/terra/Users/gz2294/PreMode.final/analysis/cohort/"
logit.dir <- "/share/terra/Users/gz2294/PreMode.final/analysis/cohort/CHPs.v4.esm.small.StarAttn.MSA.StarPool/"
meta.logit.dir <- "/share/terra/Users/gz2294/PreMode.final/analysis/5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim//"
pfam.data <- read.csv(paste0(input.dir, "PF00130.csv"))
pretrain <- read.csv(paste0(logit.dir, "PF00130.pretrain.csv"))
pfam <- "P15056"
logits <- NULL
for (fold in 0:4) {
  logit.data <- read.csv(paste0(logit.dir, pfam, '.fold.', fold, '.csv'))
  pfam.data[,paste0('model.', fold)] <- logit.data$logits
  if (is.null(logits)) {
    logits <- logit.data$logits
  } else {
    logits <- logits + logit.data$logits
  }
}
logits <- logits / 5
pfam.data$pretrain.logits <- pretrain$logits
# ensembl logits
pfam.data$ensembl.logits <- logits
# meta logits
library(caret)
meta.model <- readRDS(paste0(meta.logit.dir, pfam, '.meta.RDS'))
pfam.data$meta.logits <- predict(meta.model, as.matrix(pfam.data[,paste0('model.', 0:4)]), type = 'prob')[,2]

pfam.data <- pfam.data[pfam.data$pretrain.logits >= 0.8,]
pfam.data <- pfam.data[pfam.data$uniprotID==pfam,]
ggplot(pfam.data, aes(x=meta.logits, col=cohort)) + geom_density()

pfam.data <- pfam.data[pfam.data$cohort=="NDD",]
pfam.data <- pfam.data[!is.na(pfam.data$IID),]
pfam.data$HPO <- phenotype_data$HPO[match(pfam.data$IID, phenotype_data$IID)]
pfam.data$HPO_annotation <- phenotype_data$HPO_annotation[match(pfam.data$IID, phenotype_data$IID)]
pfam.data.other.variants <- phenotype_data[phenotype_data$IID %in% pfam.data$IID & !(phenotype_data$HGNC %in% "BRAF")&(phenotype_data$vclass %in% c("LGD", "Dmis")),]
pfam.data <- pfam.data[!pfam.data$IID %in% pfam.data.other.variants$IID,]
# add other gene HPO, ion channel genes in this case
control.gene <- c("CHD8", "SCN2A")
noonan.gene <- c("KRAS", "MAP2K1", "MRAS", "NRAS", 
                 "PTPN11", "RAF1", "RASA2", "RIT1", 
                 "RRAS2", "SOS1", "SOS2")
pfam.data.noonan <- phenotype_data[phenotype_data$HGNC%in%noonan.gene&(phenotype_data$vclass %in% c("LGD", "Dmis")),]
# make sure they don't carry other variants
pfam.data.noonan.other.variants <- phenotype_data[phenotype_data$IID %in% pfam.data.noonan$IID & !(phenotype_data$HGNC %in% noonan.gene)&(phenotype_data$vclass %in% c("LGD", "Dmis")),]
pfam.data.noonan <- pfam.data.noonan[!pfam.data.noonan$IID %in% pfam.data.noonan.other.variants$IID,]
pfam.data.noonan <- pfam.data.noonan[!duplicated(pfam.data.noonan$IID),]

pfam.data.control <- phenotype_data[phenotype_data$HGNC%in%control.gene&(phenotype_data$vclass %in% c("LGD", "Dmis")),]
# make sure they don't carry other variants
pfam.data.control.other.variants <- phenotype_data[phenotype_data$IID %in% pfam.data.control$IID & !(phenotype_data$HGNC %in% control.gene)&(phenotype_data$vclass %in% c("LGD", "Dmis")),]
pfam.data.control <- pfam.data.control[!pfam.data.control$IID %in% pfam.data.control.other.variants$IID,]
pfam.data.control <- pfam.data.control[!duplicated(pfam.data.control$IID),]

pfam.data$label <- "BRAF-carrier"
# pfam.data$label[pfam.data$meta.logits<=0.5] <- paste0("BRAF:", pfam.data$aaChg[pfam.data$meta.logits<=0.5], "-carrier")
tmp <- rbind(read.csv('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8/P15056/testing.csv'),
             read.csv('~/Data/DMS/Itan.CKB.Cancer.good.batch/pfams.0.8/P15056/training.csv'))
# pfam.data$label[!pfam.data$aaChg %in% tmp$aaChg] <- paste0("BRAF:", pfam.data$aaChg[!pfam.data$aaChg %in% tmp$aaChg], "-carrier")
pfam.data.noonan$label <- "noonan-carrier"
pfam.data.control$label <- "control-carrier"

source('~/Pipeline/bind_rows.R')
pfam.data.compare <- my.bind.rows(my.bind.rows(pfam.data, pfam.data.control), pfam.data.noonan)
pfam.data.compare <- pfam.data.compare[!is.na(pfam.data.compare$HPO),]

all.pheno <- strsplit(pfam.data.compare$HPO_annotation, split = " \\| ")
unique.pheno <- unique(unlist(all.pheno))

pheno.matrix <- matrix(0, nrow = dim(pfam.data.compare)[1], ncol = length(unique.pheno))
colnames(pheno.matrix) <- unique.pheno
rownames(pheno.matrix) <- pfam.data.compare$IID
for (r in 1:dim(pfam.data.compare)[1]) {
  pheno.matrix[r,all.pheno[[r]]] <- 1
}
pheno.matrix.anno <- data.frame(gene=pfam.data.compare$HGNC, 
                                meta.logits=pfam.data.compare$meta.logits,
                                label=pfam.data.compare$label,
                                aaChg=pfam.data.compare$aaChg,
                                row.names = pfam.data.compare$IID)
source('~/Pipeline/plot.genes.by.group.pca.R')
p1 <- plot.complex.pca(pheno.matrix[,colSums(pheno.matrix)>=3 & 
                                      (dim(pheno.matrix)[1]-colSums(pheno.matrix))>=3],
                       pheno.matrix.anno, 
                       font.size = 2,
                       text.group = 'aaChg',
                       color.group = 'meta.logits', i=1, plot.loading = T)

p2 <- plot.complex.pca(pheno.matrix[,colSums(pheno.matrix)>=3 &
                                      (dim(pheno.matrix)[1]-colSums(pheno.matrix))>=3],
                       pheno.matrix.anno, 
                       font.size = 2,
                       color.group = 'label', i=1, plot.loading = T)
p <- patchwork::wrap_elements(p1) + patchwork::wrap_elements(p2)
ggsave('figs/NDD.BRAF.phenotype.pca.pdf', p, height = 6, width = 12)

# source('~/Pipeline/cluster.leiden.R')
# umap.mat <- cluster.leiden(input = pheno.matrix[,colSums(pheno.matrix)>=3 &
#                                                   (dim(pheno.matrix)[1]-colSums(pheno.matrix))>=3],
#                            npc = 20)
# umap.mat$anno <- pheno.matrix.anno$label
# umap.mat$meta.logits <- pheno.matrix.anno$meta.logits
# p3 <- ggplot(umap.mat, aes(x=UMAP1, y=UMAP2, col=anno)) + geom_point() 
# p4 <- ggplot(umap.mat, aes(x=UMAP1, y=UMAP2, col=meta.logits)) + geom_point() + scale_color_gradient2(low = "blue", high="red", midpoint = 0.5)
noonan.terms <- read.csv('figs/noonan.txt', header = F)$V1
p

pheno.matrix <- pheno.matrix[,colSums(pheno.matrix)>=3 & 
                               (dim(pheno.matrix)[1]-colSums(pheno.matrix))>=3]
# calculate distance
pheno.matrix.dist <- as.matrix(proxy::dist(pheno.matrix, method = 'manhattan'))
rn <- paste0(pheno.matrix.anno$gene, ":", pheno.matrix.anno$aaChg)
rn[is.na(pheno.matrix.anno$aaChg)] <- pheno.matrix.anno$gene[is.na(pheno.matrix.anno$aaChg)]
colnames(pheno.matrix.dist) <- rn
rownames(pheno.matrix.dist) <- rn
# train a random forest to predict noonan
meta_model_fit <- train(pheno.matrix[pheno.matrix.anno$label!="BRAF-carrier",], 
                        as.factor(pheno.matrix.anno$label[pheno.matrix.anno$label!="BRAF-carrier"]))

BRAF.pheno <- predict(meta_model_fit, 
                      pheno.matrix[pheno.matrix.anno$label=="BRAF-carrier",], type = 'prob')[,2]

source('~/Pipeline/plot.genes.scores.heatmap.R')
pheno.matrix.anno$gene[is.na(pheno.matrix.anno$gene)] <- "BRAF"
plot.complex.heatmap(pheno.matrix.dist, 
                     col.anno = pheno.matrix.anno[,c("gene", "label")], 
                     col_font_size = 5,
                     row_font_size = 5,
                     # row.anno = pheno.matrix.anno[,c("gene", "label")],
                     # numeric.row.anno = data.frame(meta.logits=pheno.matrix.anno[,c("meta.logits")]),
                     numeric.col.anno = data.frame(meta.logits=pheno.matrix.anno[,c("meta.logits")]),
                     savefilename = 'figs/05.BRAF.phenotype.pdf',
                     height = 8,
                     legend_name = 'Dist',
                     cluster_rows = F,
                     cluster_columns = F,
                     width = 10)

rownames(pheno.matrix) <- pheno.matrix.anno$aaChg
split <- factor(pheno.matrix.anno$label)
plot.complex.heatmap(t(pheno.matrix), 
                     col.anno = pheno.matrix.anno[,c("gene", "label")], 
                     col_font_size = 5,
                     row_font_size = 5,
                     # row.anno = pheno.matrix.anno[,c("gene", "label")],
                     # numeric.row.anno = data.frame(meta.logits=pheno.matrix.anno[,c("meta.logits")]),
                     numeric.col.anno = data.frame(meta.logits=pheno.matrix.anno[,c("meta.logits")]),
                     savefilename = 'figs/05.BRAF.phenotype.heatmap.pdf',
                     legend_name = 'Phenotype',
                     col_split = split,
                     cluster_rows = F,
                     cluster_columns = T,
                     height = 18,
                     width = 12)


to.plot <- data.frame(meta.logits=pheno.matrix.anno$meta.logits[pheno.matrix.anno$label=="BRAF-carrier"],
                      Noonan.pheno=BRAF.pheno,
                      label=pfam.data.compare$aaChg[pheno.matrix.anno$label=="BRAF-carrier"])
ggplot(to.plot, aes(x=meta.logits, y=BRAF.pheno, label=label)) + 
  geom_point() + ggrepel::geom_text_repel() + theme_bw()
ggsave("figs/05.BRAF.phenotype.rf.pdf", width = 5, height = 4)

