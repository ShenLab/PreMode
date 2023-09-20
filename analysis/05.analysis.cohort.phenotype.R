library(dplyr)
library(tidyr)
library(patchwork)
#directory = /share/terra/Users/demiz/Cohort/
cohorts <- c("CHD", "ASD", "NDD")
# good.pfams <- read.csv('/share/terra/Users/demiz/Cohort/pfams.second.txt', header = F)$V1
good.pfams <- c("IonChannel.notest", "IonChannel.notest", "IonChannel.notest", "IonChannel.notest",
                "O00555", "Q99250", 
                "PF00130", 
                "P15056",
                "PF07679"
                # "Epi.SCN2A"
)
pretrain.file.name <- c(rep('IonChannel', 6),
                        rep('PF00130', 2),
                        "PF07679"
                        # "Epi.SCN2A"
)
gene.names <- c("O00555", "Q99250", "P35498", "Q9UQD0",
                "O00555", "Q99250", 
                "P15056", 
                "P15056",
                "P22607"
                # "Q99250"
)

input.dir <- "/share/terra/Users/gz2294/PreMode.final/analysis/cohort/"
logit.dir <- "/share/terra/Users/gz2294/PreMode.final/analysis/cohort/CHPs.v4.esm.small.StarAttn.MSA.StarPool/"
meta.logit.dir <- "/share/terra/Users/gz2294/PreMode.final/analysis/5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim//"
result.dir <- "/share/terra/Users/demiz/Cohort/NDD/"
plot.list <- list()
#/share/terra/Users/gz2294/PreMode.final/analysis/cohort/ASD.result.CADD20.RDS
extTADA.result <- readRDS("/share/terra/Users/gz2294/PreMode.final/analysis/cohort/NDD.result.CADD20.RDS")
extTADA.result <- extTADA.result$dataFDR
phenotype <- data.frame(matrix(ncol = 11, nrow = 0))
colnames(phenotype) <- c("IID", "gene", "variant", "pretrain.logits", "logit.0", "logit.1", "logit.2", "logit.3", "logit.4", "ensembl.logits", "phenotypes")
phenotype_data <- read.delim("/share/terra/Users/gz2294/ld1/Data/geneDx.anno.hg19/GDX_denovos_May_2019.phenotype.tsv") #for NDD
source('~/Pipeline/annotate.var.class.R')
source('~/Pipeline/bind_rows.R')
phenotype_data <- annotate.var.class(phenotype_data)
phenotype_data <- annotate.Dmis(phenotype_data, "REVEL", 0.5)
ALL <- read.csv('/share/terra/Users/gz2294/Data/DMS/Itan.CKB.Cancer/ALL.csv', row.names = 1)
ALL$unique.id <- paste(ALL$uniprotID, ALL$ref, ALL$pos.orig, ALL$alt, sep = ":")
ion.channel <- read.csv('/share/terra/Users/gz2294/Data/DMS/Ion_Channel/all.af2update.heyne.csv', row.names = 1)
ion.channel$unique.id <- paste(ion.channel$uniprotID, ion.channel$ref, ion.channel$pos.orig, ion.channel$alt, sep = ":")
to.del <- readRDS("/share/terra/Users/gz2294/Data/DMS/Itan.CKB.Cancer/to.del.conflict.with.chps.RDS")
ALL <- ALL[!ALL$unique.id %in% to.del,]
ALL <- ALL[!ALL$unique.id %in% ion.channel$unique.id,]
heyne <- dplyr::bind_rows(ALL, ion.channel)

for (i in 1:length(good.pfams)) {   
  pfam <- good.pfams[i]
  print(paste0("Begin ", pfam))
  logits <- NULL
  pfam.data <- read.csv(paste0(input.dir, pretrain.file.name[i], ".csv"))
  pfam.data$unique.id <- paste0(pfam.data$uniprotID, pfam.data$aaChg)
  pretrain <- read.csv(paste0(logit.dir, pretrain.file.name[i], ".pretrain.csv"))
  pretrain$unique.id <- paste0(pretrain$uniprotID, pretrain$aaChg)
  for (fold in 0:4) {
    logit.data <- read.csv(paste0(logit.dir, pfam, '.fold.', fold, '.csv'))
    logit.data$unique.id <- paste0(logit.data$uniprotID, logit.data$aaChg)
    # logit.data <- logit.data[logit.data$uniprotID==gene.names[i],]
    pfam.data[,paste0('model.', fold)] <- logit.data$logits[match(pfam.data$unique.id, logit.data$unique.id)]
    if (is.null(logits)) {
      logits <- logit.data$logits
    } else {
      logits <- logits + logit.data$logits
    }
  }
  logits <- logits / 5
  pfam.data$pretrain.logits <- pretrain$logits[match(pfam.data$unique.id, pretrain$unique.id)]
  # ensembl logits
  pfam.data$ensembl.logits <- logits[match(pfam.data$unique.id, logit.data$unique.id)]
  # meta logits
  library(caret)
  if (pfam == "Epi.SCN2A") {
    meta.model <- readRDS(paste0(meta.logit.dir, 'IonChannel.meta.RDS'))
  } else {
    meta.model <- readRDS(paste0(meta.logit.dir, pfam, '.meta.RDS'))
  }
  pfam.data$meta.logits <- predict(meta.model, as.matrix(pfam.data[,paste0('model.', 0:4)]), type = 'prob')[,2]
  
  pfam.data <- pfam.data[pfam.data$pretrain.logits >= 0.8,]
  ggplot(pfam.data, aes(x=meta.logits, col=cohort)) + geom_density()
  pfam.data <- pfam.data[pfam.data$cohort=="NDD",]
  # pfam.data <- pfam.data[pfam.data$extTADA.PP > 0.5,]
  if(nrow(pfam.data)>0){
    pfam.data$log.bf.ratio <- log(extTADA.result$BFdn[match(pfam.data$GeneID, extTADA.result$Gene), 2]) - log(extTADA.result$BFdn[match(pfam.data$GeneID, extTADA.result$Gene), 1])
    pfam.data$HGNC <- extTADA.result$HGNC[match(pfam.data$GeneID, extTADA.result$Gene)]
  }
  pfam.data <- pfam.data[!is.na(pfam.data$IID),]
  # pfam.data <- pfam.data[pfam.data$uniprotID==gene.names[i],]
  pfam.data$HPO <- phenotype_data$HPO[match(pfam.data$IID, phenotype_data$IID)]
  pfam.data$HPO_annotation <- phenotype_data$HPO_annotation[match(pfam.data$IID, phenotype_data$IID)]
  
  pfam.data.gene <- pfam.data[pfam.data$uniprotID==gene.names[i],]
  # add LGD variants from phenotype_data
  pfam.data.gene.LGD <- phenotype_data[phenotype_data$vclass=="LGD"&phenotype_data$HGNC==pfam.data.gene$HGNC[1],]
  if (dim(pfam.data.gene.LGD)[1] != 0) {
    pfam.data.gene.LGD$meta.logits <- 0
  }
  if (dim(pfam.data.gene)[1] != 0) {
    pfam.data.gene$vclass <- "mis"
  }
  pfam.data.gene <- my.bind.rows(pfam.data.gene, pfam.data.gene.LGD)
  pfam.data.gene <- pfam.data.gene[!is.na(pfam.data.gene$HPO),]
  # remove patients if they have other damaging de novo variants
  to.drop <- c()
  for (k in 1:dim(pfam.data.gene)) {
    if (sum(phenotype_data$IID==pfam.data.gene$IID[k]) != 1) {
      IID.all.variants <- phenotype_data$vclass[phenotype_data$IID==pfam.data.gene$IID[k] & 
                                                  !is.na(phenotype_data$HGNC) & 
                                                  phenotype_data$HGNC != pfam.data.gene$HGNC[k]]
      if ("LGD" %in% IID.all.variants | "Dmis" %in% IID.all.variants) {
        to.drop <- c(to.drop, k)
      }
    }
  }
  pfam.data.gene <- pfam.data.gene[-to.drop,]
  all.pheno <- strsplit(pfam.data.gene$HPO_annotation, split = " \\| ")
  unique.pheno <- unique(unlist(all.pheno))
  
  pheno.matrix <- matrix(0, nrow = dim(pfam.data.gene)[1], ncol = length(unique.pheno))
  colnames(pheno.matrix) <- unique.pheno
  rownames(pheno.matrix) <- pfam.data.gene$IID
  for (r in 1:dim(pfam.data.gene)[1]) {
    pheno.matrix[r,all.pheno[[r]]] <- 1
  }
  source('~/Pipeline/plot.genes.by.group.pca.R')
  heyne$unique.id <- paste0(heyne$uniprotID, ":", heyne$aaChg)
  pfam.data.gene$unique.id <- paste0(pfam.data.gene$uniprotID, ":", pfam.data.gene$aaChg)
  pfam.data.gene$in.heyne <- pfam.data.gene$uniprotID %in% heyne$uniprotID & pfam.data.gene$aaChg %in% heyne$aaChg
  pfam.data.gene$heyne.label <- heyne$score[match(pfam.data.gene$unique.id, heyne$unique.id)]
  pfam.data.gene$heyne.label[pfam.data.gene$heyne.label==1 & !is.na(pfam.data.gene$heyne.label)] <- "GoF"
  pfam.data.gene$heyne.label[pfam.data.gene$heyne.label==0 & !is.na(pfam.data.gene$heyne.label)] <- "LoF"
  # print(paste0(sum(pfam.data.gene$in.heyne), '/', dim(pfam.data.gene)[1]))
  pfam.data.gene$vclass[pfam.data.gene$in.heyne] <- pfam.data.gene$heyne.label[pfam.data.gene$in.heyne]
  pheno.matrix.anno <- data.frame(gene=pfam.data.gene$HGNC, 
                                  meta.logits=pfam.data.gene$meta.logits,
                                  ensembl.logits=pfam.data.gene$ensembl.logits,
                                  logits=pfam.data.gene$ensembl.logits,
                                  vclass=paste0(pfam.data.gene$HGNC, ":", pfam.data.gene$vclass),
                                  aaChg=pfam.data.gene$aaChg,
                                  row.names = pfam.data.gene$IID)
  pheno.matrix.anno$vclass[pfam.data.gene$vclass=="mis"] <- NA
  pheno.matrix.anno$logits.label <- NA
  pheno.matrix.anno$logits.label[pfam.data.gene$meta.logits > 0.5] <- "High"
  pheno.matrix.anno$logits.label[pfam.data.gene$meta.logits <= 0.5] <- "Low"
  
  corr.result <- data.frame(pheno=colnames(pheno.matrix))
  for (k in 1:dim(corr.result)[1]) {
    corr.result$pearson.r[k] <- cor.test(pfam.data.gene$meta.logits, pheno.matrix[,k])$estimate
    corr.result$spearman.r[k] <- cor.test(pfam.data.gene$meta.logits, pheno.matrix[,k], method = 'spearman')$estimate
    corr.result$auc[k] <- PRROC::roc.curve(scores.class0 = pfam.data.gene$meta.logits[pheno.matrix[,k]==0],
                                           scores.class1 = pfam.data.gene$meta.logits[pheno.matrix[,k]==1])$auc
    corr.result$fisher.p[k] <- fisher.test(table(pheno.matrix.anno$logits.label, pheno.matrix[,k]))$p.value
    corr.result$fisher.est[k] <- fisher.test(table(pheno.matrix.anno$logits.label, pheno.matrix[,k]))$estimate
    if (sum(pheno.matrix[,k]==1) >= 2 & sum(pheno.matrix[,k]==0) >= 2) {
      t.test.res <- t.test(pfam.data.gene$meta.logits[pheno.matrix[,k]==0],
                           pfam.data.gene$meta.logits[pheno.matrix[,k]==1])
      corr.result$t.statistic[k] <- t.test.res$statistic
      corr.result$t.p[k] <- t.test.res$p.value
    } else {
      corr.result$t.statistic[k] <- NA
      corr.result$t.p[k] <- NA
    }
  }
  corr.result$t.q <- p.adjust(corr.result$t.p, method = 'fdr')
  corr.result <- corr.result[colSums(pheno.matrix)>=2 & (dim(pheno.matrix)[1]-colSums(pheno.matrix))>=2,]
  corr.result <- corr.result[order(corr.result$spearman.r, decreasing = T),]
  p <- ggplot(corr.result, aes(x=factor(pheno, levels = corr.result$pheno), y=spearman.r)) + 
    geom_point() +
    # geom_abline(intercept = 0.75, slope = 0, col = 'blue', linetype="dotted") +
    # geom_abline(intercept = 0.25, slope = 0, col = 'blue', linetype="dotted") +
    geom_abline(intercept = 0, slope = 0, col = 'blue', linetype="dotted") +
    xlab('Phenotypes') + ylab('<--Loss Gain-->') + theme_bw() +
    coord_flip() 
  ggsave(plot = p, filename = paste0('figs/phenotype/05.', pfam, '.', gene.names[i], '.phenotype.pdf'), height = 10, width = 5)
  write.csv(corr.result, file = paste0('figs/phenotype/05.', pfam, '.', gene.names[i], '.phenotype.csv'))
  # for PCA, we should remove phenotypes with only 1 patients
  p1 <- plot.complex.pca(pheno.matrix[,colSums(pheno.matrix)>=3 & (dim(pheno.matrix)[1]-colSums(pheno.matrix))>=3],
                         pheno.matrix.anno, 
                         font.size = 1.5,
                         color.group = 'meta.logits',
                         text.group = 'aaChg',
                         i=1, plot.loading = T, )
  p2 <- plot.complex.pca(pheno.matrix[,colSums(pheno.matrix)>=3 & (dim(pheno.matrix)[1]-colSums(pheno.matrix))>=3],
                         pheno.matrix.anno, 
                         font.size = 1.5,
                         color.group = 'vclass', i=1, plot.loading = T, )
  p <- patchwork::wrap_elements(p1) + patchwork::wrap_elements(p2)
  # visualize all seizures, epelip*
  to.visualize <- corr.result$pheno
  to.visualize <- to.visualize[colSums(pheno.matrix[,to.visualize])>=3 &
                                 (dim(pheno.matrix)[1]-colSums(pheno.matrix[,to.visualize]))>=3]
  p.density <- list()
  for (k in 1:length(to.visualize)) {
    pfam.data.gene$visualize <- as.factor(pheno.matrix[,to.visualize[k]])
    p.dens <- ggplot(pfam.data.gene, aes(x=meta.logits, col=visualize)) +
      geom_density() + ggtitle(paste0(to.visualize[k], '\n rho=', round(corr.result$spearman.r[corr.result$pheno==to.visualize[k]], 2))) +
      labs(col=paste0('pheno: ', 
                      sum(pheno.matrix[,to.visualize[k]]==0), 
                      '/', sum(pheno.matrix[,to.visualize[k]]==1))) + 
      theme_bw() + ggeasy::easy_center_title()
    p.density[[k]] <- p.dens
  }
  p.all <- patchwork::wrap_plots(p.density, ncol = 8)
  ggsave(paste0('figs/phenotype/05.', pfam, '.', gene.names[i], '.phenotype.density.pdf'), p.all, 
         height = 2*ceiling(length(to.visualize)/8), width = 24)
  ggsave(paste0('figs/phenotype/05.', pfam, '.', gene.names[i], '.phenotype.pca.pdf'), p, height = 5, width = 10)
}   #end of going through 1 Pfam
