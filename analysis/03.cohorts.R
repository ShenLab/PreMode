library(ggplot2)
library(ggrepel)
library(ggpubr)
cohorts <- c("CHD", "ASD", "NDD")
good.pfams <- read.csv('../scripts/pfams.txt', header = F)$V1
good.pfams <- good.pfams[startsWith(good.pfams, "PF")]
good.pfams <- unique(as.character(as.data.frame(strsplit(good.pfams, '\\.'))[1,]))
good.pfams <- good.pfams[good.pfams != "PF_IPR000719"]
good.pfams <- c("IonChannel.split", good.pfams)
result.dir <- 'cohort/CHPs.v4.esm.small.StarAttn.MSA.StarPool/'
meta.logit.dir <- "/share/terra/Users/gz2294/PreMode.final/analysis/5genes.all.mut/CHPs.v4.esm.torchmdnet.small.TriAttn.StarPool.1dim/"

for (i in 1:length(good.pfams)) {
  pfam <- good.pfams[i]
  print(paste0("Begin ", pfam))
  logits <- NULL
  for (fold in 0:4) {
    pfam.data <- read.csv(paste0(result.dir, pfam, '.fold.', fold, '.csv'))
    pfam.cohort.data <- list()
    for (c in cohorts) {
      c.extTADA.result <- readRDS(paste0("/share/terra/Users/gz2294/PreMode.final/analysis/cohort/", c, ".result.CADD20.RDS"))$dataFDR
      c.pfam.data <- pfam.data[pfam.data$cohort==c,]
      c.pfam.data$log.bf.LGD <- log(c.extTADA.result$BFdn[match(c.pfam.data$GeneID, c.extTADA.result$Gene), 1])
      c.pfam.data$log.bf.mis <- log(c.extTADA.result$BFdn[match(c.pfam.data$GeneID, c.extTADA.result$Gene), 2])
      c.pfam.data$log.bf.ratio <-  c.pfam.data$log.bf.mis - c.pfam.data$log.bf.LGD
      c.pfam.data$extTADA.PP <- c.extTADA.result$PP[match(c.pfam.data$GeneID, c.extTADA.result$Gene)]
      c.pfam.data$HGNC <- c.extTADA.result$HGNC[match(c.pfam.data$GeneID, c.extTADA.result$Gene)]
      pfam.cohort.data[[c]] <- c.pfam.data
    }
    pfam.data <- rbind(pfam.cohort.data$NDD, pfam.cohort.data$ASD, pfam.cohort.data$CHD)
    if (is.null(logits)) {
      logits <- pfam.data$logits
    } else {
      logits <- cbind(logits, pfam.data$logits)
    }
  }
  pretrain.logits <- read.csv(paste0(result.dir, pfam, '.pretrain.csv'))
  pretrain.logits$unique.id <- paste0(pretrain.logits$uniprotID, ":", pretrain.logits$ref, pretrain.logits$pos.orig, pretrain.logits$alt)
  colnames(logits) <- paste0("model.", 0:4)
  pfam.data$ensembl.logits <- rowMeans(logits) 
  pfam.data$unique.id <- paste0(pfam.data$uniprotID, ":", pfam.data$ref, pfam.data$pos.orig, pfam.data$alt)
  pfam.data$pretrain.logits <- pretrain.logits$logits[match(pfam.data$unique.id, pretrain.logits$unique.id)]
  meta.model <- readRDS(paste0(meta.logit.dir, pfam, '.meta.RDS'))
  pfam.data$meta.logits <- predict(meta.model, logits, type = 'prob')[,2]
  write.csv(pfam.data, file = paste0('figs/', pfam, '.cohorts.csv'))
}

plot.list.1 <- list()
plot.list.2 <- list()
for (i in 1:length(good.pfams)) {
  pfam <- good.pfams[i]
  pfam.data <- read.csv(paste0('figs/', pfam, '.cohorts.csv'), row.names = 1)
  pfam.data.plot <- pfam.data[pfam.data$pretrain.logits >= 0.8 &
                                pfam.data$extTADA.PP >= 0.8 & 
                                # pfam.data$cohort %in% c('ASD', 'NDD') &
                                !is.na(pfam.data$extTADA.PP),]
  # pfam.data.plot <- pfam.data.plot[order(pfam.data.plot$meta.logits),]
  pfam.data.plot$LABEL <- pfam.data.plot$HGNC
  pfam.data.plot$LABEL[duplicated(paste0(pfam.data.plot$LABEL, pfam.data.plot$cohort))] <- NA
  if (pfam == "IonChannel.split") {
    pfam <- "IonChannel"
  }
  p1 <- ggplot(pfam.data.plot, aes(x=log.bf.ratio, y=meta.logits, col=cohort, label=LABEL)) +
    geom_point(alpha=0.5) + ylim(0, 1) +
    # stat_regline_equation(
    #   aes(label =  paste(..eq.label.., ..adj.rr.label.., sep = "~~~~")),
    #   formula = y~x,
    #   # position = "jitter",
    # ) + 
    ggrepel::geom_text_repel() +
    xlab('Enrichment Dmis/LGD') +
    ggtitle(pfam) + theme_bw() + ggeasy::easy_center_title() 
  if (dim(pfam.data.plot)[1] > 0) {
    pfam.data.plot$label <- 'Low enrichment'
    pfam.data.plot$label[pfam.data.plot$log.bf.ratio>=10] <- 'High enrichment'
    p2 <- ggplot(pfam.data.plot, aes(x=meta.logits, col=label)) +
      geom_density(alpha=0.5) +
      xlab('<--Loss   Logits   Gain-->') +
      ggtitle(pfam) + theme_bw() + ggeasy::easy_center_title() 
    ggsave(paste0('figs/', pfam, '.cohorts.pdf'), p1, height = 4, width = 4)
  }
  plot.list.1[[i]] <- p1
  plot.list.2[[i]] <- p2
}
library(patchwork)
p <- plot.list[[1]] + plot.list[[2]] + plot.list[[3]] + plot.list[[4]] + 
  plot.list[[5]] + plot.list[[9]] + plot_layout(ncol = 3)
ggsave(paste0('figs/', 'ALL.cohorts.pdf'), p, height = 8, width = 15)



