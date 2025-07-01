# parse dnv table to HGNC, columns:
# Reauired: VarID, Score, HGNC, aaChg
# Optional: offset
source('/share/pascal/Users/gz2294/Pipeline/parse.variant.wt.sequence.R')

dnv.table.seperate.multiple.effects <- function(
    dnv.table, njobs=42, split.chr=";", 
    cols.to.split=c("TransIDs", "TransEffs", "AAChg", "Symbol", "GeneID", "HGNC", "GeneEff"),
    carno.only=TRUE) {
  # expect to receive standard output from VEP
  duplicated.effects.idx <- grep(split.chr, dnv.table$AAChg)
  if (length(duplicated.effects.idx)!=0) {
    dnv.table.nodup <- dnv.table[-duplicated.effects.idx,]
    dnv.table.dup <- dnv.table[duplicated.effects.idx,]
    library(doParallel)
    cl <- makeCluster(njobs)
    registerDoParallel(cl)
    cols.to.split <- cols.to.split[cols.to.split %in% colnames(dnv.table.dup)]
    dnv.table.dup.expand <- foreach (i = 1:dim(dnv.table.dup)[1], .combine = rbind, .multicombine=TRUE) %dopar% {
      row.number <- strsplit(dnv.table.dup[i, cols.to.split[1]], split = split.chr)[[1]]
      tmp <- dnv.table.dup[rep(i, length(row.number)),]
      for (k in cols.to.split) {
        split.res <- strsplit(dnv.table.dup[i, k], split = split.chr)[[1]][1:nrow(tmp)]
        tmp[,k] <- split.res
      }
      if (carno.only) {
        tmp <- tmp[match(dnv.table.dup$TransCanon[i], tmp$TransIDs),]
      }
      tmp
    }
    stopCluster(cl)
    dnv.table <- rbind(dnv.table.nodup, dnv.table.dup.expand)
  }
  dnv.table
}


dnv.table.to.uniprot.by.HGNC <- function(
    dnv.table, VarID.column, Score.column,
    HGNC.column, aaChg.column, offset.column=NA) {
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  result <- data.frame(VarID = dnv.table[,VarID.column],
                       aaChg = dnv.table[,aaChg.column],
                       uniprotID = NA,
                       ref = NA,
                       pos = NA,
                       alt = NA,
                       score = dnv.table[,Score.column],
                       sequence = NA,
                       wt = NA,
                       sequence.len = NA,
                       seq.start = NA,
                       seq.end = NA,
                       pos.orig = NA,
                       sequence.orig = NA,
                       wt.orig = NA,
                       sequence.len.orig = NA,
                       HGNC = dnv.table[,HGNC.column])
  unique_genes <- as.character(unique(result$HGNC))
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/swissprot.ID.mapping.tsv')
  uniprot_ID <- vector("list", length(unique_genes))
  uniprot_wt.sequence <- vector("list", length(unique_genes))
  for (i in 1:length(unique_genes)) {
    uniprot_ID[[i]] <- uniprot_ID.mapping$Entry[grep(paste0('\\b', unique_genes[i], '\\b'),
                                                     uniprot_ID.mapping$Gene.Names)]
    uniprot_wt.sequence[[i]] <- uniprot_ID.mapping$Sequence[grep(paste0('\\b', unique_genes[i], '\\b'),
                                                                 uniprot_ID.mapping$Gene.Names)]
    if (length(uniprot_ID[[i]]>1)) {
      print(unique_genes[i])
    }
  }
  # next match unreviewed IDs
  empty_ids <- c()
  for (i in 1:length(uniprot_ID)) {
    if (length(uniprot_ID[[i]])==0) {
      empty_ids <- c(empty_ids, i)
    }
  }
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/uniprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  for (id in empty_ids) {
    uniprot_ID[[id]] <- uniprot_ID.mapping$Entry[grep(paste0('\\b', unique_genes[id], '\\b'),
                                                      uniprot_ID.mapping$Gene.Names)]
    uniprot_wt.sequence[[id]] <- uniprot_ID.mapping$Sequence[grep(paste0('\\b', unique_genes[id], '\\b'),
                                                                  uniprot_ID.mapping$Gene.Names)]
  }
  for (i in 1:dim(result)[1]) {
    aaChg <- result$aaChg[i]
    uniprot_IDs <- uniprot_ID[[match(result$HGNC[i], unique_genes)]]
    wt.sequences <- uniprot_wt.sequence[[match(result$HGNC[i], unique_genes)]]
    j = 1
    while (is.na(result$sequence[i]) & j <= length(uniprot_IDs)) {
      result$uniprotID[i] <- uniprot_IDs[j]
      wt.sequence <- wt.sequences[j]
      # parse variant
      substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset)
      result$ref[i] <- substitute_res$ref
      result$pos[i] <- substitute_res$pos
      result$alt[i] <- substitute_res$alt
      result$sequence[i] <- substitute_res$sequence
      result$wt[i] <- substitute_res$wt
      result$sequence.len[i] <- substitute_res$sequence.len
      result$seq.start[i] = substitute_res$seq.start
      result$seq.end[i] = substitute_res$seq.end
      result$pos.orig[i] = substitute_res$pos.orig
      result$sequence.orig[i] = substitute_res$sequence.orig
      result$wt.orig[i] = substitute_res$wt.orig
      result$sequence.len.orig[i] = substitute_res$sequence.len.orig
      j = j + 1
    }
  }
  result <- dplyr::bind_cols(dnv.table[,!colnames(dnv.table) %in% colnames(result)], result)
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}


dnv.table.to.uniprot.by.uniprotID <- function(
    dnv.table, VarID.column, Score.column,
    uniprotID.column, aaChg.column, offset.column=NA) {
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  result <- data.frame(VarID = dnv.table[,VarID.column],
                       aaChg = dnv.table[,aaChg.column],
                       uniprotID = dnv.table[,uniprotID.column],
                       ref = NA,
                       pos = NA,
                       alt = NA,
                       score = dnv.table[,Score.column],
                       sequence = NA,
                       wt = NA,
                       sequence.len = NA,
                       seq.start = NA,
                       seq.end = NA,
                       pos.orig = NA,
                       sequence.orig = NA,
                       wt.orig = NA,
                       sequence.len.orig = NA,
                       HGNC = NA)
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/uniprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  for (i in 1:dim(result)[1]) {
    aaChg <- result$aaChg[i]
    uniprot_IDs <- result$uniprotID[i]
    if (grepl("-", uniprot_IDs)) {
      url.request <- paste0('https://rest.uniprot.org/uniprotkb/', uniprot_IDs, '.fasta')
      txt <- strsplit(RCurl::getURL(url.request), split = '\n')[[1]]
      wt.sequences <- paste(txt[2:length(txt)], collapse = '')
    } else {
      wt.sequences <- uniprot_ID.mapping$Sequence[match(uniprot_IDs, uniprot_ID.mapping$Entry)]
    }
    j = 1
    while (is.na(result$sequence[i]) & j <= length(uniprot_IDs)) {
      wt.sequence <- wt.sequences[j]
      # parse variant
      substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset[i])
      result$ref[i] <- substitute_res$ref
      result$pos[i] <- substitute_res$pos
      result$alt[i] <- substitute_res$alt
      result$sequence[i] <- substitute_res$sequence
      result$wt[i] <- substitute_res$wt
      result$sequence.len[i] <- substitute_res$sequence.len
      result$seq.start[i] = substitute_res$seq.start
      result$seq.end[i] = substitute_res$seq.end
      result$pos.orig[i] = substitute_res$pos.orig
      result$sequence.orig[i] = substitute_res$sequence.orig
      result$wt.orig[i] = substitute_res$wt.orig
      result$sequence.len.orig[i] = substitute_res$sequence.len.orig
      j = j + 1
    }
  }
  result <- dplyr::bind_cols(dnv.table[,!colnames(dnv.table) %in% colnames(result)], result)
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}


dnv.table.to.uniprot.by.HGNC.parallel <- function(
    dnv.table, VarID.column, Score.column,
    HGNC.column, aaChg.column, offset.column=NA, njobs=42) {
  library(doParallel)
  cl <- makeCluster(njobs)
  registerDoParallel(cl)
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  prompt <- data.frame(VarID = dnv.table[,VarID.column],
                       aaChg = dnv.table[,aaChg.column],
                       score = dnv.table[,Score.column],
                       HGNC = dnv.table[,HGNC.column])
  unique_genes <- as.character(unique(prompt$HGNC))
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/swissprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  # first match reviewed IDs
  uniprot_ID <- foreach (i = 1:length(unique_genes)) %dopar% {
    tmp <- uniprot_ID.mapping$Entry[grep(paste0('\\b', unique_genes[i], '\\b'),
                                         uniprot_ID.mapping$Gene.Names)]
    if (length(tmp>1)) {
      print(unique_genes[i])
    }
    tmp
  }
  uniprot_wt.sequence <- foreach (i = 1:length(unique_genes)) %dopar% {
    tmp <- uniprot_ID.mapping$Sequence[grep(paste0('\\b', unique_genes[i], '\\b'),
                                            uniprot_ID.mapping$Gene.Names)]
    tmp
  }
  # next match unreviewed IDs
  empty_ids <- c()
  for (i in 1:length(uniprot_ID)) {
    if (length(uniprot_ID[[i]])==0) {
      empty_ids <- c(empty_ids, i)
    }
  }
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/uniprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  for (id in empty_ids) {
    uniprot_ID[[id]] <- uniprot_ID.mapping$Entry[grep(paste0('\\b', unique_genes[id], '\\b'),
                                                      uniprot_ID.mapping$Gene.Names)]
    uniprot_wt.sequence[[id]] <- uniprot_ID.mapping$Sequence[grep(paste0('\\b', unique_genes[id], '\\b'),
                                                                  uniprot_ID.mapping$Gene.Names)]
  }
  result <- foreach (i = 1:dim(prompt)[1], .combine = rbind, .multicombine=TRUE) %dopar% {
    source('/share/pascal/Users/gz2294/Pipeline/parse.variant.wt.sequence.R')
    substitute_res <- list(ref=NA, pos=NA, alt=NA, wt = NA,
                           sequence = NA, sequence.len = NA,
                           seq.start = NA, seq.end = NA,
                           pos.orig = NA, sequence.orig = NA,
                           wt.orig = NA, sequence.len.orig = NA)
    aaChg <- prompt$aaChg[i]
    uniprot_IDs <- uniprot_ID[[match(prompt$HGNC[i], unique_genes)]]
    wt.sequences <- uniprot_wt.sequence[[match(prompt$HGNC[i], unique_genes)]]
    j = 1
    while (is.na(substitute_res$sequence) & j <= length(uniprot_IDs)) {
      matched_uniprot_ID <- uniprot_IDs[j]
      wt.sequence <- wt.sequences[j]
      # parse variant
      substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset[i])
      j = j + 1
    }
    tmp <- data.frame(VarID = prompt$VarID[i],
                      aaChg = prompt$aaChg[i],
                      uniprotID = matched_uniprot_ID,
                      ref = substitute_res$ref,
                      pos = substitute_res$pos,
                      alt = substitute_res$alt,
                      score = prompt$score[i],
                      sequence = substitute_res$sequence,
                      wt = substitute_res$wt,
                      sequence.len = substitute_res$sequence.len,
                      seq.start = substitute_res$seq.start,
                      seq.end = substitute_res$seq.end,
                      pos.orig = substitute_res$pos.orig,
                      sequence.orig = substitute_res$sequence.orig,
                      wt.orig = substitute_res$wt.orig,
                      sequence.len.orig = substitute_res$sequence.len.orig,
                      HGNC = prompt$HGNC[i])
    tmp
  }
  stopCluster(cl)
  result <- dplyr::bind_cols(dnv.table[,!colnames(dnv.table) %in% colnames(result)], result)
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}


dnv.table.to.uniprot.by.HGNC.af2.parallel <- function(
    dnv.table, VarID.column, Score.column,
    HGNC.column, aaChg.column, offset.column=NA, njobs=42) {
  library(doParallel)
  cl <- makeCluster(njobs)
  registerDoParallel(cl)
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  prompt <- data.frame(VarID = dnv.table[,VarID.column],
                       aaChg = dnv.table[,aaChg.column],
                       score = dnv.table[,Score.column],
                       HGNC = dnv.table[,HGNC.column])
  unique_genes <- as.character(unique(prompt$HGNC))
  af2.mapping <- read.csv('/share/pascal/Users/gz2294/Data/af2_uniprot/swissprot_and_human.csv', row.names = 1)
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/swissprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  # first match reviewed IDs
  uniprot_ID <- foreach (i = 1:length(unique_genes)) %dopar% {
    tmp <- af2.mapping$uniprotID[match(unique_genes[i], af2.mapping$HGNC)]
    if (is.na(tmp)) {
      tmp <- uniprot_ID.mapping$Entry[grep(paste0('\\b', unique_genes[i], '\\b'),
                                           uniprot_ID.mapping$Gene.Names)]
      if (length(tmp>1)) {
        print(unique_genes[i])
      }
    }
    tmp
  }
  uniprot_wt.sequence <- foreach (i = 1:length(unique_genes)) %dopar% {
    tmp <- af2.mapping$seq[match(unique_genes[i], af2.mapping$HGNC)]
    if (is.na(tmp)) {
      tmp <- uniprot_ID.mapping$Sequence[grep(paste0('\\b', unique_genes[i], '\\b'),
                                              uniprot_ID.mapping$Gene.Names)]
    }
    tmp
  }
  # next match unreviewed IDs
  empty_ids <- c()
  for (i in 1:length(uniprot_ID)) {
    if (length(uniprot_ID[[i]])==0) {
      empty_ids <- c(empty_ids, i)
    }
  }
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/uniprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  for (id in empty_ids) {
    uniprot_ID[[id]] <- uniprot_ID.mapping$Entry[grep(paste0('\\b', unique_genes[id], '\\b'),
                                                      uniprot_ID.mapping$Gene.Names)]
    uniprot_wt.sequence[[id]] <- uniprot_ID.mapping$Sequence[grep(paste0('\\b', unique_genes[id], '\\b'),
                                                                  uniprot_ID.mapping$Gene.Names)]
  }
  result <- foreach (i = 1:dim(prompt)[1], .combine = rbind, .multicombine=TRUE) %dopar% {
    source('/share/pascal/Users/gz2294/Pipeline/parse.variant.wt.sequence.R')
    substitute_res <- list(ref=NA, pos=NA, alt=NA, wt = NA,
                           sequence = NA, sequence.len = NA,
                           seq.start = NA, seq.end = NA,
                           pos.orig = NA, sequence.orig = NA,
                           wt.orig = NA, sequence.len.orig = NA)
    aaChg <- prompt$aaChg[i]
    uniprot_IDs <- uniprot_ID[[match(prompt$HGNC[i], unique_genes)]]
    wt.sequences <- uniprot_wt.sequence[[match(prompt$HGNC[i], unique_genes)]]
    j = 1
    while (is.na(substitute_res$sequence) & j <= length(uniprot_IDs)) {
      matched_uniprot_ID <- uniprot_IDs[j]
      wt.sequence <- wt.sequences[j]
      # parse variant
      substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset[i])
      j = j + 1
    }
    tmp <- data.frame(VarID = prompt$VarID[i],
                      aaChg = prompt$aaChg[i],
                      uniprotID = matched_uniprot_ID,
                      ref = substitute_res$ref,
                      pos = substitute_res$pos,
                      alt = substitute_res$alt,
                      score = prompt$score[i],
                      sequence = substitute_res$sequence,
                      wt = substitute_res$wt,
                      sequence.len = substitute_res$sequence.len,
                      seq.start = substitute_res$seq.start,
                      seq.end = substitute_res$seq.end,
                      pos.orig = substitute_res$pos.orig,
                      sequence.orig = substitute_res$sequence.orig,
                      wt.orig = substitute_res$wt.orig,
                      sequence.len.orig = substitute_res$sequence.len.orig,
                      HGNC = prompt$HGNC[i])
    tmp
  }
  stopCluster(cl)
  result <- dplyr::bind_cols(dnv.table[,!colnames(dnv.table) %in% colnames(result)], result)
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}



dnv.table.to.uniprot.by.uniprotID.parallel <- function(
    dnv.table, VarID.column, Score.column,
    uniprotID.column, aaChg.column, offset.column=NA, length.column=NA, 
    match.length=FALSE,  njobs=42) {
  library(doParallel)
  cl <- makeCluster(njobs)
  registerDoParallel(cl)
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  if (match.length) {
    prompt <- data.frame(VarID = dnv.table[,VarID.column],
                         aaChg = dnv.table[,aaChg.column],
                         score = dnv.table[,Score.column],
                         uniprotID = dnv.table[,uniprotID.column],
                         seq.match.length = dnv.table[,length.column])
  } else {
    prompt <- data.frame(VarID = dnv.table[,VarID.column],
                         aaChg = dnv.table[,aaChg.column],
                         score = dnv.table[,Score.column],
                         uniprotID = dnv.table[,uniprotID.column])
  }
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/uniprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  result <- foreach (i = 1:dim(prompt)[1], .combine = rbind, .multicombine=TRUE) %dopar% {
    source('/share/pascal/Users/gz2294/Pipeline/parse.variant.wt.sequence.R')
    substitute_res <- list(ref=NA, pos=NA, alt=NA, wt = NA,
                           sequence = NA, sequence.len = NA,
                           seq.start = NA, seq.end = NA,
                           pos.orig = NA, sequence.orig = NA,
                           wt.orig = NA, sequence.len.orig = NA)
    aaChg <- prompt$aaChg[i]
    uniprot_IDs <- prompt$uniprotID[i]
    if (grepl("-", uniprot_IDs)) {
      url.request <- paste0('https://rest.uniprot.org/uniprotkb/', uniprot_IDs, '.fasta')
      txt <- strsplit(RCurl::getURL(url.request), split = '\n')[[1]]
      wt.sequences <- paste(txt[2:length(txt)], collapse = '')
    } else {
      wt.sequences <- uniprot_ID.mapping$Sequence[match(uniprot_IDs, uniprot_ID.mapping$Entry)]
    }
    if (match.length) {
      if (!is.na(wt.sequences) & nchar(wt.sequences) != prompt$seq.match.length[i]) {
        wt.sequences <- NA
      }
    }
    j = 1
    while (is.na(substitute_res$sequence) & j <= length(uniprot_IDs)) {
      matched_uniprot_ID <- uniprot_IDs[j]
      wt.sequence <- wt.sequences[j]
      # parse variant
      substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset[i])
      j = j + 1
    }
    tmp <- data.frame(VarID = prompt$VarID[i],
                      aaChg = prompt$aaChg[i],
                      uniprotID = matched_uniprot_ID,
                      ref = substitute_res$ref,
                      pos = substitute_res$pos,
                      alt = substitute_res$alt,
                      score = prompt$score[i],
                      sequence = substitute_res$sequence,
                      wt = substitute_res$wt,
                      sequence.len = substitute_res$sequence.len,
                      seq.start = substitute_res$seq.start,
                      seq.end = substitute_res$seq.end,
                      pos.orig = substitute_res$pos.orig,
                      sequence.orig = substitute_res$sequence.orig,
                      wt.orig = substitute_res$wt.orig,
                      sequence.len.orig = substitute_res$sequence.len.orig,
                      HGNC = NA)
    tmp
  }
  stopCluster(cl)
  result <- dplyr::bind_cols(dnv.table[,!colnames(dnv.table) %in% colnames(result)], result)
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}


dnv.table.to.uniprot.by.af2.uniprotID.parallel <- function(
    dnv.table, VarID.column, Score.column,
    uniprotID.column, aaChg.column, offset.column=NA, length.column=NA, 
    match.length=FALSE, njobs=96) {
  library(doParallel)
  cl <- makeCluster(njobs)
  registerDoParallel(cl)
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  if (match.length) {
    prompt <- data.frame(VarID = dnv.table[,VarID.column],
                         aaChg = dnv.table[,aaChg.column],
                         score = dnv.table[,Score.column],
                         uniprotID = dnv.table[,uniprotID.column],
                         seq.match.length = dnv.table[,length.column])
  } else {
    prompt <- data.frame(VarID = dnv.table[,VarID.column],
                         aaChg = dnv.table[,aaChg.column],
                         score = dnv.table[,Score.column],
                         uniprotID = dnv.table[,uniprotID.column])
  }
  uniprot_ID.mapping <- read.csv('/share/pascal/Users/gz2294/Data/af2_uniprot/swissprot_and_human.csv')
  # order to make sure first use longest transcript
  # uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  result <- foreach (i = 1:dim(prompt)[1], .combine = rbind, .multicombine=TRUE) %dopar% {
    source('/share/pascal/Users/gz2294/Pipeline/parse.variant.wt.sequence.R')
    substitute_res <- list(ref=NA, pos=NA, alt=NA, wt = NA,
                           sequence = NA, sequence.len = NA,
                           seq.start = NA, seq.end = NA,
                           pos.orig = NA, sequence.orig = NA,
                           wt.orig = NA, sequence.len.orig = NA)
    aaChg <- prompt$aaChg[i]
    uniprot_IDs <- prompt$uniprotID[i]
    if (grepl("-", uniprot_IDs) | !uniprot_IDs %in% uniprot_ID.mapping$uniprotID) {
      url.request <- paste0('https://rest.uniprot.org/uniprotkb/', uniprot_IDs, '.fasta')
      txt <- strsplit(RCurl::getURL(url.request), split = '\n')[[1]]
      wt.sequences <- paste(txt[2:length(txt)], collapse = '')
    } else {
      wt.sequences <- uniprot_ID.mapping$seq[match(uniprot_IDs, uniprot_ID.mapping$uniprotID)]
    }
    if (match.length) {
      if (!is.na(wt.sequences) & nchar(wt.sequences) != prompt$seq.match.length[i]) {
        wt.sequences <- NA
      }
    }
    j = 1
    while (is.na(substitute_res$sequence) & j <= length(uniprot_IDs)) {
      matched_uniprot_ID <- uniprot_IDs[j]
      wt.sequence <- wt.sequences[j]
      # parse variant
      substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset[i])
      j = j + 1
    }
    tmp <- data.frame(VarID = prompt$VarID[i],
                      aaChg = prompt$aaChg[i],
                      uniprotID = matched_uniprot_ID,
                      ref = substitute_res$ref,
                      pos = substitute_res$pos,
                      alt = substitute_res$alt,
                      score = prompt$score[i],
                      sequence = substitute_res$sequence,
                      wt = substitute_res$wt,
                      sequence.len = substitute_res$sequence.len,
                      seq.start = substitute_res$seq.start,
                      seq.end = substitute_res$seq.end,
                      pos.orig = substitute_res$pos.orig,
                      sequence.orig = substitute_res$sequence.orig,
                      wt.orig = substitute_res$wt.orig,
                      sequence.len.orig = substitute_res$sequence.len.orig)
    tmp
  }
  stopCluster(cl)
  result <- dplyr::bind_cols(dnv.table[,!colnames(dnv.table) %in% colnames(result)], result)
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}


dnv.table.to.uniprot.by.ensembl_geneID.parallel <- function(
    dnv.table, VarID.column, Score.column,
    ensembl.column, aaChg.column, offset.column=NA, njobs=42) {
  library(doParallel)
  cl <- makeCluster(njobs)
  registerDoParallel(cl)
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  prompt <- data.frame(VarID = dnv.table[,VarID.column],
                       aaChg = dnv.table[,aaChg.column],
                       score = dnv.table[,Score.column],
                       ensembl = dnv.table[,ensembl.column])
  unique_genes <- as.character(unique(prompt$ensembl))
  ensembl.mapping <- read.csv('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/ensembl.uniprot.ID.mapping.csv')
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/uniprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  # first match reviewed IDs
  uniprot_ID <- foreach (i = 1:length(unique_genes)) %dopar% {
    tmp <- ensembl.mapping$uniprot_gn_id[grep(paste0('\\b', unique_genes[i], '\\b'),
                                              ensembl.mapping$ensembl_gene_id)]
    if (length(tmp>1)) {
      print(unique_genes[i])
    }
    tmp
  }
  uniprot_wt.sequence <- foreach (i = 1:length(unique_genes)) %dopar% {
    tmp <- uniprot_ID.mapping$Sequence[grep(paste0('\\b', uniprot_ID[[i]], '\\b'),
                                            uniprot_ID.mapping$Gene.Names)]
    tmp
  }
  # next match unreviewed IDs
  empty_ids <- c()
  for (i in 1:length(uniprot_ID)) {
    if (length(uniprot_ID[[i]])==0) {
      empty_ids <- c(empty_ids, i)
    }
  }
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/uniprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  for (id in empty_ids) {
    uniprot_ID[[id]] <- uniprot_ID.mapping$Entry[grep(paste0('\\b', unique_genes[id], '\\b'),
                                                      uniprot_ID.mapping$Gene.Names)]
    uniprot_wt.sequence[[id]] <- uniprot_ID.mapping$Sequence[grep(paste0('\\b', unique_genes[id], '\\b'),
                                                                  uniprot_ID.mapping$Gene.Names)]
  }
  result <- foreach (i = 1:dim(prompt)[1], .combine = rbind, .multicombine=TRUE) %dopar% {
    source('/share/pascal/Users/gz2294/Pipeline/parse.variant.wt.sequence.R')
    substitute_res <- list(ref=NA, pos=NA, alt=NA, wt = NA,
                           sequence = NA, sequence.len = NA,
                           seq.start = NA, seq.end = NA,
                           pos.orig = NA, sequence.orig = NA,
                           wt.orig = NA, sequence.len.orig = NA)
    aaChg <- prompt$aaChg[i]
    uniprot_IDs <- uniprot_ID[[match(prompt$ensembl[i], unique_genes)]]
    wt.sequences <- uniprot_wt.sequence[[match(prompt$ensembl[i], unique_genes)]]
    j = 1
    while (is.na(substitute_res$sequence) & j <= length(uniprot_IDs)) {
      matched_uniprot_ID <- uniprot_IDs[j]
      wt.sequence <- wt.sequences[j]
      # parse variant
      substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset[i])
      j = j + 1
    }
    tmp <- data.frame(VarID = prompt$VarID[i],
                      aaChg = prompt$aaChg[i],
                      uniprotID = matched_uniprot_ID,
                      ref = substitute_res$ref,
                      pos = substitute_res$pos,
                      alt = substitute_res$alt,
                      score = prompt$score[i],
                      sequence = substitute_res$sequence,
                      wt = substitute_res$wt,
                      sequence.len = substitute_res$sequence.len,
                      seq.start = substitute_res$seq.start,
                      seq.end = substitute_res$seq.end,
                      pos.orig = substitute_res$pos.orig,
                      sequence.orig = substitute_res$sequence.orig,
                      wt.orig = substitute_res$wt.orig,
                      sequence.len.orig = substitute_res$sequence.len.orig,
                      ensembl = prompt$ensembl[i])
    tmp
  }
  stopCluster(cl)
  result <- dplyr::bind_cols(dnv.table[,!colnames(dnv.table) %in% colnames(result)], result)
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}


dnv.table.to.uniprot.by.ensembl_transcriptID.parallel <- function(
    dnv.table, VarID.column, Score.column,
    ensembl.column, aaChg.column, offset.column=NA, njobs=42) {
  library(doParallel)
  cl <- makeCluster(njobs)
  registerDoParallel(cl)
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  prompt <- data.frame(VarID = dnv.table[,VarID.column],
                       aaChg = dnv.table[,aaChg.column],
                       score = dnv.table[,Score.column],
                       ensembl = dnv.table[,ensembl.column])
  unique_genes <- as.character(unique(prompt$ensembl))
  ensembl.mapping <- read.csv('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/ensembl.uniprot.ID.mapping.csv')
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/uniprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  # first match reviewed IDs
  uniprot_ID <- foreach (i = 1:length(unique_genes)) %dopar% {
    tmp <- ensembl.mapping$uniprot_gn_id[grep(paste0('\\b', unique_genes[i], '\\b'),
                                              ensembl.mapping$ensembl_gene_id)]
    if (length(tmp>1)) {
      print(unique_genes[i])
    }
    tmp
  }
  uniprot_wt.sequence <- foreach (i = 1:length(unique_genes)) %dopar% {
    tmp <- uniprot_ID.mapping$Sequence[grep(paste0('\\b', uniprot_ID[[i]], '\\b'),
                                            uniprot_ID.mapping$Gene.Names)]
    tmp
  }
  # next match unreviewed IDs
  empty_ids <- c()
  for (i in 1:length(uniprot_ID)) {
    if (length(uniprot_ID[[i]])==0) {
      empty_ids <- c(empty_ids, i)
    }
  }
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/uniprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  for (id in empty_ids) {
    uniprot_ID[[id]] <- uniprot_ID.mapping$Entry[grep(paste0('\\b', unique_genes[id], '\\b'),
                                                      uniprot_ID.mapping$Gene.Names)]
    uniprot_wt.sequence[[id]] <- uniprot_ID.mapping$Sequence[grep(paste0('\\b', unique_genes[id], '\\b'),
                                                                  uniprot_ID.mapping$Gene.Names)]
  }
  result <- foreach (i = 1:dim(prompt)[1], .combine = rbind, .multicombine=TRUE) %dopar% {
    source('/share/pascal/Users/gz2294/Pipeline/parse.variant.wt.sequence.R')
    substitute_res <- list(ref=NA, pos=NA, alt=NA, wt = NA,
                           sequence = NA, sequence.len = NA,
                           seq.start = NA, seq.end = NA,
                           pos.orig = NA, sequence.orig = NA,
                           wt.orig = NA, sequence.len.orig = NA)
    aaChg <- prompt$aaChg[i]
    uniprot_IDs <- uniprot_ID[[match(prompt$ensembl[i], unique_genes)]]
    wt.sequences <- uniprot_wt.sequence[[match(prompt$ensembl[i], unique_genes)]]
    j = 1
    while (is.na(substitute_res$sequence) & j <= length(uniprot_IDs)) {
      matched_uniprot_ID <- uniprot_IDs[j]
      wt.sequence <- wt.sequences[j]
      # parse variant
      substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset[i])
      j = j + 1
    }
    tmp <- data.frame(VarID = prompt$VarID[i],
                      aaChg = prompt$aaChg[i],
                      uniprotID = matched_uniprot_ID,
                      ref = substitute_res$ref,
                      pos = substitute_res$pos,
                      alt = substitute_res$alt,
                      score = prompt$score[i],
                      sequence = substitute_res$sequence,
                      wt = substitute_res$wt,
                      sequence.len = substitute_res$sequence.len,
                      seq.start = substitute_res$seq.start,
                      seq.end = substitute_res$seq.end,
                      pos.orig = substitute_res$pos.orig,
                      sequence.orig = substitute_res$sequence.orig,
                      wt.orig = substitute_res$wt.orig,
                      sequence.len.orig = substitute_res$sequence.len.orig,
                      ensembl = prompt$ensembl[i])
    tmp
  }
  stopCluster(cl)
  result <- dplyr::bind_cols(dnv.table[,!colnames(dnv.table) %in% colnames(result)], result)
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}


dnv.table.to.uniprot.by.ensembl_transcriptID.from.dir.parallel <- function(
    dnv.table, VarID.column, Score.column,
    ensembl.column, aaChg.column, ensembl.dir, offset.column=NA, njobs=42) {
  library(doParallel)
  cl <- makeCluster(njobs)
  registerDoParallel(cl)
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  prompt <- data.frame(VarID = dnv.table[,VarID.column],
                       aaChg = dnv.table[,aaChg.column],
                       score = dnv.table[,Score.column],
                       ensembl = dnv.table[,ensembl.column])
  unique_genes <- as.character(unique(prompt$ensembl))
  ensembl.mapping <- read.csv('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/ensembl.uniprot.ID.mapping.csv')
  uniprot_ID.mapping <- read.delim('/share/pascal/Users/gz2294/Data/Protein/uniprot.ID/uniprot.ID.mapping.tsv')
  # order to make sure first use longest transcript
  uniprot_ID.mapping <- uniprot_ID.mapping[order(uniprot_ID.mapping$Length, decreasing = T),]
  # first match reviewed IDs
  uniprot_ID <- as.list(unique_genes)
  uniprot_wt.sequence <- foreach (i = 1:length(unique_genes)) %dopar% {
    tmp <- read.csv(paste0(ensembl.dir, unique_genes[i], '.csv'), row.names = 1)
    tmp <- tmp$X0
    tmp
  }
  result <- foreach (i = 1:dim(prompt)[1], .combine = rbind, .multicombine=TRUE) %dopar% {
    source('/share/pascal/Users/gz2294/Pipeline/parse.variant.wt.sequence.R')
    substitute_res <- list(ref=NA, pos=NA, alt=NA, wt = NA,
                           sequence = NA, sequence.len = NA,
                           seq.start = NA, seq.end = NA,
                           pos.orig = NA, sequence.orig = NA,
                           wt.orig = NA, sequence.len.orig = NA)
    aaChg <- prompt$aaChg[i]
    uniprot_IDs <- uniprot_ID[[match(prompt$ensembl[i], unique_genes)]]
    wt.sequences <- uniprot_wt.sequence[[match(prompt$ensembl[i], unique_genes)]]
    j = 1
    while (is.na(substitute_res$sequence) & j <= length(uniprot_IDs)) {
      matched_uniprot_ID <- uniprot_IDs[j]
      wt.sequence <- wt.sequences[j]
      # parse variant
      substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset[i])
      j = j + 1
    }
    tmp <- data.frame(VarID = prompt$VarID[i],
                      aaChg = prompt$aaChg[i],
                      uniprotID = matched_uniprot_ID,
                      ref = substitute_res$ref,
                      pos = substitute_res$pos,
                      alt = substitute_res$alt,
                      score = prompt$score[i],
                      sequence = substitute_res$sequence,
                      wt = substitute_res$wt,
                      sequence.len = substitute_res$sequence.len,
                      seq.start = substitute_res$seq.start,
                      seq.end = substitute_res$seq.end,
                      pos.orig = substitute_res$pos.orig,
                      sequence.orig = substitute_res$sequence.orig,
                      wt.orig = substitute_res$wt.orig,
                      sequence.len.orig = substitute_res$sequence.len.orig,
                      ensembl = prompt$ensembl[i])
    tmp
  }
  stopCluster(cl)
  result <- dplyr::bind_cols(dnv.table[,!colnames(dnv.table) %in% colnames(result)], result)
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}


dnv.table.to.uniprot.by.wt.seq.parallel <- function(
    dnv.table, VarID.column, Score.column,
    wt.seq.column, aaChg.column, offset.column=NA, njobs=42) {
  library(doParallel)
  cl <- makeCluster(njobs)
  registerDoParallel(cl)
  if (!is.na(offset.column)) {
    offset <- as.numeric(dnv.table[,offset.column])
  } else {
    offset <- rep(0, dim(dnv.table)[1])
  }
  prompt <- data.frame(VarID = dnv.table[,VarID.column],
                       aaChg = dnv.table[,aaChg.column],
                       score = dnv.table[,Score.column],
                       wt.orig = dnv.table[,wt.seq.column])
  result <- foreach (i = 1:dim(prompt)[1], .combine = rbind, .multicombine=TRUE) %dopar% {
    source('/share/pascal/Users/gz2294/Pipeline/parse.variant.wt.sequence.R')
    substitute_res <- list(ref=NA, pos=NA, alt=NA, wt = NA,
                           sequence = NA, sequence.len = NA,
                           seq.start = NA, seq.end = NA,
                           pos.orig = NA, sequence.orig = NA,
                           wt.orig = NA, sequence.len.orig = NA)
    aaChg <- prompt$aaChg[i]
    uniprot_ID <- NA
    wt.sequence <- prompt$wt.orig[i]
    substitute_res <- parse_one_substitute(aaChg, wt.sequence, offset[i])
    tmp <- data.frame(VarID = prompt$VarID[i],
                      aaChg = prompt$aaChg[i],
                      uniprotID = NA,
                      ref = substitute_res$ref,
                      pos = substitute_res$pos,
                      alt = substitute_res$alt,
                      score = prompt$score[i],
                      sequence = substitute_res$sequence,
                      wt = substitute_res$wt,
                      sequence.len = substitute_res$sequence.len,
                      seq.start = substitute_res$seq.start,
                      seq.end = substitute_res$seq.end,
                      pos.orig = substitute_res$pos.orig,
                      sequence.orig = substitute_res$sequence.orig,
                      wt.orig = substitute_res$wt.orig,
                      sequence.len.orig = substitute_res$sequence.len.orig)
    tmp
  }
  stopCluster(cl)
  result <- dplyr::bind_cols(dnv.table[,!colnames(dnv.table) %in% colnames(result)], result)
  result.noNA <- result[!is.na(result$sequence),]
  output <- list(result=result,
                 result.noNA=result.noNA)
  output
}

source('/share/pascal/Users/gz2294/Pipeline/uniprot.table.add.annotation.R')




