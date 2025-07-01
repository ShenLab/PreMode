# remove all the homologous proteins from pretrain
pretrain <- rbind(read.csv('../archive/data.files/pretrain.orig/training.csv', row.names = 1),
                  read.csv('../archive/data.files/pretrain.orig/testing.csv', row.names = 1))
# read homologous data frame
geneid2paralog <- read.csv('pretrain/revision.split.by.paralogues/geneid2paralog.csv', row.names = 1)
uniprot2geneid <- read.csv('pretrain/revision.split.by.paralogues/uniprot2geneid.csv', row.names = 1)
# define the transfer learning tasks
tf.tasks <- read.csv('../scripts/gene.txt', header = F)
tf.uids <- gsub('.clean', '', tf.tasks$V1)
# add DMS uids: PTEN PTEN.bin CCR5 CXCR4 NUDT15 SNCA CYP2C9 GCK ASPA Stab
tf.uids <- c(tf.uids, 'P60484', 'P51681', 'P61073', 'Q9NV35', 'P37840', 'P11712', 'P35557', 'P45381')
stab <- rbind(read.csv('../archive/data.files/Stab/test.seed.0.csv'),
              read.csv('../archive/data.files/Stab/train.seed.0.csv'))
tf.uids <- c(tf.uids, unique(stab$uniprotID))
# add ProteinGym uids
task.dic <- rbind(read.csv('../analysis/single.assays.txt', header = F), read.csv('../analysis/multiple.assays.txt', header = F))
for (task in task.dic$V1) {
  if (file.exists(paste0('../analysis/PreMode/', task, '/', 'testing.fold.0.csv'))) {
    tmp <- read.csv(paste0('../analysis/PreMode/', task, '/', 'testing.fold.0.csv'))
    tf.uids <- c(tf.uids, tmp$uniprotID[1])
  }
}
tf.uids <- unique(tf.uids)
tf.geneids <- uniprot2geneid$ensembl_gene_id[match(tf.uids, uniprot2geneid$uniprot_gn_id)]
tf.homologues <- geneid2paralog$hsapiens_paralog_ensembl_gene[geneid2paralog$ensembl_gene_id %in% tf.geneids]
to.remove <- unique(c(tf.geneids, tf.homologues))
to.remove.uids <- uniprot2geneid$uniprot_gn_id[match(to.remove, uniprot2geneid$ensembl_gene_id)]
# remove the uids from pretrain
pretrain <- pretrain[!pretrain$uniprotID %in% to.remove.uids,]
# split into 4 parts
split.by.uniprotID <- function(freq_table, number_to_select) {
  set.seed(0)
  selected = 0
  selected_uniprotIDs = c()
  candidates = freq_table[freq_table$Freq <= number_to_select - selected,]
  while ((selected < number_to_select) & (dim(candidates)[1] > 0)) {
    selected_uniprotID = sample(as.character(candidates$Var1), size = 1)
    selected_uniprotIDs <- c(selected_uniprotIDs, selected_uniprotID)
    selected = selected + freq_table$Freq[freq_table$Var1 == selected_uniprotID]
    # update freq_table and candidates
    freq_table = freq_table[!freq_table$Var1 %in% selected_uniprotID,]
    candidates = freq_table[freq_table$Freq <= number_to_select - selected,]
  }
  result = list(selected_uniprotIDs, freq_table)
  result
}

training_freq_table <- as.data.frame(table(pretrain$uniprotID))

quarter.size <- floor(dim(pretrain)[1] / 4)

tmp <- split.by.uniprotID(freq_table = training_freq_table, quarter.size)
quarter.1 <- tmp[[1]]
left_freq_table <- tmp[[2]]

tmp <- split.by.uniprotID(freq_table = left_freq_table, quarter.size)
quarter.2 <- tmp[[1]]
left_freq_table <- tmp[[2]]

tmp <- split.by.uniprotID(freq_table = left_freq_table, quarter.size)
quarter.3 <- tmp[[1]]
left_freq_table <- tmp[[2]]

tmp <- split.by.uniprotID(freq_table = left_freq_table, dim(pretrain)[1]-quarter.size*3)
quarter.4 <- tmp[[1]]
left_freq_table <- tmp[[2]]

# write out the 4 quarters
dir.create('pretrain/revision.remove.all.paralog/', recursive = T, showWarnings = F)
write.csv(pretrain[pretrain$uniprotID %in% quarter.1,], "pretrain/revision.remove.all.paralog/training.0.csv", na = ".")
write.csv(pretrain[pretrain$uniprotID %in% quarter.2,], "pretrain/revision.remove.all.paralog/training.1.csv", na = ".")
write.csv(pretrain[pretrain$uniprotID %in% quarter.3,], "pretrain/revision.remove.all.paralog/training.2.csv", na = ".")
write.csv(pretrain[pretrain$uniprotID %in% quarter.4,], "pretrain/revision.remove.all.paralog/training.3.csv", na = ".")
write.csv(pretrain, 'pretrain/revision.remove.all.paralog/training.csv', na = ".")
