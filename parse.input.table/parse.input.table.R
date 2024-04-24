source('parse.input.table/dnv.table.to.uniprot.R')
# Define the directory containing the CSV files
args <- commandArgs(trailingOnly = T)
input <- args[1]
output <- args[2]

dnv.table <- read.csv(input, row.names = 1)
dnv.table$VarID <- paste0(dnv.table$uniprotID, ":", dnv.table$aaChg)
uniprot.table <- dnv.table.to.uniprot.by.af2.uniprotID.parallel(dnv.table, 'VarID', 'score', 'uniprotID', 'aaChg')
write.csv(uniprot.table$result.noNA, output)