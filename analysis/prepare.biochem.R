prepare.unique.id <- function (uniprot.table) {
  # prepare unique.id
  uniprot.table$unique.id <- paste0(uniprot.table$uniprotID, ":", uniprot.table$ref, uniprot.table$pos.orig, uniprot.table$alt)
  uniprot.table
}

prepare.biochemical <- function (uniprot.table) {
  # prepare sse
  sse <- diag(8)[as.numeric(factor(uniprot.table$secondary_struc, levels = c(' ', 'B', 'E', 'G', 'H', 'I', 'S', 'T'))),]
  sse[is.na(sse)] <- 0
  ref.biochem <- cbind(as.numeric(uniprot.table$ref %in% c('A', 'I', 'L', 'M', 'V')),
                       as.numeric(uniprot.table$ref %in% c('F', 'W', 'Y')),
                       as.numeric(uniprot.table$ref %in% c('F', 'W', 'Y', 'A', 'I', 'L', 'M', 'V')),
                       as.numeric(uniprot.table$ref %in% c('H', 'K', 'R')),
                       as.numeric(uniprot.table$ref %in% c('D', 'E')),
                       as.numeric(uniprot.table$ref %in% c('N', 'Q', 'S', 'T')),
                       as.numeric(uniprot.table$ref %in% c('H', 'K', 'R', 'D', 'E', 'N', 'Q', 'S', 'T')),
                       as.numeric(uniprot.table$ref %in% c('C', 'P', 'G')))
  alt.biochem <- cbind(as.numeric(uniprot.table$alt %in% c('A', 'I', 'L', 'M', 'V')),
                       as.numeric(uniprot.table$alt %in% c('F', 'W', 'Y')),
                       as.numeric(uniprot.table$alt %in% c('F', 'W', 'Y', 'A', 'I', 'L', 'M', 'V')),
                       as.numeric(uniprot.table$alt %in% c('H', 'K', 'R')),
                       as.numeric(uniprot.table$alt %in% c('D', 'E')),
                       as.numeric(uniprot.table$alt %in% c('N', 'Q', 'S', 'T')),
                       as.numeric(uniprot.table$alt %in% c('H', 'K', 'R', 'D', 'E', 'N', 'Q', 'S', 'T')),
                       as.numeric(uniprot.table$alt %in% c('C', 'P', 'G')))
  uniprot.table$rsa[is.na(uniprot.table$rsa)] <- mean(uniprot.table$rsa, na.rm = T)
  uniprot.table$conservation.entropy[is.na(uniprot.table$conservation.entropy)] <- mean(uniprot.table$conservation.entropy, na.rm = T)
  uniprot.table$conservation.alt[is.na(uniprot.table$conservation.alt)] <- mean(uniprot.table$conservation.alt, na.rm = T)
  uniprot.table$conservation.ref[is.na(uniprot.table$conservation.ref)] <- mean(uniprot.table$conservation.ref, na.rm = T)
  uniprot.table$pLDDT[is.na(uniprot.table$pLDDT)] <- mean(uniprot.table$pLDDT, na.rm = T)
  if ("FoldXddG" %in% colnames(uniprot.table)) {
    uniprot.table$FoldXddG[is.na(uniprot.table$FoldXddG)] <- mean(uniprot.table$FoldXddG, na.rm = T)
    out <- cbind(sse, ref.biochem, alt.biochem, uniprot.table$rsa, 
                 uniprot.table$conservation.ref,
                 uniprot.table$conservation.alt,
                 uniprot.table$conservation.entropy,
                 uniprot.table$FoldXddG,
                 uniprot.table$pLDDT/100)
  } else {
    out <- cbind(sse, ref.biochem, alt.biochem, uniprot.table$rsa, 
                 uniprot.table$conservation.ref,
                 uniprot.table$conservation.alt,
                 uniprot.table$conservation.entropy,
                 uniprot.table$pLDDT/100)
  }
  out
}
