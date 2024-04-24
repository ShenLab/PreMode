parse_one_substitute <- function(aaChg, wt.sequence, offset=0, seq.lim=1001) {
  if (is.na(wt.sequence)) {
    ref = NA
    pos = NA
    alt = NA 
    wt = NA
    sequence = NA
    sequence.len = NA
    seq.start = NA
    seq.end = NA
    pos.orig = NA
    sequence.orig = NA
    wt.orig = NA
    sequence.len.orig = NA
  } else {
    protein.dictionary <- c(
      "A"="Ala", "R"="Arg", "N"="Asn", "D"="Asp", "C"="Cys", "Q"="Gln", "E"="Glu",
      "G"="Gly", "H"="His", "I"="Ile", "L"="Leu", "K"="Lys", "M"="Met", "F"="Phe",
      "P"="Pro", "O"="Pyl", "S"="Ser", "U"="Sec", "T"="Thr", "W"="Trp", "Y"="Tyr",
      "V"="Val", "B"="Asx", "Z"="Glx", "X"="Xaa", "J"="Xle"
    )
    protein.reverse.dictionary <- names(protein.dictionary)
    names(protein.reverse.dictionary) <- protein.dictionary
    pos_raw <- regmatches(aaChg, gregexpr('[0-9]+', aaChg))[[1]]
    pos <- as.numeric(pos_raw) + offset
    if (length(pos) == 0) {
      # probably synonymous variant
      ref <- NA
      pos <- NA
      alt <- NA
      if (aaChg == "p.=" | aaChg == "p.(=)" | aaChg == "_wt") {
        newseq <- wt.sequence
      } else {
        newseq <- NA
      }
    } else if (length(pos) == 2 & !grepl('fs', aaChg)) {
      # probably delins
      ref_aa_start <- strsplit(substr(aaChg, 3, 
                                      nchar(aaChg)),
                               split = pos_raw[1])[[1]][1]
      remains <- strsplit(substr(aaChg, 3, 
                                 nchar(aaChg)),
                          split = pos_raw[1])[[1]][2]
      ref_aa_end <- strsplit(substr(remains, 2, 
                                    nchar(remains)),
                             split = pos_raw[2])[[1]][1]
      remains <- strsplit(substr(remains, 2, 
                                 nchar(remains)),
                          split = pos_raw[2])[[1]][2]
      if (remains == "del") {
        alt <- ""
      } else {
        alt <- strsplit(remains, split = 'delins')[[1]][2]
      }
      if (nchar(ref_aa_start) > 1) {
        ref_aa_start <- as.character(protein.reverse.dictionary[ref_aa_start])
      }
      if (nchar(ref_aa_end) > 1) {
        ref_aa_end <- as.character(protein.reverse.dictionary[ref_aa_end])
      }
      if (nchar(alt) > 1) {
        alt <- as.character(protein.reverse.dictionary[alt])
      }
      if (ref_aa_start == substr(wt.sequence, pos[1], pos[1]) &
          ref_aa_end == substr(wt.sequence, pos[2], pos[2])) {
        newseq <- wt.sequence
        ref <- substr(wt.sequence, pos[1], pos[2])
        substr(newseq, pos[1], pos[1]) <- alt
        for (i in (pos[1]+1):pos[2]) {
          newseq <- paste(unlist(strsplit(newseq, ""))[-(pos[1]+1)], collapse = "")
        }
        pos <- pos[1]
      } else {
        ref <- substr(wt.sequence, pos[1], pos[2])
        alt <- NA
        newseq <- NA
        pos <- NA
      }
    } else if (length(pos) == 2 & grepl('fs', aaChg)) {
      pos <- pos[1]
      ref <- substr(wt.sequence, pos, pos)
      alt <- NA
      newseq <- NA
    } else if (pos > nchar(wt.sequence)) {
      # out of bound
      ref <- NA
      pos <- pos
      alt <- NA
      if (aaChg == "p.=" | aaChg == "p.(=)" | aaChg == "_wt") {
        newseq <- wt.sequence
      } else {
        newseq <- NA
      }
    } else {
      # possible missense variant
      ref_alt_raw <- strsplit(substr(aaChg, 3, 
                                     nchar(aaChg)),
                              split = pos_raw)[[1]]
      ref <- ref_alt_raw[1]
      if (nchar(ref) > 1) {
        ref <- as.character(protein.reverse.dictionary[ref])
      }
      if (ref == substr(wt.sequence, pos, pos)) {
        newseq <- wt.sequence
        if (ref_alt_raw[2] == "~" | ref_alt_raw[2] == "del") {
          alt <- NA
          newseq <- paste(unlist(strsplit(wt.sequence, ""))[-pos], collapse = "")
        } else if (ref_alt_raw[2] == "*" | ref_alt_raw[2] == "Ter") {
          alt <- NA
          newseq <- substr(wt.sequence, 1, pos-1)
        } else if (ref_alt_raw[2] == "=") {
          # do nothing
          alt <- NA
        } else {
          alt <- ref_alt_raw[2]
          if (nchar(alt) > 1) {
            alt <- as.character(protein.reverse.dictionary[alt])
          }
          substr(newseq, pos, pos) <- alt
        }
      } else {
        ref <- substr(wt.sequence, pos, pos)
        alt <- NA
        newseq <- NA
      }
    }
    if (!is.na(newseq) & nchar(newseq)<=1) {
      newseq <- NA
    }
    # crop sequence depends on sequence length
    sequence.len.orig <- nchar(newseq)
    sequence.orig <- newseq
    pos.orig <- pos
    wt.orig <- wt.sequence
    if (!is.na(sequence.len.orig) & 
        !is.na(pos.orig) &
        (sequence.len.orig > seq.lim | nchar(wt.orig) > seq.lim)) {
      sequence.len <- seq.lim
      if (pos.orig < (seq.lim+1)/2) {
        sequence <- substr(sequence.orig, 1, seq.lim)
        wt <- substr(wt.orig, 1, seq.lim)
        pos <- pos.orig
        seq.start <- 1
        seq.end <- seq.lim
      } else if (pos.orig + (seq.lim-1)/2 > sequence.len.orig) {
        sequence <- substr(sequence.orig,
                           sequence.len.orig-seq.lim+1,
                           sequence.len.orig)
        wt <- substr(wt.orig,
                     sequence.len.orig-seq.lim+1,
                     sequence.len.orig)
        pos <- pos.orig - sequence.len.orig + seq.lim
        seq.start <- sequence.len.orig - seq.lim + 1
        seq.end <- sequence.len.orig
      } else {
        sequence <- substr(sequence.orig,
                           pos.orig-(seq.lim-1)/2,
                           pos.orig+(seq.lim-1)/2)
        wt <- substr(wt.sequence,
                     pos.orig-(seq.lim-1)/2,
                     pos.orig+(seq.lim-1)/2)
        pos <- (seq.lim+1)/2
        seq.start <- pos.orig-(seq.lim-1)/2
        seq.end <- pos.orig+(seq.lim-1)/2
      }
    } else {
      sequence.len <- sequence.len.orig
      sequence <- sequence.orig
      wt <- wt.sequence
      pos <- pos.orig
      seq.start <- 1
      seq.end <- sequence.len.orig
    }
  }
  result <- list(ref=ref, pos=pos, alt=alt, 
                 wt = wt,
                 sequence = sequence,
                 sequence.len = sequence.len,
                 seq.start = seq.start,
                 seq.end = seq.end,
                 pos.orig = pos.orig,
                 sequence.orig = sequence.orig,
                 wt.orig = wt.orig,
                 sequence.len.orig = sequence.len.orig)
  result
}