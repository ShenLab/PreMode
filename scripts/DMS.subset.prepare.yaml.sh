#!/bin/bash
# $1 is the name of the scripts folder
# pretrain.seed.0.yaml: main file, the pretrain model
# first select the best model for TL based on validation dataset in pretrain
# prepare yaml files for subset tasks
for gene in PTEN PTEN.bin CCR5 CXCR4 NUDT15 SNCA CYP2C9 GCK ASPA fluorescence
do
  mkdir $1/$gene.subsets/
  for subset in 1 2 4 6
  do
    mkdir $1/$gene.subsets/subset.$subset
    for seed in {0..4}
    do
      cp $1/$gene/$gene.seed.$seed.yaml $1/$gene.subsets/subset.$subset/seed.$seed.yaml
      # change training dataset
      sed -i "s|training.csv|/training."$subset"."$seed".csv|g" $1/$gene.subsets/subset.$subset/seed.$seed.yaml
      # change output log
      sed -i "s|TL."$gene"|TL."$gene".subset."$subset"|g" $1/$gene.subsets/subset.$subset/seed.$seed.yaml
    done
  done
done
