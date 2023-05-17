#!/bin/bash
# $1 is the name of the scripts folder
# pretrain.seed.0.yaml: main file, the pretrain model
# first select the best model for TL based on validation dataset in pretrain
cd /share/terra/Users/gz2294/RESCVE.final
# prepare yaml files for subset tasks
for gene in PTEN CCR5 CXCR4 NUDT15 VKORC1 IonChannel
do
  mkdir $1/$gene.subsets/
  for subset in 1 2 4 6
  do
    mkdir $1/$gene.subsets/subset.$subset
    for seed in 0 1 2
    do
      cp $1/$gene/$gene.seed.$seed.yaml $1/$gene.subsets/subset.$subset/seed.$seed.yaml
      # change training dataset
      sed -i "s|"$gene"/training.csv|"$gene"."$subset".seed."$seed"/training.csv|g" $1/$gene.subsets/subset.$subset/seed.$seed.yaml
      # change output log
      sed -i "s|TL."$gene"|TL."$gene".subset."$subset"|g" $1/$gene.subsets/subset.$subset/seed.$seed.yaml
    done
  done
done
