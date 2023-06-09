#!/bin/bash
# $1 is the name of the scripts folder
# $2 are the tasks to run, seperated by comma
# $3 is the gpu ids that used for training, seperated by comma
# $4 is an optional argument that, if present, skips the check for finished tasks
cd /share/terra/Users/gz2294/RESCVE.final
IFS=',' read -ra arr <<< $2
CUDA_VISIBLE_DEVICES=$3
for gene in ${arr[@]}
do
  echo "Begin "$gene
  for subset in 1 2 4 6
  do
    for seed in {0..2}
    do
      # check if task has finished, unless the skip argument is present
      if [ -z "$4" ]; then
        logdir=$(cat $1/$gene.subsets/subset.$subset/seed.$seed.yaml | grep log_dir | sed 's/.*: //')
        num_epochs=$(cat $1/$gene.subsets/subset.$subset/seed.$seed.yaml | grep num_epochs | sed 's/.*: //')
        if [ -f $logdir/model.epoch.$num_epochs.pt ]; then
          echo "Skip "$gene" subset "$subset" seed "$seed
          continue
        fi
      fi
      python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene.subsets/subset.$subset/seed.$seed.yaml --test-by batch
    done
  done
done
