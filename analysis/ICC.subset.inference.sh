#!/bin/bash
# $1 is the name of the scripts folder
# $2 are the tasks to run, seperated by comma
# $3 is the gpu ids that used for training, seperated by comma
# $4 is an optional argument that, if present, skips the check for finished tasks
IFS=',' read -ra arr <<< $3
CUDA_VISIBLE_DEVICES=$4
for gene in ${arr[@]}
do
  echo "Begin "$gene
  for subset in 1 2 4 6
  do
    for seed in {0..4}
    do
      logdir=$(cat $1/$gene.subset.$subset.5fold/$gene.subset.$subset.fold.$seed.yaml | grep log_dir | sed 's/.*: //')
      num_epochs=$(cat $1/$gene.subset.$subset.5fold/$gene.subset.$subset.fold.$seed.yaml | grep num_epochs | sed 's/.*: //')
      data_file_train=$(cat $1/$gene.subset.$subset.5fold/$gene.subset.$subset.fold.$seed.yaml | grep data_file_train: | sed 's/.*: //')
      # check if task has finished, unless the skip argument is present
      if [ -f $logdir/FOLD.3/model.epoch.$num_epochs.pt ]; then
      if [ ! -f $2/$gene/testing.subset.$subset.fold.$seed.4fold.csv ]; then
	      python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
          --conf $1/$gene.subset.$subset.5fold/$gene.subset.$subset.fold.$seed.yaml \
		      --mode interpret_4_fold --interpret-by both --out-dir $2/$gene/testing.subset.$subset.fold.$seed.4fold.csv
      fi
      if [ ! -f $2/$gene/training.subset.$subset.fold.$seed.4fold.csv ]; then
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
            --conf $1/$gene.subset.$subset.5fold/$gene.subset.$subset.fold.$seed.yaml \
            --data-file-test $data_file_train \
            --mode interpret_4_fold --interpret-by both --out-dir $2/$gene/training.subset.$subset.fold.$seed.4fold.csv
      fi
      fi
    done
  done
done
