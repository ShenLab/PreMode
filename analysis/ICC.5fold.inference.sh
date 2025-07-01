#!/bin/bash
# $1 is the name of the scripts folder
# $2 is the name of output folder
# $3 is the gpu ids that used for training, seperated by comma
CUDA_VISIBLE_DEVICES=$3
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
for fold in {0..4}
do
    for task in $(cat scripts/gene.txt) $(cat scripts/gene.itan.txt) $(cat scripts/gene.large.window.txt) $(cat scripts/gene.pfams.txt); do
    echo "Begin "$task" fold "$fold 
    mkdir $2/$task
    # check if task has finished, unless the skip argument is present
    logdir=$(cat $1/$task.5fold/$task.fold.$fold.yaml | grep log_dir | sed 's/.*: //')
    num_epochs=$(cat $1/$task.5fold/$task.fold.$fold.yaml | grep num_epochs | sed 's/.*: //')
    data_file_test=$(cat $1/$task.5fold/$task.fold.$fold.yaml | grep data_file_test: | sed 's/.*: //')
    data_file_train=$(cat $1/$task.5fold/$task.fold.$fold.yaml | grep data_file_train: | sed 's/.*: //')
    if [ -f $logdir/FOLD.0/model.epoch.$num_epochs.pt ] && [ -f $logdir/FOLD.1/model.epoch.$num_epochs.pt ] && [ -f $logdir/FOLD.2/model.epoch.$num_epochs.pt ] && [ -f $logdir/FOLD.3/model.epoch.$num_epochs.pt ]; then
      echo "Begin "$task" fold "$fold
      mkdir $2/$task
      if [ ! -f $2/$task/testing.fold.$fold.4fold.csv ]; then
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
            --conf $1/$task.5fold/$task.fold.$fold.yaml \
      	    --data-file-test $data_file_test \
            --mode interpret_4_fold --interpret-by both --out-dir $2/$task/testing.fold.$fold.4fold.csv
      fi
      if [ ! -f $2/$task/training.fold.$fold.4fold.csv ] && [[ ! $(cat scripts/gene.pfams.txt) == *"$task"* ]]; then
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
            --conf $1/$task.5fold/$task.fold.$fold.yaml \
            --data-file-test $data_file_train \
            --mode interpret_4_fold --interpret-by both --out-dir $2/$task/training.fold.$fold.4fold.csv
      fi
    else
      echo $task" fold "$fold" not finished"
    fi
  done
done
