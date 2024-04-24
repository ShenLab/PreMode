#!/bin/bash
# $1 is the name of the scripts seeder
# $2 are the tasks to run, seperated by comma
# $3 is the output folder
# $4 is the gpu ids that used for training, seperated by comma
# $5 is an optional argument that, if present, skips the check for finished tasks
IFS=',' read -ra arr <<< $2
output_folder=$3
mkdir -p $output_folder
CUDA_VISIBLE_DEVICES=$4
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
if [ -z "$5" ]; then
  echo "Check if tasks have finished"
fi
for seed in {0..4}
  do
  for gene in ${arr[@]}
    do
    logdir=$(cat $1/$gene/$gene.seed.$seed.yaml | grep log_dir | sed 's/.*: //')
    num_epochs=$(cat $1/$gene/$gene.seed.$seed.yaml | grep num_epochs | sed 's/.*: //')
    data_type=$(cat $1/$gene/$gene.seed.$seed.yaml | grep data_type | sed 's/.*: //')
    if [[ $data_type == "GLOF" ]]; then
      echo "Begin "$gene
      # check if task has finished, unless the skip argument is present
      if [[ -z "$5" ]] && [[ -f $logdir/FOLD.3/model.epoch.$num_epochs.pt ]]; then 
        echo "Skip "$gene
      else
        echo "Run "$gene
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene/$gene.seed.$seed.yaml --mode train_4_fold
      fi
      echo "Begin large window of "$gene
      logdir=$(cat $1/$gene/$gene.seed.$seed.large.window.yaml | grep log_dir | sed 's/.*: //')
      num_epochs=$(cat $1/$gene/$gene.seed.$seed.large.window.yaml | grep num_epochs | sed 's/.*: //')
      # check if task has finished, unless the skip argument is present
      if [[ -z "$5" ]] && [[ -f $logdir/FOLD.3/model.epoch.$num_epochs.pt ]]; then 
        echo "Skip large window of "$gene
      else
        echo "Run large window of "$gene
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene/$gene.seed.$seed.large.window.yaml --mode train_4_fold
      fi
    else 
      # for DMS tasks, we can do continue train
      echo "Begin "$gene
      if [[ -z "$5" ]] && [[ -f $logdir/model.epoch.$num_epochs.pt ]]; then 
        echo "Skip "$gene
      else
        echo "Run "$gene
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene/$gene.seed.$seed.yaml --mode continue_train
      fi
    fi
  done
done
# make inference
for seed in {0..4}
  do
  for gene in ${arr[@]}
    do
    echo "Begin "$gene
    logdir=$(cat $1/$gene/$gene.seed.$seed.yaml | grep log_dir | sed 's/.*: //')
    num_epochs=$(cat $1/$gene/$gene.seed.$seed.yaml | grep num_epochs | sed 's/.*: //')
    data_type=$(cat $1/$gene/$gene.seed.$seed.yaml | grep data_type | sed 's/.*: //')
    data_file_train=$(cat $1/$gene/$gene.seed.$seed.yaml | grep data_file_train: | sed 's/.*: //')
    # if GLOF, do the same for large window
    if [[ $data_type == "GLOF" ]]; then
      # check if task has finished
      if [[ -f $logdir/FOLD.0/model.epoch.$num_epochs.pt ]] && [[ -f $logdir/FOLD.1/model.epoch.$num_epochs.pt ]] && [[ -f $logdir/FOLD.2/model.epoch.$num_epochs.pt ]] && [[ -f $logdir/FOLD.3/model.epoch.$num_epochs.pt ]] && [[ ! -f $output_folder/$gene.training.seed.$seed.csv ]]; then 
        echo "Begin inference "$gene
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene/$gene.seed.$seed.yaml --mode interpret_4_fold --interpret-by both --out-dir $output_folder/$gene.testing.seed.$seed.csv
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene/$gene.seed.$seed.yaml --mode interpret_4_fold --interpret-by both --data-file-test $data_file_train --out-dir $output_folder/$gene.training.seed.$seed.csv
      else
        echo "Not finished "$gene
      fi
      echo "Begin large window of "$gene
      logdir=$(cat $1/$gene/$gene.seed.$seed.large.window.yaml | grep log_dir | sed 's/.*: //')
      num_epochs=$(cat $1/$gene/$gene.seed.$seed.large.window.yaml | grep num_epochs | sed 's/.*: //')
      # check if task has finished
      if [[ -f $logdir/FOLD.0/model.epoch.$num_epochs.pt ]] && [[ -f $logdir/FOLD.1/model.epoch.$num_epochs.pt ]] && [[ -f $logdir/FOLD.2/model.epoch.$num_epochs.pt ]] && [[ -f $logdir/FOLD.3/model.epoch.$num_epochs.pt ]] && [[ ! -f $output_folder/$gene.training.seed.$seed.large.window.csv ]]; then 
        echo "Begin inference large window of "$gene
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene/$gene.seed.$seed.large.window.yaml --mode interpret_4_fold --interpret-by both --out-dir $output_folder/$gene.testing.seed.$seed.large.window.csv
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene/$gene.seed.$seed.large.window.yaml --mode interpret_4_fold --interpret-by both --data-file-test $data_file_train --out-dir $output_folder/$gene.training.seed.$seed.large.window.csv
      else
        echo "Not finished large window of "$gene
      fi
    else
      # if not GLOF we don't have to do large window
      # check if task has finished
      if [[ -f $logdir/model.epoch.$num_epochs.pt ]] && [[ ! -f $output_folder/$gene.testing.seed.$seed.csv ]]; then 
        echo "Begin inference "$gene
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene/$gene.seed.$seed.yaml --mode interpret --interpret-by both --out-dir $output_folder/$gene.testing.seed.$seed.csv
      else
        echo "Not finished "$gene
      fi
    fi
  done
done
# aggregate results
# get conda home
conda_home=$(conda info --base)
for gene in ${arr[@]}; do
  $conda_home/envs/r4-base/bin/Rscript scripts/run.new.task.R $1/$gene/$gene $output_folder 
done