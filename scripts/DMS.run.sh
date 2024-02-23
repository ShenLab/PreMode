#!/bin/bash
# $1 is the name of the scripts folder
# $2 are the tasks to run, seperated by comma
# $3 is the gpu ids that used for training, seperated by comma
# $4 is an optional argument that, if present, skips the check for finished tasks
IFS=',' read -ra arr <<< $2
CUDA_VISIBLE_DEVICES=$3
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
for gene in ${arr[@]}
do
  # use original yaml as template
  if [[ $gene == "Itan.CKB.Cancer" || $gene == "ICC" ]]; then
    for task in $(cat scripts/pfams.txt); do
      echo "Begin "$task
      for seed in {0..4}
      do
        # check if task has finished, unless the skip argument is present
        if [ -z "$4" ]; then
          logdir=$(cat $1/$task/$task.seed.$seed.yaml | grep log_dir | sed 's/.*: //')
          num_epochs=$(cat $1/$task/$task.seed.$seed.yaml | grep num_epochs | sed 's/.*: //')
          if [ -f $logdir/model.epoch.$num_epochs.pt ]; then
            echo "Skip "$task
            continue
          fi
        fi
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$task/$task.seed.$seed.yaml
      done
    done
  else
    echo "Begin "$gene
    for seed in {0..4}
    do
      # check if task has finished, unless the skip argument is present
      if [ -z "$4" ]; then
        logdir=$(cat $1/$gene/$gene.seed.$seed.yaml | grep log_dir | sed 's/.*: //')
        num_epochs=$(cat $1/$gene/$gene.seed.$seed.yaml | grep num_epochs | sed 's/.*: //')
        if [ -f $logdir/model.epoch.$num_epochs.pt ]; then
          echo "Skip "$gene
          continue
        fi
      fi
      python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene/$gene.seed.$seed.yaml
    done
  fi
done
