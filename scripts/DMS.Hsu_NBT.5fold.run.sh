#!/bin/bash
# $1 is the name of the scripts folder
# $2 are the tasks to run, seperated by comma
# $3 is the gpu ids that used for training, seperated by comma
# $4 is an optional argument that, if present, skips the check for finished tasks
cd /share/terra/Users/gz2294/PreMode.final
IFS=',' read -ra arr <<< $2
CUDA_VISIBLE_DEVICES=$3
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
for gene in ${arr[@]}; do
  for fold in {0..4}; do
    # use original yaml as template
    if [[ $gene == "Hsu" || $gene == "Hsu_NBT" ]]; then
      for task in $(cat /share/terra/Users/gz2294/ld1/Data/DMS/Hsu_NBT/useful.data.csv); do
        echo "Begin "$task
        # check if task has finished, unless the skip argument is present
        if [ -z "$4" ]; then
          logdir=$(cat $1/$task.5fold/$task.fold.$fold.yaml | grep log_dir | sed 's/.*: //')
          num_epochs=$(cat $1/$task.5fold/$task.fold.$fold.yaml | grep num_epochs | sed 's/.*: //')
          if [ -f $logdir/model.epoch.$num_epochs.pt ]; then
            echo "Skip "$task
            continue
          fi
        fi
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$task.5fold/$task.fold.$fold.yaml
      done
    else
      echo "Begin "$gene
      # check if task has finished, unless the skip argument is present
      if [ -z "$4" ]; then
        logdir=$(cat $1/$gene.5fold/$gene.fold.$fold.yaml | grep log_dir | sed 's/.*: //')
        num_epochs=$(cat $1/$gene.5fold/$gene.fold.$fold.yaml | grep num_epochs | sed 's/.*: //')
        if [ -f $logdir/model.epoch.$num_epochs.pt ]; then
          echo "Skip "$gene
          continue
        fi
      fi
      python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py --conf $1/$gene.5fold/$gene.fold.$fold.yaml
    fi
  done
done
