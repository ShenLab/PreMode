#!/bin/bash
# $1 is the name of the scripts folder
# $2 is the output dir
# $3 is the gpu ids that used for training, seperated by comma
CUDA_VISIBLE_DEVICES=$3
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
mkdir $2
for fold in {0..4}
do
  for task in IonChannel $(sed -n '2,$p' scripts/pfams.txt | grep PF | sed 's/.split.uniprotID//g'); do
    echo "Begin "$task
    # check if task has finished, unless the skip argument is present
    if [ -f $2/$task.fold.$fold.csv ]; then
      echo "Skip "$task" fold "$fold
      continue
    fi
    python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
          --conf $1/$task.split.uniprotID.5fold/$task.split.uniprotID.fold.$fold.yaml \
          --data-file-test /share/terra/Users/gz2294/PreMode.final/analysis/cohort/$task.csv \
          --mode interpret --out-dir $2/$task.fold.$fold.csv
  done
done
