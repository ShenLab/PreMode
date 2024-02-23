#!/bin/bash
# $1 is the name of the scripts folder
# $2 is the name of output folder
# $3 is the gpu ids that used for training, seperated by comma
CUDA_VISIBLE_DEVICES=$3
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
for fold in {0..4}
do
  for task in $(cat scripts/pfams.txt) $(cat scripts/gene.pfams.txt); do
      echo "Begin "$task" fold "$fold 
      mkdir analysis/$2/$task
      # if [ ! -f analysis/$2/$task/testing.pretrain.fold.$fold.csv ]; then
      #   python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
      #       --conf $1/pretrain.seed.0.yaml \
      #       --data-file-test /share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.$fold/$task/testing.csv \
      #       --mode interpret --interpret-by both --use-jacob true --out-dir analysis/$2/$task/testing.pretrain.fold.$fold.csv
      # fi
      # if [ ! -f analysis/$2/$task/training.pretrain.fold.$fold.csv ]; then
      #   python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
      #       --conf $1/pretrain.seed.0.yaml \
      #       --data-file-test /share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.$fold/$task/training.csv \
      #       --mode interpret --interpret-by both --use-jacob true --out-dir analysis/$2/$task/training.pretrain.fold.$fold.csv
      # fi
    # check if task has finished, unless the skip argument is present
    logdir=$(cat $1/$task.5fold/$task.fold.$fold.yaml | grep log_dir | sed 's/.*: //')
    num_epochs=$(cat $1/$task.5fold/$task.fold.$fold.yaml | grep num_epochs | sed 's/.*: //')
    if [ -f $logdir/model.epoch.$num_epochs.pt ]; then
      echo "Begin "$task" fold "$fold
      mkdir analysis/$2/$task
      if [ ! -f analysis/$2/$task/testing.fold.$fold.csv ]; then
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
            --conf $1/$task.5fold/$task.fold.$fold.yaml \
      --data-file-test /share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.$fold/$task/testing.csv \
            --mode interpret --interpret-by both --out-dir analysis/$2/$task/testing.fold.$fold.csv
      fi
      # if [ ! -f analysis/$2/$task/training.fold.$fold.csv ]; then
      #   python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
      #       --conf $1/$task.5fold/$task.fold.$fold.yaml \
      #       --data-file-test /share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.$fold/$task/training.csv \
      #       --mode interpret --interpret-by both --out-dir analysis/$2/$task/training.fold.$fold.csv
      # fi
      # if [ ! -f analysis/$2/$task/beni.fold.$fold.csv ] && [ -f /share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.0/$task/beni.csv ]; then
      #   python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
      #       --conf $1/$task.5fold/$task.fold.$fold.yaml \
      #       --data-file-test /share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.0/$task/beni.csv \
      #       --mode interpret --interpret-by both --out-dir analysis/$2/$task/beni.fold.$fold.csv 
      # fi
    else
      echo $task" fold "$fold" not finished"
    fi
  done
done
if [ $4 == 0 ]; then 
for task in $(cat scripts/pfams.txt) $(cat scripts/gene.pfams.txt); do
    echo "Begin "$task
    # check if task has finished, unless the skip argument is present
    if [ -f analysis/$2/$task/all.mutants.pretrain.csv ]; then
      echo "Skip "$task" beni"
      continue
    fi
    # if [ -f /share/vault/Users/gz2294/PreMode/analysis/5genes.all.mut/$task.csv ]; then
    #   python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
    #       --conf $1/pretrain.seed.0.yaml \
    #       --data-file-test /share/vault/Users/gz2294/PreMode/analysis/5genes.all.mut/$task.csv \
    #       --mode interpret --interpret-by both --use-jacob true --out-dir analysis/$2/$task/all.mutants.pretrain.csv
    # fi
done
fi
