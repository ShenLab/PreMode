#!/bin/bash
# $1 is the name of the scripts folder
# $2 is the output folder
# $3 is the gpu ids that used for training, seperated by comma
CUDA_VISIBLE_DEVICES=$3
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
for fold in {0..4}
do
  for task in PTEN NUDT15 CCR5 CXCR4 VKORC1 PTEN.bin; do
    for subset in 1 2 4 6; do
    # check if task has finished, unless the skip argument is present
    logdir=$(cat $1/$task.subsets/subset.$subset/seed.$fold.yaml | grep log_dir | sed 's/.*: //')
    num_epochs=$(cat $1/$task.subsets/subset.$subset/seed.$fold.yaml | grep num_epochs | sed 's/.*: //')
    if [ -f $logdir/model.epoch.$num_epochs.pt ]; then
      echo "Begin "$task" fold "$fold" subset "$subset
      mkdir analysis/$2/$task
      # check if task has finished, unless the skip argument is present
      if [ ! -f analysis/$2/$task/testing.subset.$subset.fold.$fold.csv ]; then
        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
            --conf $1/$task.subsets/subset.$subset/seed.$fold.yaml \
            --mode interpret --interpret-by both --out-dir analysis/$2/$task/testing.subset.$subset.fold.$fold.csv
      fi
    fi
    #if [ ! -f analysis/$2/$task/testing.pretrain.subset.$subset.fold.$fold.csv ]; then
    #  python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
    #      --conf $1/pretrain.seed.0.yaml \
    #	  --data-file-test /share/terra/Users/gz2294/ld1/Data/DMS/MAVEDB/$task.$subset.seed.$fold/testing.csv \
    #      --mode interpret --interpret-by both --out-dir analysis/$2/$task/testing.pretrain.subset.$subset.fold.$fold.csv
    #fi
    #if [ ! -f analysis/$2/$task/beni.fold.$fold.csv ] && [ -f /share/terra/Users/gz2294/ld1/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.0/$task/beni.csv ]; then
    #  python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
    #      --conf $1/$task.5fold/$task.fold.$fold.yaml \
    #	  --data-file-test /share/terra/Users/gz2294/ld1/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.0/$task/beni.csv \
    #      --mode interpret --interpret-by both --out-dir analysis/$2/$task/beni.fold.$fold.csv 
    #fi
  done
done
done
#for task in $(cat scripts/pfams.txt); do
#    echo "Begin "$task
#    # check if task has finished, unless the skip argument is present
#    if [ -f analysis/$2/$task/beni.pretrain.csv ]; then
#      echo "Skip "$task" beni"
#      continue
#    fi
#    if [ -f /share/terra/Users/gz2294/ld1/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.0/$task/beni.csv ]; then
#    	python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
#          --conf $1/pretrain.seed.0.yaml \
#          --data-file-test /share/terra/Users/gz2294/ld1/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.0/$task/beni.csv \
#          --mode interpret --interpret-by both --out-dir analysis/$2/$task/beni.pretrain.csv
#    fi
#done
