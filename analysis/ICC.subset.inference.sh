#!/bin/bash
# $1 is the name of the scripts folder
# $2 are the tasks to run, seperated by comma
# $3 is the gpu ids that used for training, seperated by comma
# $4 is an optional argument that, if present, skips the check for finished tasks
cd /share/vault/Users/gz2294/PreMode
IFS=',' read -ra arr <<< $3
CUDA_VISIBLE_DEVICES=$4
for gene in ${arr[@]}
do
  echo "Begin "$gene
  for subset in {1..5}
  do
    for seed in {0..4}
    do
      # check if task has finished, unless the skip argument is present
      if [ ! -f analysis/$2/$gene/testing.subset.$subset.fold.$seed.csv ]; then
	      python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
          --conf $1/$gene.subset.$subset.5fold/$gene.subset.$subset.fold.$seed.yaml \
		      --mode interpret --interpret-by both --out-dir analysis/$2/$gene/testing.subset.$subset.fold.$seed.csv
      fi
#      if [ ! -f analysis/$2/$gene/testing.pretrain.subset.$subset.fold.$seed.csv ]; then
#        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
#            --conf $1/pretrain.seed.0.yaml \
#            --data-file-test /share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.$seed/$gene.subset.$subset/testing.csv \
#            --mode interpret --interpret-by both --out-dir analysis/$2/$gene/testing.pretrain.subset.$subset.fold.$seed.csv
#      fi
#      if [ ! -f analysis/$2/$gene/training.pretrain.subset.$subset.fold.$seed.csv ]; then
#        python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
#            --conf $1/pretrain.seed.0.yaml \
#            --data-file-test /share/vault/Users/gz2294/Data/DMS/Itan.CKB.Cancer/pfams.add.beni.0.8.seed.$seed/$gene.subset.$subset/training.csv \
#            --mode interpret --interpret-by both --out-dir analysis/$2/$gene/training.pretrain.subset.$subset.fold.$seed.csv
#      fi
    done
  done
done
