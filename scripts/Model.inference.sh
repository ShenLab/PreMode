#!/bin/bash
# $1 is the name of the scripts folder
# $2 are the files that needs inference, seperated by comma
# $3 is the gpu id to use
# $4 is an optional argument that, if present, serves as the output files
cd /share/terra/Users/gz2294/PreMode.final
IFS=',' read -ra arr <<< $2
if [ -z "$4" ]; then
  IFS=',' read -ra out <<< $4
fi
# TODO: implement output
CUDA_VISIBLE_DEVICES=$3
for gene in ${arr[@]}
do
  echo "Begin inference "$gene
  python -W ignore::UserWarning:torch_geometric.data.collate:147 train.py \
  --conf $1/pretrain.seed.0.yaml \
  --data-file-test $gene \
  --out-dir $gene.inference.csv \
  --mode interpret
done
