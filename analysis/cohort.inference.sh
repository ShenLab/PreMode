#!/bin/bash
# $1 is the name of the scripts folder
# $2 are the files that needs inference, seperated by comma
# $3 is the out dir
# $4 is the fold
# $5 is the gpu to use
IFS=',' read -ra arr <<< $2
IFS=',' read -ra folds <<< $4
CUDA_VISIBLE_DEVICES=$5
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
for gene in ${arr[@]}
do
	for fold in ${folds[@]}
	do
		echo "Begin inference "$gene
		python -W ignore::UserWarning:torch_geometric.data.collate:147 /share/terra/Users/gz2294/PreMode.final/train.py \
			--conf $1/$gene.chps.even.uniprotID.5fold/$gene.chps.even.uniprotID.fold.$fold.yaml \
			--data-file-test cohort/$gene.csv \
			--out-dir $3/$gene.fold.$fold.csv \
			--mode interpret \
			--interpret-by batch
	done
done
