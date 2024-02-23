#!/bin/bash
cd /share/terra/Users/gz2294/PreMode.final/analysis/
for i in {0..4}
do 
	for gene in IonChannel PF00130 PF07679 PF07714 PF02196 PF00047 PF00028 PF17756 PF00069 PF00454 PF00520 PF01007 PF06512 PF11933
       	do 
		python -W ignore::UserWarning:torch_geometric.data.collate:147 /share/terra/Users/gz2294/PreMode.final/train.py \
                        --conf $1/pretrain.seed.0.yaml \
                        --mode interpret --data-file-test cohort/$gene.csv \
                        --out-dir $2/$gene.pretrain.csv \
                        --interpret-by batch
		python -W ignore::UserWarning:torch_geometric.data.collate:147 /share/terra/Users/gz2294/PreMode.final/train.py \
			--conf $1/$gene.chps.even.uniprotID.5fold/$gene.chps.even.uniprotID.fold.$i.yaml \
			--mode interpret --data-file-test cohort/$gene.csv \
			--out-dir $2/$gene.fold.$i.csv \
			--interpret-by batch
	done
done
