#!/bin/bash
# $1 is the name of the scripts folder
# pretrain.seed.0.yaml: main file, the pretrain model
# first select the best model for TL based on validation dataset in pretrain
cd /share/terra/Users/gz2294/RESCVE.final
if [ ! -f $1/pretrain.seed.0.summary ] || [ ! -s $1/pretrain.seed.0.summary ]; then
  Rscript plot.test.AUC.by.step.R $1/pretrain.seed.0.yaml > $1/pretrain.seed.0.summary
fi
number=$(cat $1/pretrain.seed.0.summary | grep 'val' | grep -oE '\([0-9]+\)' | sed 's/[(|)]//g')
logdir=$(cat $1/pretrain.seed.0.yaml | grep log_dir | sed 's/.*: //')
best_model=$logdir"model.step."$number".pt"
echo "Best model is: "$best_model
# prepare yaml files for all tasks
for gene in PTEN CCR5 CXCR4 NUDT15 VKORC1 DDX3X IonChannel $(cat scripts/pfams.txt) AAV GB1 B_LAC fluorescence
do
  # use original yaml as template
  cp $1/pretrain.seed.0.yaml $1/$gene.yaml
  # ngpu should be 1
  sed -i "s|ngpus: 4|ngpus: 1|g" $1/$gene.yaml
  # learning rate should be half
  sed -i "s|lr: 1e-5|lr: 5e-6|g" $1/$gene.yaml
  sed -i "s|lr_min: 1e-6|lr_min: 1e-7|g" $1/$gene.yaml
  # change data type
  sed -i "s|data_type: ClinVar|data_type: "$gene"|g" $1/$gene.yaml
  # change loss fn
  sed -i "s|loss_fn: binary_cross_entropy|loss_fn: mse_loss|g" $1/$gene.yaml
  # change logdir
  sed -i "s|log_dir: "$logdir"|log_dir: "$logdir"TL."$gene".seed.0/|g" $1/$gene.yaml
done


# for human genes, load_model based on the best model in pretrain
for gene in PTEN CCR5 CXCR4 NUDT15 VKORC1 DDX3X IonChannel $(cat scripts/pfams.txt)
do
  # change load model
  orig_load_model=$(cat $1/pretrain.seed.0.yaml | grep ^load_model | sed 's/.*: //' | sed 's/ #.*//g')
  sed -i "s|load_model: "$orig_load_model"|load_model: "$best_model"|g" $1/$gene.yaml
  sed -i "s|partial_load_model: true|partial_load_model: false|g" $1/$gene.yaml
  # change num epochs
  sed -i "s|num_epochs: 10|num_epochs: 40|g" $1/$gene.yaml
  # warm up steps should be 20 times lower
  lr_warmup_steps=$(cat $1/pretrain.seed.0.yaml | grep lr_warmup_steps | sed 's/.*: //' | sed 's/ #.*//g')
  sed -i "s|lr_warmup_steps: "$lr_warmup_steps"|lr_warmup_steps: "$(($lr_warmup_steps/20))"|g" $1/$gene.yaml
  # num saved batches should be 20 times lower
  num_save_batches=$(cat $1/pretrain.seed.0.yaml | grep num_save_batches | sed 's/.*: //' | sed 's/ #.*//g')
  sed -i "s|num_save_batches: "$num_save_batches"|num_save_batches: "$(($num_save_batches/20))"|g" $1/$gene.yaml
done

# non human genes, done't load model
for gene in B_LAC
do
  # change num epochs
  sed -i "s|num_epochs: 10|num_epochs: 40|g" $1/$gene.yaml
  # warm up steps should be 5 times lower
  lr_warmup_steps=$(cat $1/pretrain.seed.0.yaml | grep lr_warmup_steps | sed 's/.*: //' | sed 's/ #.*//g')
  sed -i "s|lr_warmup_steps: "$lr_warmup_steps"|lr_warmup_steps: "$(($lr_warmup_steps/4))"|g" $1/$gene.yaml
  # num saved batches should be 5 times lower
  num_save_batches=$(cat $1/pretrain.seed.0.yaml | grep num_save_batches | sed 's/.*: //' | sed 's/ #.*//g')
  sed -i "s|num_save_batches: "$num_save_batches"|num_save_batches: "$(($num_save_batches/4))"|g" $1/$gene.yaml
  sed -i "s|lr_min: 1e-7|lr_min: 1e-6|g" $1/$gene.yaml
  sed -i "s|lr: 5e-6|lr: 5e-5|g" $1/$gene.yaml
  # num saved batches should be 5 times lower
  batch_size=$(cat $1/pretrain.seed.0.yaml | grep batch_size | sed 's/.*: //' | sed 's/ #.*//g')
  sed -i "s|batch_size: "$batch_size"|batch_size: 32|g" $1/$gene.yaml
  sed -i "s|num_epochs: 40|num_epochs: 200|g" $1/$gene.yaml
done

# for DMS tasks, data should be changed
for gene in PTEN CCR5 CXCR4 NUDT15 VKORC1 DDX3X 
do
  sed -i "s|ClinVar.HGMD.PrimateAI.syn|MAVEDB/"$gene"|g" $1/$gene.yaml
  # don't add contrastive loss because they are from same protein
  sed -i "s|contrastive_loss_fn: cosin_contrastive_loss|contrastive_loss_fn: null|g" $1/$gene.yaml
  # change output model to regression
  sed -i "s|BinaryClassification|Regression|g" $1/$gene.yaml
done

for gene in PTEN NUDT15 VKORC1
do
  sed -i "s|output_dim: 1|output_dim: 2|g" $1/$gene.yaml
done

for gene in CCR5 CXCR4
do
  sed -i "s|output_dim: 1|output_dim: 3|g" $1/$gene.yaml
done

for gene in PTEN CCR5 CXCR4
do
  sed -i "s|num_epochs: 40|num_epochs: 80|g" $1/$gene.yaml
done

for gene in IonChannel $(cat scripts/pfams.txt)
do
  sed -i "s|ClinVar.HGMD.PrimateAI.syn|Itan.CKB.Cancer.good.batch/pfams.0.8/"$gene"|g" $1/$gene.yaml
  sed -i "s|BinaryClassification|Tanh|g" $1/$gene.yaml
done

for gene in AAV GB1 B_LAC fluorescence
do
  sed -i "s|ClinVar.HGMD.PrimateAI.syn|"$gene"|g" $1/$gene.yaml
  # change to large learning rate
  sed -i "s|lr: 5e-6|lr: 1e-5|g" $1/$gene.yaml
  # # change to large batch size
  # sed -i "s|batch_size: 6|batch_size: 7|g" $1/$gene.yaml
  # data split fn
  sed -i "s|data_split_fn: _by_good_batch|data_split_fn: _by_anno|g" $1/$gene.yaml
  # don't add contrastive loss because they are from same protein
  sed -i "s|contrastive_loss_fn: cosin_contrastive_loss|contrastive_loss_fn: null|g" $1/$gene.yaml
  # change output model to regression
  sed -i "s|BinaryClassification|Regression|g" $1/$gene.yaml
done

# change seed
for gene in PTEN CCR5 CXCR4 NUDT15 VKORC1 DDX3X IonChannel $(cat scripts/pfams.txt) AAV GB1 B_LAC fluorescence
do
  # use original yaml as template
  mv $1/$gene.yaml $1/$gene.seed.0.yaml
  for seed in {1..4}
  do
    cp $1/$gene.seed.0.yaml $1/$gene.seed.$seed.yaml
    sed -i "s|seed: 0|seed: "$seed"|g" $1/$gene.seed.$seed.yaml
    sed -i "s|log_dir: "$logdir"TL."$gene".seed.0/|log_dir: "$logdir"TL."$gene".seed."$seed"/|g" $1/$gene.seed.$seed.yaml
  done
  # make a dir and move all yaml files into it
  mkdir -p $1/$gene
  mv $1/$gene.seed.*.yaml $1/$gene
done

bash scripts/DMS.subset.prepare.yaml.sh $1