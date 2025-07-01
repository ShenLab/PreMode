#!/bin/bash
# $1 is the name of the scripts folder
# pretrain.seed.0.yaml: main file, the pretrain model
# first select the best model for TL based on validation dataset in pretrain
if [ ! -f $1/pretrain.seed.0.summary ] || [ ! -s $1/pretrain.seed.0.summary ]; then
  Rscript plot.test.AUC.by.step.R $1/pretrain.seed.0.yaml > $1/pretrain.seed.0.summary
fi
number=$(cat $1/pretrain.seed.0.summary | grep 'val' | grep -oE '\([0-9]+\)' | sed 's/[(|)]//g')
logdir=$(cat $1/pretrain.seed.0.yaml | grep log_dir | sed 's/.*: //')
if [ -z $number ]; then
  best_model="null"
else
  best_model=$logdir"model.step."$number".pt"
fi
echo "Best model is: "$best_model
# origin hyper paramters
lr_warmup_steps=$(cat $1/pretrain.seed.0.yaml | grep lr_warmup_steps | sed 's/.*: //' | sed 's/ #.*//g')
num_save_batches=$(cat $1/pretrain.seed.0.yaml | grep num_save_batches | sed 's/.*: //' | sed 's/ #.*//g')
target_num_save_batches=400
num_epochs=$(cat $1/pretrain.seed.0.yaml | grep num_epochs | sed 's/.*: //' | sed 's/ #.*//g')
batch_size=$(cat $1/pretrain.seed.0.yaml | grep batch_size | sed 's/.*: //' | sed 's/ #.*//g')
lr=$(cat $1/pretrain.seed.0.yaml | grep lr: | sed 's/.*: //' | sed 's/ #.*//g')
half_lr=$(printf "%.1e" "$(echo "scale=10; $(printf "%f" "$lr")" | bc)")
five_lr=$(printf "%.1e" "$(echo "scale=10; $(printf "%f" "$lr") * 5" | bc)")
lr_min=$(cat $1/pretrain.seed.0.yaml | grep lr_min: | sed 's/.*: //' | sed 's/ #.*//g')
half_lr_min=$(echo "$lr_min" | awk '{ printf "%.1e", $1/10 }')
data_split=$(cat $1/pretrain.seed.0.yaml | grep data_split_fn | sed 's/.*: //' | sed 's/ #.*//g')
loss_fn=$(cat $1/pretrain.seed.0.yaml | grep ^loss_fn | sed 's/.*: //' | sed 's/ #.*//g')
drop_out=$(cat $1/pretrain.seed.0.yaml | grep drop_out | sed 's/.*: //' | sed 's/ #.*//g')
num_steps_update=$(cat $1/pretrain.seed.0.yaml | grep num_steps_update | sed 's/.*: //' | sed 's/ #.*//g')
ngpus=$(cat $1/pretrain.seed.0.yaml | grep ngpus | sed 's/.*: //' | sed 's/ #.*//g')
nworkers=$(cat $1/pretrain.seed.0.yaml | grep num_workers | sed 's/.*: //' | sed 's/ #.*//g')
target_nworkers=1
batch_size=$(cat $1/pretrain.seed.0.yaml | grep batch_size | sed 's/.*: //' | sed 's/ #.*//g')
echo "loss_fn was: "$loss_fn
if grep -q "_by_anno" $1/pretrain.seed.0.yaml; then
    echo "modify data-file-train in original yaml"
    if [ ! -f $1/pretrain.seed.0.yaml.bak ]; then
        cp $1/pretrain.seed.0.yaml $1/pretrain.seed.0.yaml.bak
    fi
    sed -i 's|_by_anno|""|g' $1/pretrain.seed.0.yaml
    changed_data=true
else
    changed_data=false
fi
# prepare yaml files for all tasks
for gene in fluorescence
do
  # use original yaml as template
  cp $1/pretrain.seed.0.yaml $1/$gene.yaml
  # ngpu should be 1
  sed -i "s|ngpus: "$ngpus"|ngpus: 1\nuse_lora: |g" $1/$gene.yaml
  # learning rate should be half
  sed -i "s|lr: "$lr"|lr: "$half_lr"|g" $1/$gene.yaml
  sed -i "s|lr_min: "$lr_min"|lr_min: "$half_lr_min"|g" $1/$gene.yaml
  # change data type
  sed -i "s|data_type: ClinVar|data_type: "$gene"|g" $1/$gene.yaml
  # change loss fn
  sed -i "s|loss_fn: "$loss_fn"|loss_fn: mse_loss|g" $1/$gene.yaml
  # change logdir
  sed -i "s|log_dir: "$logdir"|log_dir: "$logdir"TL."$gene".seed.0/|g" $1/$gene.yaml
  # change drop out rate
  sed -i "s|drop_out: "$drop_out"|drop_out: 0.1|g" $1/$gene.yaml
  # change num workers in dataloader
  sed -i "s|num_workers: "$nworkers"|num_workers: "$target_nworkers"|g" $1/$gene.yaml
  # change loaded msa
  if grep -q "loaded_msa" $1/pretrain.seed.0.yaml; then
  	sed -i "s|loaded_msa: false|loaded_msa: true|g" $1/$gene.yaml
  else
	echo "loaded_msa: true" >> $1/$gene.yaml
  fi
  # change loaded confidence
  if grep -q "loaded_confidence" $1/pretrain.seed.0.yaml; then
  	sed -i "s|loaded_confidence: false|loaded_confidence: true|g" $1/$gene.yaml
  else
	echo "loaded_confidence: true" >> $1/$gene.yaml
  fi
  if grep -q "loaded_esm" $1/pretrain.seed.0.yaml; then
	sed -i "s|loaded_esm: false|loaded_esm: true|g" $1/$gene.yaml
  else
	echo "loaded_esm: true" >> $1/$gene.yaml
  fi
done

# if original loss_fn is combined_loss or weighted_combined_loss, change loss back
if [ "$loss_fn" == "combined_loss" ] || [ "$loss_fn" == "weighted_combined_loss" ] || [ "$loss_fn" == "GP_loss" ]; then
  for gene in $(cat scripts/pfams.txt) $(cat scripts/gene.pfams.txt)
  do
    sed -i "s|loss_fn: mse_loss|loss_fn: "$loss_fn"|g" $1/$gene.yaml
  done
fi

# for non human data, change learning rates and data split fn
for gene in fluorescence
do
  sed -i "s|pretrain|"$gene"|g" $1/$gene.yaml
  # change to large learning rate
  sed -i "s|lr: "$half_lr"|lr: "$lr"|g" $1/$gene.yaml
  # data split fn
  sed -i 's|data_split_fn: ""|data_split_fn: _by_anno|g' $1/$gene.yaml
  # don't add contrastive loss because they are from same protein
  sed -i "s|contrastive_loss_fn: cosin_contrastive_loss|contrastive_loss_fn: null|g" $1/$gene.yaml
  # change output model to regression
  sed -i "s|BinaryClassification|Regression|g" $1/$gene.yaml
  # change data split to with anno
  sed -i "s|data_split_fn: ""|data_split_fn: _by_anno|g" $1/$gene.yaml
done

# change seed
for gene in fluorescence
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

if [ $changed_data = true ]; then
  echo "change data-file-train back to original yaml"
  mv $1/pretrain.seed.0.yaml.bak $1/pretrain.seed.0.yaml
fi
