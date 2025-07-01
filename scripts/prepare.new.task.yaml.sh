#!/bin/bash
# $1 is the name of the scripts folder
# $2 is the name of your task
# $3 is the training data file of your task
# $4 is the testing data file of your task
# $5 is the type of your task, either GLOF or DMS
# $6 is the number of dimensions of mode of action
# pretrain.seed.0.yaml: main file, the pretrain model
# first select the best model for TL based on validation dataset in pretrain
if [ ! -f $1/pretrain.seed.0.summary ] || [ ! -s $1/pretrain.seed.0.summary ]; then
  Rscript visualize.train.process/plot.test.AUC.by.step.R $1/pretrain.seed.0.yaml > $1/pretrain.seed.0.summary
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
target_nworkers=0
batch_size=$(cat $1/pretrain.seed.0.yaml | grep batch_size | sed 's/.*: //' | sed 's/ #.*//g')
data_file_train=$(cat $1/pretrain.seed.0.yaml | grep data_file_train: | sed 's/.*: //' | sed 's/ #.*//g')
data_file_test=$(cat $1/pretrain.seed.0.yaml | grep data_file_test: | sed 's/.*: //' | sed 's/ #.*//g')
echo "loss_fn was: "$loss_fn
changed_data=false
if grep -q "_by_anno" $1/pretrain.seed.0.yaml; then
    echo "modify data-file-train in original yaml"
    if [ ! -f $1/pretrain.seed.0.yaml.bak ]; then
        cp $1/pretrain.seed.0.yaml $1/pretrain.seed.0.yaml.bak
    fi
    sed -i 's|_by_anno|""|g' $1/pretrain.seed.0.yaml
    changed_data=true
fi
# prepare new yaml files for all tasks
for gene in $2
do
  # use original yaml as template
  cp $1/pretrain.seed.0.yaml $1/$gene.yaml
  # ngpu should be 1
  sed -i "s|ngpus: "$ngpus"|ngpus: 1\nuse_lora: |g" $1/$gene.yaml
  # learning rate should be half
  sed -i "s|lr: "$lr"|lr: "$half_lr"|g" $1/$gene.yaml
  sed -i "s|lr_min: "$lr_min"|lr_min: "$half_lr_min"|g" $1/$gene.yaml
  # change data type
  sed -i "s|data_type: ClinVar|data_type: "$5"|g" $1/$gene.yaml
  # change loss fn, if DMS, use mse_loss, if GLOF, use weighted_loss
  if [[ "DMS" == *"$5"* ]]; then
      sed -i "s|loss_fn: "$loss_fn"|loss_fn: mse_loss|g" $1/$gene.yaml
  else
    if [[ "GLOF" == *"$5"* ]]; then
      sed -i "s|loss_fn: "$loss_fn"|loss_fn: weighted_loss|g" $1/$gene.yaml
    fi
  fi
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
  if grep -q "loaded_esm" $1/pretrain.seed.0.yaml; then
	  sed -i "s|loaded_esm: false|loaded_esm: true|g" $1/$gene.yaml
  else
	  echo "loaded_esm: true" >> $1/$gene.yaml
  fi
  # change load model
  orig_load_model=$(cat $1/pretrain.seed.0.yaml | grep ^load_model | sed 's/.*: //' | sed 's/ #.*//g')
  sed -i "s|load_model: "$orig_load_model"|load_model: "$best_model"|g" $1/$gene.yaml
  sed -i "s|partial_load_model: true|partial_load_model: false|g" $1/$gene.yaml
  # change num epochs to 2 times larger if DMS
  if [[ "DMS" == *"$5"* ]]; then
      sed -i "s|num_epochs: "$num_epochs"|num_epochs: "$(($num_epochs*2))"|g" $1/$gene.yaml
  else
    if [[ "GLOF" == *"$5"* ]]; then
      sed -i "s|num_epochs: "$num_epochs"|num_epochs: "$(($num_epochs))"|g" $1/$gene.yaml
    fi
  fi
  # warm up steps should be 20 times lower
  sed -i "s|lr_warmup_steps: "$lr_warmup_steps"|lr_warmup_steps: "$(($lr_warmup_steps/20))"|g" $1/$gene.yaml
  # num saved batches should be 20 times lower
  if [[ "DMS" == *"$5"* ]]; then
      sed -i "s|num_save_batches: "$num_save_batches"|num_save_batches: "$(($target_num_save_batches))"|g" $1/$gene.yaml
  else
    if [[ "GLOF" == *"$5"* ]]; then
      sed -i "s|num_save_batches: "$num_save_batches"|num_save_batches: "$(($target_num_save_batches/80))"|g" $1/$gene.yaml
    fi
  fi
  sed -i "s|num_steps_update: "$num_steps_update"|num_steps_update: 1|g" $1/$gene.yaml
  # change the output dimension
  sed -i "s|output_dim: 1|output_dim: "$6"|g" $1/$gene.yaml
  # if is GLOF task, train/val split should be 0.75/0.25
  if [[ "GLOF" == *"$5"* ]]; then
    sed -i "s|train_size: 0.95|train_size: 0.75|g" $1/$gene.yaml
    sed -i "s|val_size: 0.05|val_size: 0.25|g" $1/$gene.yaml
  fi
  # change the data file train
  sed -i "s|data_file_train: "$data_file_train"|data_file_train: "$3"|g" $1/$gene.yaml
  # change the data file test
  sed -i "s|data_file_test: "$data_file_test"|data_file_test: "$4"|g" $1/$gene.yaml
done

# make 5 seeds
for gene in $2
do
  mkdir -p $1/$gene/
  mv $1/$gene.yaml $1/$gene/$gene.seed.0.yaml
  for seed in {1..4}
  do
    cp $1/$gene/$gene.seed.0.yaml $1/$gene/$gene.seed.$seed.yaml
    sed -i "s|log_dir: "$logdir"TL."$gene".seed.0/|log_dir: "$logdir"TL."$gene".seed."$seed"/|g" $1/$gene/$gene.seed.$seed.yaml
    sed -i "s|seed: 0|seed: "$seed"|g" $1/$gene/$gene.seed.$seed.yaml
  done
done
# make large window version for 5 seeds, if GLOF
if [[ "GLOF" == *"$5"* ]]; then
  for gene in $2
  do
    for seed in {0..4}
    do
      cp $1/$gene/$gene.seed.$seed.yaml $1/$gene/$gene.seed.$seed.large.window.yaml
      sed -i "s|max_len: 251|max_len: 1251|g" $1/$gene/$gene.seed.$seed.large.window.yaml
      # change logdir
      sed -i "s|log_dir: "$logdir"TL."$gene".seed."$seed"/|log_dir: "$logdir"TL."$gene".seed."$seed".large.window/|g" $1/$gene/$gene.seed.$seed.large.window.yaml
    done
  done
fi