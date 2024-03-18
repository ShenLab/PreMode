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
if grep -q "clinvar*.4.splits/" $1/pretrain.seed.0.yaml; then
    echo "modify data-file-train in original yaml"
    if [ ! -f $1/pretrain.seed.0.yaml.bak ]; then
        cp $1/pretrain.seed.0.yaml $1/pretrain.seed.0.yaml.bak
    fi
    data_prefix=$(cat $1/pretrain.seed.0.yaml | grep data_file_train: | sed 's/.*: //' | sed 's/.*ClinVar.HGMD.PrimateAI.syn\/\([^/]*\)\/.*/\1/')
    echo "data_prefix was: "$data_prefix
    sed -i "s|"$data_prefix"/||g" $1/pretrain.seed.0.yaml
    changed_data=true
else
    changed_data=false
fi
if grep -q "pascal/" $1/pretrain.seed.0.yaml; then
    echo "modify data-file-train in original yaml"
    if [ ! -f $1/pretrain.seed.0.yaml.bak ]; then
        cp $1/pretrain.seed.0.yaml $1/pretrain.seed.0.yaml.bak
    fi
    data_prefix=$(cat $1/pretrain.seed.0.yaml | grep data_file_train: | sed 's/.*: //' | sed 's/training.csv//')
    data_prefix_new=$(cat $1/pretrain.seed.0.yaml | grep data_file_train: | sed 's/.*: //' | sed 's/training.csv//' | sed 's/pascal/vault/')
    echo "data_prefix was: "$data_prefix
    sed -i "s|"$data_prefix"|"$data_prefix_new"|g" $1/pretrain.seed.0.yaml
    changed_data=true
else
    changed_data=false
fi
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
if grep -q "PrimateAI.Random.ESM" $1/pretrain.seed.0.yaml; then
    echo "modify data-file-train in original yaml"
    if [ ! -f $1/pretrain.seed.0.yaml.bak ]; then
        cp $1/pretrain.seed.0.yaml $1/pretrain.seed.0.yaml.bak
    fi
    sed -i 's|batch_size: '$batch_size'|batch_size: 8|g' $1/pretrain.seed.0.yaml
    changed_data=true
    target_nworkers=1
    # change PrimateAI.Random.ESM to ClinVar.HGMD.PrimateAI.syn
    # get data_prefix bewteen character 'DMS/' and '/training.csv'
    data_prefix=$(cat $1/pretrain.seed.0.yaml | grep data_file_train: | sed 's/.*: //' | sed 's/training.csv//' | sed 's/.*DMS\///')
    echo "data_prefix was: "$data_prefix
    sed -i "s|"$data_prefix"|ClinVar.HGMD.PrimateAI.syn/|g" $1/pretrain.seed.0.yaml
else
    changed_data=false
fi
# prepare yaml files for all tasks
for gene in PTEN PTEN.bin CCR5 CXCR4 NUDT15 VKORC1 DDX3X $(cat scripts/pfams.txt) $(cat scripts/gene.pfams.txt) AAV GB1 B_LAC fluorescence $(cat /share/pascal/Users/gz2294/Data/DMS/Hsu_NBT/useful.data.csv)
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
  # change data split fn
  # sed -i "s|data_split_fn: "$data_split"|data_split_fn: _by_good_batch|g" $1/$gene.yaml
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

# for human genes, load_model based on the best model in pretrain
for gene in PTEN PTEN.bin CCR5 CXCR4 NUDT15 VKORC1 DDX3X $(cat scripts/pfams.txt) $(cat scripts/gene.pfams.txt) $(cat /share/pascal/Users/gz2294/Data/DMS/Hsu_NBT/useful.data.human.csv)
do
  # change load model
  orig_load_model=$(cat $1/pretrain.seed.0.yaml | grep ^load_model | sed 's/.*: //' | sed 's/ #.*//g')
  sed -i "s|load_model: "$orig_load_model"|load_model: "$best_model"|g" $1/$gene.yaml
  sed -i "s|partial_load_model: true|partial_load_model: false|g" $1/$gene.yaml
  # change num epochs to 4 times larger
  sed -i "s|num_epochs: "$num_epochs"|num_epochs: "$(($num_epochs))"|g" $1/$gene.yaml
  # warm up steps should be 20 times lower
  sed -i "s|lr_warmup_steps: "$lr_warmup_steps"|lr_warmup_steps: "$(($lr_warmup_steps/20))"|g" $1/$gene.yaml
  # num saved batches should be 20 times lower
  if [[ "PTEN CCR5 CXCR4 NUDT15 DDX3X" == *"$gene"* ]]; then
      sed -i "s|num_save_batches: "$num_save_batches"|num_save_batches: "$(($target_num_save_batches))"|g" $1/$gene.yaml
  else
      sed -i "s|num_save_batches: "$num_save_batches"|num_save_batches: "$(($target_num_save_batches/4))"|g" $1/$gene.yaml
  fi
done

for gene in $(cat /share/pascal/Users/gz2294/Data/DMS/Hsu_NBT/useful.data.other.csv)
do
  data_size=$(cat /share/pascal/Users/gz2294/Data/DMS/$gene/train.seed.0.csv | wc -l)
  # warm up steps should be 20 times lower
  if [ $data_size -lt 20000 ]; then
    # change num epochs
    sed -i "s|num_epochs: "$num_epochs"|num_epochs: "$(($num_epochs*4))"|g" $1/$gene.yaml
    sed -i "s|lr_warmup_steps: "$lr_warmup_steps"|lr_warmup_steps: "$(($lr_warmup_steps/20))"|g" $1/$gene.yaml
    # num saved batches should be 20 times lower
    sed -i "s|num_save_batches: "$num_save_batches"|num_save_batches: "$(($num_save_batches/20))"|g" $1/$gene.yaml
  fi
done

# non human genes, done't load model
for gene in B_LAC
do
  # warm up steps should be 5 times lower
  sed -i "s|lr_warmup_steps: "$lr_warmup_steps"|lr_warmup_steps: "$(($lr_warmup_steps/4))"|g" $1/$gene.yaml
  # num saved batches should be 5 times lower
  sed -i "s|num_save_batches: "$num_save_batches"|num_save_batches: "$(($num_save_batches/4))"|g" $1/$gene.yaml
  sed -i "s|lr_min: "$half_lr_min"|lr_min: "$lr_min"|g" $1/$gene.yaml
  sed -i "s|lr: "$half_lr"|lr: "$five_lr"|g" $1/$gene.yaml
  # num saved batches should be 5 times higher
  sed -i "s|batch_size: "$batch_size"|batch_size: "$(($batch_size*5))"|g" $1/$gene.yaml
  sed -i "s|num_epochs: "$num_epochs"|num_epochs: "$(($num_epochs*5))"|g" $1/$gene.yaml
done

# for Human DMS tasks, data should be changed
for gene in PTEN PTEN.bin CCR5 CXCR4 NUDT15 VKORC1 DDX3X $(cat /share/pascal/Users/gz2294/Data/DMS/Hsu_NBT/useful.data.human.csv)
do
  sed -i "s|ClinVar.HGMD.PrimateAI.syn|MAVEDB/"$gene"|g" $1/$gene.yaml
  # add a new line that specifies "convert_to_onesite: true"
  echo "convert_to_onesite: true" >> $1/$gene.yaml
  # change num steps update
  sed -i "s|num_steps_update: "$num_steps_update"|num_steps_update: 1|g" $1/$gene.yaml
  # don't add contrastive loss because they are from same protein
  sed -i "s|contrastive_loss_fn: cosin_contrastive_loss|contrastive_loss_fn: null|g" $1/$gene.yaml
  # change output model to regression
  if [[ "PTEN.bin" == "$gene" ]]; then
      sed -i "s|loss_fn: mse_loss|loss_fn: weighted_loss_betabinomial|g" $1/$gene.yaml
  else
      sed -i "s|BinaryClassification|Regression|g" $1/$gene.yaml
      sed -i "s|loss_fn: mse_loss|loss_fn: gaussian_loss|g" $1/$gene.yaml
  fi
  
done

for gene in PTEN PTEN.bin NUDT15 VKORC1
do
  sed -i "s|output_dim: 1|output_dim: 2|g" $1/$gene.yaml
done

for gene in CCR5 CXCR4
do
  sed -i "s|output_dim: 1|output_dim: 3|g" $1/$gene.yaml
done

for gene in PTEN CCR5 CXCR4
do
  sed -i "s|num_epochs: "$num_epochs"|num_epochs: "$(($num_epochs))"|g" $1/$gene.yaml
done

for gene in $(cat scripts/pfams.txt) $(cat scripts/gene.pfams.txt)
do
  sed -i "s|ClinVar.HGMD.PrimateAI.syn|Itan.CKB.Cancer.good.batch/pfams.0.8/"$gene"|g" $1/$gene.yaml
  # sed -i "s|BinaryClassification|Tanh|g" $1/$gene.yaml
  sed -i "s|train_size: 0.95|train_size: 0.75|g" $1/$gene.yaml
  sed -i "s|val_size: 0.05|val_size: 0.25|g" $1/$gene.yaml
  sed -i "s|output_dim: 1|output_dim: 1|g" $1/$gene.yaml
  sed -i "s|num_steps_update: "$num_steps_update"|num_steps_update: 1|g" $1/$gene.yaml
  # change loss function to combined_loss
  sed -i "s|loss_fn: mse_loss|loss_fn: weighted_loss_betabinomial|g" $1/$gene.yaml
done

# for genes in gene.pfams.txt, use pre-split data
for gene in $(cat scripts/gene.pfams.txt)
do
  sed -i 's|data_split_fn: ""|data_split_fn: _by_anno|g' $1/$gene.yaml
done

# for non human data, change learning rates and data split fn
for gene in AAV GB1 B_LAC fluorescence $(cat /share/pascal/Users/gz2294/Data/DMS/Hsu_NBT/useful.data.other.csv)
do
  sed -i "s|ClinVar.HGMD.PrimateAI.syn|"$gene"|g" $1/$gene.yaml
  # change to large learning rate
  sed -i "s|lr: "$half_lr"|lr: "$lr"|g" $1/$gene.yaml
  # # change to large batch size
  # sed -i "s|batch_size: 6|batch_size: 7|g" $1/$gene.yaml
  # data split fn
  sed -i 's|data_split_fn: ""|data_split_fn: _by_anno|g' $1/$gene.yaml
  # don't add contrastive loss because they are from same protein
  sed -i "s|contrastive_loss_fn: cosin_contrastive_loss|contrastive_loss_fn: null|g" $1/$gene.yaml
  # change output model to regression
  sed -i "s|BinaryClassification|Regression|g" $1/$gene.yaml
done

# change seed
for gene in PTEN PTEN.bin CCR5 CXCR4 NUDT15 VKORC1 DDX3X $(cat scripts/pfams.txt) $(cat scripts/gene.pfams.txt) AAV GB1 B_LAC fluorescence
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

# create PTEN replicates
mkdir $1/PTEN.replicates/
for replicate in {1..8}
do
	cp $1/PTEN/PTEN.seed.0.yaml $1/PTEN.replicates/PTEN.replicate.$replicate.yaml
	sed -i "s|output_dim: 2|output_dim: 1|g" $1/PTEN.replicates/PTEN.replicate.$replicate.yaml
	sed -i "s|PTEN|PTEN.replicate."$replicate"|g" $1/PTEN.replicates/PTEN.replicate.$replicate.yaml
done

mkdir $1/PTEN.replicates.mean/
for replicate in {1..8}
do
        cp $1/PTEN/PTEN.seed.0.yaml $1/PTEN.replicates.mean/PTEN.replicate.mean.$replicate.yaml
        sed -i "s|output_dim: 2|output_dim: 1|g" $1/PTEN.replicates.mean/PTEN.replicate.mean.$replicate.yaml
        sed -i "s|PTEN|PTEN.replicate.mean."$replicate"|g" $1/PTEN.replicates.mean/PTEN.replicate.mean.$replicate.yaml
done

mkdir $1/PTEN.replicates.rest/
for replicate in {1..8}
do
        cp $1/PTEN/PTEN.seed.0.yaml $1/PTEN.replicates.rest/PTEN.replicate.rest.$replicate.yaml
        sed -i "s|output_dim: 2|output_dim: 1|g" $1/PTEN.replicates.rest/PTEN.replicate.rest.$replicate.yaml
        sed -i "s|PTEN|PTEN.replicate.rest."$replicate"|g" $1/PTEN.replicates.rest/PTEN.replicate.rest.$replicate.yaml
done

bash scripts/DMS.subset.prepare.yaml.sh $1

# run IonChannel and ICC with five fold cross validation
for gene in $(cat scripts/pfams.txt) $(cat scripts/gene.pfams.txt)
do
  mkdir $1/$gene.5fold/
  cp $1/$gene/$gene.seed.0.yaml $1/$gene.5fold/$gene.fold.0.yaml
  for fold in {1..4}
  do
    cp $1/$gene.5fold/$gene.fold.0.yaml $1/$gene.5fold/$gene.fold.$fold.yaml
    sed -i "s|pfams.0.8|pfams.0.8.seed."$fold"|g" $1/$gene.5fold/$gene.fold.$fold.yaml
    sed -i "s|TL."$gene".seed.0|TL."$gene".seed.0.fold."$fold"|g" $1/$gene.5fold/$gene.fold.$fold.yaml
  done
done

# run ICC genes with five fold cross validation in subsets
for gene in $(cat scripts/pfams.txt | grep -v 'uniprot' | grep -v 'PF' | grep -v 'Ion')
do
  for subset in {1..5}
  do
    mkdir $1/$gene.subset.$subset.5fold/
    cp -r $1/$gene.5fold/* $1/$gene.subset.$subset.5fold/
    for fold in {0..4}
    do
      sed -i "s|"$gene"|"$gene".subset."$subset"|g" $1/$gene.subset.$subset.5fold/$gene.fold.$fold.yaml
      sed -i 's|data_split_fn: ""|data_split_fn: _by_anno|g' $1/$gene.subset.$subset.5fold/$gene.fold.$fold.yaml
      mv $1/$gene.subset.$subset.5fold/$gene.fold.$fold.yaml $1/$gene.subset.$subset.5fold/$gene.subset.$subset.fold.$fold.yaml
    done
  done
done

# run DMS with five fold cross validation
for gene in PTEN PTEN.bin NUDT15 CCR5 CXCR4 VKORC1
do
  mkdir $1/$gene.5fold/
  cp $1/$gene/$gene.seed.0.yaml $1/$gene.5fold/$gene.fold.0.yaml
  for fold in {1..4}
  do
    cp $1/$gene.5fold/$gene.fold.0.yaml $1/$gene.5fold/$gene.fold.$fold.yaml
    sed -i "s|training.csv|train.seed."$fold".csv|g" $1/$gene.5fold/$gene.fold.$fold.yaml
    sed -i "s|testing.csv|test.seed."$fold".csv|g" $1/$gene.5fold/$gene.fold.$fold.yaml
    sed -i "s|TL."$gene".seed.0|TL."$gene".seed.0.fold."$fold"|g" $1/$gene.5fold/$gene.fold.$fold.yaml
  done
done

# run Hsu with five fold cross validation
for gene in $(cat /share/pascal/Users/gz2294/Data/DMS/Hsu_NBT/useful.data.csv)
do
  mkdir $1/$gene.5fold/
  mv $1/$gene.yaml $1/$gene.5fold/
  for fold in {0..4}
  do
    cp $1/$gene.5fold/$gene.yaml $1/$gene.5fold/$gene.fold.$fold.yaml
    sed -i "s|training.csv|train.seed."$fold".csv|g" $1/$gene.5fold/$gene.fold.$fold.yaml
    sed -i "s|testing.csv|test.seed."$fold".csv|g" $1/$gene.5fold/$gene.fold.$fold.yaml
    sed -i "s|TL."$gene".seed.0|TL."$gene".seed.0.fold."$fold"|g" $1/$gene.5fold/$gene.fold.$fold.yaml
  done
done

if [ $changed_data = true ]; then
  echo "change data-file-train back to original yaml"
  mv $1/pretrain.seed.0.yaml.bak $1/pretrain.seed.0.yaml
  #sed -i "s|ClinVar.HGMD.PrimateAI.syn/|ClinVar.HGMD.PrimateAI.syn/"$data_prefix"/|g" $1/pretrain.seed.0.yaml
fi
