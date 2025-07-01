rm -r scripts/PreMode/ProteinGym.random.5/$1.5fold
cp -r scripts/PreMode/PTEN/ scripts/PreMode/ProteinGym.random.5/$1.5fold
for s in {0..4};
do
	sed -i "s|data_file_train: ./data.files/PTEN/training.csv|data_file_train: /share/vault/Users/gz2294/Data/DMS/ProteinGym/cv_folds_substitutions_singles/fold_random_5/$1/fold.$s/training.csv|g" scripts/PreMode/ProteinGym.random.5/$1.5fold/PTEN.seed.$s.yaml
	sed -i "s|data_file_train_ddp_prefix: ./data.files/PTEN/training|data_file_train_ddp_prefix: /share/vault/Users/gz2294/Data/DMS/ProteinGym/cv_folds_substitutions_singles/fold_random_5/$1/fold.$s/training|g" scripts/PreMode/ProteinGym.random.5/$1.5fold/PTEN.seed.$s.yaml
	sed -i "s|data_file_test: ./data.files/PTEN/testing.csv|data_file_test: /share/vault/Users/gz2294/Data/DMS/ProteinGym/cv_folds_substitutions_singles/fold_random_5/$1/fold.$s/testing.csv|g" scripts/PreMode/ProteinGym.random.5/$1.5fold/PTEN.seed.$s.yaml
	sed -i "s|log_dir: ./PreMode.results/PreMode/TL.PTEN.seed.$s/|log_dir: ./PreMode.results/PreMode/ProteinGym.random.5/TL.$1.fold.$s/|g" scripts/PreMode/ProteinGym.random.5/$1.5fold/PTEN.seed.$s.yaml
	sed -i "s|output_dim: 2|output_dim: 1|g" scripts/PreMode/ProteinGym.random.5/$1.5fold/PTEN.seed.$s.yaml
	# change the num epochs based on number of updates
	ntrain=$(wc -l /share/vault/Users/gz2294/Data/DMS/ProteinGym/cv_folds_substitutions_singles/fold_random_5/$1/fold.$s/training.csv | awk '{print $1}')
  nsteps=$(expr $ntrain / 8)
  if [ "$nsteps" -le 1 ]; then
    nsteps=1
  fi
  nepochs=$(expr 25000 / $nsteps)
  if [ "$nepochs" -lt 40 ]; then
    nepochs=40
  fi
  echo $nepochs
  sed -i "s|num_epochs: 40$|num_epochs: $nepochs|g" scripts/PreMode/ProteinGym.random.5/$1.5fold/PTEN.seed.$s.yaml
	mv scripts/PreMode/ProteinGym.random.5/$1.5fold/PTEN.seed.$s.yaml scripts/PreMode/ProteinGym.random.5/$1.5fold/$1.fold.$s.yaml
done
