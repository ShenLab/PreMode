rm -r scripts/PreMode/ProteinGym.modulo.5/$1.5fold
cp -r scripts/PreMode/PTEN/ scripts/PreMode/ProteinGym.modulo.5/$1.5fold
for s in {0..4};
do
	sed -i "s|data_file_train: ./data.files/PTEN/training.csv|data_file_train: /share/vault/Users/gz2294/Data/DMS/ProteinGym/cv_folds_substitutions_singles/fold_modulo_5/$1/fold.$s/training.csv|g" scripts/PreMode/ProteinGym.modulo.5/$1.5fold/PTEN.seed.$s.yaml
	sed -i "s|data_file_train_ddp_prefix: ./data.files/PTEN/training|data_file_train_ddp_prefix: /share/vault/Users/gz2294/Data/DMS/ProteinGym/cv_folds_substitutions_singles/fold_modulo_5/$1/fold.$s/training|g" scripts/PreMode/ProteinGym.modulo.5/$1.5fold/PTEN.seed.$s.yaml
	sed -i "s|data_file_test: ./data.files/PTEN/testing.csv|data_file_test: /share/vault/Users/gz2294/Data/DMS/ProteinGym/cv_folds_substitutions_singles/fold_modulo_5/$1/fold.$s/testing.csv|g" scripts/PreMode/ProteinGym.modulo.5/$1.5fold/PTEN.seed.$s.yaml
	sed -i "s|log_dir: ./PreMode.results/PreMode/TL.PTEN.seed.$s/|log_dir: ./PreMode.results/PreMode/ProteinGym.modulo.5/TL.$1.fold.$s/|g" scripts/PreMode/ProteinGym.modulo.5/$1.5fold/PTEN.seed.$s.yaml
	sed -i "s|output_dim: 2|output_dim: 1|g" scripts/PreMode/ProteinGym.modulo.5/$1.5fold/PTEN.seed.$s.yaml
	mv scripts/PreMode/ProteinGym.modulo.5/$1.5fold/PTEN.seed.$s.yaml scripts/PreMode/ProteinGym.modulo.5/$1.5fold/$1.fold.$s.yaml
done
