# PreMode
This is the repository for our manuscript "PreMode predicts mode-of-action of missense variants by deep graph representation learning of protein sequence and structural context" posted on bioRxiv: https://www.biorxiv.org/content/10.1101/2024.02.20.581321v2

# Data
Please use the git lfs to download all files in `data.files/` folder

Unzip the files with this script: `cd data.files/; bash unzip.files.sh; cd ../`

# Run
`python train.py --conf CONFIG.yaml`
All config files were stored in `scripts/` folder.

Here is the list of models in our manuscript:

`scripts/PreMode/` PreMode (Note previously we used torch_lightning to control the random seeds, now we used numpy.seed and torch.seed, there might be a slight difference), it takes 250 GB RAM and 4 A40 Nvidia GPUs to run, will finish in ~50h.

`scripts/ESM.LR/` Baseline Model, ESM2 (650M) + Single Layer Perceptron

`scripts/PreMode.large.window/` PreMode, window size set to 1251 AA.

`scripts/PreMode.noESM/`  PreMode, replace the ESM2 embeddings to one hot encodings of 20 AA.

`scripts/PreMode.noMSA/`  PreMode, remove the MSA input.

`scripts/PreMode.noPretrain/` PreMode, but didn't pretrain on ClinVar/HGMD.

`scripts/PreMode.noStructure/` PreMode, remove the AF2 predicted structure input.

`scripts/PreMode.ptm/` PreMode, add the onehot encoding of post transcriptional modification sites as input.

`scripts/PreMode.mean.var/` PreMode, it will output both predicted value (mean) and confidence (var), used in adaptive learning tasks.

# New Experiment
1. Please prepare a folder under `scripts/` and create a file named `pretrain.seed.0.yaml` inside the folder:
2. Run pretrain in pathogenicity task: `python train.py --conf scripts/NEW_FOLDER/pretrain.seed.0.yaml`
3. Prepare transfer learning config files: `bash scripts/DMS.prepare.yaml.sh scripts/NEW_FOLDER/`
4. Run transfer learning: `bash scripts/DMS.5fold.run.sh scripts/NEW_FOLDER TASK_NAME GPU_ID`. To reuse the transfer learning tasks in our paper using 8 GPU cards, just do `bash transfer.all.sh scripts/NEW_FOLDER`. If you only have one GPU card, then do `bash transfer.all.in.one.card.sh scripts/NEW_FOLDER`
5. Save inference results: `bash scripts/DMS.5fold.run.sh scripts/NEW_FOLDER TASK_NAME GPU_ID`
5. Test AUCs for genetic (GoF/LoF) mode of action tasks: `Rscript analysis/02.02.ICC.folders.compare.R scripts/NEW_FOLDER/`

