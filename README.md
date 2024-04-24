# PreMode
This is the repository for our manuscript "PreMode predicts mode-of-action of missense variants by deep graph representation learning of protein sequence and structural context" posted on bioRxiv: https://www.biorxiv.org/content/10.1101/2024.02.20.581321v2

# Data
Please use the git lfs to download all files in `data.files/` folder

Unzip the files with this script: `bash unzip.files.sh`

Unfortunately we are not allowed to share the HGMD data, so in the `data.files/pretrain/training.*` files we removed all the pathogenic variants from HGMD (49218 in total). This might affect the plots of `analysis/figs/fig.sup.14.pdf` and `analysis/figs/fig.sup.15.pdf` if you re-run the R codes in `analysis/` folder.

We shared the trained weights of our models trained using HGMD instead. 

# Run
Please install the necessary packages using `mamba env create -f PreMode.yaml`

Then, inside the PreMode conda environment, run `python train.py --conf CONFIG.yaml`, where `CONFIG.yaml` is a config file stored in `scripts/` folder.

Here is the list of models in our manuscript:

`scripts/PreMode/` PreMode, it takes 250 GB RAM and 4 A40 Nvidia GPUs to run, will finish in ~50h.

`scripts/ESM.LR/` Baseline Model, ESM2 (650M) + Single Layer Perceptron

`scripts/PreMode.large.window/` PreMode, window size set to 1251 AA.

`scripts/PreMode.noESM/`  PreMode, replace the ESM2 embeddings to one hot encodings of 20 AA.

`scripts/PreMode.noMSA/`  PreMode, remove the MSA input.

`scripts/PreMode.noPretrain/` PreMode, but didn't pretrain on ClinVar/HGMD.

`scripts/PreMode.noStructure/` PreMode, remove the AF2 predicted structure input.

`scripts/PreMode.ptm/` PreMode, add the onehot encoding of post transcriptional modification sites as input.

`scripts/PreMode.mean.var/` PreMode, it will output both predicted value (mean) and confidence (var), used in adaptive learning tasks.

# Figures in our manuscript

Please install the necessary R packages using `mamba env create -f r4-base.yaml`

Please go to `analysis/` folder and run the corresponding R scripts.

# New Experiment (Start from scratch and use our G/LoF datasets)
1. Please prepare a folder under `scripts/` and create a file named `pretrain.seed.0.yaml` inside the folder, check `scripts/PreMode/pretrain.seed.0.yaml` for example. 
2. Run pretrain in pathogenicity task: `python train.py --conf scripts/NEW_FOLDER/pretrain.seed.0.yaml`
3. Prepare transfer learning config files: `bash scripts/DMS.prepare.yaml.sh scripts/NEW_FOLDER/`
4. Run transfer learning: `bash scripts/DMS.5fold.run.sh scripts/NEW_FOLDER TASK_NAME GPU_ID`. If you have multiple tasks, just separate each task by comma in the TASK_NAME, like "task_1,task_2,task_3". 
4. (Optional) To reuse the transfer learning tasks in our paper using 8 GPU cards, just do `bash transfer.all.sh scripts/NEW_FOLDER`. If you only have one GPU card, then do `bash transfer.all.in.one.card.sh scripts/NEW_FOLDER`
5. Save inference results: `bash scripts/DMS.5fold.inference.sh scripts/NEW_FOLDER analysis/NEW_FOLDER TASK_NAME GPU_ID`
6. You'll get a folder `analysis/NEW_FOLDER/TASK_NAME` with 5 `.csv` files, each file has 4 columns `logits.FOLD.[0-3]`. Each column represent the G/LoF prediction at one cross-validation (closer to 0 means more likely GoF, closer to 1 means more likely LoF). We suggest averaging the predictions at 4 columns. 

# New Experiment (Only transfer learning, user defined mode-of-action datasets)
1.  Prepare a `.csv` file for training and inference, there are two accepted formats: 
+ Format 1 (only for missense variants):
    | uniprotID | aaChg  | score | ENST |
    | :-: | :-: | :-: | :-: |
    | P15056 | p.V600E | 1 | ENST00000646891 |
    | P15056 | p.G446V | -1 | ENST00000646891 |
  + `uniprotID`: the uniprot ID of the protein.
  + `aaChg`: the amino acid change induced by missense variant.
  + `score`: 1 for GoF, -1 for LoF. For inference it is not required. For DMS, this could be experimental readouts. If you have multiplexed assays, you can change it to `score.1, score.2, score.3, ..., score.N`.
  + `ENST` (optional): the ensemble transcript ID that matched the uniprotID.
+ Format 2 (can be missense variant or multiple variants):
    | uniprotID | ref | alt | pos.orig | score | ENST | wt.orig | sequence.len.orig
    | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
    | P15056 | V | E | 600 | 1 | ENST00000646891 | ... | 766 |
    | P15056 | G | V | 446 | -1 | ENST00000646891 | ... | 766 |
    | P15056 | G;V | V;F | 446;471 | -1 | ENST00000646891 | ... | 766 |
  + `uniprotID`: the uniprot ID of the protein.
  + `ref`: the reference amino acid, if multiple variants, separated by ";".
  + `alt`: the alternative, if multiple variants, separated by ";" in the same order of "ref".
  + `pos.orig`: the amino acid change position, if multiple variants, separated by ";" in the same order of "ref".
  + `score`: same as above.
  + `ENST` (optional): same as above.
  + `wt.orig`: the wild type protein sequence, in the uniprot format.
  + `sequence.len.orig`: the wild type protein sequence length.

+ If you prepared your input in Format 1, please run `bash parse.input.table/parse.input.table.sh YOUR_FILE TRANSFORMED_FILE` to transform it to Format 2, note it will drop some lines if your aaChg doesn't match the corresponding alphafold sequence.
2.  Prepare a config file for training the model and inference. 
    ```
    bash scripts/prepare.new.task.yaml.sh PRETRAIN_MODEL_NAME YOUR_TASK_NAME YOUR_TRAINING_FILE YOUR_INFERENCE_FILE TASK_TYPE MODE_OF_ACTION_N
    ```
  + `PRETRAIN_MODEL_NAME` could be one of the following:
    + `scripts/PreMode/`: Default PreMode
    + `scripts/PreMode.ptm`: PreMode + ptm as input
    + `scripts/PreMode.noStructure`: PreMode without structure input
    + `scripts/PreMode.noESM`: PreMode without ESM input
    + `scripts/PreMode.noMSA`: PreMode without MSA input
    + `scripts/ESM.SLP`: ESM embedding + Single Layer Perceptron
  + `YOUR_TASK_NAME` can be anything on your preference
  + `YOUR_TRAINING_FILE` is the training file you prepared in step 1.
  + `YOUR_INFERENCE_FILE` is the inference file you prepared in step 1.
  + `TASK_TYPE` could be `DMS` or `GLOF`.
  + `MODE_OF_ACTION_N` The number of dimensions of mode-of-action. For `GLOF` this is usually 1. For multiplexed `DMS` dataset, this could be the number of biochemical properties measured. Note that if it is larger than 1, then you have to make sure the `score` column in step 1 is replaced to `score.1, score.2, ..., score.N` correspondingly.
  
3.  Run your config file
    ```
    bash scripts/run.new.task.sh PRETRAIN_MODEL_NAME YOUR_TASK_NAME OUTPUT_FOLDER GPU_ID 
    ```

4.  You'll get a file in the `OUTPUT_FOLDER` named as `YOUR_TASK_NAME.inference.result.csv`. 
  + If your `TASK_TYPE` is `GLOF`, then the column `logits` will be the inference results. Closer to 0 means GoF, Closer to 1 means LoF.
  + If your `TASK_TYPE` is `DMS` and `MODE_OF_ACTION_N` is 1, then the column `logits` will be the inference results. If your `MODE_OF_ACTION_N` is larger than 1, then you will get multiple columns of `logits.*`, each represent a predicted DMS measurement.
