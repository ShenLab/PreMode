# PreMode
This is the repository for our manuscript "Predicting mode of action of variants by graph representation of protein structural context" that submit to NeurIPS 2023 and under review.

# Data
Please unzip `MAVE.tgz`, `ICC.tgz`, `pathogenicity.tgz` and put them under your data directory.

For use of `MAVE.tgz`, we curated from MAVEDB. (DOI: 10.1186/s13059-019-1845-6)

For use of `ICC.tgz`, we curated from these papers and please cite us and them. (DOI: 10.1016/j.ajhg.2021.10.007; 10.1126/scitranslmed.aay6848; 10.1158/2159-8290.CD-17-0321; 10.1126/scisignal.2004088; 10.1093/nar/gky1015; 10.1089/gtmb.2010.0036; 10.1016/j.tibs.2019.03.009)

For use of `pathogenicity.tgz`, please cite PrimatAI. (DOI: 10.1038/s41588-018-0167-z)


# Run
`python train.py --conf CONFIG.yaml`
All config files were stored in `scripts/` folder.

Here is the list of models in our manuscript:

`scripts/CHPs.v1.SAGPool.ct/` PreMode

`scripts/CHPs.v1.noGraph.SAGPool.ct/` PreMode-1D

`scripts/CHPs.v1.SAGPool.2KNN.ct.seed.0.yaml` PreMode-2KNN

`scripts/CHPs.v1.aa5dim.SAGPool.ct/`  PreMode-AAchem

`scripts/CHPs.v1.onehot.SAGPool.ct/`  PreMode-onehot

`scripts/CHPs.v1.swissprot.SAGPool.ct/` PreMode-onehot (parameters initialized from pretrain on swissprot)

`scripts/CHPs.v1.swissprot.noGraph.SAGPool.ct/` PreMode-onehot-1D (parameters initialized from pretrain on swissprot)

# New Experiment
1. Please prepare a folder under `scripts/` and create a file named `pretrain.seed.0.yaml` inside the folder:
2. Run training in pathogenicity task: `python train.py --conf scripts/NEW_FOLDER/pretrain.seed.0.yaml`
3. Prepare transfer learning config files: `bash scripts/DMS.prepare.yaml.sh scripts/NEW_FOLDER/`
4. Run transfer learning: `bash scripts/DMS.run.sh scripts/NEW_FOLDER TASK_NAME GPU_ID`
5. Plot results for GoF/LoF mode of action tasks: `Rscript ICC.test.AUC.by.step.R scripts/NEW_FOLDER/`
6. Plot results for partial Loss of function mode of action tasks: `Rscript MAVE.test.R.by.step.R scripts/NEW_FOLDER/`
