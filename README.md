# PreMode
This is the repository for our manuscript "Predicting mode of action of variants by graph representation of protein structural context" that submit to NeurIPS 2023.

# Data
Please unzip `MAVE.tgz`, `ICC.tgz`, `pathogenicity.tgz` and put them under your data directory.

For use of `MAVE.tgz`, please cite the MAVEDB.
@article{RN36,
   author = {Esposito, D. and Weile, J. and Shendure, J. and Starita, L. M. and Papenfuss, A. T. and Roth, F. P. and Fowler, D. M. and Rubin, A. F.},
   title = {MaveDB: an open-source platform to distribute and interpret data from multiplexed assays of variant effect},
   journal = {Genome Biol},
   volume = {20},
   number = {1},
   pages = {223},
   ISSN = {1474-760X (Electronic)
1474-7596 (Linking)},
   DOI = {10.1186/s13059-019-1845-6},
   url = {https://www.ncbi.nlm.nih.gov/pubmed/31679514},
   year = {2019},
   type = {Journal Article}
}

For use of `ICC.tgz`, please cite the original papers.

For use of `pathogenicity.tgz`, please cite PrimatAI.


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
