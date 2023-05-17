# PreMode
This is the repository for our manuscript that submit to NeurIPS.

# Data
Please unzip `MAVE.tgz`, `ICC.tgz`, `pathogenicity.tgz` and put them under your data directory.
For use of `MAVE.tgz`, please cite the MAVEDB.
For use of `ICC.tgz`, please cite the original papers.
For use of `ICC.tgz`, please cite PrimatAI.


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

