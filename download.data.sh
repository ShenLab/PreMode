# download files from huggingface
mkdir -p to_huggingface
huggingface-cli download ghong/PreMode.Data --local-dir to_huggingface --local-dir-use-symlinks False
# move files to data.files/
mv to_huggingface/data.files/* data.files/
mv to_huggingface/analysis/* analysis/
# untar the files for analysis/results.tar
cd analysis/
tar -xvf results.tar
rm results.tar
cd ../
# unzip files in the data.files/ folder
cd data.files/
cat esm.files.tgz.part-* | tar -xzvf -
cat esm.MSA.tgz.part-* | tar -xzvf -
cat gMVP.MSA.tgz.part-* | tar -xzvf -
cat pretrain.tgz.part-* | tar -xzvf -
for tgzfile in `ls *.tgz`; do echo "Unzipping $tgzfile"; tar -xzvf $tgzfile; done
for gzfile in `ls pretrain/*.gz`; do echo "Unzipping $gzfile"; gunzip $gzfile; done
cd ../
# unzip files in the parse.input.table/ folder
cd parse.input.table/
cat swissprot_and_human.full.seq.csv.tgz.part-* | tar -xzvf -
cd ../
# unzip files in the analysis/5genes.all.mut/inference.results/ folder
for gzfiles in `ls analysis/5genes.all.mut/PreMode/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
# unzip files in the analysis/5genes.all.mut/ folder
for gzfiles in `ls analysis/5genes.all.mut/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
# unzip files in the analysis/ folder
for gzfiles in `ls analysis/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
# unzip files in the analysis/*/ folder
for gzfiles in `ls analysis/PreMode/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/PreMode.all/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/PreMode.noESM/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/PreMode.noMSA/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/PreMode.noPretrain/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/PreMode.noStructure/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/PreMode.ptm/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/ESM.SLP/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/esm2.inference/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/figs/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/funNCion/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
# unzip files in the PreMode.results/ folder
for gzfiles in `ls PreMode.results/PreMode.mean.var/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
