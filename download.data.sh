# download files from huggingface
mkdir -p huggingface
huggingface-cli download gzhong/PreMode.Data --local-dir huggingface --local-dir-use-symlinks False
# move files to data.files/
mv huggingface/data.files/* data.files/
mv huggingface/analysis/* analysis/
# untar the files for analysis/results.tar
cd analysis/
tar -xvf results.tar
rm results.tar
cd ../
# unzip files in the data.files/ folder
cd data.files/
cat af2.files.tgz.part-* | tar -xzvf -
cat esm.files.tgz.part-* | tar -xzvf -
cat esm.MSA.tgz.part-* | tar -xzvf -
cat gMVP.MSA.tgz.part-* | tar -xzvf -
cat pretrain.tgz.part-* | tar -xzvf -
cat MSA.tgz.part-* | tar -xzvf -
for tgzfile in `ls *.tgz`; do echo "Unzipping $tgzfile"; tar -xzvf $tgzfile; done
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
for gzfiles in `ls analysis/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
for gzfiles in `ls analysis/*/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
# unzip files in the PreMode.results/ folder
for gzfiles in `ls PreMode.results/PreMode.mean.var/*/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
