# unzip files in the data.files/ folder
cd data.files/
cat esm.files.tgz.part-* | tar -xzvf -
cat esm.MSA.tgz.part-* | tar -xzvf -
cat gMVP.MSA.tgz.part-* | tar -xzvf -
for tgzfile in `ls *.tgz`; do echo "Unzipping $tgzfile"; tar -xzvf $tgzfile; done
for gzfile in `ls pretrain/*.gz`; do echo "Unzipping $gzfile"; gunzip $gzfile; done
cd ../
# unzip files in the analysis/5genes.all.mut/inference.results/ folder
for gzfiles in `ls analysis/5genes.all.mut/PreMOde/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
# unzip files in the analysis/5genes.all.mut/ folder
for gzfiles in `ls analysis/5genes.all.mut/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
# unzip files in the analysis/ folder
for gzfiles in `ls analysis/*.gz`; do echo "Unzipping $gzfiles"; gunzip $gzfiles; done
# unzip files in the PreMode.results/ folder
cd PreMode.results/
cat PreMode.mean.var.seed.0.tgz.part-* | tar -xzvf -
for tgzfile in `ls *.tgz`; do echo "Unzipping $tgzfile"; tar -xzvf $tgzfile; done
cd ../
# unzip files in the analysis/ folder
cd analysis/
for tgzfile in `ls *.tgz`; do echo "Unzipping $tgzfile"; tar -xzvf $tgzfile; done
cd ../
