export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup bash scripts/DMS.5fold.run.sh $1 IonChannel.chps.even.uniprotID,PF07714.even.uniprotID,PF00130.even.uniprotID,PF02196.even.uniprotID,PF07679.even.uniprotID 0 > $2.log.0.txt &
nohup bash scripts/DMS.5fold.run.sh $1 PF00047.even.uniprotID,PF00028.even.uniprotID,PF17756.even.uniprotID,PF00520.even.uniprotID 1 > $2.log.1.txt &
nohup bash scripts/DMS.5fold.run.sh $1 PF01007.even.uniprotID,IPR000719.even.uniprotID,IPR001806.even.uniprotID,P15056,P21802,P07949,P04637 2 > $2.log.2.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q09428,O00555,Q14654,Q99250,Q14524,P15056.PF00130,P15056.PF07714 3 > $2.log.3.txt &
