export CUDA_VISIBLE_DEVICES=4,5,6,7
nohup bash scripts/DMS.5fold.run.sh $1 P15056,P21802,P07949,Q14524,Q14654 4 > $2.log.4.txt &
nohup bash scripts/DMS.5fold.run.sh $1 P04637,Q09428,IonChannel.chps.even.uniprotID 5 > $2.log.5.txt &
nohup bash scripts/DMS.5fold.run.sh $1 O00555,IPR000719.even.uniprotID 6 > $2.log.6.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q99250,IPR001806.even.uniprotID 7 > $2.log.7.txt &
