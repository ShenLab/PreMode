export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup bash scripts/DMS.5fold.run.sh $1 P15056,P21802,P07949,Q14524,Q14654 0 > $2.log.0.txt &
nohup bash scripts/DMS.5fold.run.sh $1 P04637,Q09428,IonChannel.chps.even.uniprotID 1 > $2.log.1.txt &
nohup bash scripts/DMS.5fold.run.sh $1 O00555,IPR000719.even.uniprotID 2 > $2.log.2.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q99250,IPR001806.even.uniprotID 3 > $2.log.3.txt &
