export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup bash scripts/DMS.5fold.run.sh $1 P15056,P21802 0 > $2.log.0.txt &
nohup bash scripts/DMS.5fold.run.sh $1 P07949,P04637,Q14524 1 > $2.log.1.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q09428,O00555 2 > $2.log.2.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q14654,Q99250 3 > $2.log.3.txt &
