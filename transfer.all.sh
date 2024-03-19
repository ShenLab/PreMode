export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup bash scripts/DMS.5fold.run.sh $1 P15056,P21802 0 > euler.log.0.txt &
nohup bash scripts/DMS.5fold.run.sh $1 P07949 1 > euler.log.1.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q09428 2 > euler.log.2.txt &
nohup bash scripts/DMS.5fold.run.sh $1 O00555 3 > euler.log.3.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q14654 4 > euler.log.4.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q99250 5 > euler.log.5.txt &
nohup bash scripts/DMS.5fold.run.sh $1 Q14524 6 > euler.log.6.txt &
nohup bash scripts/DMS.5fold.run.sh $1 P04637 7 > euler.log.7.txt &
